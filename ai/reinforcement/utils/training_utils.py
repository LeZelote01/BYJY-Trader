"""
Training Utilities for RL Agents

Utilities for training, evaluating, and managing RL trading agents.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Callable
import logging
import json
from pathlib import Path
from datetime import datetime
import time
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from core.logger import get_logger

logger = get_logger(__name__)

class RLTrainer:
    """
    RL Training Manager
    
    Manages the training process for RL agents including:
    - Training loop management
    - Performance monitoring
    - Model checkpointing
    - Training visualization
    """
    
    def __init__(self,
                 agent,
                 environment,
                 reward_function=None,
                 save_dir: str = "ai/trained_models/rl_agents",
                 checkpoint_frequency: int = 100,
                 evaluation_frequency: int = 50,
                 max_episodes: int = 2000):
        """
        Initialize RL Trainer
        
        Args:
            agent: RL agent to train
            environment: Training environment
            reward_function: Custom reward function (optional)
            save_dir: Directory to save models and logs
            checkpoint_frequency: Frequency of model checkpoints
            evaluation_frequency: Frequency of evaluations
            max_episodes: Maximum training episodes
        """
        self.agent = agent
        self.environment = environment
        self.reward_function = reward_function
        self.save_dir = Path(save_dir)
        self.checkpoint_frequency = checkpoint_frequency
        self.evaluation_frequency = evaluation_frequency
        self.max_episodes = max_episodes
        
        # Training state
        self.current_episode = 0
        self.training_start_time = None
        self.best_performance = -np.inf
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.evaluation_scores = []
        self.training_losses = []
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"RLTrainer initialized for {agent.name}")
    
    def train(self, verbose: bool = True) -> Dict:
        """
        Train the RL agent
        
        Args:
            verbose: Whether to print training progress
            
        Returns:
            Training summary
        """
        self.training_start_time = time.time()
        convergence_detected = False
        
        logger.info(f"Starting training for {self.max_episodes} episodes")
        
        for episode in range(self.max_episodes):
            self.current_episode = episode
            
            # Run episode
            episode_reward, episode_length = self._run_episode()
            
            # Update agent statistics
            self.agent.update_stats(episode_reward, episode_length)
            
            # Store training data
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Periodic evaluation
            if episode % self.evaluation_frequency == 0 and episode > 0:
                eval_score = self._evaluate_agent()
                self.evaluation_scores.append(eval_score)
                
                # Update best performance
                if eval_score > self.best_performance:
                    self.best_performance = eval_score
                    self._save_best_model()
                
                if verbose:
                    logger.info(f"Episode {episode}: Reward={episode_reward:.2f}, "
                               f"Eval Score={eval_score:.2f}, Best={self.best_performance:.2f}")
            
            # Checkpoint
            if episode % self.checkpoint_frequency == 0 and episode > 0:
                self._save_checkpoint()
            
            # Check convergence
            if not convergence_detected and self.agent.check_convergence():
                convergence_detected = True
                logger.info(f"Convergence detected at episode {episode}")
                
                # Save converged model
                self.agent.save_model(str(self.save_dir / f"{self.agent.name}_converged.pth"))
                
                # Continue training for some episodes to confirm stability
                if episode < self.max_episodes - 200:
                    logger.info("Continuing training to confirm convergence stability")
                else:
                    logger.info("Early stopping due to convergence")
                    break
            
            # Early stopping if performance degrades severely
            if episode > 500 and len(self.episode_rewards) >= 100:
                recent_performance = np.mean(self.episode_rewards[-100:])
                if recent_performance < -1000:  # Severe performance degradation
                    logger.warning("Early stopping due to severe performance degradation")
                    break
        
        # Training completed
        training_time = time.time() - self.training_start_time
        
        # Final evaluation
        final_eval_score = self._evaluate_agent(num_episodes=20)
        
        # Save final model
        self.agent.save_model(str(self.save_dir / f"{self.agent.name}_final.pth"))
        
        # Generate training summary
        summary = self._generate_training_summary(training_time, final_eval_score, convergence_detected)
        
        # Save training summary
        with open(self.save_dir / f"{self.agent.name}_training_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Training completed: {summary['total_episodes']} episodes in {training_time:.1f}s")
        
        return summary
    
    def _run_episode(self) -> Tuple[float, int]:
        """
        Run single training episode
        
        Returns:
            Tuple of (episode_reward, episode_length)
        """
        observation = self.environment.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        while not done and episode_length < 5000:  # Max episode length
            # Select action
            action = self.agent.select_action(observation, training=True)
            
            # Take environment step
            next_observation, reward, done, info = self.environment.step(action)
            
            # Apply custom reward function if provided
            if self.reward_function:
                portfolio_info = info.copy()
                portfolio_info['value'] = info.get('portfolio_value', 0)
                market_info = {'price': info.get('price', 0), 'volatility': 0.02}
                reward = self.reward_function.calculate_reward(portfolio_info, market_info)
            
            # Store transition (for agents that use experience replay)
            if hasattr(self.agent, 'store_transition'):
                self.agent.store_transition(observation, action, reward, next_observation, done)
            
            # Update observation
            observation = next_observation
            episode_reward += reward
            episode_length += 1
            
            # Agent update (for online learning agents)
            if hasattr(self.agent, 'update') and episode_length % 20 == 0:
                update_info = self.agent.update()
                if update_info.get('total_loss'):
                    self.training_losses.append(update_info['total_loss'])
        
        return episode_reward, episode_length
    
    def _evaluate_agent(self, num_episodes: int = 5) -> float:
        """
        Evaluate agent performance
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            Average evaluation score
        """
        eval_rewards = []
        
        for _ in range(num_episodes):
            observation = self.environment.reset()
            episode_reward = 0.0
            done = False
            steps = 0
            
            while not done and steps < 2000:
                action = self.agent.select_action(observation, training=False)
                observation, reward, done, info = self.environment.step(action)
                episode_reward += reward
                steps += 1
            
            eval_rewards.append(episode_reward)
        
        return np.mean(eval_rewards)
    
    def _save_checkpoint(self):
        """Save training checkpoint"""
        checkpoint_path = self.save_dir / f"{self.agent.name}_checkpoint_ep{self.current_episode}.pth"
        self.agent.save_model(str(checkpoint_path))
        
        # Save training state
        training_state = {
            'episode': self.current_episode,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'evaluation_scores': self.evaluation_scores,
            'best_performance': self.best_performance
        }
        
        state_path = self.save_dir / f"{self.agent.name}_training_state_ep{self.current_episode}.json"
        with open(state_path, 'w') as f:
            json.dump(training_state, f, indent=2)
        
        logger.debug(f"Checkpoint saved at episode {self.current_episode}")
    
    def _save_best_model(self):
        """Save best performing model"""
        best_path = self.save_dir / f"{self.agent.name}_best_performance.pth"
        self.agent.save_model(str(best_path))
        logger.debug(f"New best model saved with score {self.best_performance:.2f}")
    
    def _generate_training_summary(self, training_time: float, final_eval_score: float, converged: bool) -> Dict:
        """Generate comprehensive training summary"""
        summary = {
            'agent_name': self.agent.name,
            'training_start': str(self.training_start_time),
            'training_duration_seconds': training_time,
            'total_episodes': len(self.episode_rewards),
            'converged': converged,
            'convergence_episode': self.agent.best_episode if converged else None,
            
            # Performance metrics
            'final_evaluation_score': final_eval_score,
            'best_performance': self.best_performance,
            'mean_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'std_episode_reward': np.std(self.episode_rewards) if self.episode_rewards else 0,
            'mean_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            
            # Learning progress
            'initial_100_episodes_avg': np.mean(self.episode_rewards[:100]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards),
            'final_100_episodes_avg': np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards),
            'improvement': (np.mean(self.episode_rewards[-100:]) - np.mean(self.episode_rewards[:100])) if len(self.episode_rewards) >= 200 else 0,
            
            # Training efficiency  
            'episodes_to_convergence': self.agent.best_episode if converged else self.max_episodes,
            'convergence_efficiency': (self.agent.best_episode / self.max_episodes) if converged else 1.0,
            
            # Additional metrics
            'evaluation_scores': self.evaluation_scores,
            'agent_stats': self.agent.get_stats() if hasattr(self.agent, 'get_stats') else {}
        }
        
        return summary
    
    def plot_training_progress(self, save_path: Optional[str] = None, show: bool = True):
        """
        Plot training progress
        
        Args:
            save_path: Path to save plot
            show: Whether to show plot
        """
        if not self.episode_rewards:
            logger.warning("No training data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards, alpha=0.7, color='blue')
        if len(self.episode_rewards) >= 50:
            # Rolling average
            rolling_avg = pd.Series(self.episode_rewards).rolling(50).mean()
            axes[0, 0].plot(rolling_avg, color='red', linewidth=2, label='50-episode average')
            axes[0, 0].legend()
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Episode lengths
        axes[0, 1].plot(self.episode_lengths, alpha=0.7, color='green')
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Evaluation scores
        if self.evaluation_scores:
            eval_episodes = list(range(0, len(self.evaluation_scores) * self.evaluation_frequency, self.evaluation_frequency))
            axes[1, 0].plot(eval_episodes, self.evaluation_scores, 'o-', color='purple')
            axes[1, 0].set_title('Evaluation Scores')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Evaluation Score')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Training losses
        if self.training_losses:
            axes[1, 1].plot(self.training_losses, alpha=0.7, color='orange')
            axes[1, 1].set_title('Training Losses')
            axes[1, 1].set_xlabel('Update Step')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training progress plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def get_training_metrics(self) -> Dict:
        """Get current training metrics"""
        if not self.episode_rewards:
            return {}
        
        recent_rewards = self.episode_rewards[-50:] if len(self.episode_rewards) >= 50 else self.episode_rewards
        
        return {
            'total_episodes': len(self.episode_rewards),
            'current_performance': np.mean(recent_rewards),
            'best_performance': self.best_performance,
            'improvement_rate': (recent_rewards[-1] - recent_rewards[0]) / len(recent_rewards) if len(recent_rewards) > 1 else 0,
            'convergence_progress': min(self.agent.episode_count / 2000, 1.0),  # Progress toward convergence threshold
            'training_stability': 1.0 / (np.std(recent_rewards) + 1e-8),
            'last_evaluation_score': self.evaluation_scores[-1] if self.evaluation_scores else 0
        }


def create_training_environment(symbol: str = "BTCUSDT", 
                              initial_balance: float = 10000.0,
                              **kwargs):
    """
    Create training environment for RL agents
    
    Args:
        symbol: Trading symbol
        initial_balance: Initial balance
        **kwargs: Additional environment parameters
        
    Returns:
        Configured trading environment
    """
    from ..environment.trading_env import TradingEnvironment
    
    return TradingEnvironment(
        symbol=symbol,
        initial_balance=initial_balance,
        **kwargs
    )


def create_reward_function(reward_type: str = "profit_risk", **kwargs):
    """
    Create reward function for RL training
    
    Args:
        reward_type: Type of reward function
        **kwargs: Reward function parameters
        
    Returns:
        Configured reward function
    """
    from ..rewards.reward_functions import (
        ProfitRiskReward, SharpeReward, MaxDrawdownReward, 
        MultiObjectiveReward, CalmarReward
    )
    
    reward_functions = {
        'profit_risk': ProfitRiskReward,
        'sharpe': SharpeReward,
        'max_drawdown': MaxDrawdownReward,
        'multi_objective': MultiObjectiveReward,
        'calmar': CalmarReward
    }
    
    if reward_type not in reward_functions:
        raise ValueError(f"Unknown reward type: {reward_type}")
    
    return reward_functions[reward_type](**kwargs)


def compare_agents(agents: List, environment, num_episodes: int = 50) -> Dict:
    """
    Compare performance of multiple RL agents
    
    Args:
        agents: List of RL agents to compare
        environment: Environment for testing
        num_episodes: Number of evaluation episodes per agent
        
    Returns:
        Comparison results
    """
    results = {}
    
    for agent in agents:
        logger.info(f"Evaluating agent: {agent.name}")
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            observation = environment.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False
            
            while not done and episode_length < 2000:
                action = agent.select_action(observation, training=False)
                observation, reward, done, info = environment.step(action)
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        # Calculate statistics
        results[agent.name] = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'win_rate': sum(1 for r in episode_rewards if r > 0) / len(episode_rewards),
            'sharpe_ratio': np.mean(episode_rewards) / (np.std(episode_rewards) + 1e-8),
            'all_rewards': episode_rewards
        }
    
    # Rank agents
    ranking = sorted(results.items(), key=lambda x: x[1]['mean_reward'], reverse=True)
    
    logger.info("Agent Comparison Results:")
    for i, (name, stats) in enumerate(ranking):
        logger.info(f"{i+1}. {name}: Mean Reward = {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    
    return {
        'results': results,
        'ranking': [name for name, _ in ranking]
    }


import pandas as pd  # Add this import

def hyperparameter_search(agent_class, 
                         environment, 
                         param_grid: Dict,
                         num_trials: int = 20,
                         episodes_per_trial: int = 500) -> Dict:
    """
    Perform hyperparameter search for RL agent
    
    Args:
        agent_class: RL agent class
        environment: Training environment
        param_grid: Dictionary of hyperparameter ranges
        num_trials: Number of trials
        episodes_per_trial: Episodes per trial
        
    Returns:
        Search results
    """
    import itertools
    import random
    
    logger.info(f"Starting hyperparameter search with {num_trials} trials")
    
    # Generate parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    all_combinations = list(itertools.product(*param_values))
    
    if len(all_combinations) > num_trials:
        # Random sample if too many combinations
        combinations = random.sample(all_combinations, num_trials)
    else:
        combinations = all_combinations
    
    results = []
    
    for i, params in enumerate(combinations):
        param_dict = dict(zip(param_names, params))
        
        logger.info(f"Trial {i+1}/{len(combinations)}: {param_dict}")
        
        try:
            # Create agent with parameters
            agent = agent_class(**param_dict)
            
            # Train agent
            trainer = RLTrainer(
                agent=agent,
                environment=environment,
                max_episodes=episodes_per_trial,
                checkpoint_frequency=episodes_per_trial,  # No checkpoints during search
                evaluation_frequency=episodes_per_trial//4
            )
            
            training_summary = trainer.train(verbose=False)
            
            # Store results
            result = {
                'trial': i + 1,
                'parameters': param_dict,
                'final_performance': training_summary['final_evaluation_score'],
                'mean_reward': training_summary['mean_episode_reward'],
                'convergence': training_summary['converged'],
                'episodes_to_convergence': training_summary.get('episodes_to_convergence', episodes_per_trial)
            }
            
            results.append(result)
            
            logger.info(f"Trial {i+1} completed: Performance = {result['final_performance']:.2f}")
            
        except Exception as e:
            logger.error(f"Trial {i+1} failed: {e}")
            continue
    
    # Analyze results
    if results:
        # Sort by performance
        results.sort(key=lambda x: x['final_performance'], reverse=True)
        
        best_params = results[0]['parameters']
        best_performance = results[0]['final_performance']
        
        logger.info(f"Best hyperparameters: {best_params}")
        logger.info(f"Best performance: {best_performance:.2f}")
        
        # Create summary
        summary = {
            'best_parameters': best_params,
            'best_performance': best_performance,
            'all_results': results,
            'num_trials': len(results),
            'search_space': param_grid
        }
        
        return summary
    else:
        logger.error("No successful trials in hyperparameter search")
        return {}