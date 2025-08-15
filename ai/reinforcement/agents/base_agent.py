"""
Base RL Agent Class

Abstract base class for all reinforcement learning agents
with common functionality and interface.
"""

import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import json

from core.logger import get_logger

logger = get_logger(__name__)

class BaseRLAgent(ABC):
    """
    Abstract Base Class for RL Agents
    
    Provides common functionality for all RL agents including:
    - Model saving/loading
    - Training statistics tracking
    - Performance evaluation
    - Configuration management
    """
    
    def __init__(self,
                 name: str,
                 observation_dim: int,
                 action_dim: int,
                 learning_rate: float = 3e-4,
                 device: str = 'cpu'):
        """
        Initialize Base RL Agent
        
        Args:
            name: Agent name
            observation_dim: Observation space dimension
            action_dim: Action space dimension
            learning_rate: Learning rate
            device: Computing device ('cpu' or 'cuda')
        """
        self.name = name
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.device = device
        
        # Training statistics
        self.episode_count = 0
        self.step_count = 0
        self.total_reward = 0.0
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        
        # Performance tracking
        self.best_reward = -np.inf
        self.best_episode = 0
        self.convergence_threshold = 1e-4
        self.convergence_episodes = 100
        
        # Model storage
        self.models_dir = Path(f"ai/trained_models/rl_agents/{self.name}")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized {self.name} agent with obs_dim={observation_dim}, action_dim={action_dim}")
    
    @abstractmethod
    def select_action(self, observation: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Select action given observation
        
        Args:
            observation: Current observation
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        pass
    
    @abstractmethod
    def update(self, experiences: Dict) -> Dict:
        """
        Update agent with experiences
        
        Args:
            experiences: Dictionary of experiences (states, actions, rewards, etc.)
            
        Returns:
            Dictionary of training metrics
        """
        pass
    
    @abstractmethod
    def save_model(self, filepath: str):
        """Save agent model"""
        pass
    
    @abstractmethod
    def load_model(self, filepath: str):
        """Load agent model"""
        pass
    
    def update_stats(self, episode_reward: float, episode_length: int, loss: float = None):
        """Update training statistics"""
        self.episode_count += 1
        self.total_reward += episode_reward
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        
        if loss is not None:
            self.losses.append(loss)
        
        # Update best performance
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            self.best_episode = self.episode_count
            
            # Save best model
            self.save_model(str(self.models_dir / f"{self.name}_best.pth"))
        
        # Keep only recent statistics
        max_history = 1000
        if len(self.episode_rewards) > max_history:
            self.episode_rewards = self.episode_rewards[-max_history:]
            self.episode_lengths = self.episode_lengths[-max_history:]
            if self.losses:
                self.losses = self.losses[-max_history:]
    
    def get_stats(self) -> Dict:
        """Get current training statistics"""
        if not self.episode_rewards:
            return {}
        
        recent_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) >= 100 else self.episode_rewards
        
        stats = {
            'episode_count': self.episode_count,
            'step_count': self.step_count,
            'mean_reward': np.mean(self.episode_rewards),
            'recent_mean_reward': np.mean(recent_rewards),
            'best_reward': self.best_reward,
            'best_episode': self.best_episode,
            'mean_episode_length': np.mean(self.episode_lengths),
            'std_reward': np.std(self.episode_rewards),
            'total_reward': self.total_reward
        }
        
        if self.losses:
            recent_losses = self.losses[-100:] if len(self.losses) >= 100 else self.losses
            stats['mean_loss'] = np.mean(self.losses)
            stats['recent_mean_loss'] = np.mean(recent_losses)
        
        return stats
    
    def check_convergence(self) -> bool:
        """Check if agent has converged"""
        if len(self.episode_rewards) < self.convergence_episodes:
            return False
        
        # Check if recent performance is stable
        recent_rewards = self.episode_rewards[-self.convergence_episodes:]
        reward_std = np.std(recent_rewards)
        reward_mean = np.mean(recent_rewards)
        
        # Convergence if coefficient of variation is below threshold
        cv = reward_std / (abs(reward_mean) + 1e-8)
        converged = cv < self.convergence_threshold
        
        if converged:
            logger.info(f"Agent {self.name} converged after {self.episode_count} episodes")
        
        return converged
    
    def save_config(self, config: Dict):
        """Save agent configuration"""
        config_path = self.models_dir / f"{self.name}_config.json"
        
        # Add training stats to config
        config['training_stats'] = self.get_stats()
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved agent config to {config_path}")
    
    def load_config(self) -> Dict:
        """Load agent configuration"""
        config_path = self.models_dir / f"{self.name}_config.json"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded agent config from {config_path}")
            return config
        
        return {}
    
    def reset_stats(self):
        """Reset training statistics"""
        self.episode_count = 0
        self.step_count = 0
        self.total_reward = 0.0
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.best_reward = -np.inf
        self.best_episode = 0
        
        logger.info(f"Reset statistics for agent {self.name}")
    
    def evaluate(self, env, num_episodes: int = 10) -> Dict:
        """
        Evaluate agent performance
        
        Args:
            env: Environment to evaluate on
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Evaluation metrics
        """
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            observation = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                action = self.select_action(observation, training=False)
                observation, reward, done, info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                if episode_length > 10000:  # Prevent infinite episodes
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            logger.debug(f"Evaluation episode {episode + 1}: reward={episode_reward:.2f}, length={episode_length}")
        
        eval_stats = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'episodes': num_episodes
        }
        
        logger.info(f"Evaluation completed: mean_reward={eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
        
        return eval_stats