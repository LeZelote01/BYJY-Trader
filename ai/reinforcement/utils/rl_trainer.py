"""
Main RL Training Script for Phase 3.3

Comprehensive training script for RL agents with full pipeline including
environment setup, agent training, evaluation, and model saving.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from core.logger import get_logger
from .training_utils import RLTrainer, create_training_environment, create_reward_function
from ..agents.ppo_agent import PPOAgent
from ..agents.a3c_agent import A3CAgent

logger = get_logger(__name__)

def train_ppo_agent(config: dict) -> dict:
    """
    Train PPO agent with given configuration
    
    Args:
        config: Training configuration
        
    Returns:
        Training results
    """
    logger.info("Starting PPO agent training")
    
    # Create environment
    env = create_training_environment(
        symbol=config.get('symbol', 'BTCUSDT'),
        initial_balance=config.get('initial_balance', 10000.0),
        lookback_window=config.get('lookback_window', 50)
    )
    
    # Create reward function
    reward_function = None
    if config.get('reward_function'):
        reward_function = create_reward_function(
            reward_type=config['reward_function'],
            **config.get('reward_params', {})
        )
    
    # Create PPO agent
    agent = PPOAgent(
        observation_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        learning_rate=config.get('learning_rate', 3e-4),
        hidden_dims=config.get('hidden_dims', [256, 256, 128]),
        clip_ratio=config.get('clip_ratio', 0.2),
        value_coef=config.get('value_coef', 0.5),
        entropy_coef=config.get('entropy_coef', 0.01),
        batch_size=config.get('batch_size', 64),
        epochs_per_update=config.get('epochs_per_update', 10)
    )
    
    # Create trainer
    trainer = RLTrainer(
        agent=agent,
        environment=env,
        reward_function=reward_function,
        save_dir=config.get('save_dir', 'ai/trained_models/rl_agents/ppo'),
        checkpoint_frequency=config.get('checkpoint_freq', 100),
        evaluation_frequency=config.get('eval_freq', 50),
        max_episodes=config.get('max_episodes', 2000)
    )
    
    # Train agent
    results = trainer.train(verbose=config.get('verbose', True))
    
    # Plot training progress
    if config.get('plot_results', True):
        plot_path = Path(config.get('save_dir', 'ai/trained_models/rl_agents/ppo')) / f"ppo_training_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        trainer.plot_training_progress(save_path=str(plot_path), show=False)
    
    logger.info("PPO agent training completed")
    return results

def train_a3c_agent(config: dict) -> dict:
    """
    Train A3C agent with given configuration
    
    Args:
        config: Training configuration
        
    Returns:
        Training results
    """
    logger.info("Starting A3C agent training")
    
    # Environment creator function
    def env_creator():
        return create_training_environment(
            symbol=config.get('symbol', 'BTCUSDT'),
            initial_balance=config.get('initial_balance', 10000.0),
            lookback_window=config.get('lookback_window', 50)
        )
    
    # Create sample environment for dimensions
    sample_env = env_creator()
    
    # Create A3C agent
    agent = A3CAgent(
        observation_dim=sample_env.observation_space.shape[0],
        action_dim=sample_env.action_space.shape[0],
        learning_rate=config.get('learning_rate', 3e-4),
        hidden_dims=config.get('hidden_dims', [256, 128]),
        num_workers=config.get('num_workers', 4),
        gamma=config.get('gamma', 0.99),
        entropy_coef=config.get('entropy_coef', 0.01),
        value_coef=config.get('value_coef', 0.5)
    )
    
    # Create workers
    agent.create_workers(env_creator)
    
    # Train in parallel
    max_episodes_per_worker = config.get('max_episodes', 2000) // config.get('num_workers', 4)
    agent.train_parallel(max_episodes_per_worker)
    
    # Save trained model
    save_dir = Path(config.get('save_dir', 'ai/trained_models/rl_agents/a3c'))
    save_dir.mkdir(parents=True, exist_ok=True)
    agent.save_model(str(save_dir / f"a3c_trained_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"))
    
    # Get training statistics
    results = agent.update()
    
    logger.info("A3C agent training completed")
    return results

def compare_agent_performance(config: dict) -> dict:
    """
    Compare performance of different RL agents
    
    Args:
        config: Comparison configuration
        
    Returns:
        Comparison results
    """
    from .training_utils import compare_agents
    
    logger.info("Starting agent comparison")
    
    # Create environment
    env = create_training_environment(
        symbol=config.get('symbol', 'BTCUSDT'),
        initial_balance=config.get('initial_balance', 10000.0),
        lookback_window=config.get('lookback_window', 50)
    )
    
    # Create agents to compare
    agents = []
    
    # PPO Agent
    if 'ppo' in config.get('agents', ['ppo', 'a3c']):
        ppo_agent = PPOAgent(
            observation_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            **config.get('ppo_params', {})
        )
        
        # Load trained model if available
        ppo_model_path = config.get('ppo_model_path')
        if ppo_model_path and Path(ppo_model_path).exists():
            ppo_agent.load_model(ppo_model_path)
            logger.info(f"Loaded PPO model from {ppo_model_path}")
        
        agents.append(ppo_agent)
    
    # A3C Agent
    if 'a3c' in config.get('agents', ['ppo', 'a3c']):
        a3c_agent = A3CAgent(
            observation_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            **config.get('a3c_params', {})
        )
        
        # Load trained model if available
        a3c_model_path = config.get('a3c_model_path')
        if a3c_model_path and Path(a3c_model_path).exists():
            a3c_agent.load_model(a3c_model_path)
            logger.info(f"Loaded A3C model from {a3c_model_path}")
        
        agents.append(a3c_agent)
    
    if not agents:
        raise ValueError("No agents specified for comparison")
    
    # Compare agents
    results = compare_agents(
        agents=agents,
        environment=env,
        num_episodes=config.get('num_episodes', 50)
    )
    
    # Save comparison results
    save_dir = Path(config.get('save_dir', 'ai/trained_models/rl_agents/comparisons'))
    save_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = save_dir / f"agent_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Comparison results saved to {results_path}")
    return results

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train RL agents for trading')
    parser.add_argument('--config', type=str, required=True, help='Configuration file path')
    parser.add_argument('--agent', type=str, choices=['ppo', 'a3c', 'compare'], 
                       default='ppo', help='Agent type to train')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Set verbosity
    config['verbose'] = config.get('verbose', args.verbose)
    
    # Train or compare agents
    try:
        if args.agent == 'ppo':
            results = train_ppo_agent(config)
        elif args.agent == 'a3c':
            results = train_a3c_agent(config)
        elif args.agent == 'compare':
            results = compare_agent_performance(config)
        
        logger.info("Training/comparison completed successfully")
        logger.info(f"Results: {results}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()