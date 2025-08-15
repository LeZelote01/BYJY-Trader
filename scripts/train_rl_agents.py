"""
Training Script for RL Agents - Phase 3.3

Comprehensive script for training PPO and A3C agents for autonomous trading.
This script provides a complete training pipeline with evaluation and model saving.
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
import sys
import os
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.logger import get_logger
from ai.reinforcement.agents.ppo_agent import PPOAgent
from ai.reinforcement.agents.a3c_agent import A3CAgent
from ai.reinforcement.environment.trading_env import TradingEnvironment
from ai.reinforcement.rewards.reward_functions import ProfitRiskReward, SharpeReward, MultiObjectiveReward
from ai.reinforcement.utils.training_utils import RLTrainer, compare_agents, hyperparameter_search

logger = get_logger(__name__)

def create_default_config():
    """Create default training configuration"""
    return {
        "training": {
            "agent_type": "ppo",
            "symbol": "BTCUSDT", 
            "initial_balance": 10000.0,
            "max_episodes": 2000,
            "checkpoint_frequency": 100,
            "evaluation_frequency": 50,
            "save_dir": "ai/trained_models/rl_agents"
        },
        "environment": {
            "lookback_window": 50,
            "transaction_cost": 0.001,
            "max_position_size": 1.0
        },
        "reward": {
            "type": "profit_risk",
            "profit_weight": 1.0,
            "risk_weight": 0.5,
            "drawdown_penalty": 2.0,
            "volatility_penalty": 0.1
        },
        "ppo": {
            "learning_rate": 3e-4,
            "hidden_dims": [256, 256, 128],
            "clip_ratio": 0.2,
            "value_coef": 0.5,
            "entropy_coef": 0.01,
            "batch_size": 64,
            "epochs_per_update": 10,
            "gae_lambda": 0.95,
            "gamma": 0.99
        },
        "a3c": {
            "learning_rate": 3e-4,
            "hidden_dims": [256, 128],
            "num_workers": 4,
            "gamma": 0.99,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "update_freq": 20
        }
    }

def train_ppo_agent(config: dict, run_id: str) -> dict:
    """
    Train PPO agent with comprehensive logging and evaluation
    
    Args:
        config: Training configuration
        run_id: Unique run identifier
        
    Returns:
        Training results
    """
    logger.info(f"=== Starting PPO Training - Run {run_id} ===")
    
    # Create environment
    logger.info("Creating training environment...")
    env = TradingEnvironment(
        symbol=config["training"]["symbol"],
        initial_balance=config["training"]["initial_balance"],
        lookback_window=config["environment"]["lookback_window"],
        transaction_cost=config["environment"]["transaction_cost"],
        max_position_size=config["environment"]["max_position_size"]
    )
    logger.info(f"Environment created: obs_dim={env.observation_space.shape[0]}, action_dim={env.action_space.shape[0]}")
    
    # Create reward function
    reward_function = None
    reward_config = config["reward"]
    
    if reward_config["type"] == "profit_risk":
        reward_function = ProfitRiskReward(
            profit_weight=reward_config["profit_weight"],
            risk_weight=reward_config["risk_weight"],
            drawdown_penalty=reward_config.get("drawdown_penalty", 2.0),
            volatility_penalty=reward_config.get("volatility_penalty", 0.1)
        )
    elif reward_config["type"] == "sharpe":
        reward_function = SharpeReward()
    elif reward_config["type"] == "multi_objective":
        reward_function = MultiObjectiveReward()
    
    if reward_function:
        logger.info(f"Using reward function: {reward_function.name}")
    
    # Create PPO agent
    ppo_config = config["ppo"]
    logger.info("Creating PPO agent...")
    
    agent = PPOAgent(
        observation_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        learning_rate=ppo_config["learning_rate"],
        hidden_dims=ppo_config["hidden_dims"],
        clip_ratio=ppo_config["clip_ratio"],
        value_coef=ppo_config["value_coef"],
        entropy_coef=ppo_config["entropy_coef"],
        batch_size=ppo_config["batch_size"],
        epochs_per_update=ppo_config["epochs_per_update"],
        gae_lambda=ppo_config["gae_lambda"],
        gamma=ppo_config["gamma"]
    )
    
    logger.info(f"PPO agent created with {sum(p.numel() for p in agent.network.parameters())} parameters")
    
    # Create save directory
    save_dir = Path(config["training"]["save_dir"]) / "ppo" / run_id
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training configuration
    config_path = save_dir / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Training config saved to {config_path}")
    
    # Create trainer
    trainer = RLTrainer(
        agent=agent,
        environment=env,
        reward_function=reward_function,
        save_dir=str(save_dir),
        checkpoint_frequency=config["training"]["checkpoint_frequency"],
        evaluation_frequency=config["training"]["evaluation_frequency"],
        max_episodes=config["training"]["max_episodes"]
    )
    
    # Run training with comprehensive monitoring
    logger.info("Starting training...")
    training_start = datetime.now()
    
    try:
        results = trainer.train(verbose=True)
        
        training_end = datetime.now()
        training_duration = (training_end - training_start).total_seconds()
        
        logger.info(f"Training completed in {training_duration:.1f} seconds")
        logger.info(f"Final performance: {results.get('final_evaluation_score', 0):.2f}")
        logger.info(f"Best performance: {results.get('best_performance', 0):.2f}")
        logger.info(f"Converged: {results.get('converged', False)}")
        
        # Generate and save training plots
        plot_path = save_dir / f"ppo_training_progress_{run_id}.png"
        trainer.plot_training_progress(save_path=str(plot_path), show=False)
        logger.info(f"Training plots saved to {plot_path}")
        
        # Detailed evaluation on fresh environment
        logger.info("Running final evaluation...")
        eval_env = TradingEnvironment(
            symbol=config["training"]["symbol"],
            initial_balance=config["training"]["initial_balance"],
            lookback_window=config["environment"]["lookback_window"]
        )
        
        final_eval = agent.evaluate(eval_env, num_episodes=20)
        logger.info(f"Final evaluation results: {final_eval}")
        
        # Save final model with descriptive name
        final_model_path = save_dir / f"ppo_final_{run_id}.pth"
        agent.save_model(str(final_model_path))
        logger.info(f"Final model saved to {final_model_path}")
        
        # Update results with additional info
        results.update({
            "run_id": run_id,
            "training_duration_seconds": training_duration,
            "final_evaluation": final_eval,
            "model_path": str(final_model_path),
            "config_path": str(config_path)
        })
        
        return results
        
    except Exception as e:
        logger.error(f"PPO training failed: {e}")
        raise

def train_a3c_agent(config: dict, run_id: str) -> dict:
    """
    Train A3C agent with parallel workers
    
    Args:
        config: Training configuration
        run_id: Unique run identifier
        
    Returns:
        Training results
    """
    logger.info(f"=== Starting A3C Training - Run {run_id} ===")
    
    # Environment creator for workers
    def env_creator():
        return TradingEnvironment(
            symbol=config["training"]["symbol"],
            initial_balance=config["training"]["initial_balance"],
            lookback_window=config["environment"]["lookback_window"],
            transaction_cost=config["environment"]["transaction_cost"],
            max_position_size=config["environment"]["max_position_size"]
        )
    
    # Create sample environment for dimensions
    sample_env = env_creator()
    logger.info(f"Environment specs: obs_dim={sample_env.observation_space.shape[0]}, action_dim={sample_env.action_space.shape[0]}")
    
    # Create A3C agent
    a3c_config = config["a3c"]
    logger.info(f"Creating A3C agent with {a3c_config['num_workers']} workers...")
    
    agent = A3CAgent(
        observation_dim=sample_env.observation_space.shape[0],
        action_dim=sample_env.action_space.shape[0],
        learning_rate=a3c_config["learning_rate"],
        hidden_dims=a3c_config["hidden_dims"],
        num_workers=a3c_config["num_workers"],
        gamma=a3c_config["gamma"],
        entropy_coef=a3c_config["entropy_coef"],
        value_coef=a3c_config["value_coef"],
        update_freq=a3c_config["update_freq"]
    )
    
    # Create workers
    agent.create_workers(env_creator)
    logger.info(f"A3C agent created with {a3c_config['num_workers']} workers")
    
    # Create save directory
    save_dir = Path(config["training"]["save_dir"]) / "a3c" / run_id
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = save_dir / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Start training
    training_start = datetime.now()
    logger.info("Starting A3C parallel training...")
    
    try:
        max_episodes_per_worker = config["training"]["max_episodes"] // a3c_config["num_workers"]
        agent.train_parallel(max_episodes_per_worker)
        
        training_end = datetime.now()
        training_duration = (training_end - training_start).total_seconds()
        
        logger.info(f"A3C training completed in {training_duration:.1f} seconds")
        
        # Get training statistics
        training_stats = agent.update()
        worker_stats = agent.get_worker_stats()
        
        logger.info(f"Total episodes: {training_stats.get('total_episodes', 0)}")
        logger.info(f"Mean reward: {training_stats.get('mean_reward', 0):.2f}")
        
        # Save final model
        final_model_path = save_dir / f"a3c_final_{run_id}.pth"
        agent.save_model(str(final_model_path))
        logger.info(f"A3C model saved to {final_model_path}")
        
        # Final evaluation
        logger.info("Running final evaluation...")
        eval_env = env_creator()
        final_eval = agent.evaluate(eval_env, num_episodes=10)
        
        results = {
            "run_id": run_id,
            "agent_type": "a3c",
            "training_duration_seconds": training_duration,
            "training_stats": training_stats,
            "worker_stats": worker_stats,
            "final_evaluation": final_eval,
            "model_path": str(final_model_path),
            "config_path": str(config_path)
        }
        
        logger.info(f"A3C training results: {results}")
        return results
        
    except Exception as e:
        logger.error(f"A3C training failed: {e}")
        raise

def run_agent_comparison(config: dict, models_to_compare: list) -> dict:
    """
    Compare performance of different trained models
    
    Args:
        config: Configuration for comparison
        models_to_compare: List of model paths to compare
        
    Returns:
        Comparison results
    """
    logger.info("=== Starting Agent Comparison ===")
    
    # Create evaluation environment
    env = TradingEnvironment(
        symbol=config["training"]["symbol"],
        initial_balance=config["training"]["initial_balance"],
        lookback_window=config["environment"]["lookback_window"]
    )
    
    # Load agents
    agents = []
    
    for model_info in models_to_compare:
        model_path = Path(model_info["path"])
        agent_type = model_info["type"]
        
        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            continue
        
        # Create agent
        if agent_type.lower() == "ppo":
            agent = PPOAgent(
                observation_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0]
            )
        elif agent_type.lower() == "a3c":
            agent = A3CAgent(
                observation_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0]
            )
        else:
            logger.warning(f"Unknown agent type: {agent_type}")
            continue
        
        # Load model
        try:
            agent.load_model(str(model_path))
            agent.name = f"{agent_type}_{model_path.stem}"
            agents.append(agent)
            logger.info(f"Loaded agent: {agent.name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
    
    if not agents:
        raise ValueError("No valid agents loaded for comparison")
    
    # Run comparison
    logger.info(f"Comparing {len(agents)} agents...")
    comparison_results = compare_agents(agents, env, num_episodes=50)
    
    # Save results
    save_dir = Path(config["training"]["save_dir"]) / "comparisons"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = save_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, 'w') as f:
        json.dump(comparison_results, f, indent=2, default=str)
    
    logger.info(f"Comparison results saved to {results_path}")
    logger.info("=== Agent Comparison Completed ===")
    
    return comparison_results

def run_hyperparameter_search(config: dict, agent_type: str) -> dict:
    """
    Run hyperparameter optimization for RL agents
    
    Args:
        config: Base configuration
        agent_type: Type of agent ('ppo' or 'a3c')
        
    Returns:
        Optimization results
    """
    logger.info(f"=== Starting Hyperparameter Search for {agent_type.upper()} ===")
    
    # Create environment
    env = TradingEnvironment(
        symbol=config["training"]["symbol"],
        initial_balance=config["training"]["initial_balance"],
        lookback_window=config["environment"]["lookback_window"]
    )
    
    # Define parameter grids
    if agent_type.lower() == "ppo":
        param_grid = {
            "observation_dim": [env.observation_space.shape[0]],
            "action_dim": [env.action_space.shape[0]],
            "learning_rate": [1e-4, 3e-4, 1e-3],
            "clip_ratio": [0.1, 0.2, 0.3],
            "batch_size": [32, 64, 128],
            "entropy_coef": [0.01, 0.02, 0.05]
        }
        agent_class = PPOAgent
    
    elif agent_type.lower() == "a3c":
        param_grid = {
            "observation_dim": [env.observation_space.shape[0]],
            "action_dim": [env.action_space.shape[0]],
            "learning_rate": [1e-4, 3e-4, 1e-3],
            "num_workers": [2, 4, 8],
            "entropy_coef": [0.01, 0.02, 0.05],
            "value_coef": [0.5, 1.0]
        }
        agent_class = A3CAgent
    
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    # Run optimization
    optimization_results = hyperparameter_search(
        agent_class=agent_class,
        environment=env,
        param_grid=param_grid,
        num_trials=20,
        episodes_per_trial=500
    )
    
    # Save results
    save_dir = Path(config["training"]["save_dir"]) / "optimization"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = save_dir / f"{agent_type}_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, 'w') as f:
        json.dump(optimization_results, f, indent=2, default=str)
    
    logger.info(f"Optimization results saved to {results_path}")
    logger.info("=== Hyperparameter Search Completed ===")
    
    return optimization_results

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train RL agents for autonomous trading - Phase 3.3")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--agent", type=str, choices=["ppo", "a3c", "both"], default="ppo", help="Agent type to train")
    parser.add_argument("--mode", type=str, choices=["train", "compare", "optimize"], default="train", help="Operation mode")
    parser.add_argument("--models", type=str, nargs="+", help="Model paths for comparison")
    parser.add_argument("--run-id", type=str, help="Custom run ID")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "auto"], default="auto", help="Computing device")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Load configuration
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded config from {args.config}")
    else:
        config = create_default_config()
        logger.info("Using default configuration")
        
        # Save default config
        default_config_path = "ai/reinforcement/default_config.json"
        os.makedirs(os.path.dirname(default_config_path), exist_ok=True)
        with open(default_config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Default config saved to {default_config_path}")
    
    # Generate run ID
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    try:
        if args.mode == "train":
            if args.agent == "ppo":
                results = train_ppo_agent(config, run_id)
                logger.info("PPO training completed successfully")
                
            elif args.agent == "a3c":
                results = train_a3c_agent(config, run_id)
                logger.info("A3C training completed successfully")
                
            elif args.agent == "both":
                logger.info("Training both PPO and A3C agents...")
                
                ppo_results = train_ppo_agent(config, f"{run_id}_ppo")
                a3c_results = train_a3c_agent(config, f"{run_id}_a3c")
                
                results = {
                    "ppo_results": ppo_results,
                    "a3c_results": a3c_results
                }
                
                logger.info("Both agents trained successfully")
            
            # Print final summary
            logger.info("=== TRAINING SUMMARY ===")
            if isinstance(results, dict) and "ppo_results" in results:
                for agent_type, agent_results in results.items():
                    logger.info(f"{agent_type.upper()}: {agent_results.get('final_evaluation_score', 0):.2f}")
            else:
                logger.info(f"Final Score: {results.get('final_evaluation_score', 0):.2f}")
        
        elif args.mode == "compare":
            if not args.models:
                logger.error("Models must be specified for comparison mode")
                return
            
            models_info = []
            for model_path in args.models:
                # Infer agent type from path
                agent_type = "ppo" if "ppo" in model_path.lower() else "a3c"
                models_info.append({"path": model_path, "type": agent_type})
            
            results = run_agent_comparison(config, models_info)
            logger.info("Agent comparison completed successfully")
        
        elif args.mode == "optimize":
            results = run_hyperparameter_search(config, args.agent)
            logger.info("Hyperparameter optimization completed successfully")
        
        logger.info("=== ALL OPERATIONS COMPLETED SUCCESSFULLY ===")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()