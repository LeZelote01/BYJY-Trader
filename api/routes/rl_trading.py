"""
API Routes for Reinforcement Learning Trading

Provides REST API endpoints for controlling RL trading agents,
monitoring training, and managing agent deployments.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import logging
import json
import asyncio
from pathlib import Path
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.logger import get_logger
from core.database import DatabaseManager
from ai.reinforcement.agents.ppo_agent import PPOAgent
from ai.reinforcement.agents.a3c_agent import A3CAgent
from ai.reinforcement.environment.trading_env import TradingEnvironment
from ai.reinforcement.utils.training_utils import RLTrainer, create_reward_function
from ai.reinforcement.portfolio.rl_portfolio_manager import RLPortfolioManager

logger = get_logger(__name__)
router = APIRouter()

# Global training state
active_trainings = {}
deployed_agents = {}
training_logs = {}

# Request/Response Models
class TrainingConfig(BaseModel):
    agent_type: str = Field(..., description="Type of agent: 'ppo' or 'a3c'")
    symbol: str = Field(default="BTCUSDT", description="Trading symbol")
    initial_balance: float = Field(default=10000.0, description="Initial balance for training")
    max_episodes: int = Field(default=2000, description="Maximum training episodes")
    learning_rate: float = Field(default=3e-4, description="Learning rate")
    reward_function: Optional[str] = Field(default="profit_risk", description="Reward function type")
    save_name: Optional[str] = Field(default=None, description="Name for saved model")
    
    # PPO specific parameters
    clip_ratio: Optional[float] = Field(default=0.2, description="PPO clip ratio")
    batch_size: Optional[int] = Field(default=64, description="Batch size for PPO")
    epochs_per_update: Optional[int] = Field(default=10, description="Epochs per update for PPO")
    
    # A3C specific parameters
    num_workers: Optional[int] = Field(default=4, description="Number of A3C workers")
    
    # Environment parameters
    lookback_window: Optional[int] = Field(default=50, description="Lookback window for observations")
    transaction_cost: Optional[float] = Field(default=0.001, description="Transaction cost")

class AgentDeploymentConfig(BaseModel):
    agent_name: str = Field(..., description="Name of the agent")
    model_path: str = Field(..., description="Path to trained model")
    symbol: str = Field(default="BTCUSDT", description="Trading symbol")
    initial_balance: float = Field(default=10000.0, description="Initial balance")
    max_position_size: float = Field(default=1.0, description="Maximum position size")

class TradingAction(BaseModel):
    agent_name: str = Field(..., description="Name of deployed agent")
    symbol: str = Field(..., description="Trading symbol")
    current_price: float = Field(..., description="Current market price")
    market_data: Optional[Dict] = Field(default=None, description="Additional market data")

@router.get("/rl/health")
async def health_check():
    """Health check for RL system"""
    return {
        "status": "healthy",
        "active_trainings": len(active_trainings),
        "deployed_agents": len(deployed_agents),
        "timestamp": datetime.now().isoformat()
    }

@router.get("/rl/training/status")
async def get_training_status():
    """Get status of all active trainings"""
    status = {}
    
    for training_id, trainer in active_trainings.items():
        if hasattr(trainer, 'get_training_metrics'):
            status[training_id] = trainer.get_training_metrics()
        else:
            status[training_id] = {
                "status": "running",
                "agent_type": trainer.agent.name if hasattr(trainer, 'agent') else "unknown"
            }
    
    return {
        "active_trainings": status,
        "total_trainings": len(active_trainings)
    }

@router.post("/rl/training/start")
async def start_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    """Start RL agent training"""
    try:
        # Generate training ID
        training_id = f"{config.agent_type}_{config.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Add training to background tasks
        background_tasks.add_task(run_training, training_id, config.dict())
        
        return {
            "message": "Training started",
            "training_id": training_id,
            "config": config.dict(),
            "status": "starting"
        }
        
    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        raise HTTPException(status_code=500, detail=f"Training start failed: {str(e)}")

@router.get("/rl/training/{training_id}/status")
async def get_training_status_by_id(training_id: str):
    """Get status of specific training"""
    if training_id not in active_trainings:
        raise HTTPException(status_code=404, detail="Training not found")
    
    trainer = active_trainings[training_id]
    
    try:
        if hasattr(trainer, 'get_training_metrics'):
            metrics = trainer.get_training_metrics()
        else:
            metrics = {"status": "running"}
        
        # Add training logs if available
        if training_id in training_logs:
            metrics["logs"] = training_logs[training_id][-100:]  # Last 100 log entries
        
        return {
            "training_id": training_id,
            "status": "active",
            "metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"Failed to get training status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rl/training/{training_id}/stop")
async def stop_training(training_id: str):
    """Stop specific training"""
    if training_id not in active_trainings:
        raise HTTPException(status_code=404, detail="Training not found")
    
    try:
        trainer = active_trainings[training_id]
        
        # Stop training if method exists
        if hasattr(trainer, 'stop_training'):
            trainer.stop_training()
        
        # Remove from active trainings
        del active_trainings[training_id]
        
        return {
            "message": "Training stopped",
            "training_id": training_id,
            "status": "stopped"
        }
        
    except Exception as e:
        logger.error(f"Failed to stop training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/rl/models")
async def list_trained_models():
    """List all available trained models"""
    models_dir = Path("ai/trained_models/rl_agents")
    
    if not models_dir.exists():
        return {"models": []}
    
    models = []
    
    for agent_dir in models_dir.iterdir():
        if agent_dir.is_dir():
            for model_file in agent_dir.glob("*.pth"):
                # Try to load model info
                info = {
                    "agent_type": agent_dir.name,
                    "model_name": model_file.stem,
                    "model_path": str(model_file),
                    "created": datetime.fromtimestamp(model_file.stat().st_mtime).isoformat(),
                    "size_mb": round(model_file.stat().st_size / (1024 * 1024), 2)
                }
                
                # Load model config if available
                config_file = model_file.with_suffix('.json')
                if config_file.exists():
                    try:
                        with open(config_file, 'r') as f:
                            model_config = json.load(f)
                        info["config"] = model_config
                    except Exception as e:
                        logger.warning(f"Could not load config for {model_file}: {e}")
                
                models.append(info)
    
    # Sort by creation date
    models.sort(key=lambda x: x['created'], reverse=True)
    
    return {"models": models}

@router.post("/rl/agents/deploy")
async def deploy_agent(config: AgentDeploymentConfig):
    """Deploy trained RL agent for live trading"""
    try:
        model_path = Path(config.model_path)
        
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Model file not found")
        
        # Determine agent type from path
        agent_type = model_path.parent.name.lower()
        
        # Create environment for agent
        env = TradingEnvironment(
            symbol=config.symbol,
            initial_balance=config.initial_balance,
            lookbook_window=50
        )
        
        # Create and load agent
        if agent_type == 'ppo':
            agent = PPOAgent(
                observation_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0]
            )
        elif agent_type == 'a3c':
            agent = A3CAgent(
                observation_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0]
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        # Load trained model
        agent.load_model(str(model_path))
        
        # Create portfolio manager
        portfolio_manager = RLPortfolioManager(
            initial_cash=config.initial_balance,
            max_position_size=config.max_position_size
        )
        
        # Store deployed agent
        deployed_agents[config.agent_name] = {
            "agent": agent,
            "environment": env,
            "portfolio_manager": portfolio_manager,
            "config": config.dict(),
            "deployed_at": datetime.now().isoformat(),
            "status": "active"
        }
        
        logger.info(f"Deployed RL agent: {config.agent_name}")
        
        return {
            "message": "Agent deployed successfully",
            "agent_name": config.agent_name,
            "agent_type": agent_type,
            "status": "deployed"
        }
        
    except Exception as e:
        logger.error(f"Failed to deploy agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/rl/agents")
async def list_deployed_agents():
    """List all deployed agents"""
    agents_info = {}
    
    for agent_name, agent_data in deployed_agents.items():
        portfolio_state = agent_data["portfolio_manager"].get_portfolio_state()
        
        agents_info[agent_name] = {
            "agent_name": agent_name,
            "agent_type": agent_data["agent"].__class__.__name__,
            "deployed_at": agent_data["deployed_at"],
            "status": agent_data["status"],
            "portfolio_value": portfolio_state.get("total_value", 0),
            "total_return": portfolio_state.get("total_return", 0),
            "positions": portfolio_state.get("positions", {}),
            "config": agent_data["config"]
        }
    
    return {"deployed_agents": agents_info}

@router.post("/rl/agents/{agent_name}/action")
async def get_agent_action(agent_name: str, action_request: TradingAction):
    """Get trading action from deployed RL agent"""
    if agent_name not in deployed_agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    try:
        agent_data = deployed_agents[agent_name]
        agent = agent_data["agent"]
        env = agent_data["environment"]
        portfolio_manager = agent_data["portfolio_manager"]
        
        # Get current observation from environment
        # Note: In a real implementation, you'd update the environment with current market data
        current_obs = env._get_observation()
        
        # Get action from agent
        action = agent.select_action(current_obs, training=False)
        
        # Execute action through portfolio manager
        execution_info = portfolio_manager.execute_action(
            symbol=action_request.symbol,
            target_position=float(action[0]),  # Position target
            current_price=action_request.current_price,
            confidence=float(action[1]) if len(action) > 1 else 1.0  # Confidence
        )
        
        # Get updated portfolio state
        portfolio_state = portfolio_manager.get_portfolio_state()
        
        return {
            "agent_name": agent_name,
            "action": action.tolist(),
            "execution_info": execution_info,
            "portfolio_state": portfolio_state,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get agent action: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/rl/agents/{agent_name}/portfolio")
async def get_agent_portfolio(agent_name: str):
    """Get portfolio state of deployed agent"""
    if agent_name not in deployed_agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    try:
        agent_data = deployed_agents[agent_name]
        portfolio_manager = agent_data["portfolio_manager"]
        
        portfolio_state = portfolio_manager.get_portfolio_state()
        performance_summary = portfolio_manager.get_performance_summary()
        
        return {
            "agent_name": agent_name,
            "portfolio_state": portfolio_state,
            "performance_summary": performance_summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get portfolio state: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rl/agents/{agent_name}/reset")
async def reset_agent_portfolio(agent_name: str):
    """Reset agent portfolio to initial state"""
    if agent_name not in deployed_agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    try:
        agent_data = deployed_agents[agent_name]
        portfolio_manager = agent_data["portfolio_manager"]
        
        # Reset portfolio
        portfolio_manager.reset()
        
        return {
            "message": "Portfolio reset successfully",
            "agent_name": agent_name,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to reset portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/rl/agents/{agent_name}")
async def undeploy_agent(agent_name: str):
    """Undeploy RL agent"""
    if agent_name not in deployed_agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    try:
        # Get final portfolio state
        portfolio_manager = deployed_agents[agent_name]["portfolio_manager"]
        final_performance = portfolio_manager.get_performance_summary()
        
        # Remove agent
        del deployed_agents[agent_name]
        
        logger.info(f"Undeployed RL agent: {agent_name}")
        
        return {
            "message": "Agent undeployed successfully",
            "agent_name": agent_name,
            "final_performance": final_performance,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to undeploy agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background training function
async def run_training(training_id: str, config: Dict):
    """Run training in background"""
    try:
        training_logs[training_id] = []
        
        # Create environment
        env = TradingEnvironment(
            symbol=config.get('symbol', 'BTCUSDT'),
            initial_balance=config.get('initial_balance', 10000.0),
            lookback_window=config.get('lookback_window', 50),
            transaction_cost=config.get('transaction_cost', 0.001)
        )
        
        # Create reward function
        reward_function = None
        if config.get('reward_function'):
            reward_function = create_reward_function(
                reward_type=config['reward_function'],
                profit_weight=1.0,
                risk_weight=0.5
            )
        
        # Create agent based on type
        if config['agent_type'].lower() == 'ppo':
            agent = PPOAgent(
                observation_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0],
                learning_rate=config.get('learning_rate', 3e-4),
                clip_ratio=config.get('clip_ratio', 0.2),
                batch_size=config.get('batch_size', 64),
                epochs_per_update=config.get('epochs_per_update', 10)
            )
        elif config['agent_type'].lower() == 'a3c':
            agent = A3CAgent(
                observation_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0],
                learning_rate=config.get('learning_rate', 3e-4),
                num_workers=config.get('num_workers', 4)
            )
        else:
            raise ValueError(f"Unknown agent type: {config['agent_type']}")
        
        # Create trainer
        save_dir = f"ai/trained_models/rl_agents/{config['agent_type'].lower()}"
        trainer = RLTrainer(
            agent=agent,
            environment=env,
            reward_function=reward_function,
            save_dir=save_dir,
            max_episodes=config.get('max_episodes', 2000)
        )
        
        # Store trainer
        active_trainings[training_id] = trainer
        
        # Log start
        training_logs[training_id].append({
            "timestamp": datetime.now().isoformat(),
            "message": f"Started training {config['agent_type']} agent",
            "level": "info"
        })
        
        # Run training
        results = trainer.train(verbose=False)
        
        # Save model with custom name if provided
        if config.get('save_name'):
            custom_path = Path(save_dir) / f"{config['save_name']}.pth"
            agent.save_model(str(custom_path))
            training_logs[training_id].append({
                "timestamp": datetime.now().isoformat(),
                "message": f"Model saved as {custom_path}",
                "level": "info"
            })
        
        # Log completion
        training_logs[training_id].append({
            "timestamp": datetime.now().isoformat(),
            "message": f"Training completed: {results}",
            "level": "info"
        })
        
        # Remove from active trainings
        if training_id in active_trainings:
            del active_trainings[training_id]
        
        logger.info(f"Training {training_id} completed successfully")
        
    except Exception as e:
        error_msg = f"Training {training_id} failed: {str(e)}"
        logger.error(error_msg)
        
        if training_id in training_logs:
            training_logs[training_id].append({
                "timestamp": datetime.now().isoformat(),
                "message": error_msg,
                "level": "error"
            })
        
        # Remove from active trainings
        if training_id in active_trainings:
            del active_trainings[training_id]