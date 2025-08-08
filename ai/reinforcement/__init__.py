"""
Reinforcement Learning Module for BYJY-Trader Phase 3.3

This module implements reinforcement learning agents for autonomous trading,
including PPO/A3C agents, trading environment simulation, reward functions,
and portfolio management.

Components:
- environment: Trading environment simulation
- agents: RL agents (PPO, A3C)
- rewards: Reward function implementations
- portfolio: RL-based portfolio management
- utils: Utilities and helpers
"""

from .environment.trading_env import TradingEnvironment
from .agents.ppo_agent import PPOAgent
from .agents.a3c_agent import A3CAgent
from .rewards.reward_functions import ProfitRiskReward, SharpeReward
from .portfolio.rl_portfolio_manager import RLPortfolioManager

__all__ = [
    'TradingEnvironment',
    'PPOAgent', 
    'A3CAgent',
    'ProfitRiskReward',
    'SharpeReward',
    'RLPortfolioManager'
]

__version__ = "3.3.0"