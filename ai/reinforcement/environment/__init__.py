"""
Trading Environment Module for Reinforcement Learning

Implements realistic trading environments for RL agent training,
including market simulation, historical data integration, and
state/action/reward management.
"""

from .trading_env import TradingEnvironment
from .market_simulator import MarketSimulator

__all__ = ['TradingEnvironment', 'MarketSimulator']