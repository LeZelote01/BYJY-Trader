"""
🚀 Module Trading BYJY-Trader - Phase 2.3
Système complet de stratégies de trading automatisées
"""

from .engine.trading_engine import TradingEngine
from .strategies.base_strategy import BaseStrategy
from .strategies.trend_following import TrendFollowingStrategy
from .strategies.mean_reversion import MeanReversionStrategy

__version__ = "2.3.0"
__author__ = "BYJY-Trader AI"

__all__ = [
    "TradingEngine",
    "BaseStrategy", 
    "TrendFollowingStrategy",
    "MeanReversionStrategy"
]