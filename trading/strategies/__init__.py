"""
📊 Trading Strategies Module
Collection de stratégies de trading algorithmiques
"""

from .base_strategy import BaseStrategy
from .trend_following import TrendFollowingStrategy
from .mean_reversion import MeanReversionStrategy

__all__ = [
    "BaseStrategy",
    "TrendFollowingStrategy", 
    "MeanReversionStrategy"
]