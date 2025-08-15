# Data Collection Module for BYJY-Trader
# Phase 2.1 - Historical Data Collector

__version__ = "2.1.0"
__author__ = "BYJY-Trader Team"
__description__ = "Historical Data Collection System with Multi-Source Support"

from .collectors.base_collector import BaseCollector
from .collectors.binance_collector import BinanceCollector  
from .collectors.yahoo_collector import YahooCollector
from .collectors.coingecko_collector import CoinGeckoCollector
from .processors.feature_engine import FeatureEngine
from .storage.data_manager import DataManager
from .feeds.realtime_feed import RealtimeFeed

__all__ = [
    'BaseCollector',
    'BinanceCollector', 
    'YahooCollector',
    'CoinGeckoCollector',
    'FeatureEngine',
    'DataManager',
    'RealtimeFeed'
]