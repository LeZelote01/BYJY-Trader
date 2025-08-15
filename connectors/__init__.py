"""
🔌 BYJY-Trader Connectors Module - Phase 2.4

Module de connecteurs pour exchanges de trading en temps réel.
Supporte Binance, Coinbase, Kraken, Bybit avec WebSocket feeds.

Architecture:
- base/: Classes abstraites communes
- exchanges/: Implémentations par exchange  
- security/: Sécurité et authentification
- feeds/: Flux données temps réel
- testing/: Tests et simulation

Version: 2.4.0
"""

from .base.base_connector import BaseConnector
from .base.order_manager import OrderManager
from .base.feed_manager import FeedManager
from .base.exchange_config import ExchangeConfig

from .exchanges.binance.binance_connector import BinanceConnector
from .exchanges.coinbase.coinbase_connector import CoinbaseConnector

__version__ = "2.4.0"
__all__ = [
    "BaseConnector",
    "OrderManager", 
    "FeedManager",
    "ExchangeConfig",
    "BinanceConnector",
    "CoinbaseConnector"
]