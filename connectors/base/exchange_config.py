"""
🔧 Exchange Configuration Manager - Phase 2.4

Configuration centralisée pour tous les exchanges supportés.
Endpoints, limites, et paramètres par exchange.

Exchanges supportés:
- Binance (Spot + WebSocket)
- Coinbase Advanced (Spot + WebSocket)
- Kraken Pro (Spot + WebSocket)
- Bybit (Spot + Derivatives + WebSocket)
"""

from typing import Dict, Any, NamedTuple
from dataclasses import dataclass
import os


@dataclass
class ExchangeEndpoints:
    """Endpoints d'un exchange"""
    rest_base_url: str
    websocket_base_url: str
    sandbox_rest_url: str = None
    sandbox_websocket_url: str = None


@dataclass 
class ExchangeLimits:
    """Limites d'un exchange"""
    requests_per_minute: int
    requests_per_second: int
    max_order_size: float
    min_order_size: float
    websocket_connections: int


class ExchangeConfig:
    """
    Configuration centralisée des exchanges
    
    Fournit endpoints, limites et paramètres pour chaque exchange
    selon le mode sandbox/production.
    """
    
    # Configuration endpoints par exchange
    ENDPOINTS = {
        "binance": ExchangeEndpoints(
            rest_base_url="https://api.binance.com",
            websocket_base_url="wss://stream.binance.com:9443/ws",
            sandbox_rest_url="https://testnet.binance.vision",
            sandbox_websocket_url="wss://testnet.binance.vision/ws"
        ),
        
        "coinbase": ExchangeEndpoints(
            rest_base_url="https://api.coinbase.com/api/v3/brokerage",
            websocket_base_url="wss://advanced-trade-ws.coinbase.com",
            sandbox_rest_url="https://api.coinbase.com/api/v3/brokerage",  # Même URL
            sandbox_websocket_url="wss://advanced-trade-ws-sandbox.coinbase.com"
        ),
        
        "kraken": ExchangeEndpoints(
            rest_base_url="https://api.kraken.com",
            websocket_base_url="wss://ws.kraken.com",
            sandbox_rest_url="https://api.kraken.com",  # Pas de sandbox séparé
            sandbox_websocket_url="wss://ws.kraken.com"
        ),
        
        "bybit": ExchangeEndpoints(
            rest_base_url="https://api.bybit.com",
            websocket_base_url="wss://stream.bybit.com/v5/public/spot",
            sandbox_rest_url="https://api-testnet.bybit.com",
            sandbox_websocket_url="wss://stream-testnet.bybit.com/v5/public/spot"
        )
    }
    
    # Configuration limites par exchange
    LIMITS = {
        "binance": ExchangeLimits(
            requests_per_minute=1200,
            requests_per_second=20,
            max_order_size=100000.0,
            min_order_size=0.00001,
            websocket_connections=10
        ),
        
        "coinbase": ExchangeLimits(
            requests_per_minute=300,  # Conservative pour Coinbase
            requests_per_second=5,
            max_order_size=1000000.0,
            min_order_size=0.0001,
            websocket_connections=5
        ),
        
        "kraken": ExchangeLimits(
            requests_per_minute=900,  # 15 req/sec * 60
            requests_per_second=15,
            max_order_size=500000.0,
            min_order_size=0.00001,
            websocket_connections=8
        ),
        
        "bybit": ExchangeLimits(
            requests_per_minute=600,
            requests_per_second=10,
            max_order_size=1000000.0,
            min_order_size=0.00001,
            websocket_connections=20
        )
    }
    
    # Configuration générale par exchange
    EXCHANGE_INFO = {
        "binance": {
            "name": "Binance",
            "type": "crypto",
            "supports_spot": True,
            "supports_futures": False,  # Pas implémenté dans cette phase
            "supports_websocket": True,
            "auth_method": "HMAC-SHA256",
            "symbol_format": "BTCUSDT",
            "order_types": ["market", "limit", "stop_loss", "take_profit", "oco"],
            "time_in_force": ["GTC", "IOC", "FOK"],
            "precision": 8
        },
        
        "coinbase": {
            "name": "Coinbase Advanced Trade",
            "type": "crypto",
            "supports_spot": True,
            "supports_futures": False,
            "supports_websocket": True,
            "auth_method": "JWT-RS256",
            "symbol_format": "BTC-USD",
            "order_types": ["market", "limit"],
            "time_in_force": ["GTC", "IOC", "FOK"],
            "precision": 8
        },
        
        "kraken": {
            "name": "Kraken Pro",
            "type": "crypto",
            "supports_spot": True,
            "supports_futures": False,
            "supports_websocket": True,
            "auth_method": "HMAC-SHA512",
            "symbol_format": "BTCUSD",
            "order_types": ["market", "limit", "stop-loss", "take-profit"],
            "time_in_force": ["GTC", "IOC", "FOK"],
            "precision": 10
        },
        
        "bybit": {
            "name": "Bybit",
            "type": "crypto",
            "supports_spot": True,
            "supports_futures": True,
            "supports_websocket": True,
            "auth_method": "HMAC-SHA256",
            "symbol_format": "BTCUSDT",
            "order_types": ["market", "limit", "stop", "conditional"],
            "time_in_force": ["GTC", "IOC", "FOK"],
            "precision": 8
        }
    }
    
    @classmethod
    def get_endpoints(cls, exchange: str, sandbox: bool = True) -> ExchangeEndpoints:
        """
        Récupérer endpoints pour un exchange
        
        Args:
            exchange: Nom exchange (binance, coinbase, kraken, bybit)
            sandbox: Mode sandbox/testnet
            
        Returns:
            ExchangeEndpoints avec URLs appropriées
        """
        if exchange not in cls.ENDPOINTS:
            raise ValueError(f"Exchange {exchange} non supporté")
        
        endpoints = cls.ENDPOINTS[exchange]
        
        if sandbox:
            # Utiliser endpoints sandbox si disponibles
            return ExchangeEndpoints(
                rest_base_url=endpoints.sandbox_rest_url or endpoints.rest_base_url,
                websocket_base_url=endpoints.sandbox_websocket_url or endpoints.websocket_base_url,
                sandbox_rest_url=endpoints.sandbox_rest_url,
                sandbox_websocket_url=endpoints.sandbox_websocket_url
            )
        else:
            # Utiliser endpoints production
            return endpoints
    
    @classmethod
    def get_limits(cls, exchange: str) -> ExchangeLimits:
        """
        Récupérer limites pour un exchange
        
        Args:
            exchange: Nom exchange
            
        Returns:
            ExchangeLimits avec toutes les limites
        """
        if exchange not in cls.LIMITS:
            raise ValueError(f"Exchange {exchange} non supporté")
        
        return cls.LIMITS[exchange]
    
    @classmethod
    def get_config(cls, exchange: str) -> Dict[str, Any]:
        """
        Récupérer configuration générale pour un exchange
        
        Args:
            exchange: Nom exchange
            
        Returns:
            Dict avec configuration complète
        """
        if exchange not in cls.EXCHANGE_INFO:
            raise ValueError(f"Exchange {exchange} non supporté")
        
        return cls.EXCHANGE_INFO[exchange]
    
    @classmethod
    def get_supported_exchanges(cls) -> Dict[str, Dict[str, Any]]:
        """
        Récupérer liste des exchanges supportés
        
        Returns:
            Dict avec infos de tous les exchanges supportés
        """
        return cls.EXCHANGE_INFO.copy()
    
    @classmethod
    def normalize_symbol(cls, exchange: str, symbol: str) -> str:
        """
        Normaliser format symbole selon l'exchange
        
        Args:
            exchange: Nom exchange
            symbol: Symbole à normaliser (ex: BTC/USD, BTCUSD, BTC-USD)
            
        Returns:
            Symbole normalisé selon format exchange
        """
        # Nettoyer symbole de base
        clean_symbol = symbol.replace("/", "").replace("-", "").upper()
        
        if exchange == "coinbase":
            # Coinbase utilise format BTC-USD
            if len(clean_symbol) >= 6:
                base = clean_symbol[:-3]  # Enlever les 3 derniers chars (USD)
                quote = clean_symbol[-3:]
                return f"{base}-{quote}"
            return symbol
        
        elif exchange in ["binance", "bybit"]:
            # Binance et Bybit utilisent BTCUSDT
            return clean_symbol
        
        elif exchange == "kraken":
            # Kraken utilise des noms spéciaux parfois
            # Mapping des symboles courants
            kraken_mapping = {
                "BTCUSD": "XBTUSD",
                "ETHUSD": "ETHUSD", 
                "ADAUSD": "ADAUSD",
                "SOLUSD": "SOLUSD"
            }
            return kraken_mapping.get(clean_symbol, clean_symbol)
        
        return clean_symbol
    
    @classmethod
    def get_websocket_channels(cls, exchange: str) -> Dict[str, str]:
        """
        Récupérer channels WebSocket disponibles par exchange
        
        Args:
            exchange: Nom exchange
            
        Returns:
            Dict avec channels disponibles
        """
        channels = {
            "binance": {
                "ticker": "@ticker",
                "depth": "@depth",
                "trades": "@trade",
                "kline": "@kline_1m",
                "user": "USER_DATA_STREAM"
            },
            
            "coinbase": {
                "ticker": "ticker",
                "level2": "level2", 
                "matches": "matches",
                "heartbeat": "heartbeat",
                "user": "user"
            },
            
            "kraken": {
                "ticker": "ticker",
                "ohlc": "ohlc",
                "trade": "trade",
                "book": "book",
                "spread": "spread"
            },
            
            "bybit": {
                "tickers": "tickers.{symbol}",
                "orderbook": "orderbook.1.{symbol}",
                "trades": "publicTrade.{symbol}",
                "kline": "kline.1m.{symbol}"
            }
        }
        
        return channels.get(exchange, {})
    
    @classmethod
    def validate_api_credentials(cls, exchange: str, api_key: str, api_secret: str) -> Dict[str, Any]:
        """
        Valider format des credentials API
        
        Args:
            exchange: Nom exchange
            api_key: Clé API
            api_secret: Secret API
            
        Returns:
            Dict avec résultat validation
        """
        if not api_key or not api_secret:
            return {
                "valid": False,
                "error": "API key et secret requis"
            }
        
        # Validation spécifique par exchange
        if exchange == "binance":
            if len(api_key) != 64:
                return {"valid": False, "error": "Binance API key doit faire 64 caractères"}
            if len(api_secret) != 64:
                return {"valid": False, "error": "Binance secret doit faire 64 caractères"}
        
        elif exchange == "coinbase":
            # Coinbase utilise JWT avec clé privée
            if not api_key.startswith("organizations/") and not api_key.startswith("users/"):
                return {"valid": False, "error": "Format API name Coinbase invalide"}
            # api_secret contient la clé privée JWT
        
        elif exchange == "kraken":
            if len(api_key) < 56:
                return {"valid": False, "error": "Kraken API key trop courte"}
            if len(api_secret) < 80:
                return {"valid": False, "error": "Kraken private key trop courte"}
        
        elif exchange == "bybit":
            if len(api_key) < 20:
                return {"valid": False, "error": "Bybit API key trop courte"}
            if len(api_secret) < 30:
                return {"valid": False, "error": "Bybit secret trop court"}
        
        return {"valid": True}
    
    @classmethod
    def get_minimum_order_amounts(cls, exchange: str) -> Dict[str, float]:
        """
        Récupérer montants minimum d'ordre par exchange
        
        Args:
            exchange: Nom exchange
            
        Returns:
            Dict avec montants minimums courants
        """
        minimums = {
            "binance": {
                "BTCUSDT": 0.00001,
                "ETHUSDT": 0.0001,
                "ADAUSDT": 1.0,
                "SOLUSDT": 0.01,
                "default": 0.00001
            },
            
            "coinbase": {
                "BTC-USD": 0.0001,
                "ETH-USD": 0.001,
                "ADA-USD": 1.0,
                "SOL-USD": 0.01,
                "default": 0.0001
            },
            
            "kraken": {
                "XBTUSD": 0.0001,
                "ETHUSD": 0.001,
                "ADAUSD": 10.0,
                "SOLUSD": 0.1,
                "default": 0.0001
            },
            
            "bybit": {
                "BTCUSDT": 0.00001,
                "ETHUSDT": 0.0001,
                "ADAUSDT": 1.0,
                "SOLUSDT": 0.01,
                "default": 0.00001
            }
        }
        
        return minimums.get(exchange, {"default": 0.0001})
    
    @classmethod
    def get_fee_structure(cls, exchange: str) -> Dict[str, Any]:
        """
        Structure des frais par exchange (approximative)
        
        Args:
            exchange: Nom exchange
            
        Returns:
            Dict avec structure frais
        """
        fees = {
            "binance": {
                "spot_maker": 0.001,    # 0.1%
                "spot_taker": 0.001,    # 0.1%
                "futures_maker": 0.0002, # 0.02%
                "futures_taker": 0.0004, # 0.04%
                "withdrawal_fees": "Variable selon asset"
            },
            
            "coinbase": {
                "spot_maker": 0.005,    # 0.5%
                "spot_taker": 0.005,    # 0.5%
                "advanced_maker": 0.005, # 0.5% (peut être réduit avec volume)
                "advanced_taker": 0.005, # 0.5%
                "withdrawal_fees": "Variable selon asset"
            },
            
            "kraken": {
                "spot_maker": 0.0016,   # 0.16%
                "spot_taker": 0.0026,   # 0.26%
                "futures_maker": 0.0002, # 0.02%
                "futures_taker": 0.0005, # 0.05%
                "withdrawal_fees": "Variable selon asset"
            },
            
            "bybit": {
                "spot_maker": 0.001,    # 0.1%
                "spot_taker": 0.001,    # 0.1%
                "derivatives_maker": 0.0001, # 0.01%
                "derivatives_taker": 0.0006, # 0.06%
                "withdrawal_fees": "Variable selon asset"
            }
        }
        
        return fees.get(exchange, {})
    
    @classmethod
    def get_environment_config(cls) -> Dict[str, Any]:
        """
        Configuration basée sur variables d'environnement
        
        Returns:
            Dict avec configuration d'environnement
        """
        return {
            "default_sandbox": os.getenv("TRADING_SANDBOX", "true").lower() == "true",
            "default_timeout": int(os.getenv("EXCHANGE_TIMEOUT", "30")),
            "max_retries": int(os.getenv("EXCHANGE_RETRIES", "3")),
            "log_level": os.getenv("EXCHANGE_LOG_LEVEL", "INFO"),
            "enable_websockets": os.getenv("ENABLE_WEBSOCKETS", "true").lower() == "true",
            "rate_limit_buffer": float(os.getenv("RATE_LIMIT_BUFFER", "0.1")),  # 10% buffer
        }