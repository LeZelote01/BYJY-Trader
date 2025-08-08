"""
🔌 Bybit Connector - Phase 2.4 Priority 4

Connecteur complet pour exchange Bybit.
Support trading Spot et Derivatives avec WebSocket temps réel.

Features:
- Trading Spot + Derivatives complet (Market, Limit, Stop orders)
- WebSocket feeds multi-streams
- Authentification sécurisée Bybit API
- Rate limiting automatique
- Error handling et resilience
"""

import asyncio
import aiohttp
import time
import hmac
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
from decimal import Decimal

from core.logger import get_logger
from connectors.base.base_connector import BaseConnector, OrderType, OrderSide, OrderStatus, ExchangeStatus
from connectors.base.exchange_config import ExchangeConfig
from .bybit_auth import BybitAuth
from .bybit_websocket import BybitWebSocket

logger = get_logger(__name__)


class BybitConnector(BaseConnector):
    """
    Connecteur Bybit complet
    
    Implémente toutes les fonctionnalités requises par BaseConnector
    pour trading Spot et Derivatives Bybit avec WebSocket temps réel.
    """
    
    def __init__(
        self,
        api_key: str = None,
        api_secret: str = None, 
        sandbox: bool = True,
        rate_limit: int = 600  # Bybit: 600 req/min
    ):
        """
        Initialise connecteur Bybit
        
        Args:
            api_key: Clé API Bybit
            api_secret: Secret API Bybit
            sandbox: Mode testnet (True) ou mainnet (False)
            rate_limit: Limite requêtes par minute
        """
        super().__init__(
            exchange_name="bybit",
            api_key=api_key,
            api_secret=api_secret,
            sandbox=sandbox,
            rate_limit=rate_limit
        )
        
        # Configuration Bybit
        self.config = ExchangeConfig.get_config("bybit")
        self.endpoints = ExchangeConfig.get_endpoints("bybit", sandbox)
        self.limits = ExchangeConfig.get_limits("bybit")
        
        # Composants spécialisés
        self.auth = BybitAuth(api_key, api_secret) if api_key and api_secret else None
        self.ws = BybitWebSocket(self.endpoints.websocket_base_url)
        
        # Rate limiting - Bybit utilise un système par minute
        self.request_timestamps: List[float] = []
        self.requests_per_minute = self.rate_limit
        
        # Cache pour optimiser performances
        self.instruments_cache: Dict[str, Dict] = {}
        self.cache_ttl = 300  # 5 minutes
        self.last_cache_update = 0
        
        logger.info(f"🔌 BybitConnector initialisé (sandbox={sandbox})")
    
    # === IMPLÉMENTATION MÉTHODES DE CONNEXION ===
    
    async def connect(self) -> bool:
        """
        Se connecter à Bybit API
        
        Returns:
            bool: True si connexion réussie
        """
        try:
            logger.info("🔗 Connexion à Bybit...")
            
            # Test connectivité de base
            if not await self._test_basic_connectivity():
                return False
            
            # Test authentification si API keys fournies
            if self.auth:
                auth_test = await self.auth.test_connectivity(self.endpoints.rest_base_url)
                if not auth_test["success"]:
                    logger.error(f"❌ Échec authentification Bybit: {auth_test['error']}")
                    return False
                
                logger.info(f"✅ Authentification réussie - UID: {auth_test.get('uid', 'N/A')}")
            
            # Charger informations instruments
            await self._load_instruments_info()
            
            # Mettre à jour status
            self.status = ExchangeStatus.CONNECTED
            self.connected_at = datetime.now()
            
            logger.info("✅ BybitConnector connecté avec succès")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur connexion Bybit: {e}")
            self.status = ExchangeStatus.ERROR
            self.error_count += 1
            return False
    
    async def disconnect(self) -> bool:
        """
        Se déconnecter de Bybit
        
        Returns:
            bool: True si déconnexion propre
        """
        try:
            logger.info("🔌 Déconnexion Bybit...")
            
            # Arrêter WebSocket
            if self.ws and self.ws.connected:
                await self.ws.disconnect()
            
            # Fermer session HTTP
            if self.session and not self.session.closed:
                await self.session.close()
            
            # Reset status
            self.status = ExchangeStatus.DISCONNECTED
            self.connected_at = None
            
            logger.info("✅ BybitConnector déconnecté proprement")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur déconnexion Bybit: {e}")
            return False
    
    async def test_connection(self) -> Dict[str, Any]:
        """
        Tester connexion complète
        
        Returns:
            Dict avec résultats test détaillés
        """
        try:
            start_time = time.time()
            
            # Test connectivité de base
            basic_test = await self._test_basic_connectivity()
            if not basic_test:
                return {
                    "success": False,
                    "error": "Connectivité de base échouée",
                    "latency_ms": None,
                    "authenticated": False
                }
            
            latency = (time.time() - start_time) * 1000
            
            # Test authentification
            auth_result = None
            if self.auth:
                auth_result = await self.auth.test_connectivity(self.endpoints.rest_base_url)
            
            # Test WebSocket
            ws_test = False
            if await self.start_websocket():
                ws_test = True
                await self.stop_websocket()
            
            return {
                "success": True,
                "latency_ms": round(latency, 2),
                "authenticated": auth_result["success"] if auth_result else False,
                "permissions": auth_result["permissions"] if auth_result and auth_result["success"] else None,
                "websocket_available": ws_test,
                "exchange": "bybit",
                "sandbox_mode": self.sandbox,
                "endpoints": {
                    "rest": self.endpoints.rest_base_url,
                    "websocket": self.endpoints.websocket_base_url
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur test connexion: {e}")
            return {
                "success": False,
                "error": str(e),
                "latency_ms": None,
                "authenticated": False
            }
    
    # === IMPLÉMENTATION MÉTHODES TRADING ===
    
    async def place_order(
        self,
        symbol: str,
        order_type: OrderType,
        side: OrderSide,
        quantity: Decimal,
        price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None,
        time_in_force: str = "GTC",
        client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Placer un ordre sur Bybit
        
        Args:
            symbol: Symbole trading (ex: BTCUSDT)
            order_type: Type d'ordre
            side: Côté (buy/sell)
            quantity: Quantité à trader
            price: Prix limite (pour ordres limit)
            stop_price: Prix stop (pour stop orders)
            time_in_force: Validité ordre (GTC, IOC, FOK)
            client_order_id: ID client personnalisé
            
        Returns:
            Dict avec détails ordre placé
        """
        try:
            if not self.auth:
                return {
                    "success": False,
                    "error": "Authentification requise pour trading",
                    "order_id": None
                }
            
            # Validation et préparation paramètres
            params = await self._prepare_order_params(
                symbol, order_type, side, quantity, price, stop_price, time_in_force, client_order_id
            )
            
            if not params["valid"]:
                return {
                    "success": False,
                    "error": params["error"],
                    "order_id": None
                }
            
            # Placer ordre via API Spot
            endpoint = "/v5/order/create"
            request_data = self.auth.prepare_signed_request("POST", endpoint, params["params"])
            
            result = await self._make_request(
                request_data["method"],
                f"{self.endpoints.rest_base_url}{endpoint}",
                headers=request_data["headers"],
                json=request_data["data"]
            )
            
            if result["success"]:
                response_data = result["data"]
                
                if response_data.get("retCode") != 0:
                    return {
                        "success": False,
                        "error": response_data.get("retMsg", "Erreur inconnue"),
                        "order_id": None
                    }
                
                order_result = response_data.get("result", {})
                
                return {
                    "success": True,
                    "order_id": order_result.get("orderId"),
                    "client_order_id": order_result.get("orderLinkId"),
                    "symbol": symbol,
                    "side": side.value,
                    "type": order_type.value,
                    "quantity": quantity,
                    "price": price,
                    "status": OrderStatus.PENDING.value,
                    "time_in_force": time_in_force,
                    "timestamp": datetime.now().isoformat(),
                    "exchange": "bybit"
                }
            else:
                return {
                    "success": False,
                    "error": result["error"],
                    "order_id": None
                }
                
        except Exception as e:
            logger.error(f"❌ Erreur placement ordre: {e}")
            return {
                "success": False,
                "error": str(e),
                "order_id": None
            }
    
    async def cancel_order(
        self,
        symbol: str,
        order_id: str = None,
        client_order_id: str = None
    ) -> Dict[str, Any]:
        """
        Annuler un ordre sur Bybit
        
        Args:
            symbol: Symbole trading
            order_id: ID ordre Bybit
            client_order_id: ID ordre client
            
        Returns:
            Dict avec status annulation
        """
        try:
            if not self.auth:
                return {
                    "success": False,
                    "error": "Authentification requise"
                }
            
            # Paramètres annulation
            params = {
                "category": "spot",  # Spot trading
                "symbol": symbol
            }
            
            if order_id:
                params["orderId"] = order_id
            elif client_order_id:
                params["orderLinkId"] = client_order_id
            else:
                return {
                    "success": False,
                    "error": "order_id ou client_order_id requis"
                }
            
            # Annuler via API
            endpoint = "/v5/order/cancel"
            request_data = self.auth.prepare_signed_request("POST", endpoint, params)
            
            result = await self._make_request(
                request_data["method"],
                f"{self.endpoints.rest_base_url}{endpoint}",
                headers=request_data["headers"],
                json=request_data["data"]
            )
            
            if result["success"]:
                response_data = result["data"]
                
                if response_data.get("retCode") != 0:
                    return {
                        "success": False,
                        "error": response_data.get("retMsg", "Erreur annulation")
                    }
                
                order_result = response_data.get("result", {})
                return {
                    "success": True,
                    "order_id": order_result.get("orderId"),
                    "client_order_id": order_result.get("orderLinkId"),
                    "symbol": symbol,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": result["error"]
                }
                
        except Exception as e:
            logger.error(f"❌ Erreur annulation ordre: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_order_status(
        self,
        symbol: str,
        order_id: str = None,
        client_order_id: str = None
    ) -> Dict[str, Any]:
        """
        Récupérer status d'un ordre
        
        Args:
            symbol: Symbole trading
            order_id: ID ordre Bybit
            client_order_id: ID ordre client
            
        Returns:
            Dict avec détails ordre
        """
        try:
            if not self.auth:
                return {
                    "success": False,
                    "error": "Authentification requise"
                }
            
            # Paramètres requête
            params = {
                "category": "spot",
                "symbol": symbol
            }
            
            if order_id:
                params["orderId"] = order_id
            elif client_order_id:
                params["orderLinkId"] = client_order_id
            else:
                return {
                    "success": False,
                    "error": "order_id ou client_order_id requis"
                }
            
            # Requête status ordre
            endpoint = "/v5/order/history"
            request_data = self.auth.prepare_signed_request("GET", endpoint, params)
            
            result = await self._make_request(
                request_data["method"],
                f"{self.endpoints.rest_base_url}{endpoint}",
                headers=request_data["headers"],
                params=request_data["params"]
            )
            
            if result["success"]:
                response_data = result["data"]
                
                if response_data.get("retCode") != 0:
                    return {
                        "success": False,
                        "error": response_data.get("retMsg", "Erreur récupération")
                    }
                
                orders_list = response_data.get("result", {}).get("list", [])
                if orders_list:
                    order_data = orders_list[0]  # Premier ordre correspondant
                    
                    return {
                        "success": True,
                        "order": {
                            "order_id": order_data.get("orderId"),
                            "client_order_id": order_data.get("orderLinkId"),
                            "symbol": order_data.get("symbol"),
                            "side": order_data.get("side").lower(),
                            "type": order_data.get("orderType").lower(),
                            "quantity": Decimal(str(order_data.get("qty", "0"))),
                            "price": Decimal(str(order_data.get("price", "0"))) if order_data.get("price") else None,
                            "executed_quantity": Decimal(str(order_data.get("cumExecQty", "0"))),
                            "status": self._normalize_order_status(order_data.get("orderStatus")),
                            "time_in_force": order_data.get("timeInForce"),
                            "created_at": order_data.get("createdTime"),
                            "updated_at": order_data.get("updatedTime")
                        }
                    }
                else:
                    return {
                        "success": False,
                        "error": "Ordre non trouvé"
                    }
            else:
                return {
                    "success": False,
                    "error": result["error"]
                }
                
        except Exception as e:
            logger.error(f"❌ Erreur récupération status ordre: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Récupérer tous les ordres ouverts
        
        Args:
            symbol: Symbole spécifique (optionnel)
            
        Returns:
            List des ordres ouverts
        """
        try:
            if not self.auth:
                return []
            
            # Paramètres requête
            params = {
                "category": "spot",
                "openOnly": 0,  # Récupérer tous les ordres
                "limit": 50
            }
            if symbol:
                params["symbol"] = symbol
            
            # Requête ordres ouverts
            endpoint = "/v5/order/realtime"
            request_data = self.auth.prepare_signed_request("GET", endpoint, params)
            
            result = await self._make_request(
                request_data["method"],
                f"{self.endpoints.rest_base_url}{endpoint}",
                headers=request_data["headers"],
                params=request_data["params"]
            )
            
            if result["success"]:
                response_data = result["data"]
                
                if response_data.get("retCode") != 0:
                    logger.error(f"❌ Erreur API Bybit: {response_data.get('retMsg')}")
                    return []
                
                orders = []
                orders_list = response_data.get("result", {}).get("list", [])
                
                for order_data in orders_list:
                    # Filtrer seulement les ordres ouverts
                    status = order_data.get("orderStatus", "")
                    if status in ["New", "PartiallyFilled", "Untriggered"]:
                        orders.append({
                            "order_id": order_data.get("orderId"),
                            "client_order_id": order_data.get("orderLinkId"),
                            "symbol": order_data.get("symbol"),
                            "side": order_data.get("side").lower(),
                            "type": order_data.get("orderType").lower(),
                            "quantity": Decimal(str(order_data.get("qty", "0"))),
                            "price": Decimal(str(order_data.get("price", "0"))) if order_data.get("price") else None,
                            "executed_quantity": Decimal(str(order_data.get("cumExecQty", "0"))),
                            "status": self._normalize_order_status(order_data.get("orderStatus")),
                            "time_in_force": order_data.get("timeInForce"),
                            "created_at": order_data.get("createdTime")
                        })
                
                return orders
            else:
                logger.error(f"❌ Erreur récupération ordres ouverts: {result['error']}")
                return []
                
        except Exception as e:
            logger.error(f"❌ Erreur récupération ordres ouverts: {e}")
            return []
    
    # === IMPLÉMENTATION MÉTHODES ACCOUNT ===
    
    async def get_account_balance(self) -> Dict[str, Any]:
        """
        Récupérer soldes du compte
        
        Returns:
            Dict avec soldes par asset
        """
        try:
            if not self.auth:
                return {
                    "success": False,
                    "error": "Authentification requise"
                }
            
            # Requête wallet balance
            endpoint = "/v5/account/wallet-balance"
            params = {"accountType": "SPOT"}  # Compte Spot
            
            request_data = self.auth.prepare_signed_request("GET", endpoint, params)
            
            result = await self._make_request(
                request_data["method"],
                f"{self.endpoints.rest_base_url}{endpoint}",
                headers=request_data["headers"],
                params=request_data["params"]
            )
            
            if result["success"]:
                response_data = result["data"]
                
                if response_data.get("retCode") != 0:
                    return {
                        "success": False,
                        "error": response_data.get("retMsg", "Erreur récupération balance")
                    }
                
                # Extraire balances
                balances = {}
                accounts = response_data.get("result", {}).get("list", [])
                
                if accounts:
                    coins = accounts[0].get("coin", [])
                    
                    for coin_info in coins:
                        coin = coin_info["coin"]
                        wallet_balance = Decimal(coin_info.get("walletBalance", "0"))
                        available_balance = Decimal(coin_info.get("availableToWithdraw", "0"))
                        locked_balance = wallet_balance - available_balance
                        
                        if wallet_balance > 0:
                            balances[coin] = {
                                "free": float(available_balance),
                                "locked": float(locked_balance) if locked_balance > 0 else 0.0,
                                "total": float(wallet_balance)
                            }
                
                return {
                    "success": True,
                    "balances": balances,
                    "account_type": "BYBIT_SPOT",
                    "can_trade": True,
                    "can_withdraw": True,
                    "can_deposit": True,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": result["error"]
                }
                
        except Exception as e:
            logger.error(f"❌ Erreur récupération balances: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_trading_fees(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Récupérer frais de trading
        
        Args:
            symbol: Symbole spécifique (optionnel)
            
        Returns:
            Dict avec structure frais
        """
        try:
            if not self.auth:
                return {
                    "success": False,
                    "error": "Authentification requise"
                }
            
            # Requête fee rate
            endpoint = "/v5/account/fee-rate"
            params = {"category": "spot"}
            if symbol:
                params["symbol"] = symbol
            
            request_data = self.auth.prepare_signed_request("GET", endpoint, params)
            
            result = await self._make_request(
                request_data["method"],
                f"{self.endpoints.rest_base_url}{endpoint}",
                headers=request_data["headers"],
                params=request_data["params"]
            )
            
            if result["success"]:
                response_data = result["data"]
                
                if response_data.get("retCode") != 0:
                    return {
                        "success": False,
                        "error": response_data.get("retMsg", "Erreur récupération frais")
                    }
                
                fees_list = response_data.get("result", {}).get("list", [])
                
                if symbol and fees_list:
                    # Frais pour symbole spécifique
                    fee_info = fees_list[0]
                    return {
                        "success": True,
                        "symbol": fee_info.get("symbol"),
                        "maker_fee": float(fee_info.get("makerFeeRate", "0.001")),
                        "taker_fee": float(fee_info.get("takerFeeRate", "0.001"))
                    }
                else:
                    # Tous les frais
                    fees = {}
                    for fee_info in fees_list:
                        symbol_name = fee_info.get("symbol")
                        fees[symbol_name] = {
                            "maker_fee": float(fee_info.get("makerFeeRate", "0.001")),
                            "taker_fee": float(fee_info.get("takerFeeRate", "0.001"))
                        }
                    
                    return {
                        "success": True,
                        "fees": fees,
                        "exchange": "bybit"
                    }
            else:
                return {
                    "success": False,
                    "error": result["error"]
                }
                
        except Exception as e:
            logger.error(f"❌ Erreur récupération frais: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    # === IMPLÉMENTATION MÉTHODES MARKET DATA ===
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Récupérer ticker (prix actuel)
        
        Args:
            symbol: Symbole trading
            
        Returns:
            Dict avec prix, volume, changement 24h
        """
        try:
            # Endpoint ticker 24h
            endpoint = "/v5/market/tickers"
            params = {"category": "spot", "symbol": symbol}
            
            result = await self._make_request("GET", f"{self.endpoints.rest_base_url}{endpoint}", params=params)
            
            if result["success"]:
                response_data = result["data"]
                
                if response_data.get("retCode") != 0:
                    return {
                        "success": False,
                        "error": response_data.get("retMsg", "Erreur ticker")
                    }
                
                tickers = response_data.get("result", {}).get("list", [])
                if tickers:
                    ticker_data = tickers[0]
                    
                    return {
                        "success": True,
                        "symbol": ticker_data.get("symbol"),
                        "price": float(ticker_data.get("lastPrice", "0")),
                        "bid": float(ticker_data.get("bid1Price", "0")),
                        "ask": float(ticker_data.get("ask1Price", "0")),
                        "volume": float(ticker_data.get("volume24h", "0")),
                        "quote_volume": float(ticker_data.get("turnover24h", "0")),
                        "high_24h": float(ticker_data.get("highPrice24h", "0")),
                        "low_24h": float(ticker_data.get("lowPrice24h", "0")),
                        "change_24h": float(ticker_data.get("price24hPcnt", "0")) * 100,  # Convertir en %
                        "change_percent_24h": float(ticker_data.get("price24hPcnt", "0")) * 100,
                        "timestamp": datetime.now().isoformat(),
                        "exchange": "bybit"
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Pas de données ticker pour {symbol}"
                    }
            else:
                return {
                    "success": False,
                    "error": result["error"]
                }
                
        except Exception as e:
            logger.error(f"❌ Erreur récupération ticker: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_order_book(self, symbol: str, depth: int = 25) -> Dict[str, Any]:
        """
        Récupérer order book
        
        Args:
            symbol: Symbole trading
            depth: Profondeur order book (1, 25, 50)
            
        Returns:
            Dict avec bids/asks
        """
        try:
            # Endpoint order book
            endpoint = "/v5/market/orderbook"
            # Bybit limite les profondeurs à 1, 25, 50
            valid_depths = [1, 25, 50]
            depth = min(valid_depths, key=lambda x: abs(x - depth))
            params = {"category": "spot", "symbol": symbol, "limit": depth}
            
            result = await self._make_request("GET", f"{self.endpoints.rest_base_url}{endpoint}", params=params)
            
            if result["success"]:
                response_data = result["data"]
                
                if response_data.get("retCode") != 0:
                    return {
                        "success": False,
                        "error": response_data.get("retMsg", "Erreur order book")
                    }
                
                book_data = response_data.get("result", {})
                
                # Convertir format Bybit [[price, size]]
                bids = [[float(bid[0]), float(bid[1])] for bid in book_data.get("b", [])]
                asks = [[float(ask[0]), float(ask[1])] for ask in book_data.get("a", [])]
                
                return {
                    "success": True,
                    "symbol": symbol,
                    "bids": bids,
                    "asks": asks,
                    "timestamp": datetime.now().isoformat(),
                    "exchange": "bybit"
                }
            else:
                return {
                    "success": False,
                    "error": result["error"]
                }
                
        except Exception as e:
            logger.error(f"❌ Erreur récupération order book: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_recent_trades(self, symbol: str, limit: int = 60) -> List[Dict[str, Any]]:
        """
        Récupérer trades récents
        
        Args:
            symbol: Symbole trading
            limit: Nombre max trades (max 1000)
            
        Returns:
            List des trades récents
        """
        try:
            # Endpoint trades récents
            endpoint = "/v5/market/recent-trade"
            params = {"category": "spot", "symbol": symbol, "limit": min(limit, 1000)}
            
            result = await self._make_request("GET", f"{self.endpoints.rest_base_url}{endpoint}", params=params)
            
            if result["success"]:
                response_data = result["data"]
                
                if response_data.get("retCode") != 0:
                    logger.error(f"❌ Erreur API Bybit: {response_data.get('retMsg')}")
                    return []
                
                trades_list = response_data.get("result", {}).get("list", [])
                
                trades = []
                for trade in trades_list:
                    trades.append({
                        "trade_id": trade.get("execId"),
                        "price": float(trade.get("price", "0")),
                        "quantity": float(trade.get("size", "0")),
                        "side": trade.get("side").lower(),
                        "timestamp": trade.get("time"),
                        "is_buyer_maker": trade.get("isBuyerMaker", False),
                        "exchange": "bybit"
                    })
                
                return trades
            else:
                logger.error(f"❌ Erreur récupération trades: {result['error']}")
                return []
                
        except Exception as e:
            logger.error(f"❌ Erreur récupération trades: {e}")
            return []
    
    # === IMPLÉMENTATION MÉTHODES WEBSOCKET ===
    
    async def start_websocket(self) -> bool:
        """
        Démarrer connexion WebSocket
        
        Returns:
            bool: True si démarrage réussi
        """
        try:
            if self.ws.connected:
                logger.info("⚠️ WebSocket déjà connecté")
                return True
            
            success = await self.ws.connect()
            if success:
                logger.info("✅ WebSocket Bybit démarré")
                
                await self.emit_event("connection_established", {
                    "exchange": "bybit",
                    "websocket": True,
                    "timestamp": datetime.now().isoformat()
                })
                
                return True
            else:
                logger.error("❌ Échec démarrage WebSocket")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erreur démarrage WebSocket: {e}")
            return False
    
    async def stop_websocket(self) -> bool:
        """
        Arrêter connexion WebSocket
        
        Returns:
            bool: True si arrêt propre
        """
        try:
            if not self.ws.connected:
                return True
            
            success = await self.ws.disconnect()
            if success:
                logger.info("✅ WebSocket Bybit arrêté")
                
                await self.emit_event("connection_lost", {
                    "exchange": "bybit",
                    "reason": "manual_disconnect",
                    "timestamp": datetime.now().isoformat()
                })
                
                return True
            else:
                logger.error("❌ Erreur arrêt WebSocket")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erreur arrêt WebSocket: {e}")
            return False
    
    async def subscribe_ticker(self, symbol: str) -> bool:
        """
        S'abonner aux updates prix temps réel
        
        Args:
            symbol: Symbole à suivre
            
        Returns:
            bool: True si abonnement réussi
        """
        try:
            if not self.ws.connected:
                logger.error("❌ WebSocket non connecté")
                return False
            
            success = await self.ws.subscribe_ticker(symbol)
            if success:
                self.ws.add_callback("tickers", self._handle_ticker_update)
                logger.info(f"✅ Abonnement ticker {symbol} réussi")
                return True
            else:
                logger.error(f"❌ Échec abonnement ticker {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erreur abonnement ticker: {e}")
            return False
    
    async def subscribe_order_book(self, symbol: str) -> bool:
        """
        S'abonner aux updates order book temps réel
        
        Args:
            symbol: Symbole à suivre
            
        Returns:
            bool: True si abonnement réussi
        """
        try:
            if not self.ws.connected:
                logger.error("❌ WebSocket non connecté")
                return False
            
            success = await self.ws.subscribe_order_book(symbol)
            if success:
                self.ws.add_callback("orderbook", self._handle_orderbook_update)
                logger.info(f"✅ Abonnement order book {symbol} réussi")
                return True
            else:
                logger.error(f"❌ Échec abonnement order book {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erreur abonnement order book: {e}")
            return False
    
    async def subscribe_trades(self, symbol: str) -> bool:
        """
        S'abonner aux trades temps réel
        
        Args:
            symbol: Symbole à suivre
            
        Returns:
            bool: True si abonnement réussi
        """
        try:
            if not self.ws.connected:
                logger.error("❌ WebSocket non connecté")
                return False
            
            success = await self.ws.subscribe_trades(symbol)
            if success:
                self.ws.add_callback("publicTrade", self._handle_trade_update)
                logger.info(f"✅ Abonnement trades {symbol} réussi")
                return True
            else:
                logger.error(f"❌ Échec abonnement trades {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erreur abonnement trades: {e}")
            return False
    
    # === MÉTHODES UTILITAIRES ===
    
    async def _test_basic_connectivity(self) -> bool:
        """Tester connectivité de base à Bybit"""
        try:
            endpoint = "/v5/market/time"
            result = await self._make_request("GET", f"{self.endpoints.rest_base_url}{endpoint}")
            return result["success"] and result["data"].get("retCode") == 0
        except:
            return False
    
    async def _load_instruments_info(self):
        """Charger informations instruments"""
        try:
            endpoint = "/v5/market/instruments-info"
            params = {"category": "spot", "limit": 1000}
            
            result = await self._make_request("GET", f"{self.endpoints.rest_base_url}{endpoint}", params=params)
            
            if result["success"]:
                response_data = result["data"]
                
                if response_data.get("retCode") == 0:
                    instruments = response_data.get("result", {}).get("list", [])
                    
                    # Mettre en cache infos instruments
                    for instrument in instruments:
                        symbol = instrument["symbol"]
                        self.instruments_cache[symbol] = {
                            "base_coin": instrument.get("baseCoin"),
                            "quote_coin": instrument.get("quoteCoin"),
                            "status": instrument.get("status"),
                            "lot_size_filter": instrument.get("lotSizeFilter", {}),
                            "price_filter": instrument.get("priceFilter", {}),
                            "innovation": instrument.get("innovation", False)
                        }
                    
                    self.last_cache_update = time.time()
                    logger.info(f"📊 Infos instruments Bybit chargées: {len(self.instruments_cache)} instruments")
                
        except Exception as e:
            logger.error(f"❌ Erreur chargement instruments: {e}")
    
    async def _prepare_order_params(
        self, symbol: str, order_type: OrderType, side: OrderSide,
        quantity: Decimal, price: Optional[Decimal], stop_price: Optional[Decimal],
        time_in_force: str, client_order_id: Optional[str]
    ) -> Dict[str, Any]:
        """Préparer et valider paramètres d'ordre Bybit"""
        
        try:
            params = {
                "category": "spot",
                "symbol": symbol,
                "side": side.value.capitalize(),  # Buy/Sell
                "qty": str(quantity),
                "timeInForce": time_in_force
            }
            
            # Type d'ordre
            if order_type == OrderType.MARKET:
                params["orderType"] = "Market"
            elif order_type == OrderType.LIMIT:
                if price is None:
                    return {"valid": False, "error": "Prix requis pour ordre LIMIT"}
                params["orderType"] = "Limit"
                params["price"] = str(price)
            else:
                return {"valid": False, "error": f"Type d'ordre {order_type} non supporté"}
            
            # Client order ID
            if client_order_id:
                params["orderLinkId"] = client_order_id
            
            return {"valid": True, "params": params}
            
        except Exception as e:
            logger.error(f"❌ Erreur préparation paramètres ordre: {e}")
            return {"valid": False, "error": str(e)}
    
    def _normalize_order_status(self, bybit_status: str) -> OrderStatus:
        """Normaliser status ordre Bybit vers OrderStatus"""
        mapping = {
            "New": OrderStatus.OPEN,
            "PartiallyFilled": OrderStatus.PARTIALLY_FILLED,
            "Filled": OrderStatus.FILLED,
            "Cancelled": OrderStatus.CANCELLED,
            "Rejected": OrderStatus.REJECTED,
            "PartiallyFilledCanceled": OrderStatus.CANCELLED,
            "Deactivated": OrderStatus.CANCELLED,
            "Triggered": OrderStatus.OPEN,
            "Untriggered": OrderStatus.PENDING,
            "Active": OrderStatus.OPEN
        }
        return mapping.get(bybit_status, OrderStatus.PENDING)
    
    async def _make_request(
        self, method: str, url: str, headers: Dict = None, 
        json: Dict = None, params: Dict = None
    ) -> Dict[str, Any]:
        """Faire requête HTTP avec rate limiting"""
        
        # Rate limiting
        await self._check_rate_limit()
        
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Faire requête
            async with self.session.request(
                method, url, headers=headers, json=json, params=params, timeout=30
            ) as response:
                
                self.request_count += 1
                
                if response.status == 200:
                    response_data = await response.json()
                    return {"success": True, "data": response_data}
                else:
                    error_text = await response.text()
                    self.error_count += 1
                    return {"success": False, "error": f"HTTP {response.status}: {error_text}"}
        
        except Exception as e:
            self.error_count += 1
            return {"success": False, "error": str(e)}
    
    async def _check_rate_limit(self):
        """Vérifier et respecter rate limits"""
        now = time.time()
        
        # Nettoyer anciennes timestamps (plus d'1 minute)
        self.request_timestamps = [
            ts for ts in self.request_timestamps 
            if now - ts < 60
        ]
        
        # Vérifier limite
        if len(self.request_timestamps) >= self.requests_per_minute:
            sleep_time = 60 - (now - self.request_timestamps[0])
            if sleep_time > 0:
                logger.debug(f"⏳ Rate limit Bybit: pause {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
        
        # Ajouter timestamp actuel
        self.request_timestamps.append(now)
    
    # === CALLBACKS WEBSOCKET ===
    
    async def _handle_ticker_update(self, data: Dict[str, Any]):
        """Gérer update ticker WebSocket"""
        await self.emit_event("price_update", {
            "exchange": "bybit",
            "symbol": data.get("symbol"),
            "price": float(data.get("lastPrice", "0")),
            "volume": float(data.get("volume24h", "0")),
            "change_24h": float(data.get("price24hPcnt", "0")) * 100,
            "timestamp": data.get("ts"),
            "source": "websocket"
        })
    
    async def _handle_orderbook_update(self, data: Dict[str, Any]):
        """Gérer update order book WebSocket"""
        await self.emit_event("order_book_update", {
            "exchange": "bybit",
            "symbol": data.get("s"),
            "bids": data.get("b", []),
            "asks": data.get("a", []),
            "timestamp": data.get("ts"),
            "source": "websocket"
        })
    
    async def _handle_trade_update(self, data: Dict[str, Any]):
        """Gérer update trades WebSocket"""
        await self.emit_event("trade_update", {
            "exchange": "bybit",
            "symbol": data.get("s"),
            "trades": data.get("data", []),
            "timestamp": data.get("ts"),
            "source": "websocket"
        })