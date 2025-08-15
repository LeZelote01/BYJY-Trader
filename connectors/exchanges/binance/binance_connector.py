"""
🔌 Binance Connector - Phase 2.4 Priority 1

Connecteur complet pour exchange Binance.
Support trading Spot avec WebSocket temps réel.

Features:
- Trading Spot complet (Market, Limit, Stop orders)
- WebSocket feeds multi-streams
- Authentification sécurisée 
- Rate limiting automatique
- Error handling et resilience
"""

import asyncio
import aiohttp
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from decimal import Decimal

from core.logger import get_logger
from connectors.base.base_connector import BaseConnector, OrderType, OrderSide, OrderStatus, ExchangeStatus
from connectors.base.exchange_config import ExchangeConfig
from .binance_auth import BinanceAuth
from .binance_websocket import BinanceWebSocket

logger = get_logger(__name__)


class BinanceConnector(BaseConnector):
    """
    Connecteur Binance complet
    
    Implémente toutes les fonctionnalités requises par BaseConnector
    pour trading Spot Binance avec WebSocket temps réel.
    """
    
    def __init__(
        self,
        api_key: str = None,
        api_secret: str = None, 
        sandbox: bool = True,
        rate_limit: int = 1200
    ):
        """
        Initialise connecteur Binance
        
        Args:
            api_key: Clé API Binance
            api_secret: Secret API Binance
            sandbox: Mode testnet (True) ou mainnet (False)
            rate_limit: Limite requêtes par minute
        """
        super().__init__(
            exchange_name="binance",
            api_key=api_key,
            api_secret=api_secret,
            sandbox=sandbox,
            rate_limit=rate_limit
        )
        
        # Configuration Binance
        self.config = ExchangeConfig.get_config("binance")
        self.endpoints = ExchangeConfig.get_endpoints("binance", sandbox)
        self.limits = ExchangeConfig.get_limits("binance")
        
        # Composants spécialisés
        self.auth = BinanceAuth(api_key, api_secret) if api_key and api_secret else None
        self.ws = BinanceWebSocket(self.endpoints.websocket_base_url)
        
        # Rate limiting
        self.request_timestamps: List[float] = []
        self.requests_per_minute = self.limits.requests_per_minute
        
        # Cache pour optimiser perfs
        self.symbol_info_cache: Dict[str, Dict] = {}
        self.cache_ttl = 300  # 5 minutes
        self.last_cache_update = 0
        
        logger.info(f"🔌 BinanceConnector initialisé (sandbox={sandbox})")
    
    # === IMPLÉMENTATION MÉTHODES DE CONNEXION ===
    
    async def connect(self) -> bool:
        """
        Se connecter à Binance API
        
        Returns:
            bool: True si connexion réussie
        """
        try:
            logger.info("🔗 Connexion à Binance...")
            
            # Test connectivité de base
            if not await self._test_basic_connectivity():
                return False
            
            # Test authentification si API keys fournies
            if self.auth:
                auth_test = await self.auth.test_connectivity(self.endpoints.rest_base_url)
                if not auth_test["success"]:
                    logger.error(f"❌ Échec authentification Binance: {auth_test['error']}")
                    return False
                
                logger.info(f"✅ Authentification réussie - Type: {auth_test['account_type']}")
            
            # Charger informations symboles
            await self._load_exchange_info()
            
            # Mettre à jour status
            self.status = ExchangeStatus.CONNECTED
            self.connected_at = datetime.now()
            
            logger.info("✅ BinanceConnector connecté avec succès")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur connexion Binance: {e}")
            self.status = ExchangeStatus.ERROR
            self.error_count += 1
            return False
    
    async def disconnect(self) -> bool:
        """
        Se déconnecter de Binance
        
        Returns:
            bool: True si déconnexion propre
        """
        try:
            logger.info("🔌 Déconnexion Binance...")
            
            # Arrêter WebSocket
            if self.ws and self.ws.connected:
                await self.ws.disconnect()
            
            # Fermer session HTTP
            if self.session and not self.session.closed:
                await self.session.close()
            
            # Reset status
            self.status = ExchangeStatus.DISCONNECTED
            self.connected_at = None
            
            logger.info("✅ BinanceConnector déconnecté proprement")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur déconnexion Binance: {e}")
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
                "exchange": "binance",
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
        Placer un ordre sur Binance
        
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
            
            # Placer ordre via API
            endpoint = f"{self.endpoints.rest_base_url}/api/v3/order"
            request_data = self.auth.prepare_signed_request("POST", endpoint, params["params"])
            
            result = await self._make_request(
                request_data["method"],
                request_data["url"],
                headers=request_data["headers"],
                data=request_data["data"]
            )
            
            if result["success"]:
                order_data = result["data"]
                
                # Normaliser réponse
                return {
                    "success": True,
                    "order_id": str(order_data.get("orderId")),
                    "client_order_id": order_data.get("clientOrderId"),
                    "symbol": order_data.get("symbol"),
                    "side": order_data.get("side").lower(),
                    "type": order_data.get("type").lower(),
                    "quantity": Decimal(str(order_data.get("origQty", "0"))),
                    "price": Decimal(str(order_data.get("price", "0"))) if order_data.get("price") else None,
                    "status": self._normalize_order_status(order_data.get("status")),
                    "time_in_force": order_data.get("timeInForce"),
                    "fills": order_data.get("fills", []),
                    "timestamp": datetime.now().isoformat(),
                    "exchange": "binance"
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
        Annuler un ordre sur Binance
        
        Args:
            symbol: Symbole trading
            order_id: ID ordre Binance
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
            params = {"symbol": symbol.upper()}
            if order_id:
                params["orderId"] = order_id
            elif client_order_id:
                params["origClientOrderId"] = client_order_id
            else:
                return {
                    "success": False,
                    "error": "order_id ou client_order_id requis"
                }
            
            # Annuler via API
            endpoint = f"{self.endpoints.rest_base_url}/api/v3/order"
            request_data = self.auth.prepare_signed_request("DELETE", endpoint, params)
            
            result = await self._make_request(
                request_data["method"],
                request_data["url"],
                headers=request_data["headers"],
                data=request_data["data"]
            )
            
            if result["success"]:
                order_data = result["data"]
                return {
                    "success": True,
                    "order_id": str(order_data.get("orderId")),
                    "client_order_id": order_data.get("clientOrderId"),
                    "symbol": order_data.get("symbol"),
                    "status": self._normalize_order_status(order_data.get("status")),
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
            order_id: ID ordre Binance
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
            params = {"symbol": symbol.upper()}
            if order_id:
                params["orderId"] = order_id
            elif client_order_id:
                params["origClientOrderId"] = client_order_id
            else:
                return {
                    "success": False,
                    "error": "order_id ou client_order_id requis"
                }
            
            # Requête status
            endpoint = f"{self.endpoints.rest_base_url}/api/v3/order"
            request_data = self.auth.prepare_signed_request("GET", endpoint, params)
            
            result = await self._make_request(
                request_data["method"],
                request_data["url"],
                headers=request_data["headers"]
            )
            
            if result["success"]:
                order_data = result["data"]
                return {
                    "success": True,
                    "order": {
                        "order_id": str(order_data.get("orderId")),
                        "client_order_id": order_data.get("clientOrderId"),
                        "symbol": order_data.get("symbol"),
                        "side": order_data.get("side").lower(),
                        "type": order_data.get("type").lower(),
                        "quantity": Decimal(str(order_data.get("origQty", "0"))),
                        "price": Decimal(str(order_data.get("price", "0"))) if order_data.get("price") else None,
                        "executed_quantity": Decimal(str(order_data.get("executedQty", "0"))),
                        "status": self._normalize_order_status(order_data.get("status")),
                        "time_in_force": order_data.get("timeInForce"),
                        "created_at": order_data.get("time"),
                        "updated_at": order_data.get("updateTime")
                    }
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
            params = {}
            if symbol:
                params["symbol"] = symbol.upper()
            
            # Requête ordres ouverts
            endpoint = f"{self.endpoints.rest_base_url}/api/v3/openOrders"
            request_data = self.auth.prepare_signed_request("GET", endpoint, params)
            
            result = await self._make_request(
                request_data["method"],
                request_data["url"],
                headers=request_data["headers"]
            )
            
            if result["success"]:
                orders = []
                for order_data in result["data"]:
                    orders.append({
                        "order_id": str(order_data.get("orderId")),
                        "client_order_id": order_data.get("clientOrderId"),
                        "symbol": order_data.get("symbol"),
                        "side": order_data.get("side").lower(),
                        "type": order_data.get("type").lower(),
                        "quantity": Decimal(str(order_data.get("origQty", "0"))),
                        "price": Decimal(str(order_data.get("price", "0"))) if order_data.get("price") else None,
                        "executed_quantity": Decimal(str(order_data.get("executedQty", "0"))),
                        "status": self._normalize_order_status(order_data.get("status")),
                        "time_in_force": order_data.get("timeInForce"),
                        "created_at": order_data.get("time")
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
            
            # Requête account info
            endpoint = f"{self.endpoints.rest_base_url}/api/v3/account"
            request_data = self.auth.prepare_signed_request("GET", endpoint)
            
            result = await self._make_request(
                request_data["method"],
                request_data["url"],
                headers=request_data["headers"]
            )
            
            if result["success"]:
                account_data = result["data"]
                
                # Extraire balances non-nulles
                balances = {}
                for balance in account_data.get("balances", []):
                    asset = balance["asset"]
                    free = Decimal(balance["free"])
                    locked = Decimal(balance["locked"])
                    
                    if free > 0 or locked > 0:
                        balances[asset] = {
                            "free": float(free),
                            "locked": float(locked),
                            "total": float(free + locked)
                        }
                
                return {
                    "success": True,
                    "balances": balances,
                    "account_type": account_data.get("accountType", "SPOT"),
                    "can_trade": account_data.get("canTrade", False),
                    "can_withdraw": account_data.get("canWithdraw", False),
                    "can_deposit": account_data.get("canDeposit", False),
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
            
            # Paramètres requête
            params = {}
            if symbol:
                params["symbol"] = symbol.upper()
            
            # Requête trading fees
            endpoint = f"{self.endpoints.rest_base_url}/api/v3/tradeFee"
            request_data = self.auth.prepare_signed_request("GET", endpoint, params)
            
            result = await self._make_request(
                request_data["method"],
                request_data["url"],
                headers=request_data["headers"]
            )
            
            if result["success"]:
                fees_data = result["data"]
                
                if symbol:
                    # Frais pour symbole spécifique
                    if fees_data:
                        fee_info = fees_data[0]
                        return {
                            "success": True,
                            "symbol": fee_info.get("symbol"),
                            "maker_fee": float(fee_info.get("makerCommission", "0")),
                            "taker_fee": float(fee_info.get("takerCommission", "0"))
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"Pas de frais trouvés pour {symbol}"
                        }
                else:
                    # Tous les frais
                    fees = {}
                    for fee_info in fees_data:
                        symbol_name = fee_info.get("symbol")
                        fees[symbol_name] = {
                            "maker_fee": float(fee_info.get("makerCommission", "0")),
                            "taker_fee": float(fee_info.get("takerCommission", "0"))
                        }
                    
                    return {
                        "success": True,
                        "fees": fees
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
            endpoint = f"{self.endpoints.rest_base_url}/api/v3/ticker/24hr"
            params = {"symbol": symbol.upper()}
            
            result = await self._make_request("GET", endpoint, params=params)
            
            if result["success"]:
                ticker_data = result["data"]
                return {
                    "success": True,
                    "symbol": ticker_data.get("symbol"),
                    "price": float(ticker_data.get("lastPrice", "0")),
                    "bid": float(ticker_data.get("bidPrice", "0")),
                    "ask": float(ticker_data.get("askPrice", "0")),
                    "volume": float(ticker_data.get("volume", "0")),
                    "quote_volume": float(ticker_data.get("quoteVolume", "0")),
                    "high_24h": float(ticker_data.get("highPrice", "0")),
                    "low_24h": float(ticker_data.get("lowPrice", "0")),
                    "change_24h": float(ticker_data.get("priceChange", "0")),
                    "change_percent_24h": float(ticker_data.get("priceChangePercent", "0")),
                    "timestamp": datetime.now().isoformat(),
                    "exchange": "binance"
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
    
    async def get_order_book(self, symbol: str, depth: int = 100) -> Dict[str, Any]:
        """
        Récupérer order book
        
        Args:
            symbol: Symbole trading
            depth: Profondeur (5, 10, 20, 50, 100, 500, 1000, 5000)
            
        Returns:
            Dict avec bids/asks
        """
        try:
            # Valider depth
            valid_depths = [5, 10, 20, 50, 100, 500, 1000, 5000]
            if depth not in valid_depths:
                depth = min(valid_depths, key=lambda x: abs(x - depth))
            
            # Endpoint depth
            endpoint = f"{self.endpoints.rest_base_url}/api/v3/depth"
            params = {
                "symbol": symbol.upper(),
                "limit": depth
            }
            
            result = await self._make_request("GET", endpoint, params=params)
            
            if result["success"]:
                depth_data = result["data"]
                
                # Convertir bids/asks en format standard
                bids = [[float(price), float(qty)] for price, qty in depth_data.get("bids", [])]
                asks = [[float(price), float(qty)] for price, qty in depth_data.get("asks", [])]
                
                return {
                    "success": True,
                    "symbol": symbol.upper(),
                    "bids": bids,
                    "asks": asks,
                    "last_update_id": depth_data.get("lastUpdateId"),
                    "timestamp": datetime.now().isoformat(),
                    "exchange": "binance"
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
    
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Récupérer trades récents
        
        Args:
            symbol: Symbole trading
            limit: Nombre max trades (max 1000)
            
        Returns:
            List des trades récents
        """
        try:
            # Limiter à max 1000
            limit = min(limit, 1000)
            
            # Endpoint trades
            endpoint = f"{self.endpoints.rest_base_url}/api/v3/trades"
            params = {
                "symbol": symbol.upper(),
                "limit": limit
            }
            
            result = await self._make_request("GET", endpoint, params=params)
            
            if result["success"]:
                trades_data = result["data"]
                
                trades = []
                for trade in trades_data:
                    trades.append({
                        "trade_id": str(trade.get("id")),
                        "price": float(trade.get("price", "0")),
                        "quantity": float(trade.get("qty", "0")),
                        "side": "sell" if trade.get("isBuyerMaker", False) else "buy",
                        "timestamp": trade.get("time"),
                        "exchange": "binance"
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
            
            # Connecter WebSocket
            success = await self.ws.connect()
            if success:
                logger.info("✅ WebSocket Binance démarré")
                
                # Émettre événement connexion
                await self.emit_event("connection_established", {
                    "exchange": "binance",
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
                logger.info("✅ WebSocket Binance arrêté")
                
                await self.emit_event("connection_lost", {
                    "exchange": "binance",
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
            
            # S'abonner au ticker
            success = await self.ws.subscribe_ticker(symbol)
            if success:
                # Ajouter callback pour transmettre données
                self.ws.add_callback("ticker", self._handle_ticker_update)
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
            
            success = await self.ws.subscribe_depth(symbol)
            if success:
                self.ws.add_callback("depth", self._handle_depth_update)
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
                self.ws.add_callback("trade", self._handle_trade_update)
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
        """Tester connectivité de base à Binance"""
        try:
            endpoint = f"{self.endpoints.rest_base_url}/api/v3/ping"
            result = await self._make_request("GET", endpoint)
            return result["success"]
        except:
            return False
    
    async def _load_exchange_info(self):
        """Charger informations symboles exchange"""
        try:
            endpoint = f"{self.endpoints.rest_base_url}/api/v3/exchangeInfo"
            result = await self._make_request("GET", endpoint)
            
            if result["success"]:
                exchange_info = result["data"]
                
                # Mettre en cache infos symboles
                for symbol_info in exchange_info.get("symbols", []):
                    symbol = symbol_info["symbol"]
                    self.symbol_info_cache[symbol] = {
                        "base_asset": symbol_info["baseAsset"],
                        "quote_asset": symbol_info["quoteAsset"], 
                        "status": symbol_info["status"],
                        "filters": symbol_info["filters"]
                    }
                
                self.last_cache_update = time.time()
                logger.info(f"📊 Infos exchange chargées: {len(self.symbol_info_cache)} symboles")
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement exchange info: {e}")
    
    async def _prepare_order_params(
        self, symbol: str, order_type: OrderType, side: OrderSide,
        quantity: Decimal, price: Optional[Decimal], stop_price: Optional[Decimal],
        time_in_force: str, client_order_id: Optional[str]
    ) -> Dict[str, Any]:
        """Préparer et valider paramètres d'ordre"""
        
        # Paramètres de base
        params = {
            "symbol": symbol.upper(),
            "side": side.value.upper(),
            "type": self._get_binance_order_type(order_type),
            "quantity": str(quantity)
        }
        
        # Ajouter prix selon type
        if order_type == OrderType.LIMIT:
            if price is None:
                return {"valid": False, "error": "Prix requis pour ordre LIMIT"}
            params["price"] = str(price)
            params["timeInForce"] = time_in_force
        
        elif order_type == OrderType.STOP_LOSS:
            if stop_price is None:
                return {"valid": False, "error": "Stop price requis pour STOP_LOSS"}
            params["stopPrice"] = str(stop_price)
        
        # Client order ID
        if client_order_id:
            params["newClientOrderId"] = client_order_id
        
        return {"valid": True, "params": params}
    
    def _get_binance_order_type(self, order_type: OrderType) -> str:
        """Convertir OrderType vers format Binance"""
        mapping = {
            OrderType.MARKET: "MARKET",
            OrderType.LIMIT: "LIMIT",
            OrderType.STOP_LOSS: "STOP_LOSS_LIMIT",
            OrderType.TAKE_PROFIT: "TAKE_PROFIT_LIMIT",
            OrderType.OCO: "OCO"
        }
        return mapping.get(order_type, "MARKET")
    
    def _normalize_order_status(self, binance_status: str) -> OrderStatus:
        """Normaliser status ordre Binance vers OrderStatus"""
        mapping = {
            "NEW": OrderStatus.OPEN,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "FILLED": OrderStatus.FILLED,
            "CANCELED": OrderStatus.CANCELLED,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.CANCELLED
        }
        return mapping.get(binance_status, OrderStatus.PENDING)
    
    async def _make_request(
        self, method: str, url: str, headers: Dict = None, 
        data: Dict = None, params: Dict = None
    ) -> Dict[str, Any]:
        """Faire requête HTTP avec rate limiting"""
        
        # Rate limiting
        await self._check_rate_limit()
        
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Faire requête
            async with self.session.request(
                method, url, headers=headers, json=data, params=params, timeout=30
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
                logger.debug(f"⏳ Rate limit: pause {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
        
        # Ajouter timestamp actuel
        self.request_timestamps.append(now)
    
    # === CALLBACKS WEBSOCKET ===
    
    async def _handle_ticker_update(self, data: Dict[str, Any]):
        """Gérer update ticker WebSocket"""
        await self.emit_event("price_update", {
            "exchange": "binance",
            "symbol": data.get("s"),
            "price": float(data.get("c", "0")),
            "volume": float(data.get("v", "0")),
            "change_24h": float(data.get("P", "0")),
            "timestamp": data.get("timestamp"),
            "source": "websocket"
        })
    
    async def _handle_depth_update(self, data: Dict[str, Any]):
        """Gérer update order book WebSocket"""
        await self.emit_event("order_book_update", {
            "exchange": "binance", 
            "symbol": data.get("s"),
            "bids": data.get("b", []),
            "asks": data.get("a", []),
            "timestamp": data.get("timestamp"),
            "source": "websocket"
        })
    
    async def _handle_trade_update(self, data: Dict[str, Any]):
        """Gérer update trades WebSocket"""
        await self.emit_event("trade_update", {
            "exchange": "binance",
            "symbol": data.get("s"),
            "trade_id": data.get("t"),
            "price": float(data.get("p", "0")),
            "quantity": float(data.get("q", "0")),
            "side": "sell" if data.get("m", False) else "buy",
            "timestamp": data.get("timestamp"),
            "source": "websocket"
        })