"""
🔌 Coinbase Advanced Trading Connector - Phase 2.4

Connecteur complet pour Coinbase Advanced Trading API.
Support trading professionnel avec WebSocket temps réel.

Features:
- Trading Spot complet (Market, Limit orders)
- WebSocket feeds multi-channels
- Authentification OAuth2 sécurisée
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
from .coinbase_auth import CoinbaseAuth
from .coinbase_websocket import CoinbaseWebSocket

logger = get_logger(__name__)


class CoinbaseConnector(BaseConnector):
    """
    Connecteur Coinbase Advanced Trading complet
    
    Implémente toutes les fonctionnalités requises par BaseConnector
    pour trading Coinbase avec WebSocket temps réel.
    """
    
    def __init__(
        self,
        api_key: str = None,
        api_secret: str = None,
        passphrase: str = None,
        sandbox: bool = True,
        rate_limit: int = 600
    ):
        """
        Initialise connecteur Coinbase
        
        Args:
            api_key: Clé API Coinbase
            api_secret: Secret API Coinbase (base64)
            passphrase: Passphrase (legacy Coinbase Pro)
            sandbox: Mode sandbox (True) ou production (False)
            rate_limit: Limite requêtes par minute
        """
        super().__init__(
            exchange_name="coinbase",
            api_key=api_key,
            api_secret=api_secret,
            sandbox=sandbox,
            rate_limit=rate_limit
        )
        
        # Configuration Coinbase
        self.config = ExchangeConfig.get_config("coinbase")
        self.endpoints = ExchangeConfig.get_endpoints("coinbase", sandbox)
        self.limits = ExchangeConfig.get_limits("coinbase")
        
        # Composants spécialisés
        self.auth = CoinbaseAuth(api_key, api_secret, passphrase) if api_key and api_secret else None
        self.ws = CoinbaseWebSocket(self.endpoints.websocket_base_url, api_key, api_secret)
        
        # Rate limiting
        self.request_timestamps: List[float] = []
        self.requests_per_minute = self.limits.requests_per_minute
        
        # Cache pour optimiser perfs
        self.product_info_cache: Dict[str, Dict] = {}
        self.cache_ttl = 300  # 5 minutes
        self.last_cache_update = 0
        
        logger.info(f"🔌 CoinbaseConnector initialisé (sandbox={sandbox})")
    
    # === IMPLÉMENTATION MÉTHODES DE CONNEXION ===
    
    async def connect(self) -> bool:
        """
        Se connecter à Coinbase API
        
        Returns:
            bool: True si connexion réussie
        """
        try:
            logger.info("🔗 Connexion à Coinbase...")
            
            # Test connectivité de base
            if not await self._test_basic_connectivity():
                return False
            
            # Test authentification si API keys fournies
            if self.auth:
                auth_test = await self.auth.test_connectivity(self.endpoints.rest_base_url)
                if not auth_test["success"]:
                    logger.error(f"❌ Échec authentification Coinbase: {auth_test['error']}")
                    return False
                
                logger.info(f"✅ Authentification réussie - Type: {auth_test['account_type']}")
            
            # Charger informations produits
            await self._load_products_info()
            
            # Mettre à jour status
            self.status = ExchangeStatus.CONNECTED
            self.connected_at = datetime.now()
            
            logger.info("✅ CoinbaseConnector connecté avec succès")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur connexion Coinbase: {e}")
            self.status = ExchangeStatus.ERROR
            self.error_count += 1
            return False
    
    async def disconnect(self) -> bool:
        """
        Se déconnecter de Coinbase
        
        Returns:
            bool: True si déconnexion propre
        """
        try:
            logger.info("🔌 Déconnexion Coinbase...")
            
            # Arrêter WebSocket
            if self.ws and self.ws.connected:
                await self.ws.disconnect()
            
            # Fermer session HTTP
            if self.session and not self.session.closed:
                await self.session.close()
            
            # Reset status
            self.status = ExchangeStatus.DISCONNECTED
            self.connected_at = None
            
            logger.info("✅ CoinbaseConnector déconnecté proprement")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur déconnexion Coinbase: {e}")
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
                "exchange": "coinbase",
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
        Placer un ordre sur Coinbase
        
        Args:
            symbol: Symbole trading (ex: BTC-USD)
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
            endpoint = f"{self.endpoints.rest_base_url}/orders"
            request_data = self.auth.prepare_signed_request(
                "POST", endpoint, body=params["params"]
            )
            
            result = await self._make_request(
                request_data["method"],
                request_data["url"],
                headers=request_data["headers"],
                json=request_data["data"]
            )
            
            if result["success"]:
                order_data = result["data"]
                
                # Normaliser réponse Coinbase
                return {
                    "success": True,
                    "order_id": order_data.get("order_id"),
                    "client_order_id": order_data.get("client_order_id"),
                    "symbol": order_data.get("product_id"),
                    "side": order_data.get("side"),
                    "type": order_data.get("order_type"),
                    "quantity": Decimal(str(order_data.get("order_configuration", {}).get("market_market_ioc", {}).get("quote_size", "0"))),
                    "price": Decimal(str(order_data.get("order_configuration", {}).get("limit_limit_gtc", {}).get("limit_price", "0"))) if order_data.get("order_configuration", {}).get("limit_limit_gtc") else None,
                    "status": self._normalize_order_status(order_data.get("status")),
                    "time_in_force": order_data.get("time_in_force"),
                    "timestamp": datetime.now().isoformat(),
                    "exchange": "coinbase"
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
        Annuler un ordre sur Coinbase
        
        Args:
            symbol: Symbole trading
            order_id: ID ordre Coinbase
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
            
            # Utiliser order_id priority sur client_order_id
            target_id = order_id or client_order_id
            if not target_id:
                return {
                    "success": False,
                    "error": "order_id ou client_order_id requis"
                }
            
            # Annuler via API
            endpoint = f"{self.endpoints.rest_base_url}/orders/batch_cancel"
            cancel_body = {
                "order_ids": [target_id]
            }
            
            request_data = self.auth.prepare_signed_request("POST", endpoint, body=cancel_body)
            
            result = await self._make_request(
                request_data["method"],
                request_data["url"],
                headers=request_data["headers"],
                json=request_data["data"]
            )
            
            if result["success"]:
                results = result["data"].get("results", [])
                if results:
                    cancel_result = results[0]
                    return {
                        "success": cancel_result.get("success", False),
                        "order_id": cancel_result.get("order_id"),
                        "failure_reason": cancel_result.get("failure_reason"),
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    return {
                        "success": False,
                        "error": "Aucun résultat d'annulation reçu"
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
            order_id: ID ordre Coinbase
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
            
            target_id = order_id or client_order_id
            if not target_id:
                return {
                    "success": False,
                    "error": "order_id ou client_order_id requis"
                }
            
            # Requête status ordre
            endpoint = f"{self.endpoints.rest_base_url}/orders/historical/{target_id}"
            request_data = self.auth.prepare_signed_request("GET", endpoint)
            
            result = await self._make_request(
                request_data["method"],
                request_data["url"],
                headers=request_data["headers"]
            )
            
            if result["success"]:
                order_data = result["data"].get("order", {})
                return {
                    "success": True,
                    "order": {
                        "order_id": order_data.get("order_id"),
                        "client_order_id": order_data.get("client_order_id"),
                        "symbol": order_data.get("product_id"),
                        "side": order_data.get("side"),
                        "type": order_data.get("order_type"),
                        "quantity": Decimal(str(order_data.get("size", "0"))),
                        "price": Decimal(str(order_data.get("price", "0"))) if order_data.get("price") else None,
                        "executed_quantity": Decimal(str(order_data.get("filled_size", "0"))),
                        "status": self._normalize_order_status(order_data.get("status")),
                        "time_in_force": order_data.get("time_in_force"),
                        "created_at": order_data.get("created_time"),
                        "updated_at": order_data.get("completion_percentage")
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
                params["product_id"] = symbol
            
            # Requête ordres ouverts
            endpoint = f"{self.endpoints.rest_base_url}/orders/historical/batch"
            request_data = self.auth.prepare_signed_request("GET", endpoint, params=params)
            
            result = await self._make_request(
                request_data["method"],
                request_data["url"],
                headers=request_data["headers"]
            )
            
            if result["success"]:
                orders = []
                for order_data in result["data"].get("orders", []):
                    # Filtrer seulement les ordres ouverts
                    status = order_data.get("status", "")
                    if status in ["OPEN", "PENDING"]:
                        orders.append({
                            "order_id": order_data.get("order_id"),
                            "client_order_id": order_data.get("client_order_id"),
                            "symbol": order_data.get("product_id"),
                            "side": order_data.get("side"),
                            "type": order_data.get("order_type"),
                            "quantity": Decimal(str(order_data.get("size", "0"))),
                            "price": Decimal(str(order_data.get("price", "0"))) if order_data.get("price") else None,
                            "executed_quantity": Decimal(str(order_data.get("filled_size", "0"))),
                            "status": self._normalize_order_status(order_data.get("status")),
                            "time_in_force": order_data.get("time_in_force"),
                            "created_at": order_data.get("created_time")
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
            
            # Requête accounts
            endpoint = f"{self.endpoints.rest_base_url}/accounts"
            request_data = self.auth.prepare_signed_request("GET", endpoint)
            
            result = await self._make_request(
                request_data["method"],
                request_data["url"],
                headers=request_data["headers"]
            )
            
            if result["success"]:
                accounts = result["data"].get("accounts", [])
                
                # Extraire balances non-nulles
                balances = {}
                for account in accounts:
                    currency = account["currency"]
                    available = Decimal(account["available_balance"]["value"])
                    hold = Decimal(account["hold"]["value"])
                    
                    if available > 0 or hold > 0:
                        balances[currency] = {
                            "free": float(available),
                            "locked": float(hold),
                            "total": float(available + hold)
                        }
                
                return {
                    "success": True,
                    "balances": balances,
                    "account_type": "COINBASE_ADVANCED",
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
            
            # Coinbase Advanced utilise une structure de frais simple
            # Les frais exacts dépendent du volume de trading
            
            return {
                "success": True,
                "fees": {
                    "maker_fee": 0.005,  # 0.5% maker fee par défaut
                    "taker_fee": 0.005,  # 0.5% taker fee par défaut
                    "note": "Frais réels dépendent du volume de trading mensuel"
                },
                "exchange": "coinbase"
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
            symbol: Symbole trading (ex: BTC-USD)
            
        Returns:
            Dict avec prix, volume, changement 24h
        """
        try:
            # Endpoint ticker
            endpoint = f"{self.endpoints.rest_base_url}/products/{symbol}/ticker"
            
            result = await self._make_request("GET", endpoint)
            
            if result["success"]:
                ticker_data = result["data"]
                return {
                    "success": True,
                    "symbol": symbol,
                    "price": float(ticker_data.get("price", "0")),
                    "bid": float(ticker_data.get("bid", "0")),
                    "ask": float(ticker_data.get("ask", "0")),
                    "volume": float(ticker_data.get("volume", "0")),
                    "high_24h": 0,  # Coinbase ticker ne fournit pas ces données
                    "low_24h": 0,
                    "change_24h": 0,
                    "change_percent_24h": 0,
                    "timestamp": datetime.now().isoformat(),
                    "exchange": "coinbase"
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
    
    async def get_order_book(self, symbol: str, depth: int = 50) -> Dict[str, Any]:
        """
        Récupérer order book
        
        Args:
            symbol: Symbole trading
            depth: Profondeur (limité par Coinbase)
            
        Returns:
            Dict avec bids/asks
        """
        try:
            # Endpoint order book
            endpoint = f"{self.endpoints.rest_base_url}/products/{symbol}/book"
            
            result = await self._make_request("GET", endpoint)
            
            if result["success"]:
                book_data = result["data"]
                
                # Convertir bids/asks en format standard
                bids = [[float(bid["price"]), float(bid["size"])] for bid in book_data.get("bids", [])]
                asks = [[float(ask["price"]), float(ask["size"])] for ask in book_data.get("asks", [])]
                
                return {
                    "success": True,
                    "symbol": symbol,
                    "bids": bids[:depth],
                    "asks": asks[:depth],
                    "timestamp": datetime.now().isoformat(),
                    "exchange": "coinbase"
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
            limit: Nombre max trades
            
        Returns:
            List des trades récents
        """
        try:
            # Endpoint trades récents
            endpoint = f"{self.endpoints.rest_base_url}/products/{symbol}/trades"
            
            result = await self._make_request("GET", endpoint)
            
            if result["success"]:
                trades_data = result["data"].get("trades", [])
                
                trades = []
                for i, trade in enumerate(trades_data[:limit]):
                    trades.append({
                        "trade_id": trade.get("trade_id"),
                        "price": float(trade.get("price", "0")),
                        "quantity": float(trade.get("size", "0")),
                        "side": trade.get("side"),
                        "timestamp": trade.get("time"),
                        "exchange": "coinbase"
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
                logger.info("✅ WebSocket Coinbase démarré")
                
                # Émettre événement connexion
                await self.emit_event("connection_established", {
                    "exchange": "coinbase",
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
                logger.info("✅ WebSocket Coinbase arrêté")
                
                await self.emit_event("connection_lost", {
                    "exchange": "coinbase",
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
            
            success = await self.ws.subscribe_order_book(symbol)
            if success:
                self.ws.add_callback("level2", self._handle_level2_update)
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
                self.ws.add_callback("matches", self._handle_match_update)
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
        """Tester connectivité de base à Coinbase"""
        try:
            endpoint = f"{self.endpoints.rest_base_url}/time"
            result = await self._make_request("GET", endpoint)
            return result["success"]
        except:
            return False
    
    async def _load_products_info(self):
        """Charger informations produits exchange"""
        try:
            endpoint = f"{self.endpoints.rest_base_url}/products"
            result = await self._make_request("GET", endpoint)
            
            if result["success"]:
                products = result["data"].get("products", [])
                
                # Mettre en cache infos produits
                for product in products:
                    product_id = product["product_id"]
                    self.product_info_cache[product_id] = {
                        "base_currency": product["base_currency_id"],
                        "quote_currency": product["quote_currency_id"],
                        "status": product["status"],
                        "base_increment": product.get("base_increment"),
                        "quote_increment": product.get("quote_increment"),
                        "min_market_funds": product.get("min_market_funds")
                    }
                
                self.last_cache_update = time.time()
                logger.info(f"📊 Infos produits Coinbase chargées: {len(self.product_info_cache)} produits")
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement infos produits: {e}")
    
    async def _prepare_order_params(
        self, symbol: str, order_type: OrderType, side: OrderSide,
        quantity: Decimal, price: Optional[Decimal], stop_price: Optional[Decimal],
        time_in_force: str, client_order_id: Optional[str]
    ) -> Dict[str, Any]:
        """Préparer et valider paramètres d'ordre Coinbase"""
        
        try:
            # Structure ordre Coinbase Advanced
            params = {
                "client_order_id": client_order_id or f"BYJY_{int(time.time())}_{symbol}",
                "product_id": symbol,
                "side": side.value.upper()
            }
            
            # Configuration ordre selon le type
            if order_type == OrderType.MARKET:
                if side == OrderSide.BUY:
                    # Pour les achats market, utiliser quote_size (montant à dépenser)
                    params["order_configuration"] = {
                        "market_market_ioc": {
                            "quote_size": str(quantity)
                        }
                    }
                else:
                    # Pour les ventes market, utiliser base_size (quantité à vendre)
                    params["order_configuration"] = {
                        "market_market_ioc": {
                            "base_size": str(quantity)
                        }
                    }
            
            elif order_type == OrderType.LIMIT:
                if price is None:
                    return {"valid": False, "error": "Prix requis pour ordre LIMIT"}
                
                params["order_configuration"] = {
                    "limit_limit_gtc": {
                        "base_size": str(quantity),
                        "limit_price": str(price)
                    }
                }
            
            else:
                return {"valid": False, "error": f"Type d'ordre {order_type} non supporté par Coinbase"}
            
            return {"valid": True, "params": params}
            
        except Exception as e:
            logger.error(f"❌ Erreur préparation paramètres ordre: {e}")
            return {"valid": False, "error": str(e)}
    
    def _normalize_order_status(self, coinbase_status: str) -> OrderStatus:
        """Normaliser status ordre Coinbase vers OrderStatus"""
        mapping = {
            "OPEN": OrderStatus.OPEN,
            "FILLED": OrderStatus.FILLED,
            "CANCELLED": OrderStatus.CANCELLED,
            "EXPIRED": OrderStatus.CANCELLED,
            "FAILED": OrderStatus.REJECTED,
            "PENDING": OrderStatus.PENDING,
            "QUEUED": OrderStatus.PENDING
        }
        return mapping.get(coinbase_status, OrderStatus.PENDING)
    
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
                logger.debug(f"⏳ Rate limit Coinbase: pause {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
        
        # Ajouter timestamp actuel
        self.request_timestamps.append(now)
    
    # === CALLBACKS WEBSOCKET ===
    
    async def _handle_ticker_update(self, data: Dict[str, Any]):
        """Gérer update ticker WebSocket"""
        await self.emit_event("price_update", {
            "exchange": "coinbase",
            "symbol": data.get("product_id"),
            "price": float(data.get("price", "0")),
            "volume": float(data.get("volume_24h", "0")),
            "change_24h": float(data.get("price_percent_chg_24h", "0")),
            "timestamp": data.get("timestamp"),
            "source": "websocket"
        })
    
    async def _handle_level2_update(self, data: Dict[str, Any]):
        """Gérer update order book WebSocket"""
        await self.emit_event("order_book_update", {
            "exchange": "coinbase",
            "symbol": data.get("product_id"),
            "changes": data.get("changes", []),
            "timestamp": data.get("timestamp"),
            "source": "websocket"
        })
    
    async def _handle_match_update(self, data: Dict[str, Any]):
        """Gérer update trades WebSocket"""
        await self.emit_event("trade_update", {
            "exchange": "coinbase",
            "symbol": data.get("product_id"),
            "trade_id": data.get("trade_id"),
            "price": float(data.get("price", "0")),
            "quantity": float(data.get("size", "0")),
            "side": data.get("side"),
            "timestamp": data.get("timestamp"),
            "source": "websocket"
        })