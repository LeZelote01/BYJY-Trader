"""
🔌 Kraken Pro Connector - Phase 2.4 Priority 3

Connecteur complet pour exchange Kraken Pro.
Support trading Spot avec WebSocket temps réel.

Features:
- Trading Spot complet (Market, Limit, Stop orders)
- WebSocket feeds multi-streams
- Authentification sécurisée Kraken API
- Rate limiting automatique
- Error handling et resilience
"""

import asyncio
import aiohttp
import time
import base64
import hashlib
import hmac
from datetime import datetime
from typing import Dict, List, Optional, Any
from decimal import Decimal
from urllib.parse import urlencode

from core.logger import get_logger
from connectors.base.base_connector import BaseConnector, OrderType, OrderSide, OrderStatus, ExchangeStatus
from connectors.base.exchange_config import ExchangeConfig
from .kraken_auth import KrakenAuth
from .kraken_websocket import KrakenWebSocket

logger = get_logger(__name__)


class KrakenConnector(BaseConnector):
    """
    Connecteur Kraken Pro complet
    
    Implémente toutes les fonctionnalités requises par BaseConnector
    pour trading Spot Kraken avec WebSocket temps réel.
    """
    
    def __init__(
        self,
        api_key: str = None,
        api_secret: str = None, 
        sandbox: bool = True,
        rate_limit: int = 15  # Kraken: 15 calls/sec
    ):
        """
        Initialise connecteur Kraken
        
        Args:
            api_key: Clé API Kraken
            api_secret: Secret API Kraken (privé)
            sandbox: Mode demo (True) ou live (False)
            rate_limit: Limite requêtes par seconde
        """
        super().__init__(
            exchange_name="kraken",
            api_key=api_key,
            api_secret=api_secret,
            sandbox=sandbox,
            rate_limit=rate_limit * 60  # Convertir en req/minute
        )
        
        # Configuration Kraken
        self.config = ExchangeConfig.get_config("kraken")
        self.endpoints = ExchangeConfig.get_endpoints("kraken", sandbox)
        self.limits = ExchangeConfig.get_limits("kraken")
        
        # Composants spécialisés
        self.auth = KrakenAuth(api_key, api_secret) if api_key and api_secret else None
        self.ws = KrakenWebSocket(self.endpoints.websocket_base_url)
        
        # Rate limiting - Kraken utilise un système de counter décrémental
        self.request_timestamps: List[float] = []
        self.requests_per_second = 15
        self.api_counter = 15  # Counter API Kraken
        
        # Cache pour optimiser performances
        self.asset_pairs_cache: Dict[str, Dict] = {}
        self.cache_ttl = 300  # 5 minutes
        self.last_cache_update = 0
        
        logger.info(f"🔌 KrakenConnector initialisé (sandbox={sandbox})")
    
    # === IMPLÉMENTATION MÉTHODES DE CONNEXION ===
    
    async def connect(self) -> bool:
        """
        Se connecter à Kraken API
        
        Returns:
            bool: True si connexion réussie
        """
        try:
            logger.info("🔗 Connexion à Kraken...")
            
            # Test connectivité de base
            if not await self._test_basic_connectivity():
                return False
            
            # Test authentification si API keys fournies
            if self.auth:
                auth_test = await self.auth.test_connectivity(self.endpoints.rest_base_url)
                if not auth_test["success"]:
                    logger.error(f"❌ Échec authentification Kraken: {auth_test['error']}")
                    return False
                
                logger.info(f"✅ Authentification réussie - Balance: {auth_test.get('balance_count', 'N/A')} assets")
            
            # Charger informations asset pairs
            await self._load_asset_pairs()
            
            # Mettre à jour status
            self.status = ExchangeStatus.CONNECTED
            self.connected_at = datetime.now()
            
            logger.info("✅ KrakenConnector connecté avec succès")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur connexion Kraken: {e}")
            self.status = ExchangeStatus.ERROR
            self.error_count += 1
            return False
    
    async def disconnect(self) -> bool:
        """
        Se déconnecter de Kraken
        
        Returns:
            bool: True si déconnexion propre
        """
        try:
            logger.info("🔌 Déconnexion Kraken...")
            
            # Arrêter WebSocket
            if self.ws and self.ws.connected:
                await self.ws.disconnect()
            
            # Fermer session HTTP
            if self.session and not self.session.closed:
                await self.session.close()
            
            # Reset status
            self.status = ExchangeStatus.DISCONNECTED
            self.connected_at = None
            
            logger.info("✅ KrakenConnector déconnecté proprement")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur déconnexion Kraken: {e}")
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
                "exchange": "kraken",
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
        Placer un ordre sur Kraken
        
        Args:
            symbol: Symbole trading (ex: BTCUSD)
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
            endpoint = "/0/private/AddOrder"
            request_data = self.auth.prepare_signed_request("POST", endpoint, params["params"])
            
            result = await self._make_request(
                request_data["method"],
                f"{self.endpoints.rest_base_url}{endpoint}",
                headers=request_data["headers"],
                data=request_data["data"]
            )
            
            if result["success"]:
                response_data = result["data"]
                
                if response_data.get("error"):
                    return {
                        "success": False,
                        "error": "; ".join(response_data["error"]),
                        "order_id": None
                    }
                
                order_result = response_data.get("result", {})
                tx_ids = order_result.get("txid", [])
                
                return {
                    "success": True,
                    "order_id": tx_ids[0] if tx_ids else None,
                    "client_order_id": client_order_id,
                    "symbol": symbol,
                    "side": side.value,
                    "type": order_type.value,
                    "quantity": quantity,
                    "price": price,
                    "status": OrderStatus.PENDING.value,
                    "time_in_force": time_in_force,
                    "timestamp": datetime.now().isoformat(),
                    "exchange": "kraken",
                    "kraken_description": order_result.get("descr", {})
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
        Annuler un ordre sur Kraken
        
        Args:
            symbol: Symbole trading
            order_id: ID ordre Kraken (txid)
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
            
            target_id = order_id or client_order_id
            if not target_id:
                return {
                    "success": False,
                    "error": "order_id requis pour Kraken"
                }
            
            # Annuler via API
            endpoint = "/0/private/CancelOrder"
            params = {"txid": target_id}
            
            request_data = self.auth.prepare_signed_request("POST", endpoint, params)
            
            result = await self._make_request(
                request_data["method"],
                f"{self.endpoints.rest_base_url}{endpoint}",
                headers=request_data["headers"],
                data=request_data["data"]
            )
            
            if result["success"]:
                response_data = result["data"]
                
                if response_data.get("error"):
                    return {
                        "success": False,
                        "error": "; ".join(response_data["error"])
                    }
                
                return {
                    "success": True,
                    "order_id": target_id,
                    "cancelled_count": response_data.get("result", {}).get("count", 0),
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
            order_id: ID ordre Kraken (txid)
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
                    "error": "order_id requis pour Kraken"
                }
            
            # Requête status ordre
            endpoint = "/0/private/QueryOrders"
            params = {"txid": target_id}
            
            request_data = self.auth.prepare_signed_request("POST", endpoint, params)
            
            result = await self._make_request(
                request_data["method"],
                f"{self.endpoints.rest_base_url}{endpoint}",
                headers=request_data["headers"],
                data=request_data["data"]
            )
            
            if result["success"]:
                response_data = result["data"]
                
                if response_data.get("error"):
                    return {
                        "success": False,
                        "error": "; ".join(response_data["error"])
                    }
                
                orders = response_data.get("result", {})
                if target_id in orders:
                    order_data = orders[target_id]
                    
                    return {
                        "success": True,
                        "order": {
                            "order_id": target_id,
                            "symbol": order_data.get("descr", {}).get("pair"),
                            "side": order_data.get("descr", {}).get("type"),
                            "type": order_data.get("descr", {}).get("ordertype"),
                            "quantity": Decimal(str(order_data.get("vol", "0"))),
                            "price": Decimal(str(order_data.get("descr", {}).get("price", "0"))) if order_data.get("descr", {}).get("price") else None,
                            "executed_quantity": Decimal(str(order_data.get("vol_exec", "0"))),
                            "status": self._normalize_order_status(order_data.get("status")),
                            "time_in_force": order_data.get("descr", {}).get("order"),
                            "created_at": order_data.get("opentm"),
                            "cost": Decimal(str(order_data.get("cost", "0"))),
                            "fee": Decimal(str(order_data.get("fee", "0")))
                        }
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Ordre {target_id} non trouvé"
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
            
            # Requête ordres ouverts
            endpoint = "/0/private/OpenOrders"
            request_data = self.auth.prepare_signed_request("POST", endpoint, {})
            
            result = await self._make_request(
                request_data["method"],
                f"{self.endpoints.rest_base_url}{endpoint}",
                headers=request_data["headers"],
                data=request_data["data"]
            )
            
            if result["success"]:
                response_data = result["data"]
                
                if response_data.get("error"):
                    logger.error(f"❌ Erreur API Kraken: {response_data['error']}")
                    return []
                
                orders = []
                open_orders = response_data.get("result", {}).get("open", {})
                
                for order_id, order_data in open_orders.items():
                    # Filtrer par symbole si spécifié
                    order_symbol = order_data.get("descr", {}).get("pair")
                    if symbol and order_symbol != symbol:
                        continue
                    
                    orders.append({
                        "order_id": order_id,
                        "symbol": order_symbol,
                        "side": order_data.get("descr", {}).get("type"),
                        "type": order_data.get("descr", {}).get("ordertype"),
                        "quantity": Decimal(str(order_data.get("vol", "0"))),
                        "price": Decimal(str(order_data.get("descr", {}).get("price", "0"))) if order_data.get("descr", {}).get("price") else None,
                        "executed_quantity": Decimal(str(order_data.get("vol_exec", "0"))),
                        "status": self._normalize_order_status(order_data.get("status")),
                        "created_at": order_data.get("opentm")
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
            
            # Requête balance
            endpoint = "/0/private/Balance"
            request_data = self.auth.prepare_signed_request("POST", endpoint, {})
            
            result = await self._make_request(
                request_data["method"],
                f"{self.endpoints.rest_base_url}{endpoint}",
                headers=request_data["headers"],
                data=request_data["data"]
            )
            
            if result["success"]:
                response_data = result["data"]
                
                if response_data.get("error"):
                    return {
                        "success": False,
                        "error": "; ".join(response_data["error"])
                    }
                
                # Extraire balances
                balances = {}
                balance_data = response_data.get("result", {})
                
                for asset, balance in balance_data.items():
                    balance_val = Decimal(str(balance))
                    if balance_val > 0:
                        balances[asset] = {
                            "free": float(balance_val),
                            "locked": 0.0,  # Kraken ne sépare pas free/locked dans balance
                            "total": float(balance_val)
                        }
                
                return {
                    "success": True,
                    "balances": balances,
                    "account_type": "KRAKEN_SPOT",
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
            
            # Requête trading fees
            endpoint = "/0/private/TradeVolume"
            params = {}
            if symbol:
                params["pair"] = symbol
            
            request_data = self.auth.prepare_signed_request("POST", endpoint, params)
            
            result = await self._make_request(
                request_data["method"],
                f"{self.endpoints.rest_base_url}{endpoint}",
                headers=request_data["headers"],
                data=request_data["data"]
            )
            
            if result["success"]:
                response_data = result["data"]
                
                if response_data.get("error"):
                    return {
                        "success": False,
                        "error": "; ".join(response_data["error"])
                    }
                
                result_data = response_data.get("result", {})
                fees_data = result_data.get("fees", {})
                
                if symbol and symbol in fees_data:
                    # Frais pour symbole spécifique
                    fee_info = fees_data[symbol]
                    return {
                        "success": True,
                        "symbol": symbol,
                        "maker_fee": float(fee_info.get("fee", "0.0016")),  # Default 0.16%
                        "taker_fee": float(fee_info.get("fee", "0.0026"))   # Default 0.26%
                    }
                else:
                    # Structure frais générale
                    fees = {}
                    for pair, fee_info in fees_data.items():
                        fees[pair] = {
                            "maker_fee": float(fee_info.get("fee", "0.0016")),
                            "taker_fee": float(fee_info.get("fee", "0.0026"))
                        }
                    
                    return {
                        "success": True,
                        "fees": fees,
                        "currency": result_data.get("currency", "USD"),
                        "volume": float(result_data.get("volume", "0"))
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
            # Endpoint ticker
            endpoint = "/0/public/Ticker"
            params = {"pair": symbol}
            
            result = await self._make_request("GET", f"{self.endpoints.rest_base_url}{endpoint}", params=params)
            
            if result["success"]:
                response_data = result["data"]
                
                if response_data.get("error"):
                    return {
                        "success": False,
                        "error": "; ".join(response_data["error"])
                    }
                
                ticker_data = response_data.get("result", {})
                
                # Kraken retourne les données avec le nom normalisé du symbole
                symbol_data = None
                for key, data in ticker_data.items():
                    if key != "error":
                        symbol_data = data
                        break
                
                if symbol_data:
                    return {
                        "success": True,
                        "symbol": symbol,
                        "price": float(symbol_data.get("c", ["0"])[0]),  # Last price
                        "bid": float(symbol_data.get("b", ["0"])[0]),    # Best bid
                        "ask": float(symbol_data.get("a", ["0"])[0]),    # Best ask
                        "volume": float(symbol_data.get("v", ["0"])[1]), # 24h volume
                        "high_24h": float(symbol_data.get("h", ["0"])[1]), # 24h high
                        "low_24h": float(symbol_data.get("l", ["0"])[1]),  # 24h low
                        "open_24h": float(symbol_data.get("o", "0")),      # 24h open
                        "change_24h": 0.0,  # Calculé si nécessaire
                        "change_percent_24h": 0.0,
                        "timestamp": datetime.now().isoformat(),
                        "exchange": "kraken"
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
    
    async def get_order_book(self, symbol: str, depth: int = 100) -> Dict[str, Any]:
        """
        Récupérer order book
        
        Args:
            symbol: Symbole trading
            depth: Profondeur order book (max 500 sur Kraken)
            
        Returns:
            Dict avec bids/asks
        """
        try:
            # Endpoint order book
            endpoint = "/0/public/Depth"
            params = {"pair": symbol, "count": min(depth, 500)}
            
            result = await self._make_request("GET", f"{self.endpoints.rest_base_url}{endpoint}", params=params)
            
            if result["success"]:
                response_data = result["data"]
                
                if response_data.get("error"):
                    return {
                        "success": False,
                        "error": "; ".join(response_data["error"])
                    }
                
                orderbook_data = response_data.get("result", {})
                
                # Kraken retourne avec nom normalisé
                book_data = None
                for key, data in orderbook_data.items():
                    if key != "error":
                        book_data = data
                        break
                
                if book_data:
                    # Convertir format Kraken [price, volume, timestamp]
                    bids = [[float(bid[0]), float(bid[1])] for bid in book_data.get("bids", [])]
                    asks = [[float(ask[0]), float(ask[1])] for ask in book_data.get("asks", [])]
                    
                    return {
                        "success": True,
                        "symbol": symbol,
                        "bids": bids,
                        "asks": asks,
                        "timestamp": datetime.now().isoformat(),
                        "exchange": "kraken"
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Pas de données order book pour {symbol}"
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
            limit: Nombre max trades (max 1000 sur Kraken)
            
        Returns:
            List des trades récents
        """
        try:
            # Endpoint trades récents
            endpoint = "/0/public/Trades"
            params = {"pair": symbol, "count": min(limit, 1000)}
            
            result = await self._make_request("GET", f"{self.endpoints.rest_base_url}{endpoint}", params=params)
            
            if result["success"]:
                response_data = result["data"]
                
                if response_data.get("error"):
                    logger.error(f"❌ Erreur API Kraken: {response_data['error']}")
                    return []
                
                trades_data = response_data.get("result", {})
                
                # Kraken retourne avec nom normalisé
                trades_list = None
                for key, data in trades_data.items():
                    if key != "error" and isinstance(data, list):
                        trades_list = data
                        break
                
                if trades_list:
                    trades = []
                    for trade in trades_list:
                        # Format Kraken: [price, volume, time, buy/sell, market/limit, miscellaneous]
                        trades.append({
                            "trade_id": f"{symbol}_{trade[2]}_{len(trades)}",
                            "price": float(trade[0]),
                            "quantity": float(trade[1]),
                            "side": "buy" if trade[3] == "b" else "sell",
                            "timestamp": trade[2],
                            "type": "market" if trade[4] == "m" else "limit",
                            "exchange": "kraken"
                        })
                    
                    return trades
                else:
                    return []
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
                logger.info("✅ WebSocket Kraken démarré")
                
                await self.emit_event("connection_established", {
                    "exchange": "kraken",
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
                logger.info("✅ WebSocket Kraken arrêté")
                
                await self.emit_event("connection_lost", {
                    "exchange": "kraken",
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
                self.ws.add_callback("book", self._handle_book_update)
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
        """Tester connectivité de base à Kraken"""
        try:
            endpoint = "/0/public/Time"
            result = await self._make_request("GET", f"{self.endpoints.rest_base_url}{endpoint}")
            return result["success"] and not result["data"].get("error")
        except:
            return False
    
    async def _load_asset_pairs(self):
        """Charger informations asset pairs"""
        try:
            endpoint = "/0/public/AssetPairs"
            result = await self._make_request("GET", f"{self.endpoints.rest_base_url}{endpoint}")
            
            if result["success"]:
                response_data = result["data"]
                
                if not response_data.get("error"):
                    pairs_data = response_data.get("result", {})
                    
                    # Mettre en cache infos pairs
                    for pair, info in pairs_data.items():
                        self.asset_pairs_cache[pair] = {
                            "altname": info.get("altname"),
                            "base": info.get("base"),
                            "quote": info.get("quote"),
                            "lot_decimals": info.get("lot_decimals", 8),
                            "pair_decimals": info.get("pair_decimals", 5),
                            "fee_volume_currency": info.get("fee_volume_currency"),
                            "margin_call": info.get("margin_call", 80),
                            "margin_stop": info.get("margin_stop", 40)
                        }
                    
                    self.last_cache_update = time.time()
                    logger.info(f"📊 Infos asset pairs Kraken chargées: {len(self.asset_pairs_cache)} paires")
                
        except Exception as e:
            logger.error(f"❌ Erreur chargement asset pairs: {e}")
    
    async def _prepare_order_params(
        self, symbol: str, order_type: OrderType, side: OrderSide,
        quantity: Decimal, price: Optional[Decimal], stop_price: Optional[Decimal],
        time_in_force: str, client_order_id: Optional[str]
    ) -> Dict[str, Any]:
        """Préparer et valider paramètres d'ordre Kraken"""
        
        try:
            params = {
                "pair": symbol,
                "type": side.value,  # buy/sell
                "volume": str(quantity)
            }
            
            # Type d'ordre
            if order_type == OrderType.MARKET:
                params["ordertype"] = "market"
            elif order_type == OrderType.LIMIT:
                if price is None:
                    return {"valid": False, "error": "Prix requis pour ordre LIMIT"}
                params["ordertype"] = "limit"
                params["price"] = str(price)
            elif order_type == OrderType.STOP_LOSS:
                if stop_price is None:
                    return {"valid": False, "error": "Prix stop requis pour ordre STOP_LOSS"}
                params["ordertype"] = "stop-loss"
                params["price"] = str(stop_price)
            else:
                return {"valid": False, "error": f"Type d'ordre {order_type} non supporté"}
            
            # Time in force (optionnel pour Kraken)
            if time_in_force and time_in_force != "GTC":
                if time_in_force == "IOC":
                    params["timeinforce"] = "IOC"
                elif time_in_force == "FOK":
                    params["timeinforce"] = "FOK"
            
            # Client order ID
            if client_order_id:
                params["userref"] = client_order_id
            
            return {"valid": True, "params": params}
            
        except Exception as e:
            logger.error(f"❌ Erreur préparation paramètres ordre: {e}")
            return {"valid": False, "error": str(e)}
    
    def _normalize_order_status(self, kraken_status: str) -> OrderStatus:
        """Normaliser status ordre Kraken vers OrderStatus"""
        mapping = {
            "open": OrderStatus.OPEN,
            "closed": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "cancelled": OrderStatus.CANCELLED,
            "expired": OrderStatus.CANCELLED,
            "pending": OrderStatus.PENDING
        }
        return mapping.get(kraken_status, OrderStatus.PENDING)
    
    async def _make_request(
        self, method: str, url: str, headers: Dict = None, 
        data: Dict = None, params: Dict = None
    ) -> Dict[str, Any]:
        """Faire requête HTTP avec rate limiting Kraken"""
        
        # Rate limiting Kraken
        await self._check_rate_limit()
        
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Faire requête
            async with self.session.request(
                method, url, headers=headers, data=data, params=params, timeout=30
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
        """Vérifier et respecter rate limits Kraken (15 req/sec)"""
        now = time.time()
        
        # Nettoyer anciennes timestamps (plus d'1 seconde)
        self.request_timestamps = [
            ts for ts in self.request_timestamps 
            if now - ts < 1
        ]
        
        # Vérifier limite
        if len(self.request_timestamps) >= self.requests_per_second:
            sleep_time = 1 - (now - self.request_timestamps[0])
            if sleep_time > 0:
                logger.debug(f"⏳ Rate limit Kraken: pause {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
        
        # Ajouter timestamp actuel
        self.request_timestamps.append(now)
    
    # === CALLBACKS WEBSOCKET ===
    
    async def _handle_ticker_update(self, data: Dict[str, Any]):
        """Gérer update ticker WebSocket"""
        await self.emit_event("price_update", {
            "exchange": "kraken",
            "symbol": data.get("symbol"),
            "price": data.get("price"),
            "volume": data.get("volume"),
            "change_24h": data.get("change_24h", 0),
            "timestamp": data.get("timestamp"),
            "source": "websocket"
        })
    
    async def _handle_book_update(self, data: Dict[str, Any]):
        """Gérer update order book WebSocket"""
        await self.emit_event("order_book_update", {
            "exchange": "kraken",
            "symbol": data.get("symbol"),
            "bids": data.get("bids", []),
            "asks": data.get("asks", []),
            "timestamp": data.get("timestamp"),
            "source": "websocket"
        })
    
    async def _handle_trade_update(self, data: Dict[str, Any]):
        """Gérer update trades WebSocket"""
        await self.emit_event("trade_update", {
            "exchange": "kraken",
            "symbol": data.get("symbol"),
            "trades": data.get("trades", []),
            "timestamp": data.get("timestamp"),
            "source": "websocket"
        })