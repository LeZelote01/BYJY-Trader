"""
📡 Feed Manager - Phase 2.4

Gestionnaire de flux de données temps réel multi-exchange.
Agrège et normalise les données WebSocket de tous les exchanges.

Features:
- Aggregation multi-exchange
- Normalisation format données
- Reconnection automatique
- Rate limiting et buffer management
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from collections import defaultdict, deque
import websockets
from decimal import Decimal

from core.logger import get_logger
from .base_connector import BaseConnector

logger = get_logger(__name__)


class FeedType:
    """Types de feeds supportés"""
    TICKER = "ticker"
    ORDER_BOOK = "order_book"
    TRADES = "trades"
    ACCOUNT = "account"
    ORDERS = "orders"


class FeedManager:
    """
    Gestionnaire centralisé des flux de données temps réel
    
    Agrège les données de multiple exchanges et les normalise
    pour une interface uniforme.
    """
    
    def __init__(self, buffer_size: int = 1000):
        """
        Initialise le feed manager
        
        Args:
            buffer_size: Taille max buffer par feed
        """
        self.buffer_size = buffer_size
        self.active_feeds: Dict[str, Dict[str, Any]] = {}
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Buffers par type de feed
        self.buffers: Dict[str, Dict[str, deque]] = {
            FeedType.TICKER: defaultdict(lambda: deque(maxlen=buffer_size)),
            FeedType.ORDER_BOOK: defaultdict(lambda: deque(maxlen=50)),  # Plus petit buffer
            FeedType.TRADES: defaultdict(lambda: deque(maxlen=buffer_size)),
            FeedType.ACCOUNT: defaultdict(lambda: deque(maxlen=100)),
            FeedType.ORDERS: defaultdict(lambda: deque(maxlen=200))
        }
        
        # Métriques et monitoring
        self.metrics = {
            "messages_received": 0,
            "messages_processed": 0,
            "reconnections": 0,
            "errors": 0,
            "last_activity": None
        }
        
        # Configuration reconnection
        self.reconnect_interval = 5.0  # secondes
        self.max_reconnect_attempts = 10
        
        logger.info("📡 FeedManager initialisé avec multi-exchange support")
    
    async def start_feed(
        self,
        connector: BaseConnector,
        feed_type: str,
        symbol: str,
        callback: Optional[Callable] = None
    ) -> bool:
        """
        Démarrer un feed de données
        
        Args:
            connector: Connecteur exchange
            feed_type: Type de feed (ticker, order_book, etc.)
            symbol: Symbole à suivre
            callback: Callback optionnel pour les données
            
        Returns:
            bool: True si démarrage réussi
        """
        try:
            feed_key = f"{connector.exchange_name}_{feed_type}_{symbol}"
            
            if feed_key in self.active_feeds:
                logger.warning(f"⚠️ Feed {feed_key} déjà actif")
                return False
            
            # Démarrer WebSocket si pas déjà fait
            if not connector.websocket:
                websocket_started = await connector.start_websocket()
                if not websocket_started:
                    logger.error(f"❌ Impossible de démarrer WebSocket pour {connector.exchange_name}")
                    return False
            
            # S'abonner au feed spécifique
            subscription_success = False
            if feed_type == FeedType.TICKER:
                subscription_success = await connector.subscribe_ticker(symbol)
            elif feed_type == FeedType.ORDER_BOOK:
                subscription_success = await connector.subscribe_order_book(symbol)
            elif feed_type == FeedType.TRADES:
                subscription_success = await connector.subscribe_trades(symbol)
            
            if not subscription_success:
                logger.error(f"❌ Échec abonnement feed {feed_type} pour {symbol}")
                return False
            
            # Enregistrer feed actif
            self.active_feeds[feed_key] = {
                "connector": connector,
                "feed_type": feed_type,
                "symbol": symbol,
                "started_at": datetime.now(),
                "last_message": None,
                "message_count": 0
            }
            
            # Ajouter callback si fourni
            if callback:
                self.subscribers[feed_key].append(callback)
            
            # Démarrer processing loop pour ce feed
            asyncio.create_task(self._process_feed(feed_key))
            
            logger.info(f"✅ Feed {feed_key} démarré avec succès")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur démarrage feed: {e}")
            return False
    
    async def stop_feed(
        self,
        connector: BaseConnector,
        feed_type: str,
        symbol: str
    ) -> bool:
        """
        Arrêter un feed de données
        
        Args:
            connector: Connecteur exchange
            feed_type: Type de feed
            symbol: Symbole
            
        Returns:
            bool: True si arrêt réussi
        """
        try:
            feed_key = f"{connector.exchange_name}_{feed_type}_{symbol}"
            
            if feed_key not in self.active_feeds:
                logger.warning(f"⚠️ Feed {feed_key} n'est pas actif")
                return False
            
            # Supprimer de la liste active
            del self.active_feeds[feed_key]
            
            # Nettoyer subscribers
            if feed_key in self.subscribers:
                del self.subscribers[feed_key]
            
            # Nettoyer buffer
            if feed_key in self.buffers[feed_type]:
                del self.buffers[feed_type][feed_key]
            
            logger.info(f"✅ Feed {feed_key} arrêté")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur arrêt feed: {e}")
            return False
    
    async def _process_feed(self, feed_key: str):
        """
        Boucle de traitement pour un feed spécifique
        """
        feed_info = self.active_feeds[feed_key]
        connector = feed_info["connector"]
        feed_type = feed_info["feed_type"]
        
        reconnect_attempts = 0
        
        while feed_key in self.active_feeds:
            try:
                # Vérifier connexion WebSocket
                if not connector.websocket or connector.websocket.closed:
                    logger.warning(f"⚠️ Connexion WebSocket fermée pour {feed_key}, reconnection...")
                    
                    if reconnect_attempts >= self.max_reconnect_attempts:
                        logger.error(f"❌ Max reconnections atteint pour {feed_key}")
                        break
                    
                    # Tentative reconnection
                    success = await self._reconnect_feed(connector)
                    if success:
                        reconnect_attempts = 0
                        self.metrics["reconnections"] += 1
                    else:
                        reconnect_attempts += 1
                        await asyncio.sleep(self.reconnect_interval * reconnect_attempts)
                        continue
                
                # Lire message WebSocket avec timeout
                try:
                    message = await asyncio.wait_for(
                        connector.websocket.recv(), 
                        timeout=30.0
                    )
                    
                    # Traiter message
                    await self._handle_message(feed_key, message)
                    
                    # Reset reconnect counter sur succès
                    reconnect_attempts = 0
                    
                except asyncio.TimeoutError:
                    logger.warning(f"⏰ Timeout WebSocket pour {feed_key}")
                    continue
                    
            except Exception as e:
                logger.error(f"❌ Erreur traitement feed {feed_key}: {e}")
                self.metrics["errors"] += 1
                await asyncio.sleep(1.0)
        
        logger.info(f"🔚 Processing feed {feed_key} terminé")
    
    async def _handle_message(self, feed_key: str, raw_message: str):
        """
        Traiter un message WebSocket
        """
        try:
            # Parse JSON
            message = json.loads(raw_message)
            
            # Extraire infos feed
            feed_info = self.active_feeds[feed_key]
            feed_type = feed_info["feed_type"]
            symbol = feed_info["symbol"]
            connector = feed_info["connector"]
            
            # Normaliser message selon le type
            normalized_data = await self._normalize_message(
                connector.exchange_name, feed_type, symbol, message
            )
            
            if not normalized_data:
                return
            
            # Stocker dans buffer
            buffer_key = f"{connector.exchange_name}_{symbol}"
            self.buffers[feed_type][buffer_key].append(normalized_data)
            
            # Notifier subscribers
            await self._notify_subscribers(feed_key, normalized_data)
            
            # Mettre à jour métriques
            feed_info["last_message"] = datetime.now()
            feed_info["message_count"] += 1
            self.metrics["messages_received"] += 1
            self.metrics["messages_processed"] += 1
            self.metrics["last_activity"] = datetime.now()
            
        except json.JSONDecodeError:
            logger.error(f"❌ Message JSON invalide pour {feed_key}: {raw_message[:100]}")
        except Exception as e:
            logger.error(f"❌ Erreur handling message {feed_key}: {e}")
    
    async def _normalize_message(
        self,
        exchange_name: str,
        feed_type: str,
        symbol: str,
        message: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Normaliser message selon format uniforme
        """
        try:
            timestamp = datetime.now()
            
            base_data = {
                "exchange": exchange_name,
                "symbol": symbol,
                "timestamp": timestamp.isoformat(),
                "feed_type": feed_type
            }
            
            # Normalisation selon type de feed
            if feed_type == FeedType.TICKER:
                return {
                    **base_data,
                    "price": message.get("price", message.get("c")),
                    "volume": message.get("volume", message.get("v")),
                    "change": message.get("change", message.get("P")),
                    "high": message.get("high", message.get("h")),
                    "low": message.get("low", message.get("l")),
                    "bid": message.get("bid", message.get("b")),
                    "ask": message.get("ask", message.get("a"))
                }
            
            elif feed_type == FeedType.ORDER_BOOK:
                return {
                    **base_data,
                    "bids": message.get("bids", []),
                    "asks": message.get("asks", []),
                    "last_update_id": message.get("lastUpdateId", message.get("u"))
                }
            
            elif feed_type == FeedType.TRADES:
                return {
                    **base_data,
                    "price": message.get("price", message.get("p")),
                    "quantity": message.get("quantity", message.get("q")),
                    "side": message.get("side", "buy" if message.get("m") else "sell"),
                    "trade_id": message.get("trade_id", message.get("t")),
                    "trade_time": message.get("trade_time", message.get("T"))
                }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Erreur normalisation message: {e}")
            return None
    
    async def _notify_subscribers(self, feed_key: str, data: Dict[str, Any]):
        """
        Notifier tous les subscribers d'un feed
        """
        if feed_key in self.subscribers:
            for callback in self.subscribers[feed_key]:
                try:
                    await callback(data)
                except Exception as e:
                    logger.error(f"❌ Erreur callback subscriber: {e}")
    
    async def _reconnect_feed(self, connector: BaseConnector) -> bool:
        """
        Reconnecter un feed après perte de connexion
        """
        try:
            # Fermer ancienne connexion
            if connector.websocket:
                await connector.stop_websocket()
            
            # Redémarrer WebSocket
            success = await connector.start_websocket()
            if success:
                logger.info(f"✅ Reconnection WebSocket réussie pour {connector.exchange_name}")
                
                # Re-souscrire aux feeds actifs pour ce connecteur
                for feed_key, feed_info in self.active_feeds.items():
                    if feed_info["connector"] == connector:
                        symbol = feed_info["symbol"]
                        feed_type = feed_info["feed_type"]
                        
                        if feed_type == FeedType.TICKER:
                            await connector.subscribe_ticker(symbol)
                        elif feed_type == FeedType.ORDER_BOOK:
                            await connector.subscribe_order_book(symbol)
                        elif feed_type == FeedType.TRADES:
                            await connector.subscribe_trades(symbol)
                
                return True
            else:
                logger.error(f"❌ Échec reconnection WebSocket pour {connector.exchange_name}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erreur reconnection: {e}")
            return False
    
    def subscribe_to_feed(
        self,
        exchange_name: str,
        feed_type: str,
        symbol: str,
        callback: Callable
    ):
        """
        S'abonner aux données d'un feed
        
        Args:
            exchange_name: Nom exchange
            feed_type: Type de feed  
            symbol: Symbole
            callback: Fonction callback
        """
        feed_key = f"{exchange_name}_{feed_type}_{symbol}"
        self.subscribers[feed_key].append(callback)
        logger.debug(f"📝 Abonnement feed {feed_key} ajouté")
    
    def get_latest_data(
        self,
        exchange_name: str,
        feed_type: str,
        symbol: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Récupérer dernières données d'un feed
        
        Args:
            exchange_name: Nom exchange
            feed_type: Type de feed
            symbol: Symbole  
            limit: Nombre max d'entrées
            
        Returns:
            List des dernières données
        """
        buffer_key = f"{exchange_name}_{symbol}"
        buffer = self.buffers[feed_type].get(buffer_key, deque())
        return list(buffer)[-limit:]
    
    def get_feed_status(self) -> Dict[str, Any]:
        """
        Récupérer status de tous les feeds
        """
        active_feeds_info = {}
        for feed_key, feed_info in self.active_feeds.items():
            active_feeds_info[feed_key] = {
                "exchange": feed_info["connector"].exchange_name,
                "feed_type": feed_info["feed_type"],
                "symbol": feed_info["symbol"],
                "started_at": feed_info["started_at"].isoformat(),
                "last_message": feed_info["last_message"].isoformat() if feed_info["last_message"] else None,
                "message_count": feed_info["message_count"]
            }
        
        return {
            "active_feeds": active_feeds_info,
            "active_feeds_count": len(self.active_feeds),
            "metrics": {
                **self.metrics,
                "last_activity": self.metrics["last_activity"].isoformat() if self.metrics["last_activity"] else None
            },
            "buffer_sizes": {
                feed_type: {k: len(v) for k, v in buffers.items()}
                for feed_type, buffers in self.buffers.items()
            }
        }