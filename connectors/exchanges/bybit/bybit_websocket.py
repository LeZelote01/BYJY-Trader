"""
📡 Bybit WebSocket V5 - Phase 2.4

WebSocket handler pour flux temps réel Bybit V5.
Support streams multiples avec authentification.

Features:
- Multiple streams simultanés
- Authentification WebSocket V5
- Reconnection automatique
- Gestion heartbeat
"""

import asyncio
import json
import time
import hmac
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Set
import websockets
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

from core.logger import get_logger

logger = get_logger(__name__)


class BybitWebSocket:
    """
    Gestionnaire WebSocket Bybit V5
    
    Gère connexions WebSocket pour différents types de streams
    avec authentification et reconnection automatique.
    """
    
    def __init__(self, base_url: str, api_key: str = None, api_secret: str = None):
        """
        Initialise le gestionnaire WebSocket Bybit
        
        Args:
            base_url: URL de base WebSocket
            api_key: Clé API pour authentification
            api_secret: Secret API pour signature
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.api_secret = api_secret
        self.websocket = None
        self.connected = False
        self.authenticated = False
        self.subscribed_topics: Set[str] = set()
        
        # Callbacks par type de topic
        self.callbacks: Dict[str, List[Callable]] = {
            "tickers": [],
            "orderbook": [],  # Order book
            "publicTrade": [],  # Trades publics
            "kline": [],      # Klines/OHLCV
            "wallet": [],     # Wallet updates (auth)
            "order": [],      # Order updates (auth)
            "execution": [],  # Execution updates (auth)
            "error": [],
            "connection": []
        }
        
        # Configuration
        self.heartbeat_interval = 20.0  # Bybit recommande 20s
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 5.0
        self.message_timeout = 60.0
        
        # Métriques
        self.metrics = {
            "messages_received": 0,
            "messages_processed": 0,
            "reconnections": 0,
            "errors": 0,
            "connected_at": None,
            "last_message_at": None
        }
        
        # Tasks asyncio
        self._listen_task = None
        self._heartbeat_task = None
        
        logger.info("📡 BybitWebSocket V5 initialisé")
    
    async def connect(self) -> bool:
        """
        Se connecter au WebSocket Bybit
        
        Returns:
            bool: True si connexion réussie
        """
        try:
            if self.connected:
                logger.warning("⚠️ WebSocket Bybit déjà connecté")
                return True
            
            # URL WebSocket appropriée
            ws_url = f"{self.base_url}/v5/public/spot"  # Public stream
            if self.api_key and self.api_secret:
                ws_url = f"{self.base_url}/v5/private"  # Private stream
            
            logger.info(f"🔗 Connexion WebSocket Bybit: {ws_url}")
            
            # Établir connexion
            self.websocket = await websockets.connect(
                ws_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5
            )
            
            self.connected = True
            self.metrics["connected_at"] = datetime.now()
            
            # Authentifier si clés fournies
            if self.api_key and self.api_secret:
                auth_success = await self._authenticate()
                if not auth_success:
                    logger.error("❌ Échec authentification WebSocket")
                    await self.disconnect()
                    return False
            
            # Démarrer tâches
            self._listen_task = asyncio.create_task(self._listen_loop())
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            # Notifier connexion
            await self._emit_event("connection", {
                "status": "connected",
                "authenticated": self.authenticated,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info("✅ WebSocket Bybit connecté avec succès")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur connexion WebSocket Bybit: {e}")
            self.connected = False
            return False
    
    async def _authenticate(self) -> bool:
        """
        Authentifier la connexion WebSocket
        
        Returns:
            bool: True si authentification réussie
        """
        try:
            # Générer signature d'authentification Bybit V5
            timestamp = str(int(time.time() * 1000))
            recv_window = "5000"
            
            # Message à signer
            message = f"GET/realtime{timestamp}"
            signature = hmac.new(
                self.api_secret.encode('utf-8'),
                message.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            # Message d'authentification
            auth_message = {
                "op": "auth",
                "args": [self.api_key, timestamp, recv_window, signature]
            }
            
            # Envoyer authentification
            await self.websocket.send(json.dumps(auth_message))
            
            # Attendre confirmation (timeout 10s)
            try:
                response = await asyncio.wait_for(
                    self.websocket.recv(),
                    timeout=10.0
                )
                
                message = json.loads(response)
                
                if message.get("success") and message.get("op") == "auth":
                    self.authenticated = True
                    logger.info("✅ Authentification WebSocket Bybit réussie")
                    return True
                else:
                    logger.error(f"❌ Réponse authentification inattendue: {message}")
                    return False
                
            except asyncio.TimeoutError:
                logger.error("❌ Timeout authentification WebSocket")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erreur authentification WebSocket: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """
        Déconnecter WebSocket proprement
        
        Returns:
            bool: True si déconnexion propre
        """
        try:
            if not self.connected:
                return True
            
            logger.info("🔌 Déconnexion WebSocket Bybit...")
            
            # Arrêter tâches
            if self._listen_task:
                self._listen_task.cancel()
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
            
            # Fermer connexion WebSocket
            if self.websocket:
                await self.websocket.close()
            
            self.connected = False
            self.authenticated = False
            self.subscribed_topics.clear()
            
            logger.info("✅ WebSocket Bybit déconnecté proprement")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur déconnexion WebSocket: {e}")
            return False
    
    async def subscribe_ticker(self, symbol: str) -> bool:
        """
        S'abonner au ticker d'un symbole
        
        Args:
            symbol: Symbole à suivre
            
        Returns:
            bool: True si abonnement réussi
        """
        topic = f"tickers.{symbol}"
        return await self._subscribe_topic(topic)
    
    async def subscribe_order_book(self, symbol: str, depth: int = 25) -> bool:
        """
        S'abonner à l'order book
        
        Args:
            symbol: Symbole à suivre
            depth: Profondeur (1, 25, 50)
            
        Returns:
            bool: True si abonnement réussi
        """
        topic = f"orderbook.{depth}.{symbol}"
        return await self._subscribe_topic(topic)
    
    async def subscribe_trades(self, symbol: str) -> bool:
        """
        S'abonner aux trades publics
        
        Args:
            symbol: Symbole à suivre
            
        Returns:
            bool: True si abonnement réussi
        """
        topic = f"publicTrade.{symbol}"
        return await self._subscribe_topic(topic)
    
    async def _subscribe_topic(self, topic: str) -> bool:
        """
        S'abonner à un topic spécifique
        
        Args:
            topic: Nom du topic Bybit
            
        Returns:
            bool: True si abonnement réussi
        """
        try:
            if not self.connected:
                logger.error("❌ WebSocket non connecté")
                return False
            
            if topic in self.subscribed_topics:
                logger.warning(f"⚠️ Déjà abonné au topic: {topic}")
                return True
            
            # Message d'abonnement Bybit V5
            subscribe_message = {
                "op": "subscribe",
                "args": [topic]
            }
            
            # Envoyer abonnement
            await self.websocket.send(json.dumps(subscribe_message))
            
            # Marquer comme abonné
            self.subscribed_topics.add(topic)
            
            logger.info(f"✅ Abonnement topic Bybit réussi: {topic}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur abonnement topic {topic}: {e}")
            return False
    
    async def _listen_loop(self):
        """
        Boucle d'écoute des messages WebSocket
        """
        reconnect_attempts = 0
        
        while self.connected:
            try:
                # Recevoir message avec timeout
                message = await asyncio.wait_for(
                    self.websocket.recv(),
                    timeout=self.message_timeout
                )
                
                # Traiter message
                await self._handle_message(message)
                
                # Reset compteur reconnexions sur succès
                reconnect_attempts = 0
                
                # Mettre à jour métriques
                self.metrics["messages_received"] += 1
                self.metrics["last_message_at"] = datetime.now()
                
            except asyncio.TimeoutError:
                logger.warning("⏰ Timeout WebSocket Bybit")
                continue
                
            except (ConnectionClosedError, ConnectionClosedOK) as e:
                logger.warning(f"🔌 Connexion WebSocket fermée: {e}")
                
                if reconnect_attempts < self.max_reconnect_attempts:
                    reconnect_attempts += 1
                    await asyncio.sleep(self.reconnect_delay * reconnect_attempts)
                    
                    if await self._reconnect():
                        self.metrics["reconnections"] += 1
                        continue
                else:
                    self.connected = False
                    break
                    
            except Exception as e:
                logger.error(f"❌ Erreur écoute WebSocket: {e}")
                self.metrics["errors"] += 1
                await asyncio.sleep(1.0)
        
        logger.info("🔚 Boucle écoute WebSocket Bybit terminée")
    
    async def _handle_message(self, raw_message: str):
        """
        Traiter un message WebSocket reçu
        """
        try:
            message = json.loads(raw_message)
            
            # Messages système (op: auth, subscribe, ping, etc.)
            if "op" in message:
                if message["op"] in ["auth", "subscribe", "unsubscribe"]:
                    if message.get("success"):
                        logger.debug(f"📝 Opération réussie: {message['op']}")
                    else:
                        logger.error(f"❌ Opération échouée: {message}")
                    return
                elif message["op"] == "pong":
                    logger.debug("💓 Pong reçu")
                    return
            
            # Messages de données (topic + data)
            if "topic" in message and "data" in message:
                topic = message["topic"]
                data = message["data"]
                
                # Déterminer type de topic
                topic_type = self._get_topic_type(topic)
                if topic_type:
                    enriched_data = {
                        "topic": topic,
                        "type": message.get("type"),  # snapshot, delta
                        "data": data,
                        "timestamp": message.get("ts", datetime.now().timestamp() * 1000),
                        "exchange": "bybit"
                    }
                    
                    await self._emit_event(topic_type, enriched_data)
                    self.metrics["messages_processed"] += 1
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Message JSON invalide: {e}")
        except Exception as e:
            logger.error(f"❌ Erreur traitement message: {e}")
    
    def _get_topic_type(self, topic: str) -> Optional[str]:
        """
        Déterminer le type de topic depuis son nom
        """
        if topic.startswith("tickers"):
            return "tickers"
        elif topic.startswith("orderbook"):
            return "orderbook"
        elif topic.startswith("publicTrade"):
            return "publicTrade"
        elif topic.startswith("kline"):
            return "kline"
        elif topic.startswith("wallet"):
            return "wallet"
        elif topic.startswith("order"):
            return "order"
        elif topic.startswith("execution"):
            return "execution"
        else:
            logger.warning(f"⚠️ Type de topic inconnu: {topic}")
            return None
    
    async def _reconnect(self) -> bool:
        """
        Reconnecter après perte de connexion
        """
        try:
            if self.websocket:
                await self.websocket.close()
            
            self.connected = False
            self.authenticated = False
            
            success = await self.connect()
            if not success:
                return False
            
            # Re-s'abonner aux topics précédents
            topics_to_resubscribe = self.subscribed_topics.copy()
            self.subscribed_topics.clear()
            
            for topic in topics_to_resubscribe:
                await self._subscribe_topic(topic)
            
            logger.info(f"✅ Reconnexion Bybit réussie avec {len(topics_to_resubscribe)} topics")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur reconnexion: {e}")
            return False
    
    async def _heartbeat_loop(self):
        """
        Boucle heartbeat pour maintenir connexion
        """
        while self.connected:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                if self.connected and self.websocket:
                    # Envoyer ping
                    ping_message = {"op": "ping"}
                    await self.websocket.send(json.dumps(ping_message))
                    logger.debug("💓 Ping Bybit envoyé")
                
            except Exception as e:
                logger.error(f"❌ Erreur heartbeat: {e}")
                break
        
        logger.info("💓 Heartbeat WebSocket Bybit terminé")
    
    def add_callback(self, topic_type: str, callback: Callable):
        """
        Ajouter callback pour un type de topic
        """
        if topic_type in self.callbacks:
            self.callbacks[topic_type].append(callback)
            logger.debug(f"📝 Callback ajouté pour {topic_type}")
    
    async def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """
        Émettre événement vers callbacks
        """
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    await callback(data)
                except Exception as e:
                    logger.error(f"❌ Erreur callback {event_type}: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Récupérer status du WebSocket
        """
        return {
            "connected": self.connected,
            "authenticated": self.authenticated,
            "subscribed_topics": list(self.subscribed_topics),
            "topics_count": len(self.subscribed_topics),
            "metrics": self.metrics,
            "base_url": self.base_url
        }