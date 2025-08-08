"""
📡 Kraken WebSocket - Phase 2.4

WebSocket handler pour flux temps réel Kraken Pro.
Support streams multiples avec reconnection automatique.

Features:
- Multiple streams simultanés
- Reconnection automatique  
- Gestion heartbeat
- Buffer messages et rate limiting
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Set
import websockets
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

from core.logger import get_logger

logger = get_logger(__name__)


class KrakenWebSocket:
    """
    Gestionnaire WebSocket Kraken Pro
    
    Gère connexions WebSocket pour différents types de streams
    avec reconnection automatique.
    """
    
    def __init__(self, base_url: str):
        """
        Initialise le gestionnaire WebSocket Kraken
        
        Args:
            base_url: URL de base WebSocket Kraken
        """
        self.base_url = base_url.rstrip('/')
        self.websocket = None
        self.connected = False
        self.subscribed_channels: Set[str] = set()
        
        # Callbacks par type de stream
        self.callbacks: Dict[str, List[Callable]] = {
            "ticker": [],
            "book": [],      # Order book
            "trade": [],     # Trades
            "ohlc": [],      # OHLC/Klines
            "spread": [],    # Best bid/ask
            "error": [],
            "connection": []
        }
        
        # Configuration
        self.heartbeat_interval = 30.0
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
        
        logger.info("📡 KrakenWebSocket initialisé")
    
    async def connect(self) -> bool:
        """
        Se connecter au WebSocket Kraken
        
        Returns:
            bool: True si connexion réussie
        """
        try:
            if self.connected:
                logger.warning("⚠️ WebSocket Kraken déjà connecté")
                return True
            
            logger.info(f"🔗 Connexion WebSocket Kraken: {self.base_url}")
            
            # Établir connexion
            self.websocket = await websockets.connect(
                self.base_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5
            )
            
            self.connected = True
            self.metrics["connected_at"] = datetime.now()
            
            # Démarrer tâches
            self._listen_task = asyncio.create_task(self._listen_loop())
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            # Notifier connexion
            await self._emit_event("connection", {
                "status": "connected",
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info("✅ WebSocket Kraken connecté avec succès")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur connexion WebSocket Kraken: {e}")
            self.connected = False
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
            
            logger.info("🔌 Déconnexion WebSocket Kraken...")
            
            # Arrêter tâches
            if self._listen_task:
                self._listen_task.cancel()
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
            
            # Fermer connexion WebSocket
            if self.websocket:
                await self.websocket.close()
            
            self.connected = False
            self.subscribed_channels.clear()
            
            logger.info("✅ WebSocket Kraken déconnecté proprement")
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
        return await self._subscribe_channel("ticker", [symbol])
    
    async def subscribe_order_book(self, symbol: str) -> bool:
        """
        S'abonner à l'order book
        
        Args:
            symbol: Symbole à suivre
            
        Returns:
            bool: True si abonnement réussi
        """
        return await self._subscribe_channel("book", [symbol])
    
    async def subscribe_trades(self, symbol: str) -> bool:
        """
        S'abonner aux trades
        
        Args:
            symbol: Symbole à suivre
            
        Returns:
            bool: True si abonnement réussi
        """
        return await self._subscribe_channel("trade", [symbol])
    
    async def _subscribe_channel(self, channel: str, pair_list: List[str]) -> bool:
        """
        S'abonner à un channel spécifique
        
        Args:
            channel: Nom du channel
            pair_list: Liste des symboles
            
        Returns:
            bool: True si abonnement réussi
        """
        try:
            if not self.connected:
                logger.error("❌ WebSocket non connecté")
                return False
            
            channel_key = f"{channel}_{'-'.join(pair_list)}"
            
            if channel_key in self.subscribed_channels:
                logger.warning(f"⚠️ Déjà abonné au channel: {channel_key}")
                return True
            
            # Message d'abonnement Kraken
            subscribe_message = {
                "event": "subscribe",
                "pair": pair_list,
                "subscription": {"name": channel}
            }
            
            # Envoyer abonnement
            await self.websocket.send(json.dumps(subscribe_message))
            
            # Marquer comme abonné
            self.subscribed_channels.add(channel_key)
            
            logger.info(f"✅ Abonnement channel Kraken réussi: {channel_key}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur abonnement channel {channel}: {e}")
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
                logger.warning("⏰ Timeout WebSocket Kraken")
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
        
        logger.info("🔚 Boucle écoute WebSocket Kraken terminée")
    
    async def _handle_message(self, raw_message: str):
        """
        Traiter un message WebSocket reçu
        """
        try:
            message = json.loads(raw_message)
            
            # Messages système
            if isinstance(message, dict):
                if "event" in message:
                    if message["event"] in ["subscriptionStatus", "systemStatus", "heartbeat"]:
                        logger.debug(f"📝 Message système: {message['event']}")
                        return
                    elif message["event"] == "error":
                        logger.error(f"❌ Erreur WebSocket Kraken: {message}")
                        await self._emit_event("error", message)
                        return
            
            # Messages de données (array format)
            if isinstance(message, list) and len(message) >= 4:
                channel_id = message[0]
                data = message[1]
                channel_name = message[2]
                pair = message[3]
                
                # Router selon le channel
                enriched_data = {
                    "channel_id": channel_id,
                    "channel_name": channel_name,
                    "pair": pair,
                    "data": data,
                    "timestamp": datetime.now().isoformat(),
                    "exchange": "kraken"
                }
                
                await self._emit_event(channel_name, enriched_data)
                self.metrics["messages_processed"] += 1
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Message JSON invalide: {e}")
        except Exception as e:
            logger.error(f"❌ Erreur traitement message: {e}")
    
    async def _reconnect(self) -> bool:
        """
        Reconnecter après perte de connexion
        """
        try:
            if self.websocket:
                await self.websocket.close()
            
            self.connected = False
            success = await self.connect()
            
            if success:
                # Re-s'abonner aux channels précédents
                channels_to_resubscribe = self.subscribed_channels.copy()
                self.subscribed_channels.clear()
                
                for channel_key in channels_to_resubscribe:
                    parts = channel_key.split("_", 1)
                    if len(parts) == 2:
                        channel = parts[0]
                        symbols = parts[1].split("-")
                        await self._subscribe_channel(channel, symbols)
                
                logger.info(f"✅ Reconnexion Kraken réussie avec {len(channels_to_resubscribe)} channels")
                return True
            
            return False
            
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
                    ping_message = {"event": "ping", "reqid": int(time.time())}
                    await self.websocket.send(json.dumps(ping_message))
                    logger.debug("💓 Heartbeat Kraken envoyé")
                
            except Exception as e:
                logger.error(f"❌ Erreur heartbeat: {e}")
                break
        
        logger.info("💓 Heartbeat WebSocket Kraken terminé")
    
    def add_callback(self, channel_type: str, callback: Callable):
        """
        Ajouter callback pour un type de channel
        """
        if channel_type in self.callbacks:
            self.callbacks[channel_type].append(callback)
            logger.debug(f"📝 Callback ajouté pour {channel_type}")
    
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
            "subscribed_channels": list(self.subscribed_channels),
            "channels_count": len(self.subscribed_channels),
            "metrics": self.metrics,
            "base_url": self.base_url
        }