"""
📡 Binance WebSocket - Phase 2.4

WebSocket handler pour flux temps réel Binance.
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


class BinanceWebSocket:
    """
    Gestionnaire WebSocket Binance
    
    Gère connexions WebSocket multiples pour différents types de streams
    avec reconnection automatique et gestion d'erreurs robuste.
    """
    
    def __init__(self, base_url: str):
        """
        Initialise le gestionnaire WebSocket
        
        Args:
            base_url: URL de base WebSocket Binance
        """
        self.base_url = base_url.rstrip('/')
        self.websocket = None
        self.connected = False
        self.subscribed_streams: Set[str] = set()
        
        # Callbacks par type de stream
        self.callbacks: Dict[str, List[Callable]] = {
            "ticker": [],
            "depth": [],
            "trade": [],
            "kline": [],
            "bookTicker": [],
            "error": [],
            "connection": []
        }
        
        # Configuration
        self.heartbeat_interval = 30.0  # secondes
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
        
        logger.info("📡 BinanceWebSocket initialisé")
    
    async def connect(self) -> bool:
        """
        Se connecter au WebSocket Binance
        
        Returns:
            bool: True si connexion réussie
        """
        try:
            if self.connected:
                logger.warning("⚠️ WebSocket déjà connecté")
                return True
            
            # URL de base pour streams combinés
            ws_url = f"{self.base_url}/ws/byjy_trader_stream"
            
            logger.info(f"🔗 Connexion WebSocket Binance: {ws_url}")
            
            # Établir connexion
            self.websocket = await websockets.connect(
                ws_url,
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
            
            logger.info("✅ WebSocket Binance connecté avec succès")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur connexion WebSocket: {e}")
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
            
            logger.info("🔌 Déconnexion WebSocket Binance...")
            
            # Arrêter tâches
            if self._listen_task:
                self._listen_task.cancel()
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
            
            # Fermer connexion WebSocket
            if self.websocket:
                await self.websocket.close()
            
            self.connected = False
            self.subscribed_streams.clear()
            
            # Notifier déconnexion
            await self._emit_event("connection", {
                "status": "disconnected",
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info("✅ WebSocket Binance déconnecté proprement")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur déconnexion WebSocket: {e}")
            return False
    
    async def subscribe_ticker(self, symbol: str) -> bool:
        """
        S'abonner au ticker d'un symbole
        
        Args:
            symbol: Symbole à suivre (ex: btcusdt)
            
        Returns:
            bool: True si abonnement réussi
        """
        stream_name = f"{symbol.lower()}@ticker"
        return await self._subscribe_stream(stream_name, "ticker")
    
    async def subscribe_depth(self, symbol: str, levels: int = 20) -> bool:
        """
        S'abonner à l'order book depth
        
        Args:
            symbol: Symbole à suivre
            levels: Nombre de niveaux (5, 10, 20)
            
        Returns:
            bool: True si abonnement réussi
        """
        stream_name = f"{symbol.lower()}@depth{levels}@100ms"
        return await self._subscribe_stream(stream_name, "depth")
    
    async def subscribe_trades(self, symbol: str) -> bool:
        """
        S'abonner aux trades temps réel
        
        Args:
            symbol: Symbole à suivre
            
        Returns:
            bool: True si abonnement réussi
        """
        stream_name = f"{symbol.lower()}@trade"
        return await self._subscribe_stream(stream_name, "trade")
    
    async def subscribe_klines(self, symbol: str, interval: str = "1m") -> bool:
        """
        S'abonner aux klines/candlesticks
        
        Args:
            symbol: Symbole à suivre
            interval: Intervalle (1m, 5m, 1h, 1d, etc.)
            
        Returns:
            bool: True si abonnement réussi
        """
        stream_name = f"{symbol.lower()}@kline_{interval}"
        return await self._subscribe_stream(stream_name, "kline")
    
    async def subscribe_book_ticker(self, symbol: str) -> bool:
        """
        S'abonner au best bid/ask prix
        
        Args:
            symbol: Symbole à suivre
            
        Returns:
            bool: True si abonnement réussi
        """
        stream_name = f"{symbol.lower()}@bookTicker"
        return await self._subscribe_stream(stream_name, "bookTicker")
    
    async def _subscribe_stream(self, stream_name: str, stream_type: str) -> bool:
        """
        S'abonner à un stream spécifique
        
        Args:
            stream_name: Nom du stream Binance
            stream_type: Type de stream pour callbacks
            
        Returns:
            bool: True si abonnement réussi
        """
        try:
            if not self.connected:
                logger.error("❌ WebSocket non connecté")
                return False
            
            if stream_name in self.subscribed_streams:
                logger.warning(f"⚠️ Déjà abonné au stream: {stream_name}")
                return True
            
            # Message d'abonnement Binance
            subscribe_message = {
                "method": "SUBSCRIBE",
                "params": [stream_name],
                "id": int(time.time())
            }
            
            # Envoyer abonnement
            await self.websocket.send(json.dumps(subscribe_message))
            
            # Marquer comme abonné
            self.subscribed_streams.add(stream_name)
            
            logger.info(f"✅ Abonnement stream réussi: {stream_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur abonnement stream {stream_name}: {e}")
            return False
    
    async def unsubscribe_stream(self, stream_name: str) -> bool:
        """
        Se désabonner d'un stream
        
        Args:
            stream_name: Nom du stream à désabonner
            
        Returns:
            bool: True si désabonnement réussi
        """
        try:
            if not self.connected:
                return True
            
            if stream_name not in self.subscribed_streams:
                return True
            
            # Message de désabonnement
            unsubscribe_message = {
                "method": "UNSUBSCRIBE",
                "params": [stream_name],
                "id": int(time.time())
            }
            
            await self.websocket.send(json.dumps(unsubscribe_message))
            self.subscribed_streams.discard(stream_name)
            
            logger.info(f"✅ Désabonnement stream réussi: {stream_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur désabonnement stream {stream_name}: {e}")
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
                logger.warning("⏰ Timeout WebSocket - Pas de message reçu")
                continue
                
            except (ConnectionClosedError, ConnectionClosedOK) as e:
                logger.warning(f"🔌 Connexion WebSocket fermée: {e}")
                
                if reconnect_attempts < self.max_reconnect_attempts:
                    reconnect_attempts += 1
                    logger.info(f"🔄 Tentative reconnexion {reconnect_attempts}/{self.max_reconnect_attempts}")
                    
                    # Attendre avant reconnexion
                    await asyncio.sleep(self.reconnect_delay * reconnect_attempts)
                    
                    # Tenter reconnexion
                    if await self._reconnect():
                        self.metrics["reconnections"] += 1
                        continue
                    
                else:
                    logger.error(f"❌ Max reconnexions atteint ({self.max_reconnect_attempts})")
                    self.connected = False
                    break
                    
            except Exception as e:
                logger.error(f"❌ Erreur écoute WebSocket: {e}")
                self.metrics["errors"] += 1
                await asyncio.sleep(1.0)
        
        logger.info("🔚 Boucle écoute WebSocket terminée")
    
    async def _handle_message(self, raw_message: str):
        """
        Traiter un message WebSocket reçu
        
        Args:
            raw_message: Message brut JSON
        """
        try:
            message = json.loads(raw_message)
            
            # Ignorer messages de confirmation d'abonnement
            if "result" in message and message["result"] is None:
                logger.debug("📝 Confirmation abonnement reçue")
                return
            
            # Messages avec erreur
            if "error" in message:
                logger.error(f"❌ Erreur WebSocket: {message['error']}")
                await self._emit_event("error", message["error"])
                return
            
            # Messages de données de stream
            if "stream" in message and "data" in message:
                stream_name = message["stream"]
                data = message["data"]
                
                # Déterminer type de stream
                stream_type = self._get_stream_type(stream_name)
                if stream_type:
                    # Enrichir data avec métadonnées
                    enriched_data = {
                        **data,
                        "stream": stream_name,
                        "timestamp": datetime.now().isoformat(),
                        "exchange": "binance"
                    }
                    
                    # Émettre vers callbacks
                    await self._emit_event(stream_type, enriched_data)
                    self.metrics["messages_processed"] += 1
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Message JSON invalide: {e}")
            logger.debug(f"Message brut: {raw_message[:200]}")
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement message: {e}")
    
    def _get_stream_type(self, stream_name: str) -> Optional[str]:
        """
        Déterminer le type de stream depuis son nom
        
        Args:
            stream_name: Nom du stream Binance
            
        Returns:
            str: Type de stream pour callbacks
        """
        if "@ticker" in stream_name:
            return "ticker"
        elif "@depth" in stream_name:
            return "depth"
        elif "@trade" in stream_name:
            return "trade"
        elif "@kline" in stream_name:
            return "kline"
        elif "@bookTicker" in stream_name:
            return "bookTicker"
        else:
            logger.warning(f"⚠️ Type de stream inconnu: {stream_name}")
            return None
    
    async def _reconnect(self) -> bool:
        """
        Reconnecter après perte de connexion
        
        Returns:
            bool: True si reconnexion réussie
        """
        try:
            # Nettoyer ancienne connexion
            if self.websocket:
                await self.websocket.close()
            
            self.connected = False
            
            # Nouvelle connexion
            success = await self.connect()
            if not success:
                return False
            
            # Re-s'abonner aux streams précédents
            streams_to_resubscribe = self.subscribed_streams.copy()
            self.subscribed_streams.clear()
            
            for stream_name in streams_to_resubscribe:
                stream_type = self._get_stream_type(stream_name)
                if stream_type:
                    await self._subscribe_stream(stream_name, stream_type)
            
            logger.info(f"✅ Reconnexion réussie avec {len(streams_to_resubscribe)} streams")
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
                    # Ping WebSocket
                    pong = await self.websocket.ping()
                    await asyncio.wait_for(pong, timeout=10)
                    
                    logger.debug("💓 Heartbeat WebSocket OK")
                
            except asyncio.TimeoutError:
                logger.warning("⚠️ Heartbeat timeout")
            except Exception as e:
                logger.error(f"❌ Erreur heartbeat: {e}")
                break
        
        logger.info("💓 Heartbeat WebSocket terminé")
    
    def add_callback(self, stream_type: str, callback: Callable):
        """
        Ajouter callback pour un type de stream
        
        Args:
            stream_type: Type de stream (ticker, depth, etc.)
            callback: Fonction callback async
        """
        if stream_type in self.callbacks:
            self.callbacks[stream_type].append(callback)
            logger.debug(f"📝 Callback ajouté pour {stream_type}")
        else:
            logger.warning(f"⚠️ Type de stream inconnu: {stream_type}")
    
    async def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """
        Émettre événement vers callbacks
        
        Args:
            event_type: Type d'événement
            data: Données à transmettre
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
        
        Returns:
            Dict avec informations de status
        """
        return {
            "connected": self.connected,
            "subscribed_streams": list(self.subscribed_streams),
            "subscribed_count": len(self.subscribed_streams),
            "metrics": {
                **self.metrics,
                "connected_at": self.metrics["connected_at"].isoformat() if self.metrics["connected_at"] else None,
                "last_message_at": self.metrics["last_message_at"].isoformat() if self.metrics["last_message_at"] else None
            },
            "base_url": self.base_url
        }
    
    def __str__(self) -> str:
        status = "connected" if self.connected else "disconnected"
        return f"<BinanceWebSocket(status={status}, streams={len(self.subscribed_streams)})>"
    
    def __repr__(self) -> str:
        return self.__str__()