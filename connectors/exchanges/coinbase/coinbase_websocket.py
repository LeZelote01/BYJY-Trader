"""
📡 Coinbase Advanced Trading WebSocket - Phase 2.4

WebSocket handler pour flux temps réel Coinbase Advanced Trading.
Support streams multiples avec authentification JWT.

Features:
- Multiple streams simultanés
- Authentification WebSocket
- Reconnection automatique
- Gestion heartbeat
"""

import asyncio
import json
import time
import jwt
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable, Set
import websockets
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

from core.logger import get_logger

logger = get_logger(__name__)


class CoinbaseWebSocket:
    """
    Gestionnaire WebSocket Coinbase Advanced Trading
    
    Gère connexions WebSocket avec authentification JWT
    et support des streams multiples.
    """
    
    def __init__(self, base_url: str, api_key: str = None, api_secret: str = None):
        """
        Initialise le gestionnaire WebSocket Coinbase
        
        Args:
            base_url: URL de base WebSocket
            api_key: Clé API pour authentification
            api_secret: Secret API pour JWT
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.api_secret = api_secret
        self.websocket = None
        self.connected = False
        self.authenticated = False
        self.subscribed_channels: Set[str] = set()
        
        # Callbacks par type de channel
        self.callbacks: Dict[str, List[Callable]] = {
            "ticker": [],
            "level2": [],  # Order book
            "matches": [],  # Trades
            "user": [],     # Account updates
            "heartbeat": [],
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
        
        logger.info("📡 CoinbaseWebSocket initialisé")
    
    def _generate_jwt(self) -> str:
        """
        Générer JWT token pour authentification WebSocket
        
        Returns:
            str: JWT token
        """
        if not self.api_key or not self.api_secret:
            raise ValueError("API key et secret requis pour JWT")
        
        # Claims JWT
        now = int(time.time())
        payload = {
            "sub": self.api_key,
            "iss": "cdp",  # Coinbase Developer Platform
            "nbf": now,
            "exp": now + 120,  # Expire dans 2 minutes
            "aud": ["public_websocket_api"]
        }
        
        # Générer JWT avec ES256 (ou HS256 selon la config)
        token = jwt.encode(
            payload, 
            self.api_secret, 
            algorithm="HS256",  # Coinbase utilise HS256 pour WebSocket
            headers={"kid": self.api_key, "nonce": str(int(time.time() * 1000))}
        )
        
        return token
    
    async def connect(self) -> bool:
        """
        Se connecter au WebSocket Coinbase
        
        Returns:
            bool: True si connexion réussie
        """
        try:
            if self.connected:
                logger.warning("⚠️ WebSocket Coinbase déjà connecté")
                return True
            
            # URL WebSocket
            ws_url = f"{self.base_url}/ws"
            
            logger.info(f"🔗 Connexion WebSocket Coinbase: {ws_url}")
            
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
            
            logger.info("✅ WebSocket Coinbase connecté avec succès")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur connexion WebSocket Coinbase: {e}")
            self.connected = False
            return False
    
    async def _authenticate(self) -> bool:
        """
        Authentifier la connexion WebSocket
        
        Returns:
            bool: True si authentification réussie
        """
        try:
            # Générer JWT token
            jwt_token = self._generate_jwt()
            
            # Message d'authentification
            auth_message = {
                "type": "subscribe",
                "channels": ["user"],
                "jwt": jwt_token
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
                
                if message.get("type") == "subscriptions":
                    channels = message.get("channels", [])
                    if any(ch.get("name") == "user" for ch in channels):
                        self.authenticated = True
                        logger.info("✅ Authentification WebSocket Coinbase réussie")
                        return True
                
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
            
            logger.info("🔌 Déconnexion WebSocket Coinbase...")
            
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
            self.subscribed_channels.clear()
            
            # Notifier déconnexion
            await self._emit_event("connection", {
                "status": "disconnected",
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info("✅ WebSocket Coinbase déconnecté proprement")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur déconnexion WebSocket: {e}")
            return False
    
    async def subscribe_ticker(self, symbol: str) -> bool:
        """
        S'abonner au ticker d'un symbole
        
        Args:
            symbol: Symbole à suivre (ex: BTC-USD)
            
        Returns:
            bool: True si abonnement réussi
        """
        return await self._subscribe_channel("ticker", [symbol])
    
    async def subscribe_order_book(self, symbol: str) -> bool:
        """
        S'abonner à l'order book level2
        
        Args:
            symbol: Symbole à suivre
            
        Returns:
            bool: True si abonnement réussi
        """
        return await self._subscribe_channel("level2", [symbol])
    
    async def subscribe_trades(self, symbol: str) -> bool:
        """
        S'abonner aux matches (trades)
        
        Args:
            symbol: Symbole à suivre
            
        Returns:
            bool: True si abonnement réussi
        """
        return await self._subscribe_channel("matches", [symbol])
    
    async def _subscribe_channel(self, channel: str, product_ids: List[str]) -> bool:
        """
        S'abonner à un channel spécifique
        
        Args:
            channel: Nom du channel
            product_ids: Liste des symboles
            
        Returns:
            bool: True si abonnement réussi
        """
        try:
            if not self.connected:
                logger.error("❌ WebSocket non connecté")
                return False
            
            channel_key = f"{channel}_{'-'.join(product_ids)}"
            
            if channel_key in self.subscribed_channels:
                logger.warning(f"⚠️ Déjà abonné au channel: {channel_key}")
                return True
            
            # Message d'abonnement Coinbase
            subscribe_message = {
                "type": "subscribe",
                "channels": [
                    {
                        "name": channel,
                        "product_ids": product_ids
                    }
                ]
            }
            
            # Ajouter JWT si authentifié
            if self.authenticated and self.api_key:
                subscribe_message["jwt"] = self._generate_jwt()
            
            # Envoyer abonnement
            await self.websocket.send(json.dumps(subscribe_message))
            
            # Marquer comme abonné
            self.subscribed_channels.add(channel_key)
            
            logger.info(f"✅ Abonnement channel Coinbase réussi: {channel_key}")
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
                logger.warning("⏰ Timeout WebSocket Coinbase - Pas de message reçu")
                continue
                
            except (ConnectionClosedError, ConnectionClosedOK) as e:
                logger.warning(f"🔌 Connexion WebSocket fermée: {e}")
                
                if reconnect_attempts < self.max_reconnect_attempts:
                    reconnect_attempts += 1
                    logger.info(f"🔄 Tentative reconnexion {reconnect_attempts}/{self.max_reconnect_attempts}")
                    
                    await asyncio.sleep(self.reconnect_delay * reconnect_attempts)
                    
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
        
        logger.info("🔚 Boucle écoute WebSocket Coinbase terminée")
    
    async def _handle_message(self, raw_message: str):
        """
        Traiter un message WebSocket reçu
        
        Args:
            raw_message: Message brut JSON
        """
        try:
            message = json.loads(raw_message)
            
            # Ignorer messages de subscription confirmation
            if message.get("type") in ["subscriptions", "heartbeat"]:
                logger.debug(f"📝 Message système: {message.get('type')}")
                return
            
            # Messages avec erreur
            if message.get("type") == "error":
                logger.error(f"❌ Erreur WebSocket Coinbase: {message}")
                await self._emit_event("error", message)
                return
            
            # Router les messages selon le type
            msg_type = message.get("type")
            
            if msg_type == "ticker":
                await self._handle_ticker_message(message)
            elif msg_type == "l2update":
                await self._handle_level2_message(message)
            elif msg_type == "match":
                await self._handle_match_message(message)
            elif msg_type in ["received", "open", "done", "change", "activate"]:
                await self._handle_user_message(message)
            
            self.metrics["messages_processed"] += 1
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Message JSON invalide: {e}")
            logger.debug(f"Message brut: {raw_message[:200]}")
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement message: {e}")
    
    async def _handle_ticker_message(self, message: Dict[str, Any]):
        """Traiter message ticker"""
        enriched_data = {
            **message,
            "timestamp": datetime.now().isoformat(),
            "exchange": "coinbase"
        }
        await self._emit_event("ticker", enriched_data)
    
    async def _handle_level2_message(self, message: Dict[str, Any]):
        """Traiter message order book level2"""
        enriched_data = {
            **message,
            "timestamp": datetime.now().isoformat(),
            "exchange": "coinbase"
        }
        await self._emit_event("level2", enriched_data)
    
    async def _handle_match_message(self, message: Dict[str, Any]):
        """Traiter message trades (match)"""
        enriched_data = {
            **message,
            "timestamp": datetime.now().isoformat(),
            "exchange": "coinbase"
        }
        await self._emit_event("matches", enriched_data)
    
    async def _handle_user_message(self, message: Dict[str, Any]):
        """Traiter message user (ordres, balances)"""
        enriched_data = {
            **message,
            "timestamp": datetime.now().isoformat(),
            "exchange": "coinbase"
        }
        await self._emit_event("user", enriched_data)
    
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
            self.authenticated = False
            
            # Nouvelle connexion
            success = await self.connect()
            if not success:
                return False
            
            # Re-s'abonner aux channels précédents
            channels_to_resubscribe = self.subscribed_channels.copy()
            self.subscribed_channels.clear()
            
            for channel_key in channels_to_resubscribe:
                # Parser le channel et symbols
                parts = channel_key.split("_", 1)
                if len(parts) == 2:
                    channel = parts[0]
                    symbols = parts[1].split("-")
                    await self._subscribe_channel(channel, symbols)
            
            logger.info(f"✅ Reconnexion Coinbase réussie avec {len(channels_to_resubscribe)} channels")
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
                    
                    logger.debug("💓 Heartbeat WebSocket Coinbase OK")
                
            except asyncio.TimeoutError:
                logger.warning("⚠️ Heartbeat timeout")
            except Exception as e:
                logger.error(f"❌ Erreur heartbeat: {e}")
                break
        
        logger.info("💓 Heartbeat WebSocket Coinbase terminé")
    
    def add_callback(self, channel_type: str, callback: Callable):
        """
        Ajouter callback pour un type de channel
        
        Args:
            channel_type: Type de channel
            callback: Fonction callback async
        """
        if channel_type in self.callbacks:
            self.callbacks[channel_type].append(callback)
            logger.debug(f"📝 Callback ajouté pour {channel_type}")
        else:
            logger.warning(f"⚠️ Type de channel inconnu: {channel_type}")
    
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
            "authenticated": self.authenticated,
            "subscribed_channels": list(self.subscribed_channels),
            "channels_count": len(self.subscribed_channels),
            "metrics": {
                **self.metrics,
                "connected_at": self.metrics["connected_at"].isoformat() if self.metrics["connected_at"] else None,
                "last_message_at": self.metrics["last_message_at"].isoformat() if self.metrics["last_message_at"] else None
            },
            "base_url": self.base_url
        }
    
    def __str__(self) -> str:
        status = "connected" if self.connected else "disconnected"
        auth = "authenticated" if self.authenticated else "public"
        return f"<CoinbaseWebSocket(status={status}, auth={auth}, channels={len(self.subscribed_channels)})>"
    
    def __repr__(self) -> str:
        return self.__str__()