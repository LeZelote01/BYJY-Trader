"""
🔌 WebSocket Manager
Gestion des connexions WebSocket pour données temps réel
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, List, Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

from core.logger import get_logger

logger = get_logger("byjy.api.websocket")

class ConnectionManager:
    """Gestionnaire des connexions WebSocket"""
    
    def __init__(self):
        # Connexions actives par type
        self.active_connections: Dict[str, Set[WebSocket]] = {
            "market_data": set(),
            "trading": set(),
            "system": set(),
            "all": set()
        }
        
    async def connect(self, websocket: WebSocket, connection_type: str = "all"):
        """Accepte une nouvelle connexion WebSocket"""
        await websocket.accept()
        
        # Ajouter à la liste générale
        self.active_connections["all"].add(websocket)
        
        # Ajouter au type spécifique si valide
        if connection_type in self.active_connections:
            self.active_connections[connection_type].add(websocket)
        
        logger.info(f"New WebSocket connection: {connection_type}")
        
        # Envoyer un message de bienvenue
        await self.send_personal_message({
            "type": "welcome",
            "message": "Connected to BYJY-Trader WebSocket",
            "connection_type": connection_type,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }, websocket)
    
    def disconnect(self, websocket: WebSocket):
        """Supprime une connexion WebSocket"""
        # Supprimer de toutes les listes
        for connection_set in self.active_connections.values():
            connection_set.discard(websocket)
        
        logger.info("WebSocket connection closed")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Envoie un message à une connexion spécifique"""
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send personal message: {e}")
                self.disconnect(websocket)
    
    async def broadcast_to_type(self, message: dict, connection_type: str):
        """Diffuse un message à toutes les connexions d'un type"""
        if connection_type not in self.active_connections:
            return
        
        disconnected = set()
        for websocket in self.active_connections[connection_type].copy():
            try:
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_text(json.dumps(message))
                else:
                    disconnected.add(websocket)
            except Exception as e:
                logger.error(f"Failed to broadcast to {connection_type}: {e}")
                disconnected.add(websocket)
        
        # Nettoyer les connexions fermées
        for websocket in disconnected:
            self.disconnect(websocket)
    
    async def broadcast_to_all(self, message: dict):
        """Diffuse un message à toutes les connexions"""
        await self.broadcast_to_type(message, "all")
    
    def get_connection_stats(self) -> Dict[str, int]:
        """Retourne les statistiques des connexions"""
        return {
            connection_type: len(connections)
            for connection_type, connections in self.active_connections.items()
        }

# Instance globale du gestionnaire
manager = ConnectionManager()

# Router pour les endpoints WebSocket
websocket_router = APIRouter()

@websocket_router.websocket("/")
async def websocket_endpoint(websocket: WebSocket):
    """Endpoint WebSocket principal"""
    await manager.connect(websocket, "all")
    
    try:
        while True:
            # Recevoir les messages du client
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                message_type = message.get("type", "unknown")
                
                logger.info(f"Received WebSocket message: {message_type}")
                
                # Traiter selon le type de message
                if message_type == "ping":
                    await manager.send_personal_message({
                        "type": "pong",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }, websocket)
                
                elif message_type == "subscribe":
                    # Gestion des abonnements
                    channel = message.get("channel", "all")
                    await manager.send_personal_message({
                        "type": "subscribed",
                        "channel": channel,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }, websocket)
                
                else:
                    # Echo du message pour test
                    await manager.send_personal_message({
                        "type": "echo",
                        "original_message": message,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }, websocket)
                    
            except json.JSONDecodeError:
                await manager.send_personal_message({
                    "type": "error",
                    "message": "Invalid JSON format",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }, websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@websocket_router.websocket("/market")
async def market_websocket(websocket: WebSocket):
    """WebSocket spécialisé pour les données de marché"""
    await manager.connect(websocket, "market_data")
    
    try:
        while True:
            # Simuler l'envoi de données de marché
            await asyncio.sleep(5)
            
            market_data = {
                "type": "market_update",
                "symbol": "BTCUSDT",
                "price": "45678.90",
                "change": "+1.25%",
                "volume": "1234.56",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            await manager.send_personal_message(market_data, websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Market WebSocket error: {e}")
        manager.disconnect(websocket)

@websocket_router.websocket("/trading")
async def trading_websocket(websocket: WebSocket):
    """WebSocket spécialisé pour les événements de trading"""
    await manager.connect(websocket, "trading")
    
    try:
        while True:
            data = await websocket.receive_text()
            
            # Echo pour les messages de trading
            await manager.send_personal_message({
                "type": "trading_response",
                "received": data,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }, websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Trading WebSocket error: {e}")
        manager.disconnect(websocket)

# Fonction pour broadcaster des mises à jour système
async def broadcast_system_update(message: dict):
    """Fonction helper pour broadcaster des mises à jour système"""
    await manager.broadcast_to_type({
        "type": "system_update",
        **message,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }, "system")

# Fonction pour broadcaster des données de marché
async def broadcast_market_data(symbol: str, data: dict):
    """Fonction helper pour broadcaster des données de marché"""
    await manager.broadcast_to_type({
        "type": "market_data",
        "symbol": symbol,
        **data,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }, "market_data")