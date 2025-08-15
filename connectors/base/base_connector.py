"""
🔌 Base Exchange Connector - Phase 2.4

Classe abstraite de base pour tous les connecteurs d'exchange.
Définit l'interface commune pour trading temps réel.

Features:
- Authentification sécurisée API keys
- Order management (Market, Limit, Stop)
- WebSocket feeds temps réel  
- Rate limiting automatique
- Error handling et resilience
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import aiohttp
from decimal import Decimal

from core.logger import get_logger

logger = get_logger(__name__)


class OrderType(Enum):
    """Types d'ordres supportés"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    OCO = "oco"  # One-Cancels-Other


class OrderSide(Enum):
    """Côtés des ordres"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Status des ordres"""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class ExchangeStatus(Enum):
    """Status de l'exchange"""
    CONNECTED = "connected"
    CONNECTING = "connecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class BaseConnector(ABC):
    """
    Connecteur exchange abstrait - Interface commune
    
    Toutes les implémentations d'exchange doivent hériter de cette classe
    et implémenter toutes les méthodes abstraites.
    """
    
    def __init__(
        self,
        exchange_name: str,
        api_key: str = None,
        api_secret: str = None,
        sandbox: bool = True,
        rate_limit: int = 1000  # Requêtes par minute
    ):
        """
        Initialise le connecteur exchange
        
        Args:
            exchange_name: Nom de l'exchange (binance, coinbase, etc.)
            api_key: Clé API chiffrée 
            api_secret: Secret API chiffré
            sandbox: Mode papier trading (True) ou live (False)
            rate_limit: Limite requêtes par minute
        """
        self.exchange_name = exchange_name
        self.api_key = api_key
        self.api_secret = api_secret
        self.sandbox = sandbox
        self.rate_limit = rate_limit
        
        # Status et monitoring
        self.status = ExchangeStatus.DISCONNECTED
        self.connected_at: Optional[datetime] = None
        self.last_heartbeat: Optional[datetime] = None
        self.request_count = 0
        self.error_count = 0
        
        # Sessions HTTP et WebSocket
        self.session: Optional[aiohttp.ClientSession] = None
        self.websocket = None
        
        # Callbacks pour events
        self.callbacks: Dict[str, List[Callable]] = {
            "order_update": [],
            "trade_update": [],
            "balance_update": [],
            "price_update": [],
            "connection_lost": [],
            "error": []
        }
        
        logger.info(f"🔌 {exchange_name} connecteur initialisé (sandbox={sandbox})")
    
    # === MÉTHODES DE CONNEXION ===
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Se connecter à l'exchange
        
        Returns:
            bool: True si connexion réussie
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """
        Se déconnecter proprement de l'exchange
        
        Returns:
            bool: True si déconnexion propre
        """
        pass
    
    @abstractmethod
    async def test_connection(self) -> Dict[str, Any]:
        """
        Tester la connexion et authentification
        
        Returns:
            Dict avec status, latence, permissions
        """
        pass
    
    # === MÉTHODES TRADING ===
    
    @abstractmethod
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
        Placer un ordre sur l'exchange
        
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
        pass
    
    @abstractmethod
    async def cancel_order(
        self,
        symbol: str,
        order_id: str = None,
        client_order_id: str = None
    ) -> Dict[str, Any]:
        """
        Annuler un ordre
        
        Args:
            symbol: Symbole trading
            order_id: ID ordre exchange
            client_order_id: ID ordre client
            
        Returns:
            Dict avec status annulation
        """
        pass
    
    @abstractmethod
    async def get_order_status(
        self,
        symbol: str,
        order_id: str = None,
        client_order_id: str = None
    ) -> Dict[str, Any]:
        """
        Récupérer le status d'un ordre
        
        Args:
            symbol: Symbole trading
            order_id: ID ordre exchange
            client_order_id: ID ordre client
            
        Returns:
            Dict avec détails ordre
        """
        pass
    
    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Récupérer tous les ordres ouverts
        
        Args:
            symbol: Symbole spécifique (optionnel)
            
        Returns:
            List des ordres ouverts
        """
        pass
    
    # === MÉTHODES ACCOUNT ===
    
    @abstractmethod
    async def get_account_balance(self) -> Dict[str, Any]:
        """
        Récupérer les soldes du compte
        
        Returns:
            Dict avec soldes par asset
        """
        pass
    
    @abstractmethod
    async def get_trading_fees(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Récupérer les frais de trading
        
        Args:
            symbol: Symbole spécifique (optionnel)
            
        Returns:
            Dict avec structure des frais
        """
        pass
    
    # === MÉTHODES MARKET DATA ===
    
    @abstractmethod
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Récupérer ticker (prix actuel)
        
        Args:
            symbol: Symbole trading
            
        Returns:
            Dict avec prix, volume, changement 24h
        """
        pass
    
    @abstractmethod
    async def get_order_book(self, symbol: str, depth: int = 100) -> Dict[str, Any]:
        """
        Récupérer order book
        
        Args:
            symbol: Symbole trading  
            depth: Profondeur order book
            
        Returns:
            Dict avec bids/asks
        """
        pass
    
    @abstractmethod
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Récupérer trades récents
        
        Args:
            symbol: Symbole trading
            limit: Nombre max trades
            
        Returns:
            List des trades récents
        """
        pass
    
    # === MÉTHODES WEBSOCKET ===
    
    @abstractmethod
    async def start_websocket(self) -> bool:
        """
        Démarrer connexion WebSocket pour données temps réel
        
        Returns:
            bool: True si démarrage réussi
        """
        pass
    
    @abstractmethod
    async def stop_websocket(self) -> bool:
        """
        Arrêter connexion WebSocket
        
        Returns:
            bool: True si arrêt propre
        """
        pass
    
    @abstractmethod
    async def subscribe_ticker(self, symbol: str) -> bool:
        """
        S'abonner aux updates prix temps réel
        
        Args:
            symbol: Symbole à suivre
            
        Returns:
            bool: True si abonnement réussi
        """
        pass
    
    @abstractmethod
    async def subscribe_order_book(self, symbol: str) -> bool:
        """
        S'abonner aux updates order book temps réel
        
        Args:
            symbol: Symbole à suivre
            
        Returns:
            bool: True si abonnement réussi
        """
        pass
    
    @abstractmethod
    async def subscribe_trades(self, symbol: str) -> bool:
        """
        S'abonner aux trades temps réel
        
        Args:
            symbol: Symbole à suivre
            
        Returns:
            bool: True si abonnement réussi
        """
        pass
    
    # === MÉTHODES COMMUNES ===
    
    def add_callback(self, event_type: str, callback: Callable):
        """
        Ajouter callback pour événement
        
        Args:
            event_type: Type d'événement
            callback: Fonction à appeler
        """
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
            logger.debug(f"✅ Callback ajouté pour {event_type}")
    
    async def emit_event(self, event_type: str, data: Dict[str, Any]):
        """
        Émettre événement vers callbacks
        
        Args:
            event_type: Type d'événement
            data: Données événement
        """
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    await callback(data)
                except Exception as e:
                    logger.error(f"❌ Erreur callback {event_type}: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Vérifier santé du connecteur
        
        Returns:
            Dict avec métriques santé
        """
        return {
            "exchange": self.exchange_name,
            "status": self.status.value,
            "connected_at": self.connected_at.isoformat() if self.connected_at else None,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "sandbox_mode": self.sandbox,
            "rate_limit": self.rate_limit
        }
    
    def __str__(self) -> str:
        return f"<{self.__class__.__name__}(exchange={self.exchange_name}, status={self.status.value})>"
    
    def __repr__(self) -> str:
        return self.__str__()