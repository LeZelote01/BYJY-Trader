"""
📋 Order Manager - Gestionnaire des Ordres
Gestion centralisée des ordres de trading
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

from core.logger import get_logger

logger = get_logger(__name__)


class OrderType(Enum):
    """Types d'ordres"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Côté de l'ordre"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Status des ordres"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL_FILLED = "partial_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Représentation d'un ordre de trading"""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_price: float = 0.0
    created_at: datetime = None
    updated_at: datetime = None
    strategy_id: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        self.updated_at = self.created_at
    
    @property
    def remaining_quantity(self) -> float:
        """Quantité restante à exécuter"""
        return self.quantity - self.filled_quantity
    
    @property
    def is_complete(self) -> bool:
        """L'ordre est-il complètement exécuté"""
        return self.status == OrderStatus.FILLED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit l'ordre en dictionnaire"""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "price": self.price,
            "stop_price": self.stop_price,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "average_price": self.average_price,
            "remaining_quantity": self.remaining_quantity,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "strategy_id": self.strategy_id
        }


class OrderManager:
    """
    Gestionnaire des ordres de trading
    Centralise la création, modification et suivi des ordres
    """
    
    def __init__(self):
        self.orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        
        # Statistiques
        self.stats = {
            "total_orders": 0,
            "filled_orders": 0,
            "cancelled_orders": 0,
            "rejected_orders": 0,
            "total_volume": 0.0
        }
        
        logger.info("OrderManager initialisé")
    
    def create_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        strategy_id: Optional[str] = None
    ) -> str:
        """
        Crée un nouvel ordre
        """
        try:
            # Validation des paramètres
            self._validate_order_params(symbol, side, quantity, order_type, price)
            
            # Génération ID unique
            order_id = str(uuid.uuid4())
            
            # Création de l'ordre
            order = Order(
                id=order_id,
                symbol=symbol.upper(),
                side=OrderSide(side.lower()),
                order_type=OrderType(order_type.lower()),
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                strategy_id=strategy_id
            )
            
            # Stockage
            self.orders[order_id] = order
            self.stats["total_orders"] += 1
            
            logger.info(f"Ordre créé - ID: {order_id}, {symbol} {side} {quantity} @ {price or 'MARKET'}")
            
            # En mode réel, ici on soumettrait l'ordre à l'exchange
            # Pour l'instant, on simule
            self._simulate_order_submission(order_id)
            
            return order_id
            
        except Exception as e:
            logger.error(f"Erreur création ordre: {str(e)}")
            raise
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Annule un ordre
        """
        try:
            if order_id not in self.orders:
                logger.warning(f"Ordre inexistant: {order_id}")
                return False
            
            order = self.orders[order_id]
            
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                logger.warning(f"Ordre non annulable - Status: {order.status}")
                return False
            
            # Annulation
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.now(timezone.utc)
            
            # Déplacement vers l'historique
            self._move_to_history(order_id)
            
            self.stats["cancelled_orders"] += 1
            
            logger.info(f"Ordre annulé - ID: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur annulation ordre: {str(e)}")
            return False
    
    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Récupère le status d'un ordre
        """
        try:
            if order_id in self.orders:
                return self.orders[order_id].to_dict()
            
            # Chercher dans l'historique
            for historical_order in self.order_history:
                if historical_order.id == order_id:
                    return historical_order.to_dict()
            
            return None
            
        except Exception as e:
            logger.error(f"Erreur récupération status ordre: {str(e)}")
            return None
    
    def get_open_orders(self, symbol: Optional[str] = None, strategy_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Récupère les ordres ouverts
        """
        try:
            open_orders = []
            
            for order in self.orders.values():
                # Filtres
                if symbol and order.symbol != symbol.upper():
                    continue
                if strategy_id and order.strategy_id != strategy_id:
                    continue
                
                # Seulement les ordres non terminés
                if order.status not in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                    open_orders.append(order.to_dict())
            
            return open_orders
            
        except Exception as e:
            logger.error(f"Erreur récupération ordres ouverts: {str(e)}")
            return []
    
    def get_order_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Récupère l'historique des ordres
        """
        try:
            # Trier par date de création décroissante
            sorted_history = sorted(
                self.order_history, 
                key=lambda x: x.created_at, 
                reverse=True
            )
            
            # Limiter le nombre de résultats
            limited_history = sorted_history[:limit]
            
            return [order.to_dict() for order in limited_history]
            
        except Exception as e:
            logger.error(f"Erreur récupération historique: {str(e)}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Retourne les statistiques du gestionnaire d'ordres
        """
        return {
            **self.stats,
            "active_orders": len(self.orders),
            "historical_orders": len(self.order_history),
            "fill_rate": (self.stats["filled_orders"] / max(1, self.stats["total_orders"])) * 100,
            "cancellation_rate": (self.stats["cancelled_orders"] / max(1, self.stats["total_orders"])) * 100
        }
    
    # Méthodes privées
    def _validate_order_params(self, symbol: str, side: str, quantity: float, order_type: str, price: Optional[float]):
        """Valide les paramètres d'un ordre"""
        if not symbol or len(symbol) < 3:
            raise ValueError("Symbole invalide")
        
        if side.lower() not in ["buy", "sell"]:
            raise ValueError("Side doit être 'buy' ou 'sell'")
        
        if quantity <= 0:
            raise ValueError("Quantité doit être positive")
        
        if order_type.lower() not in ["market", "limit", "stop", "stop_limit"]:
            raise ValueError("Type d'ordre invalide")
        
        if order_type.lower() in ["limit", "stop_limit"] and not price:
            raise ValueError(f"Prix requis pour ordre {order_type}")
        
        if price and price <= 0:
            raise ValueError("Prix doit être positif")
    
    def _simulate_order_submission(self, order_id: str):
        """
        Simule la soumission d'ordre (mode paper trading)
        En production, ici on intégrerait avec l'API d'exchange
        """
        order = self.orders[order_id]
        
        # Simulation : ordre soumis immédiatement
        order.status = OrderStatus.SUBMITTED
        order.updated_at = datetime.now(timezone.utc)
        
        # Pour les ordres market, simulation d'exécution immédiate
        if order.order_type == OrderType.MARKET:
            self._simulate_order_fill(order_id, order.quantity, order.price or 50000)  # Prix simulé
    
    def _simulate_order_fill(self, order_id: str, filled_qty: float, fill_price: float):
        """Simule l'exécution d'un ordre"""
        if order_id not in self.orders:
            return
        
        order = self.orders[order_id]
        
        # Mise à jour de l'exécution
        order.filled_quantity += filled_qty
        order.average_price = fill_price
        order.updated_at = datetime.now(timezone.utc)
        
        # Déterminer le nouveau status
        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
            self._move_to_history(order_id)
            self.stats["filled_orders"] += 1
        else:
            order.status = OrderStatus.PARTIAL_FILLED
        
        self.stats["total_volume"] += filled_qty
        
        logger.info(f"Ordre exécuté - ID: {order_id}, Qty: {filled_qty}, Prix: {fill_price}")
    
    def _move_to_history(self, order_id: str):
        """Déplace un ordre vers l'historique"""
        if order_id in self.orders:
            order = self.orders[order_id]
            self.order_history.append(order)
            del self.orders[order_id]
            
            # Limiter la taille de l'historique
            if len(self.order_history) > 1000:
                self.order_history = self.order_history[-1000:]