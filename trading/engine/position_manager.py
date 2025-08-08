"""
📊 Position Manager - Gestionnaire des Positions
Gestion centralisée des positions de trading
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

from core.logger import get_logger

logger = get_logger(__name__)


class PositionSide(Enum):
    """Côté de la position"""
    LONG = "long"
    SHORT = "short"


class PositionStatus(Enum):
    """Status des positions"""
    OPEN = "open"
    CLOSED = "closed"
    CLOSING = "closing"


@dataclass
class Position:
    """Représentation d'une position de trading"""
    id: str
    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    current_price: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: PositionStatus = PositionStatus.OPEN
    created_at: datetime = None
    updated_at: datetime = None
    closed_at: Optional[datetime] = None
    strategy_id: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        self.updated_at = self.created_at
        if self.current_price == 0.0:
            self.current_price = self.entry_price
    
    @property
    def unrealized_pnl(self) -> float:
        """PnL non réalisé"""
        if self.side == PositionSide.LONG:
            return (self.current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.current_price) * self.quantity
    
    @property
    def unrealized_pnl_percent(self) -> float:
        """PnL non réalisé en pourcentage"""
        if self.entry_price == 0:
            return 0.0
        return (self.unrealized_pnl / (self.entry_price * self.quantity)) * 100
    
    @property
    def market_value(self) -> float:
        """Valeur de marché actuelle"""
        return self.current_price * abs(self.quantity)
    
    @property
    def is_profitable(self) -> bool:
        """Position rentable"""
        return self.unrealized_pnl > 0
    
    def should_stop_loss(self) -> bool:
        """Vérifie si le stop-loss doit être déclenché"""
        if not self.stop_loss:
            return False
        
        if self.side == PositionSide.LONG:
            return self.current_price <= self.stop_loss
        else:
            return self.current_price >= self.stop_loss
    
    def should_take_profit(self) -> bool:
        """Vérifie si le take-profit doit être déclenché"""
        if not self.take_profit:
            return False
        
        if self.side == PositionSide.LONG:
            return self.current_price >= self.take_profit
        else:
            return self.current_price <= self.take_profit
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit la position en dictionnaire"""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "status": self.status.value,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_percent": self.unrealized_pnl_percent,
            "market_value": self.market_value,
            "is_profitable": self.is_profitable,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "strategy_id": self.strategy_id
        }


class PositionManager:
    """
    Gestionnaire des positions de trading
    Centralise l'ouverture, fermeture et suivi des positions
    """
    
    def __init__(self):
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        
        # Prix de marché en cache (normalement viendraient d'un feed de données)
        self.market_prices: Dict[str, float] = {}
        
        # Statistiques
        self.stats = {
            "total_positions": 0,
            "winning_positions": 0,
            "losing_positions": 0,
            "total_realized_pnl": 0.0,
            "total_unrealized_pnl": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0
        }
        
        logger.info("PositionManager initialisé")
    
    def open_position(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        strategy_id: Optional[str] = None
    ) -> str:
        """
        Ouvre une nouvelle position
        """
        try:
            # Validation des paramètres
            self._validate_position_params(symbol, side, quantity, entry_price)
            
            # Génération ID unique
            position_id = str(uuid.uuid4())
            
            # Création de la position
            position = Position(
                id=position_id,
                symbol=symbol.upper(),
                side=PositionSide(side.lower()),
                quantity=quantity,
                entry_price=entry_price,
                current_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                strategy_id=strategy_id
            )
            
            # Stockage
            self.positions[position_id] = position
            self.stats["total_positions"] += 1
            
            # Mise à jour du prix de marché
            self.market_prices[symbol.upper()] = entry_price
            
            logger.info(f"Position ouverte - ID: {position_id}, {symbol} {side} {quantity} @ {entry_price}")
            return position_id
            
        except Exception as e:
            logger.error(f"Erreur ouverture position: {str(e)}")
            raise
    
    def close_position(self, position_id: str, close_price: Optional[float] = None) -> bool:
        """
        Ferme une position
        """
        try:
            if position_id not in self.positions:
                logger.warning(f"Position inexistante: {position_id}")
                return False
            
            position = self.positions[position_id]
            
            if position.status != PositionStatus.OPEN:
                logger.warning(f"Position non ouverte - Status: {position.status}")
                return False
            
            # Prix de fermeture
            if close_price is None:
                close_price = self.market_prices.get(position.symbol, position.current_price)
            
            # Calcul du PnL réalisé
            if position.side == PositionSide.LONG:
                realized_pnl = (close_price - position.entry_price) * position.quantity
            else:
                realized_pnl = (position.entry_price - close_price) * position.quantity
            
            # Fermeture
            position.status = PositionStatus.CLOSED
            position.current_price = close_price
            position.closed_at = datetime.now(timezone.utc)
            position.updated_at = position.closed_at
            
            # Mise à jour des statistiques
            self.stats["total_realized_pnl"] += realized_pnl
            if realized_pnl > 0:
                self.stats["winning_positions"] += 1
                if realized_pnl > self.stats["largest_win"]:
                    self.stats["largest_win"] = realized_pnl
            else:
                self.stats["losing_positions"] += 1
                if realized_pnl < self.stats["largest_loss"]:
                    self.stats["largest_loss"] = realized_pnl
            
            # Déplacement vers l'historique
            self.closed_positions.append(position)
            del self.positions[position_id]
            
            logger.info(f"Position fermée - ID: {position_id}, PnL: {realized_pnl:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur fermeture position: {str(e)}")
            return False
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Récupère la position pour un symbole
        """
        try:
            symbol = symbol.upper()
            for position in self.positions.values():
                if position.symbol == symbol:
                    return position.to_dict()
            return None
            
        except Exception as e:
            logger.error(f"Erreur récupération position: {str(e)}")
            return None
    
    def get_position_by_id(self, position_id: str) -> Optional[Dict[str, Any]]:
        """
        Récupère une position par son ID
        """
        try:
            if position_id in self.positions:
                return self.positions[position_id].to_dict()
            
            # Chercher dans les positions fermées
            for closed_position in self.closed_positions:
                if closed_position.id == position_id:
                    return closed_position.to_dict()
            
            return None
            
        except Exception as e:
            logger.error(f"Erreur récupération position par ID: {str(e)}")
            return None
    
    def get_all_positions(self, include_closed: bool = False) -> List[Dict[str, Any]]:
        """
        Récupère toutes les positions
        """
        try:
            all_positions = []
            
            # Positions ouvertes
            for position in self.positions.values():
                all_positions.append(position.to_dict())
            
            # Positions fermées si demandées
            if include_closed:
                for position in self.closed_positions:
                    all_positions.append(position.to_dict())
            
            return all_positions
            
        except Exception as e:
            logger.error(f"Erreur récupération toutes positions: {str(e)}")
            return []
    
    def update_market_price(self, symbol: str, price: float):
        """
        Met à jour le prix de marché d'un symbole
        """
        try:
            symbol = symbol.upper()
            self.market_prices[symbol] = price
            
            # Mise à jour des positions concernées
            for position in self.positions.values():
                if position.symbol == symbol:
                    position.current_price = price
                    position.updated_at = datetime.now(timezone.utc)
            
            logger.debug(f"Prix mis à jour - {symbol}: {price}")
            
        except Exception as e:
            logger.error(f"Erreur mise à jour prix: {str(e)}")
    
    def check_stop_loss_take_profit(self) -> List[str]:
        """
        Vérifie les stop-loss et take-profit
        Retourne la liste des positions à fermer
        """
        positions_to_close = []
        
        try:
            for position_id, position in self.positions.items():
                should_close = False
                reason = ""
                
                if position.should_stop_loss():
                    should_close = True
                    reason = "Stop-loss déclenché"
                elif position.should_take_profit():
                    should_close = True
                    reason = "Take-profit déclenché"
                
                if should_close:
                    positions_to_close.append(position_id)
                    logger.info(f"Position à fermer - ID: {position_id}, Raison: {reason}")
            
            return positions_to_close
            
        except Exception as e:
            logger.error(f"Erreur vérification stop/take: {str(e)}")
            return []
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Résumé du portefeuille
        """
        try:
            # Calcul des métriques temps réel
            total_unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())
            total_market_value = sum(p.market_value for p in self.positions.values())
            
            self.stats["total_unrealized_pnl"] = total_unrealized_pnl
            
            # Calcul du win rate
            total_closed = self.stats["winning_positions"] + self.stats["losing_positions"]
            win_rate = (self.stats["winning_positions"] / max(1, total_closed)) * 100
            
            return {
                "open_positions_count": len(self.positions),
                "closed_positions_count": len(self.closed_positions),
                "total_market_value": total_market_value,
                "total_unrealized_pnl": total_unrealized_pnl,
                "total_realized_pnl": self.stats["total_realized_pnl"],
                "total_pnl": self.stats["total_realized_pnl"] + total_unrealized_pnl,
                "win_rate": win_rate,
                "largest_win": self.stats["largest_win"],
                "largest_loss": self.stats["largest_loss"],
                "positions_by_symbol": self._get_positions_by_symbol()
            }
            
        except Exception as e:
            logger.error(f"Erreur calcul résumé portefeuille: {str(e)}")
            return {}
    
    # Méthodes privées
    def _validate_position_params(self, symbol: str, side: str, quantity: float, entry_price: float):
        """Valide les paramètres d'une position"""
        if not symbol or len(symbol) < 3:
            raise ValueError("Symbole invalide")
        
        if side.lower() not in ["long", "short"]:
            raise ValueError("Side doit être 'long' ou 'short'")
        
        if quantity <= 0:
            raise ValueError("Quantité doit être positive")
        
        if entry_price <= 0:
            raise ValueError("Prix d'entrée doit être positif")
    
    def _get_positions_by_symbol(self) -> Dict[str, int]:
        """Nombre de positions par symbole"""
        positions_by_symbol = {}
        for position in self.positions.values():
            symbol = position.symbol
            positions_by_symbol[symbol] = positions_by_symbol.get(symbol, 0) + 1
        return positions_by_symbol