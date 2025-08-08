"""
💼 Virtual Portfolio - Portefeuille Virtuel
Gestion du portefeuille en mode simulation pour paper trading
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging

from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class VirtualPosition:
    """Position virtuelle dans le portefeuille"""
    symbol: str
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    current_price: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_commission: float = 0.0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def market_value(self) -> float:
        """Valeur de marché actuelle"""
        return abs(self.quantity) * self.current_price
    
    @property
    def total_pnl(self) -> float:
        """PnL total (réalisé + non réalisé)"""
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def position_side(self) -> str:
        """Côté de la position (long/short)"""
        if self.quantity > 0:
            return "long"
        elif self.quantity < 0:
            return "short"
        else:
            return "flat"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "avg_entry_price": self.avg_entry_price,
            "current_price": self.current_price,
            "market_value": self.market_value,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_pnl": self.total_pnl,
            "total_commission": self.total_commission,
            "position_side": self.position_side,
            "last_updated": self.last_updated.isoformat()
        }


class VirtualPortfolio:
    """
    Portefeuille virtuel pour paper trading
    Gère les positions, cash et calculs PnL
    """
    
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.cash_balance = initial_balance
        self.reserved_cash = 0.0  # Cash réservé pour ordres en attente
        
        # Positions
        self.positions: Dict[str, VirtualPosition] = {}
        
        # Historique des valeurs du portefeuille
        self.value_history: List[Dict[str, Any]] = []
        
        # Métriques
        self.total_commission_paid = 0.0
        self.total_trades = 0
        self.created_at = datetime.now(timezone.utc)
        
        logger.info(f"VirtualPortfolio initialisé avec ${initial_balance:,.2f}")
    
    @property
    def total_value(self) -> float:
        """Valeur totale du portefeuille (cash + positions)"""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash_balance + positions_value
    
    @property
    def available_balance(self) -> float:
        """Balance disponible (cash - réservé)"""
        return self.cash_balance - self.reserved_cash
    
    @property
    def total_pnl(self) -> float:
        """PnL total du portefeuille"""
        return self.total_value - self.initial_balance
    
    @property
    def total_return_percent(self) -> float:
        """Rendement total en pourcentage"""
        if self.initial_balance <= 0:
            return 0.0
        return (self.total_pnl / self.initial_balance) * 100
    
    def has_sufficient_balance(self, required_amount: float) -> bool:
        """Vérifie si le solde disponible est suffisant"""
        return self.available_balance >= required_amount
    
    def reserve_cash(self, amount: float) -> bool:
        """Réserve du cash pour un ordre en attente"""
        try:
            if self.available_balance >= amount:
                self.reserved_cash += amount
                logger.debug(f"Cash réservé: ${amount:,.2f}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Erreur réservation cash: {str(e)}")
            return False
    
    def release_reserved_cash(self, amount: float):
        """Libère du cash réservé"""
        try:
            self.reserved_cash = max(0, self.reserved_cash - amount)
            logger.debug(f"Cash libéré: ${amount:,.2f}")
            
        except Exception as e:
            logger.error(f"Erreur libération cash: {str(e)}")
    
    def add_position(self, symbol: str, quantity: float, price: float, commission: float = 0.0) -> bool:
        """
        Ajoute à une position (achat)
        """
        try:
            # Coût total (prix + commission)
            total_cost = abs(quantity) * price + commission
            
            # Vérification fonds
            if not self.has_sufficient_balance(total_cost):
                logger.warning(f"Fonds insuffisants pour position {symbol}")
                return False
            
            # Déduction cash
            self.cash_balance -= total_cost
            self.total_commission_paid += commission
            
            # Mise à jour position
            if symbol not in self.positions:
                self.positions[symbol] = VirtualPosition(
                    symbol=symbol,
                    quantity=quantity,
                    avg_entry_price=price,
                    current_price=price,
                    total_commission=commission
                )
            else:
                position = self.positions[symbol]
                
                # Calcul nouveau prix moyen
                if position.quantity * quantity >= 0:  # Même direction
                    total_quantity = position.quantity + quantity
                    total_value = (position.quantity * position.avg_entry_price) + (quantity * price)
                    
                    if total_quantity != 0:
                        new_avg_price = total_value / total_quantity
                    else:
                        new_avg_price = price
                    
                    position.quantity = total_quantity
                    position.avg_entry_price = new_avg_price
                else:
                    # Direction opposée - calcul PnL réalisé
                    quantity_to_close = min(abs(position.quantity), abs(quantity))
                    
                    if position.quantity > 0:  # Position long, vente
                        realized_pnl = quantity_to_close * (price - position.avg_entry_price)
                    else:  # Position short, achat
                        realized_pnl = quantity_to_close * (position.avg_entry_price - price)
                    
                    position.realized_pnl += realized_pnl
                    position.quantity += quantity
                
                position.total_commission += commission
                position.current_price = price
                position.last_updated = datetime.now(timezone.utc)
            
            self.total_trades += 1
            
            logger.info(f"Position ajoutée: {quantity:+.4f} {symbol} @ ${price:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur ajout position: {str(e)}")
            return False
    
    def reduce_position(self, symbol: str, quantity: float, price: float, commission: float = 0.0) -> bool:
        """
        Réduit une position (vente)
        """
        try:
            if symbol not in self.positions:
                logger.warning(f"Position {symbol} non trouvée pour réduction")
                return False
            
            position = self.positions[symbol]
            
            # Vérification quantité disponible
            if abs(quantity) > abs(position.quantity):
                logger.warning(f"Quantité insuffisante pour réduction {symbol}")
                return False
            
            # Déduction commission
            self.cash_balance -= commission
            self.total_commission_paid += commission
            position.total_commission += commission
            
            # Calcul PnL réalisé pour la partie vendue
            if position.quantity > 0:  # Long position
                realized_pnl = quantity * (price - position.avg_entry_price)
            else:  # Short position  
                realized_pnl = quantity * (position.avg_entry_price - price)
            
            # Ajout cash de la vente
            cash_received = abs(quantity) * price
            self.cash_balance += cash_received
            
            # Mise à jour position
            position.quantity -= quantity
            position.realized_pnl += realized_pnl
            position.current_price = price
            position.last_updated = datetime.now(timezone.utc)
            
            self.total_trades += 1
            
            logger.info(f"Position réduite: {-quantity:+.4f} {symbol} @ ${price:.4f} (PnL: ${realized_pnl:+.2f})")
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur réduction position: {str(e)}")
            return False
    
    def close_position(self, symbol: str, price: float, commission: float = 0.0) -> bool:
        """Ferme complètement une position"""
        try:
            if symbol not in self.positions:
                return False
            
            position = self.positions[symbol]
            return self.reduce_position(symbol, position.quantity, price, commission)
            
        except Exception as e:
            logger.error(f"Erreur fermeture position: {str(e)}")
            return False
    
    def update_market_prices(self, prices: Dict[str, float]):
        """Met à jour les prix de marché des positions"""
        try:
            for symbol, price in prices.items():
                if symbol in self.positions:
                    position = self.positions[symbol]
                    position.current_price = price
                    
                    # Calcul PnL non réalisé
                    if position.quantity != 0:
                        if position.quantity > 0:  # Long
                            position.unrealized_pnl = position.quantity * (price - position.avg_entry_price)
                        else:  # Short
                            position.unrealized_pnl = abs(position.quantity) * (position.avg_entry_price - price)
                    else:
                        position.unrealized_pnl = 0.0
                    
                    position.last_updated = datetime.now(timezone.utc)
            
            # Sauvegarde snapshot valeur portefeuille
            self._save_value_snapshot()
            
        except Exception as e:
            logger.error(f"Erreur mise à jour prix: {str(e)}")
    
    def _save_value_snapshot(self):
        """Sauvegarde un snapshot de la valeur du portefeuille"""
        try:
            snapshot = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_value": self.total_value,
                "cash_balance": self.cash_balance,
                "positions_value": sum(pos.market_value for pos in self.positions.values()),
                "total_pnl": self.total_pnl,
                "return_percent": self.total_return_percent,
                "positions_count": len([p for p in self.positions.values() if p.quantity != 0])
            }
            
            self.value_history.append(snapshot)
            
            # Limitation taille historique (garder 1000 derniers)
            if len(self.value_history) > 1000:
                self.value_history = self.value_history[-500:]
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde snapshot: {str(e)}")
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Retourne les détails d'une position"""
        try:
            if symbol in self.positions:
                return self.positions[symbol].to_dict()
            return None
            
        except Exception as e:
            logger.error(f"Erreur récupération position: {str(e)}")
            return None
    
    def get_active_positions(self) -> List[Dict[str, Any]]:
        """Retourne toutes les positions actives (quantity != 0)"""
        try:
            active = []
            for position in self.positions.values():
                if position.quantity != 0:
                    active.append(position.to_dict())
            
            return sorted(active, key=lambda x: x["market_value"], reverse=True)
            
        except Exception as e:
            logger.error(f"Erreur récupération positions actives: {str(e)}")
            return []
    
    def get_all_positions(self) -> List[Dict[str, Any]]:
        """Retourne toutes les positions (actives et fermées)"""
        try:
            return [pos.to_dict() for pos in self.positions.values()]
            
        except Exception as e:
            logger.error(f"Erreur récupération toutes positions: {str(e)}")
            return []
    
    def get_summary(self) -> Dict[str, Any]:
        """Retourne un résumé complet du portefeuille"""
        try:
            active_positions = self.get_active_positions()
            
            # Métriques positions
            long_positions = [p for p in active_positions if p["quantity"] > 0]
            short_positions = [p for p in active_positions if p["quantity"] < 0]
            
            total_realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            
            return {
                "created_at": self.created_at.isoformat(),
                "initial_balance": self.initial_balance,
                "current_values": {
                    "cash_balance": self.cash_balance,
                    "reserved_cash": self.reserved_cash,
                    "available_balance": self.available_balance,
                    "total_value": self.total_value,
                    "positions_value": sum(pos.market_value for pos in self.positions.values())
                },
                "pnl_summary": {
                    "total_pnl": self.total_pnl,
                    "realized_pnl": total_realized_pnl,
                    "unrealized_pnl": total_unrealized_pnl,
                    "total_return_percent": self.total_return_percent,
                    "total_commission_paid": self.total_commission_paid
                },
                "positions_summary": {
                    "total_positions": len(self.positions),
                    "active_positions": len(active_positions),
                    "long_positions": len(long_positions),
                    "short_positions": len(short_positions),
                    "total_trades": self.total_trades
                },
                "positions": active_positions,
                "value_history_points": len(self.value_history)
            }
            
        except Exception as e:
            logger.error(f"Erreur génération résumé: {str(e)}")
            return {"error": str(e)}
    
    def get_performance_chart_data(self, days: int = 30) -> List[Dict[str, Any]]:
        """Retourne les données pour graphique de performance"""
        try:
            if not self.value_history:
                return []
            
            # Filtrage par nombre de jours
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            filtered_data = []
            for snapshot in self.value_history:
                snapshot_date = datetime.fromisoformat(snapshot["timestamp"].replace("Z", "+00:00"))
                if snapshot_date >= cutoff_date:
                    filtered_data.append(snapshot)
            
            return filtered_data if filtered_data else self.value_history[-days:]
            
        except Exception as e:
            logger.error(f"Erreur données graphique: {str(e)}")
            return []
    
    def calculate_drawdown(self) -> Dict[str, float]:
        """Calcule le drawdown du portefeuille"""
        try:
            if len(self.value_history) < 2:
                return {"current_drawdown": 0.0, "max_drawdown": 0.0}
            
            values = [snapshot["total_value"] for snapshot in self.value_history]
            
            # Calcul peaks et drawdowns
            peak = values[0]
            max_drawdown = 0.0
            current_drawdown = 0.0
            
            for value in values:
                if value > peak:
                    peak = value
                
                drawdown = (peak - value) / peak if peak > 0 else 0
                current_drawdown = drawdown
                max_drawdown = max(max_drawdown, drawdown)
            
            return {
                "current_drawdown": current_drawdown,
                "max_drawdown": max_drawdown
            }
            
        except Exception as e:
            logger.error(f"Erreur calcul drawdown: {str(e)}")
            return {"current_drawdown": 0.0, "max_drawdown": 0.0}
    
    def reset_portfolio(self, new_balance: float = None):
        """Remet le portefeuille à zéro"""
        try:
            balance = new_balance or self.initial_balance
            
            self.initial_balance = balance
            self.cash_balance = balance
            self.reserved_cash = 0.0
            
            self.positions.clear()
            self.value_history.clear()
            
            self.total_commission_paid = 0.0
            self.total_trades = 0
            self.created_at = datetime.now(timezone.utc)
            
            logger.info(f"Portefeuille remis à zéro - Balance: ${balance:,.2f}")
            
        except Exception as e:
            logger.error(f"Erreur reset portefeuille: {str(e)}")
    
    def export_data(self) -> Dict[str, Any]:
        """Exporte toutes les données du portefeuille"""
        try:
            return {
                "portfolio_info": {
                    "created_at": self.created_at.isoformat(),
                    "initial_balance": self.initial_balance,
                    "current_total_value": self.total_value
                },
                "positions": self.get_all_positions(),
                "value_history": self.value_history,
                "summary": self.get_summary()
            }
            
        except Exception as e:
            logger.error(f"Erreur export données: {str(e)}")
            return {"error": str(e)}