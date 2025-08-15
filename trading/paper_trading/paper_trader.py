"""
📱 Paper Trader - Trader en Simulation
Système de trading sans risque financier pour tester stratégies
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import pandas as pd
import uuid

from core.logger import get_logger
from ..engine.execution_handler import TradingSignal, SignalType
from .virtual_portfolio import VirtualPortfolio
from ..strategies.base_strategy import BaseStrategy

logger = get_logger(__name__)


class PaperOrderStatus(Enum):
    """Statuts des ordres paper trading"""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    PARTIALLY_FILLED = "partially_filled"


@dataclass
class PaperOrder:
    """Ordre en mode paper trading"""
    order_id: str
    strategy_id: str
    symbol: str
    side: str  # buy/sell
    order_type: str  # market/limit/stop
    quantity: float
    price: float = None  # Pour ordres limit/stop
    status: PaperOrderStatus = PaperOrderStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    filled_quantity: float = 0.0
    commission: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "quantity": self.quantity,
            "price": self.price,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "filled_price": self.filled_price,
            "filled_quantity": self.filled_quantity,
            "commission": self.commission,
            "metadata": self.metadata
        }


@dataclass
class PaperTrade:
    """Trade exécuté en paper trading"""
    trade_id: str
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: datetime
    commission: float = 0.0
    pnl: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "timestamp": self.timestamp.isoformat(),
            "commission": self.commission,
            "pnl": self.pnl
        }


class PaperTrader:
    """
    Trader en mode simulation
    Exécute des stratégies sans risque financier réel
    """
    
    def __init__(self, initial_balance: float = 10000.0, config: Optional[Dict[str, Any]] = None):
        self.trader_id = str(uuid.uuid4())
        
        # Configuration
        self.config = {
            "commission_rate": 0.001,  # 0.1% par trade
            "slippage_rate": 0.0005,   # 0.05% slippage
            "latency_ms": 50,          # Latence simulation
            "allow_short_selling": True,
            "margin_requirement": 2.0,  # Marge pour short
            "max_positions": 10,       # Limite positions
            "risk_per_trade": 0.02,    # 2% max risque par trade
        }
        
        if config:
            self.config.update(config)
        
        # Portfolio virtuel
        self.portfolio = VirtualPortfolio(initial_balance)
        
        # Historiques
        self.orders: List[PaperOrder] = []
        self.trades: List[PaperTrade] = []
        self.pending_orders: Dict[str, PaperOrder] = {}
        
        # Stratégies actives
        self.active_strategies: Dict[str, BaseStrategy] = {}
        
        # Métriques de performance
        self.performance_metrics = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0
        }
        
        # État du trader
        self.is_active = True
        self.start_time = datetime.now(timezone.utc)
        
        logger.info(f"PaperTrader initialisé - Balance: ${initial_balance:,.2f}")
    
    def add_strategy(self, strategy: BaseStrategy) -> bool:
        """Ajoute une stratégie au paper trader"""
        try:
            if strategy.strategy_id in self.active_strategies:
                logger.warning(f"Stratégie {strategy.strategy_id} déjà active")
                return False
            
            self.active_strategies[strategy.strategy_id] = strategy
            logger.info(f"Stratégie {strategy.name} ajoutée au paper trader")
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur ajout stratégie: {str(e)}")
            return False
    
    def remove_strategy(self, strategy_id: str) -> bool:
        """Retire une stratégie du paper trader"""
        try:
            if strategy_id not in self.active_strategies:
                logger.warning(f"Stratégie {strategy_id} non trouvée")
                return False
            
            # Fermer toutes les positions de cette stratégie
            self._close_strategy_positions(strategy_id)
            
            # Annuler ordres en attente
            self._cancel_strategy_orders(strategy_id)
            
            # Retirer la stratégie
            del self.active_strategies[strategy_id]
            logger.info(f"Stratégie {strategy_id} retirée")
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur suppression stratégie: {str(e)}")
            return False
    
    def place_order(self, signal: TradingSignal, quantity: float = None) -> Optional[PaperOrder]:
        """Place un ordre basé sur un signal de stratégie"""
        try:
            if not self.is_active:
                logger.warning("Paper trader non actif")
                return None
            
            # Vérification stratégie
            if signal.strategy_id not in self.active_strategies:
                logger.error(f"Stratégie {signal.strategy_id} non trouvée")
                return None
            
            strategy = self.active_strategies[signal.strategy_id]
            
            # Calcul taille position si non spécifiée
            if quantity is None:
                quantity = strategy.calculate_position_size(signal, self.portfolio.total_value)
            
            # Vérification fonds disponibles
            required_margin = quantity * signal.suggested_price
            if not self.portfolio.has_sufficient_balance(required_margin):
                logger.warning("Fonds insuffisants pour ordre")
                return None
            
            # Création ordre
            order = PaperOrder(
                order_id=str(uuid.uuid4()),
                strategy_id=signal.strategy_id,
                symbol=signal.symbol,
                side="buy" if signal.signal_type == SignalType.BUY else "sell",
                order_type="market",
                quantity=abs(quantity),
                metadata={
                    "signal_confidence": signal.confidence,
                    "stop_loss": signal.stop_loss,
                    "take_profit": signal.take_profit,
                    "strategy_name": strategy.name
                }
            )
            
            # Ajout à l'historique et en attente
            self.orders.append(order)
            self.pending_orders[order.order_id] = order
            
            logger.info(f"Ordre paper placé: {order.side.upper()} {order.quantity} {order.symbol}")
            
            return order
            
        except Exception as e:
            logger.error(f"Erreur placement ordre: {str(e)}")
            return None
    
    def process_market_data(self, symbol: str, price_data: Dict[str, float]):
        """
        Traite les données de marché et exécute les ordres en attente
        """
        try:
            current_price = price_data.get("close", 0)
            if current_price <= 0:
                return
            
            # Mise à jour portfolio avec prix actuels
            self.portfolio.update_market_prices({symbol: current_price})
            
            # Traitement des ordres en attente pour ce symbol
            orders_to_process = [
                order for order in self.pending_orders.values() 
                if order.symbol == symbol
            ]
            
            for order in orders_to_process:
                self._execute_order(order, current_price)
            
            # Vérification des stops et take profits
            self._check_exit_conditions(symbol, current_price)
            
        except Exception as e:
            logger.error(f"Erreur traitement données marché: {str(e)}")
    
    def _execute_order(self, order: PaperOrder, current_price: float):
        """Exécute un ordre paper"""
        try:
            # Simulation latence
            execution_delay = timedelta(milliseconds=self.config["latency_ms"])
            
            # Calcul prix d'exécution avec slippage
            slippage = self.config["slippage_rate"]
            if order.side == "buy":
                execution_price = current_price * (1 + slippage)
            else:
                execution_price = current_price * (1 - slippage)
            
            # Calcul commission
            commission = order.quantity * execution_price * self.config["commission_rate"]
            
            # Vérification marge disponible
            required_value = order.quantity * execution_price + commission
            
            if order.side == "buy":
                if not self.portfolio.has_sufficient_balance(required_value):
                    order.status = PaperOrderStatus.REJECTED
                    self._remove_pending_order(order.order_id)
                    logger.warning(f"Ordre rejeté - fonds insuffisants: {order.order_id}")
                    return
            
            # Exécution ordre
            order.status = PaperOrderStatus.FILLED
            order.filled_at = datetime.now(timezone.utc) + execution_delay
            order.filled_price = execution_price
            order.filled_quantity = order.quantity
            order.commission = commission
            
            # Création trade
            trade = PaperTrade(
                trade_id=str(uuid.uuid4()),
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=execution_price,
                timestamp=order.filled_at,
                commission=commission
            )
            
            self.trades.append(trade)
            
            # Mise à jour portfolio
            if order.side == "buy":
                self.portfolio.add_position(order.symbol, order.quantity, execution_price, commission)
            else:
                self.portfolio.reduce_position(order.symbol, order.quantity, execution_price, commission)
            
            # Nettoyage
            self._remove_pending_order(order.order_id)
            
            # Mise à jour métriques
            self._update_performance_metrics(trade)
            
            logger.info(f"Ordre exécuté: {trade.side.upper()} {trade.quantity} {trade.symbol} @ ${execution_price:.4f}")
            
        except Exception as e:
            logger.error(f"Erreur exécution ordre: {str(e)}")
            order.status = PaperOrderStatus.REJECTED
            self._remove_pending_order(order.order_id)
    
    def _check_exit_conditions(self, symbol: str, current_price: float):
        """Vérifie les conditions de sortie (stop-loss, take-profit)"""
        try:
            position = self.portfolio.get_position(symbol)
            if not position:
                return
            
            position_side = "long" if position["quantity"] > 0 else "short"
            
            # Vérification stop-loss et take-profit
            for order in self.orders:
                if (order.symbol == symbol and 
                    order.status == PaperOrderStatus.FILLED and
                    order.metadata):
                    
                    stop_loss = order.metadata.get("stop_loss")
                    take_profit = order.metadata.get("take_profit")
                    
                    should_exit = False
                    exit_reason = ""
                    
                    if position_side == "long":
                        if stop_loss and current_price <= stop_loss:
                            should_exit = True
                            exit_reason = "stop_loss"
                        elif take_profit and current_price >= take_profit:
                            should_exit = True
                            exit_reason = "take_profit"
                    else:  # short position
                        if stop_loss and current_price >= stop_loss:
                            should_exit = True
                            exit_reason = "stop_loss"
                        elif take_profit and current_price <= take_profit:
                            should_exit = True
                            exit_reason = "take_profit"
                    
                    if should_exit:
                        self._create_exit_order(symbol, abs(position["quantity"]), 
                                              order.strategy_id, exit_reason)
                        break
            
        except Exception as e:
            logger.error(f"Erreur vérification conditions sortie: {str(e)}")
    
    def _create_exit_order(self, symbol: str, quantity: float, strategy_id: str, reason: str):
        """Crée un ordre de sortie"""
        try:
            position = self.portfolio.get_position(symbol)
            if not position:
                return
            
            # Détermination side opposé
            exit_side = "sell" if position["quantity"] > 0 else "buy"
            
            # Création signal de sortie
            exit_signal = TradingSignal(
                strategy_id=strategy_id,
                symbol=symbol,
                signal_type=SignalType.SELL if exit_side == "sell" else SignalType.BUY,
                confidence=1.0,  # Sortie forcée
                metadata={"exit_reason": reason}
            )
            
            # Placement ordre
            exit_order = self.place_order(exit_signal, quantity)
            if exit_order:
                logger.info(f"Ordre sortie créé: {reason} pour {symbol}")
            
        except Exception as e:
            logger.error(f"Erreur création ordre sortie: {str(e)}")
    
    def _close_strategy_positions(self, strategy_id: str):
        """Ferme toutes les positions d'une stratégie"""
        try:
            # Trouver tous les symboles avec positions de cette stratégie
            strategy_positions = []
            
            for order in self.orders:
                if (order.strategy_id == strategy_id and 
                    order.status == PaperOrderStatus.FILLED):
                    
                    position = self.portfolio.get_position(order.symbol)
                    if position and abs(position["quantity"]) > 0:
                        strategy_positions.append((order.symbol, abs(position["quantity"])))
            
            # Fermer chaque position
            for symbol, quantity in strategy_positions:
                self._create_exit_order(symbol, quantity, strategy_id, "strategy_removal")
            
        except Exception as e:
            logger.error(f"Erreur fermeture positions stratégie: {str(e)}")
    
    def _cancel_strategy_orders(self, strategy_id: str):
        """Annule tous les ordres en attente d'une stratégie"""
        try:
            orders_to_cancel = [
                order_id for order_id, order in self.pending_orders.items()
                if order.strategy_id == strategy_id
            ]
            
            for order_id in orders_to_cancel:
                self.cancel_order(order_id)
            
        except Exception as e:
            logger.error(f"Erreur annulation ordres stratégie: {str(e)}")
    
    def cancel_order(self, order_id: str) -> bool:
        """Annule un ordre en attente"""
        try:
            if order_id not in self.pending_orders:
                return False
            
            order = self.pending_orders[order_id]
            order.status = PaperOrderStatus.CANCELLED
            
            self._remove_pending_order(order_id)
            
            logger.info(f"Ordre annulé: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur annulation ordre: {str(e)}")
            return False
    
    def _remove_pending_order(self, order_id: str):
        """Retire un ordre des ordres en attente"""
        try:
            if order_id in self.pending_orders:
                del self.pending_orders[order_id]
                
        except Exception as e:
            logger.error(f"Erreur suppression ordre en attente: {str(e)}")
    
    def _update_performance_metrics(self, trade: PaperTrade):
        """Met à jour les métriques de performance"""
        try:
            self.performance_metrics["total_trades"] += 1
            
            # Calcul PnL si position fermée
            position = self.portfolio.get_position(trade.symbol)
            if position and position["quantity"] == 0:
                # Position fermée - calcul PnL final
                realized_pnl = position.get("realized_pnl", 0)
                
                if realized_pnl > 0:
                    self.performance_metrics["winning_trades"] += 1
                    
                    # Moyenne des gains
                    current_avg = self.performance_metrics["avg_win"]
                    win_count = self.performance_metrics["winning_trades"]
                    self.performance_metrics["avg_win"] = ((current_avg * (win_count - 1)) + realized_pnl) / win_count
                    
                elif realized_pnl < 0:
                    self.performance_metrics["losing_trades"] += 1
                    
                    # Moyenne des pertes
                    current_avg = self.performance_metrics["avg_loss"]
                    loss_count = self.performance_metrics["losing_trades"]
                    self.performance_metrics["avg_loss"] = ((current_avg * (loss_count - 1)) + abs(realized_pnl)) / loss_count
            
            # Calcul taux de réussite
            total = self.performance_metrics["total_trades"]
            wins = self.performance_metrics["winning_trades"]
            
            if total > 0:
                self.performance_metrics["win_rate"] = wins / total
                
                # Profit factor
                avg_win = self.performance_metrics["avg_win"]
                avg_loss = self.performance_metrics["avg_loss"]
                
                if avg_loss > 0:
                    self.performance_metrics["profit_factor"] = avg_win / avg_loss
            
        except Exception as e:
            logger.error(f"Erreur mise à jour métriques: {str(e)}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Génère un rapport de performance complet"""
        try:
            current_time = datetime.now(timezone.utc)
            trading_duration = (current_time - self.start_time).total_seconds() / 86400  # jours
            
            portfolio_summary = self.portfolio.get_summary()
            
            return {
                "trader_id": self.trader_id,
                "report_timestamp": current_time.isoformat(),
                "trading_duration_days": round(trading_duration, 2),
                
                "portfolio": portfolio_summary,
                
                "trading_metrics": self.performance_metrics,
                
                "activity": {
                    "total_orders": len(self.orders),
                    "pending_orders": len(self.pending_orders),
                    "total_trades": len(self.trades),
                    "active_strategies": len(self.active_strategies),
                    "active_positions": len([p for p in portfolio_summary["positions"] if p["quantity"] != 0])
                },
                
                "recent_trades": [
                    trade.to_dict() for trade in self.trades[-10:]  # 10 derniers trades
                ],
                
                "recent_orders": [
                    order.to_dict() for order in self.orders[-10:]  # 10 derniers ordres
                ]
            }
            
        except Exception as e:
            logger.error(f"Erreur génération rapport: {str(e)}")
            return {"error": str(e)}
    
    def reset_trader(self, new_balance: float = None):
        """Remet à zéro le paper trader"""
        try:
            balance = new_balance or self.portfolio.initial_balance
            
            # Reset portfolio
            self.portfolio = VirtualPortfolio(balance)
            
            # Vider historiques
            self.orders.clear()
            self.trades.clear()
            self.pending_orders.clear()
            
            # Reset métriques
            self.performance_metrics = {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0
            }
            
            # Reset temps
            self.start_time = datetime.now(timezone.utc)
            self.is_active = True
            
            logger.info(f"Paper trader remis à zéro - Nouvelle balance: ${balance:,.2f}")
            
        except Exception as e:
            logger.error(f"Erreur reset trader: {str(e)}")
    
    def stop_trading(self):
        """Arrête le trading et ferme toutes positions"""
        try:
            self.is_active = False
            
            # Annuler tous les ordres en attente
            for order_id in list(self.pending_orders.keys()):
                self.cancel_order(order_id)
            
            # Fermer toutes les positions
            for strategy_id in list(self.active_strategies.keys()):
                self._close_strategy_positions(strategy_id)
            
            logger.info("Paper trading arrêté")
            
        except Exception as e:
            logger.error(f"Erreur arrêt trading: {str(e)}")
    
    def get_active_positions(self) -> List[Dict[str, Any]]:
        """Retourne les positions actives"""
        return self.portfolio.get_active_positions()
    
    def get_order_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retourne l'historique des ordres"""
        try:
            recent_orders = self.orders[-limit:] if limit > 0 else self.orders
            return [order.to_dict() for order in reversed(recent_orders)]
            
        except Exception as e:
            logger.error(f"Erreur récupération historique ordres: {str(e)}")
            return []
    
    def get_trade_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retourne l'historique des trades"""
        try:
            recent_trades = self.trades[-limit:] if limit > 0 else self.trades
            return [trade.to_dict() for trade in reversed(recent_trades)]
            
        except Exception as e:
            logger.error(f"Erreur récupération historique trades: {str(e)}")
            return []