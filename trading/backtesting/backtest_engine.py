"""
🧪 Backtest Engine - Moteur de Backtesting
Exécution de tests historiques des stratégies
"""

import asyncio
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import pandas as pd

from core.logger import get_logger
from ..engine.execution_handler import TradingSignal, SignalType
from ..engine.order_manager import Order, OrderStatus, OrderSide, OrderType
from ..engine.position_manager import Position, PositionSide, PositionStatus

logger = get_logger(__name__)


class BacktestStatus(Enum):
    """Status du backtest"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class BacktestConfig:
    """Configuration du backtest"""
    strategy_id: str
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_balance: float = 10000.0
    timeframe: str = "1h"
    commission_rate: float = 0.001
    slippage_rate: float = 0.0001
    max_positions: int = 1
    enable_stop_loss: bool = True
    enable_take_profit: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "initial_balance": self.initial_balance,
            "timeframe": self.timeframe,
            "commission_rate": self.commission_rate,
            "slippage_rate": self.slippage_rate,
            "max_positions": self.max_positions,
            "enable_stop_loss": self.enable_stop_loss,
            "enable_take_profit": self.enable_take_profit
        }


@dataclass
class BacktestTrade:
    """Trade exécuté pendant le backtest"""
    id: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_percent: float
    commission: float
    duration_hours: float
    strategy_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "pnl": self.pnl,
            "pnl_percent": self.pnl_percent,
            "commission": self.commission,
            "duration_hours": self.duration_hours,
            "strategy_id": self.strategy_id
        }


@dataclass
class BacktestResult:
    """Résultat complet d'un backtest"""
    backtest_id: str
    config: BacktestConfig
    status: BacktestStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Portfolio metrics
    initial_balance: float = 0.0
    final_balance: float = 0.0
    total_return: float = 0.0
    total_return_percent: float = 0.0
    
    # Trading metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    profit_factor: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Trades executed
    trades: List[BacktestTrade] = field(default_factory=list)
    
    # Performance over time
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    
    # Error messages
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "backtest_id": self.backtest_id,
            "config": self.config.to_dict(),
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            
            # Portfolio metrics
            "initial_balance": self.initial_balance,
            "final_balance": self.final_balance,
            "total_return": self.total_return,
            "total_return_percent": self.total_return_percent,
            
            # Trading metrics  
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "average_win": self.average_win,
            "average_loss": self.average_loss,
            "profit_factor": self.profit_factor,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
            
            # Risk metrics
            "max_drawdown": self.max_drawdown,
            "max_drawdown_percent": self.max_drawdown_percent,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            
            "trades_count": len(self.trades),
            "trades": [trade.to_dict() for trade in self.trades[:100]],  # Limiter pour la réponse
            "equity_curve_points": len(self.equity_curve),
            "error_message": self.error_message
        }


class BacktestEngine:
    """
    Moteur de backtesting pour tester les stratégies sur données historiques
    """
    
    def __init__(self):
        self.active_backtests: Dict[str, BacktestResult] = {}
        self.completed_backtests: Dict[str, BacktestResult] = {}
        
        logger.info("BacktestEngine initialisé")
    
    async def run_backtest(
        self,
        strategy,
        data: pd.DataFrame,
        config: BacktestConfig
    ) -> str:
        """
        Lance un backtest avec la stratégie fournie
        """
        try:
            backtest_id = str(uuid.uuid4())
            
            # Initialisation du résultat
            result = BacktestResult(
                backtest_id=backtest_id,
                config=config,
                status=BacktestStatus.RUNNING,
                start_time=datetime.now(timezone.utc),
                initial_balance=config.initial_balance
            )
            
            self.active_backtests[backtest_id] = result
            
            logger.info(f"Démarrage backtest {backtest_id} pour stratégie {strategy.name}")
            
            # Exécution du backtest
            await self._execute_backtest(strategy, data, result)
            
            # Finalisation
            result.status = BacktestStatus.COMPLETED
            result.end_time = datetime.now(timezone.utc)
            
            # Déplacement vers les backtests complétés
            self.completed_backtests[backtest_id] = result
            del self.active_backtests[backtest_id]
            
            logger.info(f"Backtest {backtest_id} terminé - Return: {result.total_return_percent:.2f}%")
            
            return backtest_id
            
        except Exception as e:
            logger.error(f"Erreur backtest {backtest_id}: {str(e)}")
            if backtest_id in self.active_backtests:
                result = self.active_backtests[backtest_id]
                result.status = BacktestStatus.ERROR
                result.error_message = str(e)
                result.end_time = datetime.now(timezone.utc)
                
                self.completed_backtests[backtest_id] = result
                del self.active_backtests[backtest_id]
            
            raise
    
    async def _execute_backtest(
        self,
        strategy,
        data: pd.DataFrame,
        result: BacktestResult
    ):
        """
        Exécute le backtest principal
        """
        try:
            # Variables d'état
            current_balance = result.config.initial_balance
            open_positions: Dict[str, Position] = {}
            portfolio_values = []
            
            # Tri des données par timestamp (l'index est déjà timestamp)
            data_sorted = data.sort_index()
            
            logger.info(f"Backtesting sur {len(data_sorted)} points de données")
            
            # Boucle principale sur les données historiques
            for index, row in data_sorted.iterrows():
                current_time = index  # L'index est déjà le timestamp
                current_price = row['close']
                
                # Mise à jour des prix des positions ouvertes
                for position in open_positions.values():
                    position.current_price = current_price
                    position.updated_at = current_time
                
                # Vérification stop-loss et take-profit
                positions_to_close = self._check_stop_take_profit(open_positions, current_time, current_price)
                
                # Fermeture des positions
                for position_id in positions_to_close:
                    await self._close_position(
                        position_id, open_positions, current_price, 
                        current_time, result, current_balance
                    )
                
                # Génération de signal par la stratégie
                try:
                    # Préparer les données pour la stratégie (window des dernières valeurs)
                    window_size = 50  # Fenêtre de données
                    current_position = data_sorted.index.get_loc(index)
                    start_idx = max(0, current_position - window_size + 1)
                    end_idx = current_position + 1
                    window_data = data_sorted.iloc[start_idx:end_idx].copy()
                    
                    if len(window_data) >= 20:  # Minimum de données requis
                        signal = strategy.process_market_data(window_data)
                        
                        if signal and signal.signal_type != SignalType.HOLD:
                            # Vérifier si on peut ouvrir une nouvelle position
                            if len(open_positions) < result.config.max_positions:
                                await self._process_signal(
                                    signal, current_price, current_time, 
                                    open_positions, result, current_balance
                                )
                
                except Exception as e:
                    logger.warning(f"Erreur traitement signal à {current_time}: {str(e)}")
                    continue
                
                # Calcul valeur portfolio
                portfolio_value = current_balance
                for position in open_positions.values():
                    portfolio_value += position.unrealized_pnl
                
                portfolio_values.append({
                    "timestamp": current_time.isoformat() if hasattr(current_time, 'isoformat') else str(current_time),
                    "portfolio_value": portfolio_value,
                    "cash_balance": current_balance,
                    "unrealized_pnl": sum(p.unrealized_pnl for p in open_positions.values()),
                    "open_positions": len(open_positions)
                })
                
                # Limitation de la courbe d'equity (garder 1 point sur 10 pour performance)
                if len(portfolio_values) % 10 == 0:
                    result.equity_curve.append(portfolio_values[-1])
            
            # Fermeture des positions restantes
            final_price = data_sorted['close'].iloc[-1]
            final_time = data_sorted.index[-1]
            
            for position_id in list(open_positions.keys()):
                await self._close_position(
                    position_id, open_positions, final_price,
                    final_time, result, current_balance
                )
            
            # Calcul des métriques finales
            result.final_balance = current_balance
            await self._calculate_final_metrics(result, portfolio_values)
            
        except Exception as e:
            logger.error(f"Erreur exécution backtest: {str(e)}")
            raise
    
    async def _process_signal(
        self,
        signal: TradingSignal,
        current_price: float,
        current_time,
        open_positions: Dict[str, Position],
        result: BacktestResult,
        current_balance: float
    ):
        """
        Traite un signal de trading durant le backtest
        """
        try:
            # Calcul de la taille de position
            position_size = self._calculate_position_size(signal, current_balance, result.config)
            
            if position_size <= 0:
                return
            
            # Application du slippage
            entry_price = self._apply_slippage(current_price, signal.signal_type, result.config.slippage_rate)
            
            # Calcul de la commission
            commission = position_size * entry_price * result.config.commission_rate
            
            # Vérification du capital disponible
            required_capital = position_size * entry_price + commission
            if required_capital > current_balance:
                logger.warning(f"Capital insuffisant pour signal {signal.signal_type.value}")
                return
            
            # Création de la position
            position_id = str(uuid.uuid4())
            side = PositionSide.LONG if signal.signal_type == SignalType.BUY else PositionSide.SHORT
            
            position = Position(
                id=position_id,
                symbol=signal.symbol,
                side=side,
                quantity=position_size,
                entry_price=entry_price,
                current_price=entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                created_at=current_time,
                strategy_id=signal.strategy_id
            )
            
            open_positions[position_id] = position
            
            # Mise à jour du balance (déduction du capital utilisé + commission)
            # Note: En backtest, on ne déduit que la commission, le capital reste "investi"
            current_balance -= commission
            
            logger.debug(f"Position ouverte: {side.value} {position_size} @ {entry_price}")
            
        except Exception as e:
            logger.error(f"Erreur traitement signal: {str(e)}")
    
    async def _close_position(
        self,
        position_id: str,
        open_positions: Dict[str, Position],
        exit_price: float,
        exit_time,
        result: BacktestResult,
        current_balance: float
    ) -> float:
        """
        Ferme une position et calcule le PnL
        """
        try:
            position = open_positions[position_id]
            
            # Application du slippage à la sortie
            actual_exit_price = self._apply_slippage(
                exit_price, 
                SignalType.SELL if position.side == PositionSide.LONG else SignalType.BUY,
                result.config.slippage_rate
            )
            
            # Calcul du PnL
            if position.side == PositionSide.LONG:
                pnl = (actual_exit_price - position.entry_price) * position.quantity
            else:
                pnl = (position.entry_price - actual_exit_price) * position.quantity
            
            # Commission de sortie
            exit_commission = position.quantity * actual_exit_price * result.config.commission_rate
            pnl -= exit_commission
            
            # Commission d'entrée (déjà déduite du balance)
            entry_commission = position.quantity * position.entry_price * result.config.commission_rate
            total_commission = entry_commission + exit_commission
            
            # PnL en pourcentage
            pnl_percent = (pnl / (position.entry_price * position.quantity)) * 100
            
            # Durée de la position
            if hasattr(exit_time, 'timestamp') and hasattr(position.created_at, 'timestamp'):
                duration = (exit_time.timestamp() - position.created_at.timestamp()) / 3600
            else:
                duration = 1.0  # Durée par défaut
            
            # Création du trade
            trade = BacktestTrade(
                id=str(uuid.uuid4()),
                symbol=position.symbol,
                side=position.side.value,
                entry_price=position.entry_price,
                exit_price=actual_exit_price,
                quantity=position.quantity,
                entry_time=position.created_at,
                exit_time=exit_time,
                pnl=pnl,
                pnl_percent=pnl_percent,
                commission=total_commission,
                duration_hours=duration,
                strategy_id=position.strategy_id
            )
            
            result.trades.append(trade)
            
            # Mise à jour du balance
            current_balance += (position.entry_price * position.quantity) + pnl  # Capital + PnL
            
            # Suppression de la position
            del open_positions[position_id]
            
            logger.debug(f"Position fermée: PnL {pnl:.2f} ({pnl_percent:.2f}%)")
            
            return current_balance
            
        except Exception as e:
            logger.error(f"Erreur fermeture position: {str(e)}")
            return current_balance
    
    def _check_stop_take_profit(
        self,
        open_positions: Dict[str, Position],
        current_time,
        current_price: float
    ) -> List[str]:
        """
        Vérifie les conditions de stop-loss et take-profit
        """
        positions_to_close = []
        
        for position_id, position in open_positions.items():
            should_close = False
            
            if position.should_stop_loss():
                logger.debug(f"Stop-loss déclenché pour position {position_id}")
                should_close = True
            elif position.should_take_profit():
                logger.debug(f"Take-profit déclenché pour position {position_id}")
                should_close = True
            
            if should_close:
                positions_to_close.append(position_id)
        
        return positions_to_close
    
    def _calculate_position_size(
        self,
        signal: TradingSignal,
        current_balance: float,
        config: BacktestConfig
    ) -> float:
        """
        Calcule la taille de position pour le signal
        """
        try:
            # Taille basée sur un pourcentage du capital (2% par défaut)
            risk_percent = 0.02
            
            # Ajustement basé sur la confiance du signal
            adjusted_risk = risk_percent * signal.confidence
            
            # Capital à risquer
            risk_capital = current_balance * adjusted_risk
            
            # Si stop-loss défini, calculer la taille basée sur le risque
            if signal.stop_loss and signal.suggested_price:
                if signal.signal_type == SignalType.BUY:
                    risk_per_unit = signal.suggested_price - signal.stop_loss
                else:
                    risk_per_unit = signal.stop_loss - signal.suggested_price
                
                if risk_per_unit > 0:
                    position_size = risk_capital / risk_per_unit
                else:
                    position_size = risk_capital / signal.suggested_price
            else:
                # Taille basée sur le capital disponible
                position_size = risk_capital / signal.suggested_price if signal.suggested_price else 0
            
            # Limitation de la taille maximum
            max_size = current_balance * 0.1  # Maximum 10% du capital par trade
            position_size = min(position_size, max_size)
            
            return max(0, position_size)
            
        except Exception as e:
            logger.error(f"Erreur calcul position size: {str(e)}")
            return 0
    
    def _apply_slippage(self, price: float, signal_type: SignalType, slippage_rate: float) -> float:
        """
        Applique le slippage au prix
        """
        if signal_type in [SignalType.BUY, SignalType.CLOSE_SHORT]:
            return price * (1 + slippage_rate)  # Prix légèrement plus élevé
        else:
            return price * (1 - slippage_rate)  # Prix légèrement plus bas
    
    async def _calculate_final_metrics(
        self,
        result: BacktestResult,
        portfolio_values: List[Dict]
    ):
        """
        Calcule les métriques finales du backtest
        """
        try:
            # Métriques de base
            result.total_return = result.final_balance - result.initial_balance
            result.total_return_percent = (result.total_return / result.initial_balance) * 100
            
            # Métriques de trading
            result.total_trades = len(result.trades)
            result.winning_trades = len([t for t in result.trades if t.pnl > 0])
            result.losing_trades = len([t for t in result.trades if t.pnl < 0])
            
            if result.total_trades > 0:
                result.win_rate = (result.winning_trades / result.total_trades) * 100
                
                winning_trades = [t.pnl for t in result.trades if t.pnl > 0]
                losing_trades = [t.pnl for t in result.trades if t.pnl < 0]
                
                result.average_win = sum(winning_trades) / len(winning_trades) if winning_trades else 0
                result.average_loss = sum(losing_trades) / len(losing_trades) if losing_trades else 0
                
                result.largest_win = max([t.pnl for t in result.trades]) if result.trades else 0
                result.largest_loss = min([t.pnl for t in result.trades]) if result.trades else 0
                
                # Profit Factor
                gross_profit = sum(winning_trades) if winning_trades else 0
                gross_loss = abs(sum(losing_trades)) if losing_trades else 0
                result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Calcul du drawdown maximum
            if portfolio_values:
                peak_value = result.initial_balance
                max_drawdown = 0
                
                for pv in portfolio_values:
                    current_value = pv["portfolio_value"]
                    
                    if current_value > peak_value:
                        peak_value = current_value
                    
                    drawdown = peak_value - current_value
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
                
                result.max_drawdown = max_drawdown
                result.max_drawdown_percent = (max_drawdown / peak_value) * 100 if peak_value > 0 else 0
            
            # Calcul Sharpe Ratio (simplifié)
            if len(portfolio_values) > 1:
                returns = []
                for i in range(1, len(portfolio_values)):
                    prev_value = portfolio_values[i-1]["portfolio_value"]
                    curr_value = portfolio_values[i]["portfolio_value"]
                    
                    if prev_value > 0:
                        ret = (curr_value - prev_value) / prev_value
                        returns.append(ret)
                
                if returns:
                    import statistics
                    mean_return = statistics.mean(returns)
                    std_return = statistics.stdev(returns) if len(returns) > 1 else 0
                    
                    # Sharpe ratio (assumant risk-free rate = 0)
                    result.sharpe_ratio = mean_return / std_return if std_return > 0 else 0
                    
                    # Sortino ratio (seulement downside deviation)
                    negative_returns = [r for r in returns if r < 0]
                    if negative_returns:
                        downside_std = statistics.stdev(negative_returns)
                        result.sortino_ratio = mean_return / downside_std if downside_std > 0 else 0
                    
                    # Calmar ratio
                    annualized_return = mean_return * 365 * 24  # Assuming hourly data
                    result.calmar_ratio = annualized_return / result.max_drawdown_percent if result.max_drawdown_percent > 0 else 0
            
            logger.info(f"Métriques finales calculées - Trades: {result.total_trades}, Win Rate: {result.win_rate:.1f}%")
            
        except Exception as e:
            logger.error(f"Erreur calcul métriques finales: {str(e)}")
    
    def get_backtest_result(self, backtest_id: str) -> Optional[BacktestResult]:
        """
        Récupère le résultat d'un backtest
        """
        if backtest_id in self.completed_backtests:
            return self.completed_backtests[backtest_id]
        elif backtest_id in self.active_backtests:
            return self.active_backtests[backtest_id]
        return None
    
    def get_backtest_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Récupère l'historique des backtests
        """
        try:
            all_results = list(self.completed_backtests.values()) + list(self.active_backtests.values())
            
            # Trier par date de début décroissante
            sorted_results = sorted(all_results, key=lambda x: x.start_time, reverse=True)
            
            # Limiter et convertir en dict
            limited_results = sorted_results[:limit]
            
            return [result.to_dict() for result in limited_results]
            
        except Exception as e:
            logger.error(f"Erreur récupération historique backtests: {str(e)}")
            return []
    
    def cancel_backtest(self, backtest_id: str) -> bool:
        """
        Annule un backtest en cours
        """
        try:
            if backtest_id in self.active_backtests:
                result = self.active_backtests[backtest_id]
                result.status = BacktestStatus.CANCELLED
                result.end_time = datetime.now(timezone.utc)
                result.error_message = "Backtest annulé par l'utilisateur"
                
                # Déplacer vers les complétés
                self.completed_backtests[backtest_id] = result
                del self.active_backtests[backtest_id]
                
                logger.info(f"Backtest {backtest_id} annulé")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Erreur annulation backtest: {str(e)}")
            return False
    
    def get_engine_status(self) -> Dict[str, Any]:
        """
        Retourne le status du moteur de backtesting
        """
        return {
            "active_backtests": len(self.active_backtests),
            "completed_backtests": len(self.completed_backtests),
            "total_backtests": len(self.active_backtests) + len(self.completed_backtests),
            "engine_status": "operational"
        }