"""
🎮 Simulation Engine - Moteur de Simulation Trading
Orchestrateur pour simulations complètes de stratégies
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import pandas as pd
import asyncio
import threading
import time

from core.logger import get_logger
from .paper_trader import PaperTrader
from .virtual_portfolio import VirtualPortfolio
from ..strategies.base_strategy import BaseStrategy
from data.collectors.data_collector import DataCollector

logger = get_logger(__name__)


class SimulationStatus(Enum):
    """États de la simulation"""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class SimulationConfig:
    """Configuration de simulation"""
    name: str
    description: str = ""
    initial_balance: float = 10000.0
    
    # Période de simulation
    start_date: datetime = None
    end_date: datetime = None
    timeframe: str = "1h"  # 1m, 5m, 15m, 1h, 4h, 1d
    
    # Symboles à trader
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT"])
    
    # Configuration trading
    max_positions: int = 5
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    
    # Configuration stratégies
    strategies_config: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Options avancées
    speed_multiplier: float = 1.0  # Vitesse de simulation
    save_intermediate_results: bool = True
    generate_detailed_logs: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "initial_balance": self.initial_balance,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "timeframe": self.timeframe,
            "symbols": self.symbols,
            "max_positions": self.max_positions,
            "commission_rate": self.commission_rate,
            "slippage_rate": self.slippage_rate,
            "strategies_config": self.strategies_config,
            "speed_multiplier": self.speed_multiplier,
            "save_intermediate_results": self.save_intermediate_results,
            "generate_detailed_logs": self.generate_detailed_logs
        }


@dataclass
class SimulationResult:
    """Résultat d'une simulation"""
    simulation_id: str
    config: SimulationConfig
    status: SimulationStatus
    
    # Résultats financiers
    initial_balance: float
    final_balance: float
    total_return: float
    total_return_percent: float
    max_drawdown: float
    
    # Métriques trading
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Métriques risque
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    volatility: float = 0.0
    var_95: float = 0.0
    
    # Timing
    start_time: datetime = None
    end_time: datetime = None
    duration_seconds: float = 0.0
    
    # Données détaillées
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    trades_history: List[Dict[str, Any]] = field(default_factory=list)
    strategy_performance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "simulation_id": self.simulation_id,
            "config": self.config.to_dict(),
            "status": self.status.value,
            "financial_results": {
                "initial_balance": self.initial_balance,
                "final_balance": self.final_balance,
                "total_return": self.total_return,
                "total_return_percent": self.total_return_percent,
                "max_drawdown": self.max_drawdown
            },
            "trading_metrics": {
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
                "losing_trades": self.losing_trades,
                "win_rate": self.win_rate,
                "avg_win": self.avg_win,
                "avg_loss": self.avg_loss,
                "profit_factor": self.profit_factor
            },
            "risk_metrics": {
                "sharpe_ratio": self.sharpe_ratio,
                "sortino_ratio": self.sortino_ratio,
                "volatility": self.volatility,
                "var_95": self.var_95
            },
            "timing": {
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "duration_seconds": self.duration_seconds
            },
            "data_points": {
                "equity_curve_length": len(self.equity_curve),
                "trades_count": len(self.trades_history),
                "strategies_tested": len(self.strategy_performance)
            }
        }


class SimulationEngine:
    """
    Moteur de simulation pour backtesting avancé
    Orchestre paper traders et stratégies sur données historiques
    """
    
    def __init__(self):
        self.simulation_id = None
        self.status = SimulationStatus.STOPPED
        
        # Composants
        self.paper_trader: Optional[PaperTrader] = None
        self.data_collector = DataCollector()
        
        # Configuration et résultats
        self.config: Optional[SimulationConfig] = None
        self.result: Optional[SimulationResult] = None
        
        # Données de simulation
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.current_timestamp: Optional[datetime] = None
        
        # Contrôle simulation
        self._simulation_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        
        # Callbacks
        self.progress_callback: Optional[Callable] = None
        self.update_callback: Optional[Callable] = None
        
        logger.info("SimulationEngine initialisé")
    
    def prepare_simulation(self, config: SimulationConfig) -> bool:
        """Prépare une simulation avec la configuration donnée"""
        try:
            self.config = config
            self.simulation_id = f"sim_{int(time.time())}"
            
            # Validation configuration
            if not self._validate_config():
                return False
            
            # Chargement des données historiques
            logger.info("Chargement données historiques...")
            if not self._load_historical_data():
                return False
            
            # Initialisation paper trader
            trader_config = {
                "commission_rate": config.commission_rate,
                "slippage_rate": config.slippage_rate,
                "max_positions": config.max_positions
            }
            
            self.paper_trader = PaperTrader(
                initial_balance=config.initial_balance,
                config=trader_config
            )
            
            # Configuration des stratégies
            if not self._setup_strategies():
                return False
            
            self.status = SimulationStatus.STOPPED
            logger.info(f"Simulation {self.simulation_id} prête")
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur préparation simulation: {str(e)}")
            self.status = SimulationStatus.ERROR
            return False
    
    def start_simulation(self, async_mode: bool = True) -> bool:
        """Démarre la simulation"""
        try:
            if self.status != SimulationStatus.STOPPED:
                logger.warning("Simulation déjà en cours ou pas prête")
                return False
            
            if not self.config or not self.paper_trader:
                logger.error("Simulation non préparée")
                return False
            
            # Reset événements de contrôle
            self._stop_event.clear()
            self._pause_event.clear()
            
            if async_mode:
                # Démarrage en thread séparé
                self._simulation_thread = threading.Thread(
                    target=self._run_simulation_loop,
                    name=f"Simulation-{self.simulation_id}"
                )
                self._simulation_thread.start()
            else:
                # Exécution synchrone
                self._run_simulation_loop()
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur démarrage simulation: {str(e)}")
            self.status = SimulationStatus.ERROR
            return False
    
    def pause_simulation(self):
        """Met en pause la simulation"""
        if self.status == SimulationStatus.RUNNING:
            self._pause_event.set()
            self.status = SimulationStatus.PAUSED
            logger.info("Simulation mise en pause")
    
    def resume_simulation(self):
        """Reprend la simulation"""
        if self.status == SimulationStatus.PAUSED:
            self._pause_event.clear()
            self.status = SimulationStatus.RUNNING
            logger.info("Simulation reprise")
    
    def stop_simulation(self):
        """Arrête la simulation"""
        if self.status in [SimulationStatus.RUNNING, SimulationStatus.PAUSED]:
            self._stop_event.set()
            logger.info("Arrêt simulation demandé")
    
    def _validate_config(self) -> bool:
        """Valide la configuration de simulation"""
        try:
            if not self.config.symbols:
                logger.error("Aucun symbole configuré")
                return False
            
            if not self.config.start_date or not self.config.end_date:
                logger.error("Dates de simulation non configurées")
                return False
            
            if self.config.start_date >= self.config.end_date:
                logger.error("Date de début après date de fin")
                return False
            
            if self.config.initial_balance <= 0:
                logger.error("Balance initiale invalide")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur validation config: {str(e)}")
            return False
    
    def _load_historical_data(self) -> bool:
        """Charge les données historiques nécessaires"""
        try:
            self.market_data = {}
            
            for symbol in self.config.symbols:
                logger.info(f"Chargement données {symbol}...")
                
                # Utilisation du data collector existant
                df = self.data_collector.get_historical_data(
                    symbol=symbol,
                    interval=self.config.timeframe,
                    start_date=self.config.start_date,
                    end_date=self.config.end_date
                )
                
                if df is None or len(df) == 0:
                    logger.error(f"Pas de données pour {symbol}")
                    return False
                
                # Tri par timestamp
                df = df.sort_values('timestamp')
                self.market_data[symbol] = df
                
                logger.info(f"Chargé {len(df)} points de données pour {symbol}")
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur chargement données: {str(e)}")
            return False
    
    def _setup_strategies(self) -> bool:
        """Configure et ajoute les stratégies au paper trader"""
        try:
            if not self.config.strategies_config:
                logger.warning("Aucune stratégie configurée")
                return False
            
            for strategy_name, strategy_params in self.config.strategies_config.items():
                # Création dynamique de la stratégie
                strategy = self._create_strategy(strategy_name, strategy_params)
                if strategy:
                    self.paper_trader.add_strategy(strategy)
                    logger.info(f"Stratégie {strategy_name} ajoutée")
                else:
                    logger.error(f"Échec création stratégie {strategy_name}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur setup stratégies: {str(e)}")
            return False
    
    def _create_strategy(self, strategy_name: str, params: Dict[str, Any]) -> Optional[BaseStrategy]:
        """Crée une instance de stratégie dynamiquement"""
        try:
            # Import dynamique des stratégies
            if strategy_name.lower() == "trend_following":
                from ..strategies.trend_following import TrendFollowingStrategy
                return TrendFollowingStrategy(
                    symbol=params.get("symbol", self.config.symbols[0]),
                    timeframe=self.config.timeframe,
                    parameters=params.get("parameters", {})
                )
            
            elif strategy_name.lower() == "mean_reversion":
                from ..strategies.mean_reversion import MeanReversionStrategy
                return MeanReversionStrategy(
                    symbol=params.get("symbol", self.config.symbols[0]),
                    timeframe=self.config.timeframe,
                    parameters=params.get("parameters", {})
                )
            
            elif strategy_name.lower() == "momentum":
                from ..strategies.momentum import MomentumStrategy
                return MomentumStrategy(
                    symbol=params.get("symbol", self.config.symbols[0]),
                    timeframe=self.config.timeframe,
                    parameters=params.get("parameters", {})
                )
            
            elif strategy_name.lower() == "arbitrage":
                from ..strategies.arbitrage import ArbitrageStrategy
                return ArbitrageStrategy(
                    symbol=params.get("symbol", self.config.symbols[0]),
                    timeframe=self.config.timeframe,
                    parameters=params.get("parameters", {})
                )
            
            else:
                logger.error(f"Stratégie inconnue: {strategy_name}")
                return None
            
        except Exception as e:
            logger.error(f"Erreur création stratégie {strategy_name}: {str(e)}")
            return None
    
    def _run_simulation_loop(self):
        """Boucle principale de simulation"""
        try:
            self.status = SimulationStatus.RUNNING
            start_time = datetime.now(timezone.utc)
            
            logger.info(f"Démarrage simulation {self.simulation_id}")
            
            # Initialisation résultat
            self.result = SimulationResult(
                simulation_id=self.simulation_id,
                config=self.config,
                status=self.status,
                initial_balance=self.config.initial_balance,
                final_balance=0.0,
                total_return=0.0,
                total_return_percent=0.0,
                max_drawdown=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                start_time=start_time
            )
            
            # Obtenir toutes les timestamps uniques
            all_timestamps = self._get_unified_timestamps()
            total_steps = len(all_timestamps)
            
            for i, timestamp in enumerate(all_timestamps):
                # Vérification arrêt/pause
                if self._stop_event.is_set():
                    logger.info("Simulation arrêtée")
                    break
                
                while self._pause_event.is_set():
                    time.sleep(0.1)  # Attendre reprise
                
                self.current_timestamp = timestamp
                
                # Traitement données marché pour ce timestamp
                self._process_market_tick(timestamp)
                
                # Génération et traitement signaux stratégies
                self._process_strategy_signals(timestamp)
                
                # Sauvegarde intermédiaire
                if self.config.save_intermediate_results and i % 100 == 0:
                    self._save_intermediate_snapshot()
                
                # Callback de progression
                if self.progress_callback:
                    progress = (i + 1) / total_steps * 100
                    self.progress_callback(progress, i + 1, total_steps)
                
                # Contrôle vitesse simulation
                if self.config.speed_multiplier < 1.0:
                    sleep_time = (1.0 / self.config.speed_multiplier - 1.0) * 0.01
                    time.sleep(sleep_time)
            
            # Finalisation simulation
            self._finalize_simulation(start_time)
            
        except Exception as e:
            logger.error(f"Erreur boucle simulation: {str(e)}")
            self.status = SimulationStatus.ERROR
            if self.result:
                self.result.status = SimulationStatus.ERROR
    
    def _get_unified_timestamps(self) -> List[datetime]:
        """Obtient toutes les timestamps unifiées de tous les symboles"""
        try:
            all_timestamps = set()
            
            for symbol, df in self.market_data.items():
                timestamps = pd.to_datetime(df['timestamp'])
                all_timestamps.update(timestamps)
            
            return sorted(list(all_timestamps))
            
        except Exception as e:
            logger.error(f"Erreur unification timestamps: {str(e)}")
            return []
    
    def _process_market_tick(self, timestamp: datetime):
        """Traite un tick de marché pour tous les symboles"""
        try:
            for symbol, df in self.market_data.items():
                # Trouver la ligne correspondante à ce timestamp
                mask = pd.to_datetime(df['timestamp']) == timestamp
                current_data = df[mask]
                
                if len(current_data) > 0:
                    row = current_data.iloc[0]
                    price_data = {
                        "timestamp": timestamp,
                        "open": row.get("open", 0),
                        "high": row.get("high", 0),
                        "low": row.get("low", 0),
                        "close": row.get("close", 0),
                        "volume": row.get("volume", 0)
                    }
                    
                    # Envoi au paper trader
                    self.paper_trader.process_market_data(symbol, price_data)
            
        except Exception as e:
            logger.error(f"Erreur traitement tick marché: {str(e)}")
    
    def _process_strategy_signals(self, timestamp: datetime):
        """Traite les signaux de toutes les stratégies actives"""
        try:
            for strategy_id, strategy in self.paper_trader.active_strategies.items():
                symbol = strategy.symbol
                
                if symbol in self.market_data:
                    # Obtenir données jusqu'à ce timestamp
                    df = self.market_data[symbol]
                    timestamp_mask = pd.to_datetime(df['timestamp']) <= timestamp
                    historical_data = df[timestamp_mask].copy()
                    
                    if len(historical_data) > 0:
                        # Génération signal
                        signal = strategy.generate_signal(historical_data)
                        
                        # Placement ordre si signal valide
                        if signal.signal_type.value != "HOLD":
                            order = self.paper_trader.place_order(signal)
                            
                            if self.config.generate_detailed_logs and order:
                                logger.debug(f"Signal {signal.signal_type.value} - {strategy.name} - {symbol}")
            
        except Exception as e:
            logger.error(f"Erreur traitement signaux stratégies: {str(e)}")
    
    def _save_intermediate_snapshot(self):
        """Sauvegarde un snapshot intermédiaire"""
        try:
            if self.result and self.paper_trader:
                portfolio_summary = self.paper_trader.portfolio.get_summary()
                
                equity_point = {
                    "timestamp": self.current_timestamp.isoformat(),
                    "total_value": portfolio_summary["current_values"]["total_value"],
                    "cash": portfolio_summary["current_values"]["cash_balance"],
                    "positions_value": portfolio_summary["current_values"]["positions_value"],
                    "total_pnl": portfolio_summary["pnl_summary"]["total_pnl"]
                }
                
                self.result.equity_curve.append(equity_point)
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde snapshot: {str(e)}")
    
    def _finalize_simulation(self, start_time: datetime):
        """Finalise les résultats de simulation"""
        try:
            end_time = datetime.now(timezone.utc)
            
            if not self.result or not self.paper_trader:
                return
            
            # Récupération résultats finaux
            performance_report = self.paper_trader.get_performance_report()
            portfolio_summary = performance_report["portfolio"]
            trading_metrics = performance_report["trading_metrics"]
            
            # Mise à jour résultat
            self.result.status = SimulationStatus.COMPLETED
            self.result.end_time = end_time
            self.result.duration_seconds = (end_time - start_time).total_seconds()
            
            self.result.final_balance = portfolio_summary["current_values"]["total_value"]
            self.result.total_return = portfolio_summary["pnl_summary"]["total_pnl"]
            self.result.total_return_percent = portfolio_summary["pnl_summary"]["total_return_percent"]
            
            # Métriques trading
            self.result.total_trades = trading_metrics["total_trades"]
            self.result.winning_trades = trading_metrics["winning_trades"]
            self.result.losing_trades = trading_metrics["losing_trades"]
            self.result.win_rate = trading_metrics["win_rate"]
            self.result.avg_win = trading_metrics["avg_win"]
            self.result.avg_loss = trading_metrics["avg_loss"]
            self.result.profit_factor = trading_metrics["profit_factor"]
            
            # Calcul drawdown
            drawdown_info = self.paper_trader.portfolio.calculate_drawdown()
            self.result.max_drawdown = drawdown_info["max_drawdown"]
            
            # Historique trades
            self.result.trades_history = self.paper_trader.get_trade_history()
            
            # Performance par stratégie
            self._calculate_strategy_performance()
            
            self.status = SimulationStatus.COMPLETED
            
            logger.info(f"Simulation {self.simulation_id} terminée")
            logger.info(f"Rendement total: {self.result.total_return_percent:.2f}%")
            logger.info(f"Trades: {self.result.total_trades} (Win rate: {self.result.win_rate:.1%})")
            
        except Exception as e:
            logger.error(f"Erreur finalisation simulation: {str(e)}")
            self.status = SimulationStatus.ERROR
    
    def _calculate_strategy_performance(self):
        """Calcule la performance individuelle de chaque stratégie"""
        try:
            if not self.result or not self.paper_trader:
                return
            
            strategy_stats = {}
            
            for strategy_id, strategy in self.paper_trader.active_strategies.items():
                # Filtrer les trades de cette stratégie
                strategy_trades = [
                    trade for trade in self.result.trades_history
                    if trade.get("strategy_id") == strategy_id
                ]
                
                if strategy_trades:
                    total_pnl = sum(trade.get("pnl", 0) for trade in strategy_trades if trade.get("pnl"))
                    wins = len([t for t in strategy_trades if t.get("pnl", 0) > 0])
                    losses = len([t for t in strategy_trades if t.get("pnl", 0) < 0])
                    
                    strategy_stats[strategy.name] = {
                        "total_trades": len(strategy_trades),
                        "total_pnl": total_pnl,
                        "winning_trades": wins,
                        "losing_trades": losses,
                        "win_rate": wins / len(strategy_trades) if strategy_trades else 0,
                        "avg_pnl_per_trade": total_pnl / len(strategy_trades) if strategy_trades else 0
                    }
            
            self.result.strategy_performance = strategy_stats
            
        except Exception as e:
            logger.error(f"Erreur calcul performance stratégies: {str(e)}")
    
    def get_simulation_status(self) -> Dict[str, Any]:
        """Retourne le statut actuel de la simulation"""
        try:
            status_info = {
                "simulation_id": self.simulation_id,
                "status": self.status.value,
                "current_timestamp": self.current_timestamp.isoformat() if self.current_timestamp else None
            }
            
            if self.result:
                status_info.update({
                    "progress": self._calculate_progress(),
                    "current_balance": self.paper_trader.portfolio.total_value if self.paper_trader else 0,
                    "current_pnl": self.paper_trader.portfolio.total_pnl if self.paper_trader else 0,
                    "trades_count": len(self.paper_trader.trades) if self.paper_trader else 0
                })
            
            return status_info
            
        except Exception as e:
            logger.error(f"Erreur récupération statut: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_progress(self) -> float:
        """Calcule le progrès de la simulation"""
        try:
            if not self.config or not self.current_timestamp:
                return 0.0
            
            total_duration = self.config.end_date - self.config.start_date
            elapsed_duration = self.current_timestamp - self.config.start_date
            
            progress = elapsed_duration.total_seconds() / total_duration.total_seconds()
            return min(max(progress * 100, 0.0), 100.0)
            
        except Exception as e:
            return 0.0
    
    def get_results(self) -> Optional[SimulationResult]:
        """Retourne les résultats de simulation"""
        return self.result
    
    def cleanup(self):
        """Nettoie les ressources de simulation"""
        try:
            self.stop_simulation()
            
            if self._simulation_thread and self._simulation_thread.is_alive():
                self._simulation_thread.join(timeout=5.0)
            
            self.market_data.clear()
            self.paper_trader = None
            self.config = None
            self.result = None
            
            logger.info("Ressources simulation nettoyées")
            
        except Exception as e:
            logger.error(f"Erreur nettoyage simulation: {str(e)}")
    
    def set_progress_callback(self, callback: Callable[[float, int, int], None]):
        """Définit un callback pour les mises à jour de progression"""
        self.progress_callback = callback
    
    def set_update_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Définit un callback pour les mises à jour en temps réel"""
        self.update_callback = callback