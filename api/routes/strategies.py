"""
🧠 Strategies Routes
Endpoints pour la gestion des stratégies de trading
"""

import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, validator

from core.logger import get_logger
from core.data_collector import get_data_collector
from trading.engine.trading_engine import TradingEngine
from trading.strategies.trend_following import TrendFollowingStrategy
from trading.strategies.mean_reversion import MeanReversionStrategy
from trading.backtesting.backtest_engine import BacktestEngine, BacktestConfig
from trading.backtesting.report_generator import ReportGenerator
from trading.backtesting.performance_analyzer import PerformanceAnalyzer

logger = get_logger(__name__)
router = APIRouter()

# Instances globales (à remplacer par un système de DI en production)
_trading_engine = None
_backtest_engine = None
_report_generator = None
_performance_analyzer = None

def get_trading_engine() -> TradingEngine:
    global _trading_engine
    if _trading_engine is None:
        _trading_engine = TradingEngine()
    return _trading_engine

def get_backtest_engine() -> BacktestEngine:
    global _backtest_engine
    if _backtest_engine is None:
        _backtest_engine = BacktestEngine()
    return _backtest_engine

def get_report_generator() -> ReportGenerator:
    global _report_generator
    if _report_generator is None:
        _report_generator = ReportGenerator()
    return _report_generator

def get_performance_analyzer() -> PerformanceAnalyzer:
    global _performance_analyzer
    if _performance_analyzer is None:
        _performance_analyzer = PerformanceAnalyzer()
    return _performance_analyzer

# Models Pydantic

class StrategyCreateRequest(BaseModel):
    """Requête de création de stratégie"""
    name: str
    strategy_type: str  # "trend_following", "mean_reversion"
    symbol: str
    timeframe: str = "1h"
    parameters: Optional[Dict[str, Any]] = None
    
    @validator('strategy_type')
    def validate_strategy_type(cls, v):
        valid_types = ['trend_following', 'mean_reversion']
        if v not in valid_types:
            raise ValueError(f'Strategy type must be one of: {valid_types}')
        return v

class StrategyUpdateRequest(BaseModel):
    """Requête de mise à jour de stratégie"""
    parameters: Optional[Dict[str, Any]] = None
    status: Optional[str] = None  # "active", "inactive", "paused"

class BacktestRequest(BaseModel):
    """Requête de backtest"""
    strategy_id: str
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_balance: float = 10000.0
    timeframe: str = "1h"
    commission_rate: float = 0.001
    slippage_rate: float = 0.0001

class StrategyResponse(BaseModel):
    """Réponse stratégie"""
    strategy_id: str
    name: str
    strategy_type: str
    symbol: str
    timeframe: str
    status: str
    parameters: Dict[str, Any]
    metrics: Dict[str, Any]
    created_at: str

# Routes

@router.get("/")
async def get_strategies():
    """
    Récupère la liste de toutes les stratégies
    """
    try:
        trading_engine = get_trading_engine()
        engine_status = trading_engine.get_engine_status()
        
        # Conversion des stratégies pour l'API
        strategies = []
        for strategy_info in engine_status.get("active_strategies", []):
            strategies.append({
                "strategy_id": strategy_info["id"],
                "name": strategy_info["name"],
                "strategy_type": strategy_info["name"].lower().replace(" ", "_"),
                "created_at": strategy_info["created_at"],
                "metrics": strategy_info["metrics"]
            })
        
        return {
            "strategies": strategies,
            "count": len(strategies),
            "engine_status": engine_status["status"]
        }
        
    except Exception as e:
        logger.error(f"Erreur récupération stratégies: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve strategies")

@router.post("/", response_model=StrategyResponse)
async def create_strategy(strategy: StrategyCreateRequest):
    """
    Crée une nouvelle stratégie de trading
    """
    try:
        trading_engine = get_trading_engine()
        
        # Création de la stratégie selon le type
        if strategy.strategy_type == "trend_following":
            strategy_instance = TrendFollowingStrategy(
                symbol=strategy.symbol.upper(),
                timeframe=strategy.timeframe,
                parameters=strategy.parameters
            )
        elif strategy.strategy_type == "mean_reversion":
            strategy_instance = MeanReversionStrategy(
                symbol=strategy.symbol.upper(), 
                timeframe=strategy.timeframe,
                parameters=strategy.parameters
            )
        else:
            raise HTTPException(status_code=400, detail=f"Strategy type '{strategy.strategy_type}' not supported")
        
        # Ajout au trading engine
        strategy_id = trading_engine.add_strategy(strategy_instance, strategy.parameters)
        
        # Récupération des informations pour la réponse
        strategy_info = strategy_instance.get_strategy_info()
        
        response = StrategyResponse(
            strategy_id=strategy_id,
            name=strategy_info["name"],
            strategy_type=strategy.strategy_type,
            symbol=strategy_info["symbol"],
            timeframe=strategy_info["timeframe"],
            status=strategy_info["status"],
            parameters=strategy_info["parameters"],
            metrics=strategy_info["metrics"],
            created_at=strategy_info["created_at"]
        )
        
        logger.info(f"Stratégie créée: {strategy_id}")
        return response
        
    except Exception as e:
        logger.error(f"Erreur création stratégie: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create strategy: {str(e)}")

@router.get("/{strategy_id}")
async def get_strategy(strategy_id: str):
    """
    Récupère une stratégie spécifique
    """
    try:
        trading_engine = get_trading_engine()
        
        # Vérifier si la stratégie existe (simulation pour l'instant)
        engine_status = trading_engine.get_engine_status()
        
        # Simuler une stratégie trouvée
        mock_strategy = {
            "strategy_id": strategy_id,
            "name": "Trend Following BTC",
            "strategy_type": "trend_following",
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "status": "active",
            "parameters": {
                "fast_ma_period": 10,
                "slow_ma_period": 20,
                "rsi_oversold": 30,
                "rsi_overbought": 70
            },
            "metrics": {
                "total_signals": 45,
                "win_rate": 67.5,
                "total_pnl": 1250.75,
                "active_positions": 1
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_signal_at": datetime.now(timezone.utc).isoformat()
        }
        
        return mock_strategy
        
    except Exception as e:
        logger.error(f"Erreur récupération stratégie {strategy_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve strategy")

@router.put("/{strategy_id}")
async def update_strategy(strategy_id: str, update: StrategyUpdateRequest):
    """
    Met à jour une stratégie existante
    """
    try:
        trading_engine = get_trading_engine()
        
        # Pour l'instant, simuler la mise à jour
        logger.info(f"Mise à jour stratégie {strategy_id}: {update.dict()}")
        
        return {
            "strategy_id": strategy_id,
            "message": "Strategy updated successfully",
            "updated_fields": update.dict(exclude_none=True),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erreur mise à jour stratégie {strategy_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update strategy")

@router.delete("/{strategy_id}")
async def delete_strategy(strategy_id: str):
    """
    Supprime une stratégie
    """
    try:
        trading_engine = get_trading_engine()
        
        success = trading_engine.remove_strategy(strategy_id)
        
        if success:
            return {
                "strategy_id": strategy_id,
                "message": "Strategy deleted successfully",
                "deleted_at": datetime.now(timezone.utc).isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur suppression stratégie {strategy_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete strategy")

@router.post("/{strategy_id}/activate")
async def activate_strategy(strategy_id: str):
    """
    Active une stratégie
    """
    try:
        # Pour l'instant, simuler l'activation
        logger.info(f"Activation stratégie {strategy_id}")
        
        return {
            "strategy_id": strategy_id,
            "status": "active",
            "message": "Strategy activated successfully",
            "activated_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erreur activation stratégie {strategy_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to activate strategy")

@router.post("/{strategy_id}/deactivate")
async def deactivate_strategy(strategy_id: str):
    """
    Désactive une stratégie
    """
    try:
        # Pour l'instant, simuler la désactivation
        logger.info(f"Désactivation stratégie {strategy_id}")
        
        return {
            "strategy_id": strategy_id,
            "status": "inactive",
            "message": "Strategy deactivated successfully",
            "deactivated_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erreur désactivation stratégie {strategy_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to deactivate strategy")

@router.get("/{strategy_id}/signals")
async def get_strategy_signals(
    strategy_id: str,
    limit: int = Query(default=50, le=500)
):
    """
    Récupère les signaux récents d'une stratégie
    """
    try:
        # Simulation des signaux
        signals = []
        for i in range(min(limit, 20)):
            signals.append({
                "signal_id": str(uuid.uuid4()),
                "strategy_id": strategy_id,
                "symbol": "BTCUSDT",
                "signal_type": "BUY" if i % 2 == 0 else "SELL",
                "confidence": 0.75 + (i * 0.01),
                "suggested_price": 45000 + (i * 100),
                "stop_loss": 44000 + (i * 100),
                "take_profit": 47000 + (i * 100),
                "timestamp": (datetime.now(timezone.utc) - timedelta(hours=i)).isoformat(),
                "metadata": {
                    "trend_strength": 0.8,
                    "rsi": 65,
                    "macd": 0.15
                }
            })
        
        return {
            "strategy_id": strategy_id,
            "signals": signals,
            "count": len(signals),
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"Erreur récupération signaux {strategy_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve signals")

@router.post("/{strategy_id}/backtest")
async def run_backtest(backtest_req: BacktestRequest):
    """
    Lance un backtest pour une stratégie
    """
    try:
        backtest_engine = get_backtest_engine()
        data_collector = get_data_collector()
        
        # Récupération des données historiques
        historical_data = await data_collector.get_historical_data(
            symbol=backtest_req.symbol,
            interval=backtest_req.timeframe,
            start_time=backtest_req.start_date,
            end_time=backtest_req.end_date,
            source="yahoo"
        )
        
        if historical_data.empty:
            raise HTTPException(
                status_code=400, 
                detail=f"No historical data available for {backtest_req.symbol}"
            )
        
        # Création de la stratégie pour le backtest
        if "trend" in backtest_req.strategy_id.lower():
            strategy = TrendFollowingStrategy(
                symbol=backtest_req.symbol,
                timeframe=backtest_req.timeframe
            )
        else:
            strategy = MeanReversionStrategy(
                symbol=backtest_req.symbol,
                timeframe=backtest_req.timeframe
            )
        
        # Configuration du backtest
        config = BacktestConfig(
            strategy_id=backtest_req.strategy_id,
            symbol=backtest_req.symbol,
            start_date=backtest_req.start_date,
            end_date=backtest_req.end_date,
            initial_balance=backtest_req.initial_balance,
            timeframe=backtest_req.timeframe,
            commission_rate=backtest_req.commission_rate,
            slippage_rate=backtest_req.slippage_rate
        )
        
        # Lancement du backtest
        backtest_id = await backtest_engine.run_backtest(strategy, historical_data, config)
        
        return {
            "backtest_id": backtest_id,
            "strategy_id": backtest_req.strategy_id,
            "status": "running",
            "message": "Backtest started successfully",
            "started_at": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lancement backtest: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to run backtest: {str(e)}")

@router.get("/backtest/{backtest_id}")
async def get_backtest_result(backtest_id: str, include_trades: bool = Query(default=False)):
    """
    Récupère le résultat d'un backtest
    """
    try:
        backtest_engine = get_backtest_engine()
        
        result = backtest_engine.get_backtest_result(backtest_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="Backtest not found")
        
        # Conversion en dictionnaire pour l'API
        result_dict = result.to_dict()
        
        # Limiter les trades si non demandés explicitement
        if not include_trades and "trades" in result_dict:
            result_dict["trades"] = result_dict["trades"][:10]  # Premier 10 seulement
            result_dict["trades_truncated"] = True
        
        return result_dict
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur récupération backtest {backtest_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve backtest result")

@router.get("/backtest/{backtest_id}/report")
async def get_backtest_report(backtest_id: str):
    """
    Génère et récupère un rapport de backtest complet
    """
    try:
        backtest_engine = get_backtest_engine()
        report_generator = get_report_generator()
        
        result = backtest_engine.get_backtest_result(backtest_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="Backtest not found")
        
        # Génération du rapport
        report = report_generator.generate_full_report(result, save_to_file=False)
        
        return report
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur génération rapport {backtest_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate backtest report")

@router.get("/backtest/{backtest_id}/summary")
async def get_backtest_summary(backtest_id: str):
    """
    Récupère un résumé rapide du backtest
    """
    try:
        backtest_engine = get_backtest_engine()
        report_generator = get_report_generator()
        
        result = backtest_engine.get_backtest_result(backtest_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="Backtest not found")
        
        # Génération du résumé
        summary = report_generator.generate_summary_report(result)
        
        return summary
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur génération résumé {backtest_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate backtest summary")

@router.get("/backtest/")
async def get_backtest_history(limit: int = Query(default=20, le=100)):
    """
    Récupère l'historique des backtests
    """
    try:
        backtest_engine = get_backtest_engine()
        
        history = backtest_engine.get_backtest_history(limit)
        
        return {
            "backtests": history,
            "count": len(history),
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"Erreur récupération historique backtests: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve backtest history")

@router.delete("/backtest/{backtest_id}")
async def cancel_backtest(backtest_id: str):
    """
    Annule un backtest en cours
    """
    try:
        backtest_engine = get_backtest_engine()
        
        success = backtest_engine.cancel_backtest(backtest_id)
        
        if success:
            return {
                "backtest_id": backtest_id,
                "message": "Backtest cancelled successfully",
                "cancelled_at": datetime.now(timezone.utc).isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail="Backtest not found or already completed")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur annulation backtest {backtest_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to cancel backtest")