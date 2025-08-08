"""
🧪 Backtesting Routes
Endpoints pour les opérations de backtesting
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, validator

from core.config import get_settings
from core.logger import get_logger
from core.data_collector import DataCollector
from trading.backtesting import BacktestEngine, PerformanceAnalyzer, MetricsCalculator
from trading.backtesting.backtest_engine import BacktestConfig, BacktestStatus
from trading.backtesting.report_generator import ReportGenerator
from trading.strategies import TrendFollowingStrategy, MeanReversionStrategy

logger = get_logger(__name__)
router = APIRouter()

# Instances globales
backtest_engine = BacktestEngine()
performance_analyzer = PerformanceAnalyzer()
metrics_calculator = MetricsCalculator()
report_generator = ReportGenerator()
data_collector = DataCollector()

# Models Pydantic pour les requêtes/réponses

class BacktestRequest(BaseModel):
    """Requête de backtesting"""
    strategy_type: str  # "trend_following", "mean_reversion"
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_balance: float = 10000.0
    timeframe: str = "1h"
    commission_rate: float = 0.001
    slippage_rate: float = 0.0001
    
    # Paramètres spécifiques aux stratégies
    strategy_parameters: Optional[Dict[str, Any]] = None
    
    @validator('strategy_type')
    def validate_strategy_type(cls, v):
        valid_types = ["trend_following", "mean_reversion"]
        if v not in valid_types:
            raise ValueError(f'Strategy type must be one of: {valid_types}')
        return v
    
    @validator('start_date', 'end_date')
    def validate_dates(cls, v):
        # Convertir v en timezone-aware si nécessaire
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        
        if v > datetime.now(timezone.utc):
            raise ValueError('Dates cannot be in the future')
        return v
    
    @validator('initial_balance')
    def validate_balance(cls, v):
        if v <= 0:
            raise ValueError('Initial balance must be positive')
        return v

class BacktestStatusResponse(BaseModel):
    """Réponse de statut de backtesting"""
    backtest_id: str
    status: str
    progress: Optional[float] = None
    start_time: datetime
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None

class BacktestSummary(BaseModel):
    """Résumé de backtesting"""
    backtest_id: str
    strategy_id: str
    symbol: str
    status: str
    total_return_percent: float
    max_drawdown_percent: float
    total_trades: int
    win_rate: float
    sharpe_ratio: float
    start_time: datetime
    end_time: Optional[datetime] = None

# Routes

@router.get("/status")
async def get_backtesting_status():
    """
    Status du système de backtesting
    """
    try:
        engine_status = backtest_engine.get_engine_status()
        
        return {
            "status": "operational",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "engine_status": engine_status,
            "available_strategies": [
                "trend_following",
                "mean_reversion"
            ],
            "supported_timeframes": [
                "1m", "5m", "15m", "30m", "1h", "4h", "1d"
            ],
            "supported_symbols": [
                "BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT"
            ]
        }
    
    except Exception as e:
        logger.error(f"Failed to get backtesting status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve backtesting status")

@router.post("/run", response_model=BacktestStatusResponse)
async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """
    Lance un nouveau backtesting
    """
    try:
        logger.info(f"Starting backtest for {request.strategy_type} on {request.symbol}")
        
        # Validation de la période
        if request.end_date <= request.start_date:
            raise HTTPException(status_code=400, detail="End date must be after start date")
        
        # Vérification de la période (max 1 an pour éviter les timeouts)
        max_days = 365
        if (request.end_date - request.start_date).days > max_days:
            raise HTTPException(status_code=400, detail=f"Maximum backtest period is {max_days} days")
        
        # Configuration du backtest
        config = BacktestConfig(
            strategy_id=f"{request.strategy_type}_{request.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_balance=request.initial_balance,
            timeframe=request.timeframe,
            commission_rate=request.commission_rate,
            slippage_rate=request.slippage_rate
        )
        
        # Lancement du backtesting en arrière-plan
        background_tasks.add_task(
            _execute_backtest_task,
            request.strategy_type,
            config,
            request.strategy_parameters or {}
        )
        
        # Création d'un résultat temporaire pour retourner l'ID
        from trading.backtesting.backtest_engine import BacktestResult
        temp_result = BacktestResult(
            backtest_id=config.strategy_id,
            config=config,
            status=BacktestStatus.PENDING,
            start_time=datetime.now(timezone.utc)
        )
        
        # Ajouter aux backtests actifs
        backtest_engine.active_backtests[config.strategy_id] = temp_result
        
        return BacktestStatusResponse(
            backtest_id=config.strategy_id,
            status="pending",
            start_time=datetime.now(timezone.utc)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start backtest: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start backtest: {str(e)}")

@router.get("/results/{backtest_id}")
async def get_backtest_result(backtest_id: str, detailed: bool = Query(False)):
    """
    Récupère les résultats d'un backtesting
    """
    try:
        result = backtest_engine.get_backtest_result(backtest_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="Backtest not found")
        
        if result.status == BacktestStatus.RUNNING:
            return {
                "backtest_id": backtest_id,
                "status": "running",
                "message": "Backtest still in progress"
            }
        
        if result.status == BacktestStatus.ERROR:
            return {
                "backtest_id": backtest_id,
                "status": "error",
                "error_message": result.error_message
            }
        
        # Résultats complets ou résumé
        if detailed:
            return result.to_dict()
        else:
            return report_generator.generate_summary_report(result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get backtest result {backtest_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve backtest result")

@router.get("/results/{backtest_id}/report")
async def get_backtest_report(backtest_id: str, report_type: str = Query("summary")):
    """
    Génère et récupère un rapport de backtesting
    """
    try:
        result = backtest_engine.get_backtest_result(backtest_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="Backtest not found")
        
        if result.status != BacktestStatus.COMPLETED:
            raise HTTPException(status_code=400, detail="Backtest not completed")
        
        if report_type == "full":
            report = report_generator.generate_full_report(result, save_to_file=False)
        elif report_type == "summary":
            report = report_generator.generate_summary_report(result)
        else:
            raise HTTPException(status_code=400, detail="Invalid report type. Use 'summary' or 'full'")
        
        return report
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate report for {backtest_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate report")

@router.get("/results/{backtest_id}/analysis")
async def get_backtest_analysis(backtest_id: str):
    """
    Récupère l'analyse de performance détaillée
    """
    try:
        result = backtest_engine.get_backtest_result(backtest_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="Backtest not found")
        
        if result.status != BacktestStatus.COMPLETED:
            raise HTTPException(status_code=400, detail="Backtest not completed")
        
        # Analyse de performance
        performance_analysis = performance_analyzer.analyze_backtest_performance(result)
        
        # Métriques détaillées
        detailed_metrics = metrics_calculator.calculate_all_metrics(result)
        
        return {
            "backtest_id": backtest_id,
            "performance_analysis": performance_analysis,
            "detailed_metrics": detailed_metrics,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get analysis for {backtest_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve analysis")

@router.get("/history")
async def get_backtest_history(
    limit: int = Query(default=50, le=200),
    strategy_type: Optional[str] = Query(None),
    symbol: Optional[str] = Query(None),
    status: Optional[str] = Query(None)
):
    """
    Récupère l'historique des backtestings
    """
    try:
        history = backtest_engine.get_backtest_history(limit)
        
        # Filtrage optionnel
        filtered_history = []
        for bt in history:
            include = True
            
            if strategy_type and strategy_type not in bt.get("config", {}).get("strategy_id", ""):
                include = False
            
            if symbol and bt.get("config", {}).get("symbol") != symbol:
                include = False
            
            if status and bt.get("status") != status:
                include = False
            
            if include:
                # Créer un résumé pour l'historique
                summary = {
                    "backtest_id": bt.get("backtest_id"),
                    "strategy_id": bt.get("config", {}).get("strategy_id"),
                    "symbol": bt.get("config", {}).get("symbol"),
                    "status": bt.get("status"),
                    "total_return_percent": bt.get("total_return_percent", 0),
                    "max_drawdown_percent": bt.get("max_drawdown_percent", 0),
                    "total_trades": bt.get("total_trades", 0),
                    "win_rate": bt.get("win_rate", 0),
                    "sharpe_ratio": bt.get("sharpe_ratio", 0),
                    "start_time": bt.get("start_time"),
                    "end_time": bt.get("end_time")
                }
                filtered_history.append(summary)
        
        return {
            "backtests": filtered_history,
            "total_count": len(filtered_history),
            "filters_applied": {
                "strategy_type": strategy_type,
                "symbol": symbol,
                "status": status,
                "limit": limit
            }
        }
    
    except Exception as e:
        logger.error(f"Failed to get backtest history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve backtest history")

@router.delete("/results/{backtest_id}")
async def cancel_backtest(backtest_id: str):
    """
    Annule un backtesting en cours
    """
    try:
        success = backtest_engine.cancel_backtest(backtest_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Backtest not found or not cancellable")
        
        return {
            "backtest_id": backtest_id,
            "status": "cancelled",
            "cancelled_at": datetime.now(timezone.utc).isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel backtest {backtest_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to cancel backtest")

@router.post("/compare")
async def compare_backtests(backtest_ids: List[str]):
    """
    Compare plusieurs backtestings
    """
    try:
        if len(backtest_ids) < 2:
            raise HTTPException(status_code=400, detail="At least 2 backtests required for comparison")
        
        if len(backtest_ids) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 backtests can be compared")
        
        results = []
        for backtest_id in backtest_ids:
            result = backtest_engine.get_backtest_result(backtest_id)
            if not result:
                raise HTTPException(status_code=404, detail=f"Backtest {backtest_id} not found")
            if result.status != BacktestStatus.COMPLETED:
                raise HTTPException(status_code=400, detail=f"Backtest {backtest_id} not completed")
            results.append(result)
        
        # Génération du rapport de comparaison
        comparison_report = report_generator.generate_comparison_report(results)
        
        return comparison_report
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to compare backtests: {e}")
        raise HTTPException(status_code=500, detail="Failed to compare backtests")

# Fonction d'arrière-plan pour l'exécution du backtesting
async def _execute_backtest_task(strategy_type: str, config: BacktestConfig, strategy_params: Dict):
    """
    Exécute le backtesting en arrière-plan
    """
    try:
        logger.info(f"Starting backtest execution for {config.strategy_id}")
        
        # Récupération des données historiques
        logger.info(f"Fetching historical data for {config.symbol}")
        data = await _fetch_historical_data(config.symbol, config.start_date, config.end_date, config.timeframe)
        
        if data is None or data.empty:
            raise Exception("No historical data available for the specified period")
        
        # Création de la stratégie
        strategy = _create_strategy(strategy_type, config.symbol, strategy_params)
        
        # Exécution du backtest
        logger.info(f"Executing backtest for {config.strategy_id}")
        backtest_id = await backtest_engine.run_backtest(strategy, data, config)
        
        logger.info(f"Backtest completed successfully: {backtest_id}")
        
    except Exception as e:
        logger.error(f"Backtest execution failed for {config.strategy_id}: {str(e)}")
        
        # Marquer le backtest comme échoué
        if config.strategy_id in backtest_engine.active_backtests:
            result = backtest_engine.active_backtests[config.strategy_id]
            result.status = BacktestStatus.ERROR
            result.error_message = str(e)
            result.end_time = datetime.now(timezone.utc)
            
            # Déplacer vers les complétés
            backtest_engine.completed_backtests[config.strategy_id] = result
            del backtest_engine.active_backtests[config.strategy_id]

async def _fetch_historical_data(symbol: str, start_date: datetime, end_date: datetime, timeframe: str):
    """
    Récupère les données historiques pour le backtesting
    """
    try:
        # Utilisation du data collector existant
        data = await data_collector.get_historical_data(
            symbol=symbol,
            start_time=start_date,
            end_time=end_date,
            interval=timeframe
        )
        
        if data is not None and not data.empty:
            # S'assurer que les colonnes nécessaires sont présentes
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if all(col in data.columns for col in required_columns):
                logger.info(f"Historical data fetched: {len(data)} points for {symbol}")
                return data
            else:
                logger.error(f"Missing required columns in historical data for {symbol}")
                return None
        else:
            logger.error(f"No historical data returned for {symbol}")
            return None
    
    except Exception as e:
        logger.error(f"Failed to fetch historical data for {symbol}: {str(e)}")
        return None

def _create_strategy(strategy_type: str, symbol: str, params: Dict):
    """
    Crée une instance de stratégie selon le type
    """
    try:
        if strategy_type == "trend_following":
            strategy = TrendFollowingStrategy(symbol=symbol)
            
            # Application des paramètres personnalisés
            if params:
                if 'fast_period' in params:
                    strategy.parameters['fast_period'] = params['fast_period']
                if 'slow_period' in params:
                    strategy.parameters['slow_period'] = params['slow_period']
                if 'signal_period' in params:
                    strategy.parameters['signal_period'] = params['signal_period']
        
        elif strategy_type == "mean_reversion":
            strategy = MeanReversionStrategy(symbol=symbol)
            
            # Application des paramètres personnalisés
            if params:
                if 'rsi_period' in params:
                    strategy.parameters['rsi_period'] = params['rsi_period']
                if 'bb_period' in params:
                    strategy.parameters['bb_period'] = params['bb_period']
                if 'bb_std' in params:
                    strategy.parameters['bb_std'] = params['bb_std']
        
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        logger.info(f"Strategy created: {strategy_type} for {symbol}")
        return strategy
        
    except Exception as e:
        logger.error(f"Failed to create strategy {strategy_type}: {str(e)}")
        raise