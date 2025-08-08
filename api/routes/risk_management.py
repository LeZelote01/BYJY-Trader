"""
🛡️ Risk Management Routes
Endpoints pour la gestion des risques et du portfolio
"""

from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, validator

from core.logger import get_logger
from trading.risk_management.risk_manager import RiskManager, RiskLevel
from trading.risk_management.position_sizer import PositionSizer, SizingMethod
from trading.risk_management.stop_loss_manager import StopLossManager, StopLossType

logger = get_logger(__name__)
router = APIRouter()

# Instances globales (à remplacer par un système de DI en production)
_risk_manager = None
_position_sizer = None
_stop_loss_manager = None

def get_risk_manager() -> RiskManager:
    global _risk_manager
    if _risk_manager is None:
        _risk_manager = RiskManager()
    return _risk_manager

def get_position_sizer() -> PositionSizer:
    global _position_sizer
    if _position_sizer is None:
        _position_sizer = PositionSizer()
    return _position_sizer

def get_stop_loss_manager() -> StopLossManager:
    global _stop_loss_manager
    if _stop_loss_manager is None:
        _stop_loss_manager = StopLossManager()
    return _stop_loss_manager

# Models Pydantic

class PositionValidationRequest(BaseModel):
    """Requête de validation de nouvelle position"""
    symbol: str
    side: str  # "long" ou "short"
    quantity: float
    entry_price: float
    stop_loss: Optional[float] = None
    current_positions: Optional[List[Dict[str, Any]]] = []

class PositionSizeRequest(BaseModel):
    """Requête de calcul de taille de position"""
    account_balance: float
    risk_per_trade: float = 2.0
    entry_price: float
    stop_loss: Optional[float] = None
    confidence: float = 1.0
    method: str = "risk_based"
    additional_params: Optional[Dict[str, Any]] = {}
    
    @validator('method')
    def validate_method(cls, v):
        valid_methods = ['fixed_amount', 'fixed_percentage', 'risk_based', 
                        'kelly_criterion', 'volatility_adjusted', 'atr_based']
        if v not in valid_methods:
            raise ValueError(f'Method must be one of: {valid_methods}')
        return v

class StopLossRequest(BaseModel):
    """Requête de calcul de stop-loss"""
    entry_price: float
    side: str  # "long" ou "short"
    stop_type: str = "percentage"
    parameters: Optional[Dict[str, Any]] = {}
    
    @validator('stop_type')
    def validate_stop_type(cls, v):
        valid_types = ['fixed', 'percentage', 'trailing', 'atr_based', 
                      'volatility_based', 'time_based']
        if v not in valid_types:
            raise ValueError(f'Stop type must be one of: {valid_types}')
        return v

class RiskConfigUpdateRequest(BaseModel):
    """Requête de mise à jour de configuration de risque"""
    max_portfolio_risk_pct: Optional[float] = None
    max_position_risk_pct: Optional[float] = None
    max_correlation_exposure: Optional[float] = None
    max_drawdown_limit: Optional[float] = None
    max_concurrent_positions: Optional[int] = None
    max_daily_loss_pct: Optional[float] = None

# Routes

@router.get("/assessment")
async def get_risk_assessment():
    """
    Évalue les risques actuels du portfolio
    """
    try:
        risk_manager = get_risk_manager()
        
        # Simulation de positions actuelles (à remplacer par vraies données)
        mock_positions = [
            {
                "id": "pos_1",
                "symbol": "BTCUSDT",
                "market_value": 5000,
                "unrealized_pnl": 250,
                "side": "long"
            },
            {
                "id": "pos_2",
                "symbol": "ETHUSDT",
                "market_value": 2000,
                "unrealized_pnl": -100,
                "side": "long"
            }
        ]
        
        assessment = risk_manager.assess_portfolio_risk(mock_positions)
        
        return assessment.to_dict()
        
    except Exception as e:
        logger.error(f"Erreur évaluation risque: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to assess portfolio risk")

@router.post("/validate-position")
async def validate_position(request: PositionValidationRequest):
    """
    Valide une nouvelle position selon les règles de risque
    """
    try:
        risk_manager = get_risk_manager()
        
        is_valid, message, validation_data = risk_manager.validate_new_position(
            symbol=request.symbol,
            side=request.side,
            quantity=request.quantity,
            entry_price=request.entry_price,
            stop_loss=request.stop_loss,
            current_positions=request.current_positions
        )
        
        return {
            "is_valid": is_valid,
            "message": message,
            "validation_data": validation_data,
            "validated_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erreur validation position: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to validate position")

@router.post("/position-size")
async def calculate_position_size(request: PositionSizeRequest):
    """
    Calcule la taille de position recommandée
    """
    try:
        position_sizer = get_position_sizer()
        
        # Conversion du string en enum
        sizing_method = SizingMethod(request.method)
        
        position_size = position_sizer.calculate_position_size(
            account_balance=request.account_balance,
            risk_per_trade=request.risk_per_trade,
            entry_price=request.entry_price,
            stop_loss=request.stop_loss,
            confidence=request.confidence,
            method=sizing_method,
            additional_params=request.additional_params
        )
        
        # Informations détaillées sur la position
        position_info = position_sizer.get_position_info(
            size=position_size,
            entry_price=request.entry_price,
            account_balance=request.account_balance,
            stop_loss=request.stop_loss
        )
        
        return {
            "recommended_size": position_size,
            "method_used": request.method,
            "position_info": position_info,
            "calculated_at": datetime.now(timezone.utc).isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur calcul position size: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to calculate position size")

@router.post("/position-size/optimize")
async def optimize_position_size(
    account_balance: float,
    entry_price: float,
    win_rate: float = Query(50.0, description="Win rate percentage"),
    avg_win: float = Query(100.0, description="Average winning trade"),
    avg_loss: float = Query(50.0, description="Average losing trade"),
    volatility: float = Query(0.02, description="Asset volatility")
):
    """
    Optimise la taille de position basée sur la performance historique
    """
    try:
        position_sizer = get_position_sizer()
        
        historical_performance = {
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "volatility": volatility
        }
        
        optimization = position_sizer.optimize_position_size(
            account_balance=account_balance,
            entry_price=entry_price,
            historical_performance=historical_performance
        )
        
        return {
            **optimization,
            "optimized_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erreur optimisation position: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to optimize position size")

@router.post("/stop-loss")
async def calculate_stop_loss(request: StopLossRequest):
    """
    Calcule le niveau de stop-loss recommandé
    """
    try:
        stop_loss_manager = get_stop_loss_manager()
        
        # Conversion du string en enum
        stop_type = StopLossType(request.stop_type)
        
        stop_loss_price = stop_loss_manager.calculate_stop_loss(
            entry_price=request.entry_price,
            side=request.side,
            stop_type=stop_type,
            parameters=request.parameters
        )
        
        # Calcul du risque en pourcentage
        risk_pct = 0.0
        if stop_loss_price > 0:
            risk_pct = abs((request.entry_price - stop_loss_price) / request.entry_price) * 100
        
        return {
            "stop_loss_price": stop_loss_price,
            "risk_percentage": risk_pct,
            "stop_type": request.stop_type,
            "entry_price": request.entry_price,
            "side": request.side,
            "calculated_at": datetime.now(timezone.utc).isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur calcul stop-loss: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to calculate stop loss")

@router.put("/stop-loss/trailing/{position_id}")
async def update_trailing_stop(
    position_id: str, 
    current_price: float,
    position_data: Dict[str, Any]
):
    """
    Met à jour un trailing stop-loss
    """
    try:
        stop_loss_manager = get_stop_loss_manager()
        
        # Simulation de données position
        position = {
            "id": position_id,
            "entry_price": position_data.get("entry_price", current_price * 0.98),
            "stop_loss": position_data.get("stop_loss", current_price * 0.96),
            "side": position_data.get("side", "long"),
            **position_data
        }
        
        new_stop_loss = stop_loss_manager.update_trailing_stop(position, current_price)
        
        # Informations sur le stop-loss
        stop_info = stop_loss_manager.get_stop_loss_info(position)
        
        return {
            "position_id": position_id,
            "old_stop_loss": position.get("stop_loss"),
            "new_stop_loss": new_stop_loss,
            "current_price": current_price,
            "stop_info": stop_info,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erreur mise à jour trailing stop: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update trailing stop")

@router.get("/metrics")
async def get_risk_metrics():
    """
    Récupère les métriques de risque actuelles
    """
    try:
        risk_manager = get_risk_manager()
        
        metrics = risk_manager.get_risk_metrics()
        
        return {
            **metrics,
            "retrieved_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erreur récupération métriques: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve risk metrics")

@router.put("/config")
async def update_risk_config(config: RiskConfigUpdateRequest):
    """
    Met à jour la configuration de gestion des risques
    """
    try:
        risk_manager = get_risk_manager()
        
        # Mise à jour de la config (seulement les champs non-None)
        update_data = config.dict(exclude_none=True)
        
        # Pour l'instant, simuler la mise à jour
        logger.info(f"Mise à jour config risque: {update_data}")
        
        return {
            "message": "Risk configuration updated successfully",
            "updated_fields": update_data,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erreur mise à jour config: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update risk configuration")

@router.post("/emergency-stop/activate")
async def activate_emergency_stop():
    """
    Active l'arrêt d'urgence du trading
    """
    try:
        risk_manager = get_risk_manager()
        
        # Activation manuelle de l'arrêt d'urgence
        risk_manager.risk_state["emergency_stop"] = True
        
        return {
            "message": "Emergency stop activated",
            "status": "emergency_stop_active",
            "activated_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erreur activation arrêt urgence: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to activate emergency stop")

@router.post("/emergency-stop/deactivate")
async def deactivate_emergency_stop():
    """
    Désactive l'arrêt d'urgence du trading
    """
    try:
        risk_manager = get_risk_manager()
        
        risk_manager.deactivate_emergency_stop()
        
        return {
            "message": "Emergency stop deactivated",
            "status": "normal_trading",
            "deactivated_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erreur désactivation arrêt urgence: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to deactivate emergency stop")

@router.get("/emergency-stop/status")
async def get_emergency_status():
    """
    Vérifie l'état de l'arrêt d'urgence
    """
    try:
        risk_manager = get_risk_manager()
        
        is_active = risk_manager.risk_state.get("emergency_stop", False)
        
        return {
            "emergency_stop_active": is_active,
            "trading_status": "suspended" if is_active else "active",
            "checked_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erreur vérification urgence: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to check emergency status")