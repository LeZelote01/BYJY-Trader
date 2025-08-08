# Ensemble Predictions API Routes for BYJY-Trader
# Phase 3.1 - Enhanced AI predictions with ensemble models

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio

from ai.predictions.ensemble_predictor import EnsemblePredictor
from core.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/ensemble", tags=["Ensemble Predictions"])

# Global predictor instance
ensemble_predictor = None

async def get_ensemble_predictor():
    """Get or initialize ensemble predictor."""
    global ensemble_predictor
    if ensemble_predictor is None:
        ensemble_predictor = EnsemblePredictor()
        await ensemble_predictor.initialize()
    return ensemble_predictor

@router.on_event("startup")
async def startup_ensemble_predictor():
    """Initialize ensemble predictor on startup."""
    try:
        await get_ensemble_predictor()
        logger.info("Ensemble predictor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize ensemble predictor: {e}")

@router.get("/status")
async def get_ensemble_status():
    """
    Get ensemble predictor status and model information.
    
    Returns:
        Dict: Ensemble predictor status
    """
    try:
        predictor = await get_ensemble_predictor()
        status = await predictor.get_ensemble_status()
        
        return {
            "success": True,
            "data": status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting ensemble status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict")
async def predict_with_ensemble(
    symbol: str,
    horizon: str = "1h",
    model_preference: str = "ensemble",
    confidence_level: float = 0.95,
    include_individual: bool = False
):
    """
    Make enhanced price prediction using ensemble models.
    
    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        horizon: Prediction horizon ('15m', '1h', '4h', '1d', '7d')
        model_preference: Preferred model ('ensemble', 'lstm', 'transformer', 'xgboost', 'best')
        confidence_level: Confidence level for intervals (0.8-0.99)
        include_individual: Include individual model predictions
        
    Returns:
        Dict: Enhanced prediction results
    """
    try:
        # Validate inputs
        valid_horizons = ['15m', '1h', '4h', '1d', '7d']
        if horizon not in valid_horizons:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid horizon. Must be one of: {valid_horizons}"
            )
        
        valid_models = ['ensemble', 'lstm', 'transformer', 'xgboost', 'best']
        if model_preference not in valid_models:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model preference. Must be one of: {valid_models}"
            )
        
        if not 0.8 <= confidence_level <= 0.99:
            raise HTTPException(
                status_code=400,
                detail="Confidence level must be between 0.8 and 0.99"
            )
        
        # Get ensemble predictor and make prediction
        predictor = await get_ensemble_predictor()
        
        prediction = await predictor.predict_price_enhanced(
            symbol=symbol,
            horizon=horizon,
            model_preference=model_preference,
            confidence_level=confidence_level,
            include_individual=include_individual
        )
        
        if 'error' in prediction:
            raise HTTPException(status_code=400, detail=prediction['error'])
        
        return {
            "success": True,
            "data": prediction,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in ensemble prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/batch")
async def predict_batch_ensemble(
    symbols: List[str],
    horizon: str = "1h",
    model_preference: str = "ensemble",
    confidence_level: float = 0.95,
    include_individual: bool = False
):
    """
    Make batch predictions for multiple symbols using ensemble models.
    
    Args:
        symbols: List of trading symbols
        horizon: Prediction horizon
        model_preference: Preferred model
        confidence_level: Confidence level
        include_individual: Include individual model predictions
        
    Returns:
        Dict: Batch prediction results
    """
    try:
        if len(symbols) > 20:
            raise HTTPException(
                status_code=400,
                detail="Maximum 20 symbols allowed per batch request"
            )
        
        predictor = await get_ensemble_predictor()
        
        # Process predictions concurrently
        tasks = []
        for symbol in symbols:
            task = predictor.predict_price_enhanced(
                symbol=symbol,
                horizon=horizon,
                model_preference=model_preference,
                confidence_level=confidence_level,
                include_individual=include_individual
            )
            tasks.append(task)
        
        predictions = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        results = {
            'batch_id': f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'symbols': symbols,
            'horizon': horizon,
            'model_preference': model_preference,
            'predictions': {},
            'success_count': 0,
            'error_count': 0,
            'errors': {}
        }
        
        for symbol, prediction in zip(symbols, predictions):
            if isinstance(prediction, Exception):
                results['predictions'][symbol] = {'error': str(prediction)}
                results['errors'][symbol] = str(prediction)
                results['error_count'] += 1
            elif 'error' in prediction:
                results['predictions'][symbol] = prediction
                results['errors'][symbol] = prediction['error']
                results['error_count'] += 1
            else:
                results['predictions'][symbol] = prediction
                results['success_count'] += 1
        
        return {
            "success": True,
            "data": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch ensemble prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/multi-horizon")
async def predict_multi_horizon(
    symbol: str,
    horizons: Optional[List[str]] = None,
    model_preference: str = "ensemble",
    confidence_level: float = 0.95
):
    """
    Make predictions for multiple time horizons.
    
    Args:
        symbol: Trading symbol
        horizons: List of horizons (default: ['15m', '1h', '4h', '1d'])
        model_preference: Preferred model
        confidence_level: Confidence level
        
    Returns:
        Dict: Multi-horizon prediction results
    """
    try:
        if horizons is None:
            horizons = ['15m', '1h', '4h', '1d']
        
        valid_horizons = ['15m', '1h', '4h', '1d', '7d']
        invalid_horizons = [h for h in horizons if h not in valid_horizons]
        if invalid_horizons:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid horizons: {invalid_horizons}. Must be from: {valid_horizons}"
            )
        
        predictor = await get_ensemble_predictor()
        
        # Get predictions for all horizons
        tasks = []
        for horizon in horizons:
            task = predictor.predict_price_enhanced(
                symbol=symbol,
                horizon=horizon,
                model_preference=model_preference,
                confidence_level=confidence_level,
                include_individual=False
            )
            tasks.append(task)
        
        predictions = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        results = {
            'symbol': symbol,
            'model_preference': model_preference,
            'confidence_level': confidence_level,
            'predictions': {},
            'success_count': 0,
            'error_count': 0
        }
        
        for horizon, prediction in zip(horizons, predictions):
            if isinstance(prediction, Exception):
                results['predictions'][horizon] = {'error': str(prediction)}
                results['error_count'] += 1
            elif 'error' in prediction:
                results['predictions'][horizon] = prediction
                results['error_count'] += 1
            else:
                results['predictions'][horizon] = prediction
                results['success_count'] += 1
        
        return {
            "success": True,
            "data": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in multi-horizon prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/compare")
async def compare_models(
    symbol: str,
    horizon: str = "1h",
    confidence_level: float = 0.95
):
    """
    Compare predictions from all available models.
    
    Args:
        symbol: Trading symbol
        horizon: Prediction horizon
        confidence_level: Confidence level
        
    Returns:
        Dict: Model comparison results
    """
    try:
        predictor = await get_ensemble_predictor()
        
        # Get predictions from all models
        prediction = await predictor.predict_price_enhanced(
            symbol=symbol,
            horizon=horizon,
            model_preference='ensemble',
            confidence_level=confidence_level,
            include_individual=True
        )
        
        if 'error' in prediction:
            raise HTTPException(status_code=400, detail=prediction['error'])
        
        # Structure comparison data
        comparison = {
            'symbol': symbol,
            'horizon': horizon,
            'current_price': prediction.get('current_price'),
            'ensemble_prediction': {
                'model': 'ensemble',
                'predicted_price': prediction.get('predicted_price'),
                'price_change_percent': prediction.get('price_change_percent'),
                'prediction_quality': prediction.get('prediction_quality')
            },
            'individual_models': prediction.get('individual_predictions', {}),
            'model_agreement': prediction.get('enhanced_metrics', {}).get('model_agreement'),
            'ensemble_available': prediction.get('ensemble_details', {}).get('available', False)
        }
        
        return {
            "success": True,
            "data": comparison,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance")
async def get_ensemble_performance():
    """
    Get ensemble model performance metrics.
    
    Returns:
        Dict: Performance metrics and statistics
    """
    try:
        predictor = await get_ensemble_predictor()
        status = await predictor.get_ensemble_status()
        
        # Extract performance-related information
        performance = {
            'ensemble_available': status.get('ensemble_available', False),
            'models_status': {},
            'cache_performance': status.get('cache_stats', {}),
            'system_metrics': {
                'prediction_accuracy': 'Not yet calculated',
                'average_response_time': 'Not yet calculated',
                'total_predictions_made': 'Not yet tracked'
            }
        }
        
        # Model status
        for model_name, model_info in status.get('individual_models', {}).items():
            performance['models_status'][model_name] = {
                'trained': model_info.get('trained', False),
                'version': model_info.get('info', {}).get('version', 'unknown')
            }
        
        # Ensemble details if available
        if status.get('ensemble_available'):
            ensemble_details = status.get('ensemble_details', {})
            performance['ensemble_config'] = {
                'fusion_method': ensemble_details.get('fusion_method', 'unknown'),
                'model_weights': ensemble_details.get('model_weights', {}),
                'meta_model_trained': ensemble_details.get('meta_model_trained', False)
            }
        
        return {
            "success": True,
            "data": performance,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting ensemble performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/cache")
async def clear_ensemble_cache():
    """
    Clear ensemble prediction cache.
    
    Returns:
        Dict: Cache clear status
    """
    try:
        predictor = await get_ensemble_predictor()
        
        # Clear cache
        predictor.prediction_cache.clear()
        predictor.cache_timestamps.clear()
        
        return {
            "success": True,
            "message": "Ensemble prediction cache cleared successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error clearing ensemble cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))