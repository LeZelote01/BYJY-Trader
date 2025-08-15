# AI Predictions API Routes for BYJY-Trader
# Phase 2.2 - AI predictions and signals endpoints

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio

from ai.predictions.predictor import AIPredictor
from ai.predictions.signal_generator import SignalGenerator
from ai.models.lstm_model import LSTMModel
from core.logger import get_logger

logger = get_logger(__name__)

# Initialize AI components
predictor = AIPredictor()
signal_generator = SignalGenerator()

# Router setup
router = APIRouter(prefix="/ai", tags=["AI Predictions"])

# Pydantic models for request/response validation
class PredictionRequest(BaseModel):
    symbol: str = Field(..., description="Trading symbol")
    horizon: str = Field("1h", description="Prediction horizon (15m, 1h, 4h, 1d, 7d)")
    model: str = Field("lstm", description="AI model to use")
    confidence_level: float = Field(0.95, ge=0.5, le=0.99, description="Confidence level")

class BatchPredictionRequest(BaseModel):
    symbols: List[str] = Field(..., description="List of trading symbols")
    horizon: str = Field("1h", description="Prediction horizon")
    model: str = Field("lstm", description="AI model to use")

class MultiHorizonRequest(BaseModel):
    symbol: str = Field(..., description="Trading symbol")
    horizons: List[str] = Field(["15m", "1h", "4h", "1d"], description="List of horizons")
    model: str = Field("lstm", description="AI model to use")

class SignalRequest(BaseModel):
    symbol: str = Field(..., description="Trading symbol")
    horizons: List[str] = Field(["1h", "4h", "1d"], description="Time horizons for signal")
    model: str = Field("lstm", description="AI model to use")

class BatchSignalRequest(BaseModel):
    symbols: List[str] = Field(..., description="List of symbols")
    horizons: List[str] = Field(["1h", "4h", "1d"], description="Time horizons")
    model: str = Field("lstm", description="AI model to use")

class ModelConfigRequest(BaseModel):
    model_name: str = Field(..., description="Model name")
    config: Dict[str, Any] = Field(..., description="Model configuration")

# Global initialization flag
_initialized = False

async def ensure_initialized():
    """Ensure AI components are initialized."""
    global _initialized
    if not _initialized:
        try:
            await predictor.initialize()
            await signal_generator.initialize()
            _initialized = True
            logger.info("AI components initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing AI components: {e}")
            raise HTTPException(status_code=500, detail="AI system initialization failed")

@router.get("/health", summary="AI System Health Check")
async def ai_health_check():
    """Check AI system health and status."""
    try:
        await ensure_initialized()
        
        # Get model status
        model_status = await predictor.get_model_status()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "ai_components": {
                "predictor": "ready",
                "signal_generator": "ready"
            },
            "models": model_status["loaded_models"],
            "cache_stats": model_status["cache_stats"]
        }
        
    except Exception as e:
        logger.error(f"AI health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"AI system unhealthy: {str(e)}")

@router.get("/models/status", summary="Get AI Models Status")
async def get_models_status():
    """Get detailed status of all AI models."""
    try:
        await ensure_initialized()
        
        status = await predictor.get_model_status()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "active_models": status["active_models"],
            "models": status["loaded_models"],
            "cache_statistics": status["cache_stats"],
            "supported_horizons": ["15m", "1h", "4h", "1d", "7d"]
        }
        
    except Exception as e:
        logger.error(f"Error getting models status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predictions/single", summary="Single Symbol Prediction")
async def predict_single(request: PredictionRequest):
    """Get price prediction for a single symbol."""
    try:
        await ensure_initialized()
        
        logger.info(f"Prediction request: {request.symbol} {request.horizon} {request.model}")
        
        prediction = await predictor.predict_price(
            symbol=request.symbol,
            horizon=request.horizon,
            model_name=request.model,
            confidence_level=request.confidence_level
        )
        
        if 'error' in prediction:
            raise HTTPException(status_code=400, detail=prediction['error'])
        
        return prediction
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in single prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predictions/multi-horizon", summary="Multi-Horizon Predictions")
async def predict_multi_horizon(request: MultiHorizonRequest):
    """Get predictions for multiple time horizons."""
    try:
        await ensure_initialized()
        
        prediction = await predictor.predict_multiple_horizons(
            symbol=request.symbol,
            horizons=request.horizons,
            model_name=request.model
        )
        
        if 'error' in prediction:
            raise HTTPException(status_code=400, detail=prediction['error'])
        
        return prediction
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in multi-horizon prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predictions/batch", summary="Batch Predictions")
async def predict_batch(request: BatchPredictionRequest):
    """Get predictions for multiple symbols."""
    try:
        await ensure_initialized()
        
        if len(request.symbols) > 20:
            raise HTTPException(
                status_code=400, 
                detail="Maximum 20 symbols allowed per batch request"
            )
        
        predictions = await predictor.predict_batch(
            symbols=request.symbols,
            horizon=request.horizon,
            model_name=request.model
        )
        
        return predictions
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predictions/{symbol}", summary="Quick Symbol Prediction")
async def quick_predict(
    symbol: str,
    horizon: str = Query("1h", description="Prediction horizon"),
    model: str = Query("lstm", description="Model to use")
):
    """Quick prediction endpoint for a symbol (GET request)."""
    try:
        await ensure_initialized()
        
        prediction = await predictor.predict_price(
            symbol=symbol.upper(),
            horizon=horizon,
            model_name=model
        )
        
        if 'error' in prediction:
            raise HTTPException(status_code=400, detail=prediction['error'])
        
        return prediction
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in quick prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/signals/single", summary="Generate Trading Signal")
async def generate_signal(request: SignalRequest):
    """Generate trading signal for a symbol."""
    try:
        await ensure_initialized()
        
        signal = await signal_generator.generate_signal(
            symbol=request.symbol,
            horizons=request.horizons,
            model_name=request.model
        )
        
        if 'error' in signal:
            raise HTTPException(status_code=400, detail=signal['error'])
        
        return signal
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating signal: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/signals/batch", summary="Batch Signal Generation")
async def generate_batch_signals(request: BatchSignalRequest):
    """Generate signals for multiple symbols."""
    try:
        await ensure_initialized()
        
        if len(request.symbols) > 15:
            raise HTTPException(
                status_code=400,
                detail="Maximum 15 symbols allowed per batch signal request"
            )
        
        signals = await signal_generator.generate_batch_signals(
            symbols=request.symbols,
            horizons=request.horizons,
            model_name=request.model
        )
        
        return signals
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating batch signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/signals/{symbol}", summary="Quick Signal Generation")
async def quick_signal(
    symbol: str,
    horizons: List[str] = Query(["1h", "4h"], description="Time horizons"),
    model: str = Query("lstm", description="Model to use")
):
    """Quick signal generation for a symbol (GET request)."""
    try:
        await ensure_initialized()
        
        signal = await signal_generator.generate_signal(
            symbol=symbol.upper(),
            horizons=horizons,
            model_name=model
        )
        
        if 'error' in signal:
            raise HTTPException(status_code=400, detail=signal['error'])
        
        return signal
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in quick signal: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/train/{model_name}", summary="Train AI Model")
async def train_model(
    model_name: str,
    background_tasks: BackgroundTasks,
    symbols: List[str] = Query(["BTCUSDT", "ETHUSDT"], description="Symbols to train on"),
    epochs: int = Query(50, ge=10, le=200, description="Training epochs")
):
    """Trigger model training (background task)."""
    try:
        await ensure_initialized()
        
        if model_name not in ["lstm"]:
            raise HTTPException(status_code=400, detail=f"Unsupported model: {model_name}")
        
        # Start background training
        background_tasks.add_task(
            _train_model_background,
            model_name,
            symbols,
            epochs
        )
        
        return {
            "status": "training_started",
            "model": model_name,
            "symbols": symbols,
            "epochs": epochs,
            "message": "Model training started in background",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting model training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/performance/{model_name}", summary="Get Model Performance")
async def get_model_performance(model_name: str):
    """Get performance metrics for a model."""
    try:
        await ensure_initialized()
        
        # Get model info
        status = await predictor.get_model_status()
        
        if model_name not in status["loaded_models"]:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
        model_info = status["loaded_models"][model_name]
        
        # Return performance info (simplified)
        return {
            "model_name": model_name,
            "is_trained": model_info["is_trained"],
            "training_history": model_info.get("history_available", False),
            "model_config": model_info["config"],
            "last_updated": model_info.get("last_updated", "unknown"),
            "performance_metrics": {
                "status": "available" if model_info["is_trained"] else "not_trained",
                "note": "Detailed metrics available after training completion"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/cache/clear", summary="Clear Prediction Cache")
async def clear_cache():
    """Clear all prediction caches."""
    try:
        await ensure_initialized()
        
        await predictor.clear_cache()
        
        return {
            "status": "cache_cleared",
            "timestamp": datetime.now().isoformat(),
            "message": "All prediction caches cleared successfully"
        }
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _train_model_background(model_name: str, symbols: List[str], epochs: int):
    """Background task for model training."""
    try:
        logger.info(f"Starting background training for {model_name}")
        
        if model_name == "lstm":
            model = LSTMModel()
            
            # Get training data for each symbol
            from data.storage.data_manager import DataManager
            data_manager = DataManager()
            await data_manager.initialize_tables()
            
            # Combine data from all symbols (simplified)
            all_data = []
            for symbol in symbols:
                symbol_data = await data_manager.get_historical_data(symbol=symbol, limit=1000)
                if not symbol_data.empty:
                    all_data.append(symbol_data)
            
            if not all_data:
                logger.error("No training data available")
                return
            
            # Combine all data
            import pandas as pd
            training_data = pd.concat(all_data, ignore_index=True)
            
            # Prepare data and train
            X, y = model.prepare_data(training_data)
            
            if len(X) > 0:
                model.model_config['epochs'] = epochs
                results = model.train(X, y)
                
                # Save trained model
                model.save_model()
                
                logger.info(f"Model {model_name} training completed: {results}")
            else:
                logger.error("Could not prepare training data")
        
    except Exception as e:
        logger.error(f"Error in background training: {e}")