# AI Predictor for BYJY-Trader
# Phase 2.2 - Real-time predictions engine

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json

from ai.models.lstm_model import LSTMModel
from data.storage.data_manager import DataManager
from data.processors.feature_engine import FeatureEngine
from core.logger import get_logger

logger = get_logger(__name__)

class AIPredictor:
    """
    AI Prediction engine for real-time price predictions.
    Integrates with trained models to generate forecasts.
    """
    
    def __init__(self):
        """Initialize AI Predictor."""
        self.data_manager = DataManager()
        self.feature_engine = FeatureEngine()
        
        # Models
        self.models = {}
        self.active_models = ['lstm']
        
        # Prediction cache
        self.prediction_cache = {}
        self.cache_ttl = 300  # 5 minutes
        self.cache_timestamps = {}
        
        # Prediction horizons in minutes
        self.prediction_horizons = {
            '15m': 15,
            '1h': 60,
            '4h': 240,
            '1d': 1440,
            '7d': 10080
        }
        
        logger.info("AI Predictor initialized")
    
    async def initialize(self):
        """Initialize predictor components."""
        try:
            await self.data_manager.initialize_tables()
            await self._load_models()
            logger.info("AI Predictor ready")
            
        except Exception as e:
            logger.error(f"Error initializing AI Predictor: {e}")
            raise
    
    async def _load_models(self):
        """Load trained AI models."""
        try:
            # Load LSTM model
            lstm_model = LSTMModel()
            if lstm_model.load_model():
                self.models['lstm'] = lstm_model
                logger.info("LSTM model loaded successfully")
            else:
                logger.warning("LSTM model not found, will need training")
                self.models['lstm'] = lstm_model
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    async def predict_price(self,
                          symbol: str,
                          horizon: str = '1h',
                          model_name: str = 'lstm',
                          confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Predict future price for a symbol.
        
        Args:
            symbol: Trading symbol
            horizon: Prediction horizon (15m, 1h, 4h, 1d, 7d)
            model_name: Model to use for prediction
            confidence_level: Confidence level for intervals
            
        Returns:
            Dict: Prediction results
        """
        try:
            # Check cache first
            cache_key = f"{symbol}_{horizon}_{model_name}"
            if self._is_cached(cache_key):
                logger.debug(f"Returning cached prediction for {cache_key}")
                return self.prediction_cache[cache_key]
            
            # Validate inputs
            if horizon not in self.prediction_horizons:
                raise ValueError(f"Unsupported horizon: {horizon}")
            
            if model_name not in self.models:
                raise ValueError(f"Model not available: {model_name}")
            
            model = self.models[model_name]
            if not model.is_trained:
                return {
                    'error': f"Model {model_name} not trained",
                    'symbol': symbol,
                    'horizon': horizon
                }
            
            # Get historical data with features
            historical_data = await self._get_prediction_data(symbol)
            
            if historical_data.empty:
                return {
                    'error': 'Insufficient historical data',
                    'symbol': symbol,
                    'horizon': horizon
                }
            
            # Make prediction
            prediction_result = await self._make_prediction(
                model, historical_data, horizon, confidence_level
            )
            
            # Add metadata
            prediction_result.update({
                'symbol': symbol,
                'horizon': horizon,
                'model': model_name,
                'timestamp': datetime.now().isoformat(),
                'data_points_used': len(historical_data)
            })
            
            # Cache result
            self._cache_prediction(cache_key, prediction_result)
            
            logger.info(f"Prediction completed for {symbol} {horizon}: {prediction_result.get('predicted_price', 'N/A')}")
            return prediction_result
            
        except Exception as e:
            logger.error(f"Error predicting price for {symbol}: {e}")
            return {
                'error': str(e),
                'symbol': symbol,
                'horizon': horizon
            }
    
    async def predict_multiple_horizons(self,
                                      symbol: str,
                                      horizons: List[str] = None,
                                      model_name: str = 'lstm') -> Dict[str, Any]:
        """
        Predict multiple time horizons for a symbol.
        
        Args:
            symbol: Trading symbol
            horizons: List of horizons to predict
            model_name: Model to use
            
        Returns:
            Dict: Predictions for all horizons
        """
        if horizons is None:
            horizons = ['15m', '1h', '4h', '1d']
        
        results = {
            'symbol': symbol,
            'model': model_name,
            'timestamp': datetime.now().isoformat(),
            'predictions': {}
        }
        
        for horizon in horizons:
            try:
                prediction = await self.predict_price(symbol, horizon, model_name)
                results['predictions'][horizon] = prediction
                
            except Exception as e:
                logger.error(f"Error predicting {horizon} for {symbol}: {e}")
                results['predictions'][horizon] = {'error': str(e)}
        
        return results
    
    async def predict_batch(self,
                          symbols: List[str],
                          horizon: str = '1h',
                          model_name: str = 'lstm') -> Dict[str, Any]:
        """
        Batch predictions for multiple symbols.
        
        Args:
            symbols: List of symbols to predict
            horizon: Prediction horizon
            model_name: Model to use
            
        Returns:
            Dict: Batch prediction results
        """
        results = {
            'horizon': horizon,
            'model': model_name,
            'timestamp': datetime.now().isoformat(),
            'predictions': {},
            'success_count': 0,
            'error_count': 0
        }
        
        # Process predictions concurrently
        tasks = []
        for symbol in symbols:
            task = self.predict_price(symbol, horizon, model_name)
            tasks.append(task)
        
        predictions = await asyncio.gather(*tasks, return_exceptions=True)
        
        for symbol, prediction in zip(symbols, predictions):
            if isinstance(prediction, Exception):
                results['predictions'][symbol] = {'error': str(prediction)}
                results['error_count'] += 1
            else:
                results['predictions'][symbol] = prediction
                if 'error' not in prediction:
                    results['success_count'] += 1
                else:
                    results['error_count'] += 1
        
        logger.info(f"Batch prediction completed: {results['success_count']} success, {results['error_count']} errors")
        return results
    
    async def _get_prediction_data(self, symbol: str) -> pd.DataFrame:
        """
        Get historical data with features for prediction.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            pd.DataFrame: Historical data with features
        """
        try:
            # Get recent historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)  # 6 months of data
            
            # Retrieve historical data
            historical_data = await self.data_manager.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                limit=1000
            )
            
            if historical_data.empty:
                logger.warning(f"No historical data found for {symbol}")
                return pd.DataFrame()
            
            # Generate features if not already present
            feature_columns = [col for col in historical_data.columns 
                             if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'source', 'interval']]
            
            if len(feature_columns) < 10:  # Need sufficient features
                logger.info(f"Generating features for {symbol}")
                historical_data = self.feature_engine.generate_all_features(historical_data)
            
            # Clean data
            historical_data = historical_data.dropna()
            
            if len(historical_data) < 100:
                logger.warning(f"Insufficient clean data for {symbol}: {len(historical_data)} rows")
                return pd.DataFrame()
            
            return historical_data.sort_values('timestamp')
            
        except Exception as e:
            logger.error(f"Error getting prediction data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def _make_prediction(self,
                             model: Any,
                             data: pd.DataFrame,
                             horizon: str,
                             confidence_level: float) -> Dict[str, Any]:
        """
        Make prediction with a specific model.
        
        Args:
            model: Trained AI model
            data: Historical data with features
            horizon: Prediction horizon
            confidence_level: Confidence level
            
        Returns:
            Dict: Prediction results
        """
        try:
            # Prepare data for model
            X, _ = model.prepare_data(data, target_column='close')
            
            if len(X) == 0:
                return {'error': 'Could not prepare data for prediction'}
            
            # Use last sequence for prediction
            last_sequence = X[-1:] if len(X.shape) == 3 else X[-1:].reshape(1, -1)
            
            # Calculate number of steps based on horizon
            horizon_minutes = self.prediction_horizons[horizon]
            n_steps = max(1, horizon_minutes // 60)  # Convert to hours (simplified)
            
            if hasattr(model, 'predict_next_prices'):
                # Use model's multi-step prediction
                predictions = model.predict_next_prices(data, n_steps=n_steps)
            else:
                # Single step prediction
                predictions = model.predict(last_sequence)
            
            if len(predictions) == 0:
                return {'error': 'Model returned empty predictions'}
            
            current_price = float(data['close'].iloc[-1])
            predicted_price = float(predictions[-1] if len(predictions) > 1 else predictions[0])
            
            # Calculate prediction confidence (simplified)
            price_volatility = data['close'].pct_change().std()
            confidence_interval = predicted_price * price_volatility * 1.96  # 95% CI
            
            # Calculate additional metrics
            price_change = predicted_price - current_price
            price_change_percent = (price_change / current_price) * 100
            
            result = {
                'current_price': current_price,
                'predicted_price': predicted_price,
                'price_change': price_change,
                'price_change_percent': price_change_percent,
                'confidence_interval': {
                    'lower': predicted_price - confidence_interval,
                    'upper': predicted_price + confidence_interval
                },
                'confidence_level': confidence_level,
                'prediction_quality': self._assess_prediction_quality(data, model),
                'horizon_minutes': horizon_minutes
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {'error': str(e)}
    
    def _assess_prediction_quality(self, data: pd.DataFrame, model: Any) -> str:
        """
        Assess prediction quality based on recent model performance.
        
        Args:
            data: Historical data
            model: AI model
            
        Returns:
            str: Quality assessment (HIGH, MEDIUM, LOW)
        """
        try:
            # Simple assessment based on data quality
            data_points = len(data)
            recent_volatility = data['close'].tail(20).pct_change().std()
            
            if data_points > 500 and recent_volatility < 0.05:
                return 'HIGH'
            elif data_points > 200 and recent_volatility < 0.1:
                return 'MEDIUM'
            else:
                return 'LOW'
                
        except Exception as e:
            logger.error(f"Error assessing prediction quality: {e}")
            return 'LOW'
    
    def _is_cached(self, cache_key: str) -> bool:
        """Check if prediction is cached and valid."""
        if cache_key not in self.prediction_cache:
            return False
        
        if cache_key not in self.cache_timestamps:
            return False
        
        # Check TTL
        cached_time = self.cache_timestamps[cache_key]
        if datetime.now().timestamp() - cached_time > self.cache_ttl:
            # Remove expired cache
            del self.prediction_cache[cache_key]
            del self.cache_timestamps[cache_key]
            return False
        
        return True
    
    def _cache_prediction(self, cache_key: str, prediction: Dict[str, Any]):
        """Cache prediction result."""
        self.prediction_cache[cache_key] = prediction
        self.cache_timestamps[cache_key] = datetime.now().timestamp()
        
        # Limit cache size
        if len(self.prediction_cache) > 200:
            # Remove oldest cache entries
            oldest_key = min(self.cache_timestamps.keys(), 
                           key=lambda k: self.cache_timestamps[k])
            del self.prediction_cache[oldest_key]
            del self.cache_timestamps[oldest_key]
    
    async def clear_cache(self):
        """Clear prediction cache."""
        self.prediction_cache.clear()
        self.cache_timestamps.clear()
        logger.info("Prediction cache cleared")
    
    async def get_model_status(self) -> Dict[str, Any]:
        """
        Get status of all loaded models.
        
        Returns:
            Dict: Model status information
        """
        status = {
            'active_models': self.active_models,
            'loaded_models': {},
            'cache_stats': {
                'cached_predictions': len(self.prediction_cache),
                'cache_size_mb': len(str(self.prediction_cache)) / (1024 * 1024)
            }
        }
        
        for model_name, model in self.models.items():
            status['loaded_models'][model_name] = model.get_model_info()
        
        return status