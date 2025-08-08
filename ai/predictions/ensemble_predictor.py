# Enhanced AI Predictor with Ensemble Models for BYJY-Trader
# Phase 3.1 - Advanced prediction engine using ensemble models

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json

from ai.models.ensemble_model import EnsembleModel
from ai.models.lstm_model import LSTMModel
from ai.models.transformer_model import TransformerModel
from ai.models.xgboost_model import XGBoostModel
from data.storage.data_manager import DataManager
from data.processors.feature_engine import FeatureEngine
from core.logger import get_logger

logger = get_logger(__name__)

class EnsemblePredictor:
    """
    Enhanced AI Prediction engine using ensemble models.
    Combines LSTM, Transformer, and XGBoost for superior accuracy.
    """
    
    def __init__(self):
        """Initialize Enhanced Ensemble Predictor."""
        self.data_manager = DataManager()
        self.feature_engine = FeatureEngine()
        
        # Models
        self.ensemble_model = EnsembleModel()
        self.individual_models = {
            'lstm': LSTMModel(),
            'transformer': TransformerModel(), 
            'xgboost': XGBoostModel()
        }
        
        # Prediction cache with TTL
        self.prediction_cache = {}
        self.cache_ttl = 300  # 5 minutes
        self.cache_timestamps = {}
        
        # Performance tracking
        self.model_performance_history = {
            'ensemble': [],
            'lstm': [],
            'transformer': [],
            'xgboost': []
        }
        
        # Prediction horizons in minutes
        self.prediction_horizons = {
            '15m': 15,
            '1h': 60,
            '4h': 240,
            '1d': 1440,
            '7d': 10080
        }
        
        # Ensemble configuration
        self.ensemble_config = {
            'use_ensemble': True,
            'fallback_to_individual': True,
            'min_models_for_ensemble': 2,
            'confidence_threshold': 0.6,
            'performance_window': 100
        }
        
        logger.info("Ensemble Predictor initialized with advanced AI models")
    
    async def initialize(self):
        """Initialize predictor components."""
        try:
            await self.data_manager.initialize_tables()
            await self._load_models()
            logger.info("Enhanced Ensemble Predictor ready")
            
        except Exception as e:
            logger.error(f"Error initializing Ensemble Predictor: {e}")
            raise
    
    async def _load_models(self):
        """Load trained AI models."""
        try:
            # Try to load ensemble model first
            if self.ensemble_model.load_model():
                logger.info("Ensemble model loaded successfully")
                self.ensemble_config['use_ensemble'] = True
            else:
                logger.warning("Ensemble model not found, will use individual models")
                self.ensemble_config['use_ensemble'] = False
            
            # Load individual models as fallback/comparison
            for model_name, model in self.individual_models.items():
                try:
                    if model.load_model():
                        logger.info(f"{model_name.upper()} model loaded successfully")
                    else:
                        logger.warning(f"{model_name.upper()} model not found, will need training")
                except Exception as e:
                    logger.warning(f"Error loading {model_name} model: {e}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    async def predict_price_enhanced(self,
                                   symbol: str,
                                   horizon: str = '1h',
                                   model_preference: str = 'ensemble',
                                   confidence_level: float = 0.95,
                                   include_individual: bool = False) -> Dict[str, Any]:
        """
        Enhanced price prediction with ensemble model.
        
        Args:
            symbol: Trading symbol
            horizon: Prediction horizon (15m, 1h, 4h, 1d, 7d)
            model_preference: 'ensemble', 'lstm', 'transformer', 'xgboost', 'best'
            confidence_level: Confidence level for intervals
            include_individual: Include individual model predictions
            
        Returns:
            Dict: Enhanced prediction results
        """
        try:
            # Check cache first
            cache_key = f"{symbol}_{horizon}_{model_preference}_{confidence_level}"
            if self._is_cached(cache_key):
                logger.debug(f"Returning cached enhanced prediction for {cache_key}")
                return self.prediction_cache[cache_key]
            
            # Validate inputs
            if horizon not in self.prediction_horizons:
                raise ValueError(f"Unsupported horizon: {horizon}")
            
            # Get historical data with features
            historical_data = await self._get_prediction_data(symbol)
            
            if historical_data.empty:
                return {
                    'error': 'Insufficient historical data',
                    'symbol': symbol,
                    'horizon': horizon
                }
            
            # Make enhanced prediction
            prediction_result = await self._make_enhanced_prediction(
                historical_data, symbol, horizon, model_preference, 
                confidence_level, include_individual
            )
            
            # Add metadata
            prediction_result.update({
                'symbol': symbol,
                'horizon': horizon,
                'model_preference': model_preference,
                'timestamp': datetime.now().isoformat(),
                'data_points_used': len(historical_data),
                'ensemble_available': self.ensemble_model.is_trained,
                'prediction_version': '3.1.0'
            })
            
            # Cache result
            self._cache_prediction(cache_key, prediction_result)
            
            logger.info(f"Enhanced prediction completed for {symbol} {horizon}")
            return prediction_result
            
        except Exception as e:
            logger.error(f"Error in enhanced prediction for {symbol}: {e}")
            return {
                'error': str(e),
                'symbol': symbol,
                'horizon': horizon
            }
    
    async def _make_enhanced_prediction(self,
                                      data: pd.DataFrame,
                                      symbol: str,
                                      horizon: str,
                                      model_preference: str,
                                      confidence_level: float,
                                      include_individual: bool) -> Dict[str, Any]:
        """
        Make enhanced prediction using ensemble or individual models.
        
        Args:
            data: Historical data with features
            symbol: Trading symbol
            horizon: Prediction horizon
            model_preference: Preferred model
            confidence_level: Confidence level
            include_individual: Include individual predictions
            
        Returns:
            Dict: Enhanced prediction results
        """
        try:
            # Prepare data for models
            predictions = {}
            errors = {}
            
            # Get ensemble prediction if available and preferred
            if (model_preference in ['ensemble', 'best'] and 
                self.ensemble_model.is_trained and 
                self.ensemble_config['use_ensemble']):
                
                try:
                    ensemble_pred = await self._get_ensemble_prediction(data, horizon)
                    predictions['ensemble'] = ensemble_pred
                    logger.info("Ensemble prediction obtained")
                except Exception as e:
                    errors['ensemble'] = str(e)
                    logger.warning(f"Ensemble prediction failed: {e}")
            
            # Get individual model predictions
            individual_results = {}
            if include_individual or model_preference in ['lstm', 'transformer', 'xgboost'] or 'ensemble' not in predictions:
                
                for model_name, model in self.individual_models.items():
                    if model.is_trained:
                        try:
                            individual_pred = await self._get_individual_prediction(
                                model, model_name, data, horizon
                            )
                            individual_results[model_name] = individual_pred
                            predictions[model_name] = individual_pred
                            
                        except Exception as e:
                            errors[model_name] = str(e)
                            logger.warning(f"{model_name} prediction failed: {e}")
            
            if not predictions:
                return {'error': 'No predictions available from any model', 'errors': errors}
            
            # Select primary prediction based on preference
            primary_prediction = self._select_primary_prediction(
                predictions, model_preference, symbol
            )
            
            # Calculate enhanced metrics
            enhanced_metrics = self._calculate_enhanced_metrics(
                predictions, data, horizon
            )
            
            # Build result
            result = {
                'primary_model': primary_prediction['model'],
                'predicted_price': primary_prediction['predicted_price'],
                'current_price': primary_prediction['current_price'],
                'price_change': primary_prediction['price_change'],
                'price_change_percent': primary_prediction['price_change_percent'],
                'confidence_interval': primary_prediction.get('confidence_interval', {}),
                'prediction_quality': primary_prediction.get('prediction_quality', 'MEDIUM'),
                'enhanced_metrics': enhanced_metrics,
                'horizon_minutes': self.prediction_horizons[horizon]
            }
            
            # Add individual predictions if requested
            if include_individual and individual_results:
                result['individual_predictions'] = individual_results
            
            # Add ensemble details if available
            if 'ensemble' in predictions:
                result['ensemble_details'] = {
                    'available': True,
                    'model_weights': getattr(self.ensemble_model, 'model_weights', {}),
                    'fusion_method': getattr(self.ensemble_model, 'model_config', {}).get('fusion_method', 'weighted_average')
                }
            
            # Add errors if any
            if errors:
                result['model_errors'] = errors
            
            return result
            
        except Exception as e:
            logger.error(f"Error making enhanced prediction: {e}")
            return {'error': str(e)}
    
    async def _get_ensemble_prediction(self, data: pd.DataFrame, horizon: str) -> Dict[str, Any]:
        """Get prediction from ensemble model."""
        try:
            # Prepare data for ensemble model
            X_dict, y = self.ensemble_model.prepare_data(data, target_column='close')
            
            if not any(x is not None for x in X_dict.values()):
                raise ValueError("No data available for ensemble prediction")
            
            # Make prediction
            ensemble_pred = self.ensemble_model.predict(X_dict)
            
            if len(ensemble_pred) == 0:
                raise ValueError("Ensemble model returned empty predictions")
            
            current_price = float(data['close'].iloc[-1])
            predicted_price = float(ensemble_pred[-1])
            
            # Calculate confidence based on ensemble agreement
            confidence_metrics = self._calculate_ensemble_confidence(X_dict)
            
            result = {
                'model': 'ensemble',
                'predicted_price': predicted_price,
                'current_price': current_price,
                'price_change': predicted_price - current_price,
                'price_change_percent': ((predicted_price - current_price) / current_price) * 100,
                'confidence_interval': confidence_metrics['confidence_interval'],
                'prediction_quality': confidence_metrics['quality'],
                'ensemble_agreement': confidence_metrics['agreement_score']
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting ensemble prediction: {e}")
            raise
    
    async def _get_individual_prediction(self, model: Any, model_name: str, 
                                       data: pd.DataFrame, horizon: str) -> Dict[str, Any]:
        """Get prediction from individual model."""
        try:
            # Prepare data for model
            X, y = model.prepare_data(data, target_column='close')
            
            if len(X) == 0:
                raise ValueError(f"No data available for {model_name} prediction")
            
            # Use last sequence/sample for prediction
            if len(X.shape) == 3:  # Sequence data (LSTM, Transformer)
                X_pred = X[-1:] 
            else:  # Tabular data (XGBoost)
                X_pred = X[-1:].reshape(1, -1)
            
            # Make prediction
            pred = model.predict(X_pred)
            
            if len(pred) == 0:
                raise ValueError(f"{model_name} returned empty predictions")
            
            current_price = float(data['close'].iloc[-1])
            predicted_price = float(pred[0])
            
            # Calculate model-specific confidence
            confidence = self._calculate_individual_confidence(
                model, model_name, data, X_pred
            )
            
            result = {
                'model': model_name,
                'predicted_price': predicted_price,
                'current_price': current_price,
                'price_change': predicted_price - current_price,
                'price_change_percent': ((predicted_price - current_price) / current_price) * 100,
                'confidence_interval': confidence['confidence_interval'],
                'prediction_quality': confidence['quality']
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting {model_name} prediction: {e}")
            raise
    
    def _select_primary_prediction(self, predictions: Dict[str, Any], 
                                 preference: str, symbol: str) -> Dict[str, Any]:
        """Select primary prediction based on preference and performance."""
        try:
            # If specific model preference
            if preference in predictions:
                return predictions[preference]
            
            # If 'best' preference, select based on historical performance
            if preference == 'best':
                best_model = self._get_best_performing_model(symbol)
                if best_model in predictions:
                    return predictions[best_model]
            
            # Default priority: ensemble > transformer > lstm > xgboost
            priority = ['ensemble', 'transformer', 'lstm', 'xgboost']
            for model_name in priority:
                if model_name in predictions:
                    return predictions[model_name]
            
            # Fallback to any available prediction
            return list(predictions.values())[0]
            
        except Exception as e:
            logger.error(f"Error selecting primary prediction: {e}")
            return list(predictions.values())[0]
    
    def _calculate_enhanced_metrics(self, predictions: Dict[str, Any], 
                                  data: pd.DataFrame, horizon: str) -> Dict[str, Any]:
        """Calculate enhanced prediction metrics."""
        try:
            metrics = {
                'prediction_count': len(predictions),
                'model_agreement': 0.0,
                'volatility_adjusted_confidence': 0.0,
                'data_quality_score': 0.0,
                'market_regime': 'unknown'
            }
            
            if len(predictions) < 2:
                return metrics
            
            # Calculate model agreement
            price_predictions = [pred['predicted_price'] for pred in predictions.values()]
            if price_predictions:
                mean_pred = np.mean(price_predictions)
                std_pred = np.std(price_predictions)
                cv = std_pred / mean_pred if mean_pred != 0 else 1.0
                metrics['model_agreement'] = max(0, 1 - cv)  # Higher agreement = lower CV
            
            # Data quality score
            data_quality = self._assess_data_quality(data)
            metrics['data_quality_score'] = data_quality
            
            # Market regime detection
            metrics['market_regime'] = self._detect_market_regime(data)
            
            # Volatility-adjusted confidence
            recent_volatility = data['close'].tail(20).pct_change().std()
            base_confidence = metrics['model_agreement']
            volatility_adjustment = max(0.5, 1 - recent_volatility)  # Lower vol = higher confidence
            metrics['volatility_adjusted_confidence'] = base_confidence * volatility_adjustment
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating enhanced metrics: {e}")
            return {'error': str(e)}
    
    def _calculate_ensemble_confidence(self, X_dict: Dict) -> Dict[str, Any]:
        """Calculate confidence metrics for ensemble prediction."""
        try:
            # Get individual model predictions if available
            individual_preds = []
            
            for model_name in ['lstm', 'transformer', 'xgboost']:
                model = self.individual_models[model_name]
                X_model = X_dict.get(model_name)
                
                if model.is_trained and X_model is not None:
                    try:
                        pred = model.predict(X_model[-1:] if len(X_model.shape) == 3 else X_model[-1:].reshape(1, -1))
                        individual_preds.append(pred[0])
                    except:
                        continue
            
            # Calculate agreement
            if len(individual_preds) >= 2:
                agreement_score = 1.0 - (np.std(individual_preds) / np.mean(individual_preds))
                agreement_score = max(0, min(1, agreement_score))
            else:
                agreement_score = 0.5  # Neutral if can't calculate
            
            # Confidence interval based on agreement
            mean_pred = np.mean(individual_preds) if individual_preds else 0
            std_pred = np.std(individual_preds) if len(individual_preds) > 1 else mean_pred * 0.05
            
            confidence_interval = {
                'lower': mean_pred - 1.96 * std_pred,
                'upper': mean_pred + 1.96 * std_pred
            }
            
            # Quality assessment
            if agreement_score > 0.8:
                quality = 'HIGH'
            elif agreement_score > 0.6:
                quality = 'MEDIUM'
            else:
                quality = 'LOW'
            
            return {
                'confidence_interval': confidence_interval,
                'quality': quality,
                'agreement_score': float(agreement_score)
            }
            
        except Exception as e:
            logger.error(f"Error calculating ensemble confidence: {e}")
            return {
                'confidence_interval': {'lower': 0, 'upper': 0},
                'quality': 'LOW',
                'agreement_score': 0.0
            }
    
    def _calculate_individual_confidence(self, model: Any, model_name: str, 
                                       data: pd.DataFrame, X_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate confidence for individual model prediction."""
        try:
            # Simple confidence based on recent data quality and model type
            data_quality = self._assess_data_quality(data)
            
            # Model-specific confidence adjustments
            model_confidence_factors = {
                'lstm': 0.8,  # Good for trends
                'transformer': 0.85,  # Good for long-term patterns
                'xgboost': 0.75  # Good for non-linear patterns
            }
            
            base_confidence = model_confidence_factors.get(model_name, 0.7)
            adjusted_confidence = base_confidence * data_quality
            
            # Estimate confidence interval (simplified)
            current_price = float(data['close'].iloc[-1])
            recent_volatility = data['close'].tail(20).pct_change().std()
            
            confidence_width = current_price * recent_volatility * 2.0
            
            confidence_interval = {
                'lower': current_price - confidence_width,
                'upper': current_price + confidence_width
            }
            
            # Quality based on confidence
            if adjusted_confidence > 0.8:
                quality = 'HIGH'
            elif adjusted_confidence > 0.6:
                quality = 'MEDIUM'
            else:
                quality = 'LOW'
            
            return {
                'confidence_interval': confidence_interval,
                'quality': quality,
                'confidence_score': float(adjusted_confidence)
            }
            
        except Exception as e:
            logger.error(f"Error calculating individual confidence: {e}")
            return {
                'confidence_interval': {'lower': 0, 'upper': 0},
                'quality': 'LOW',
                'confidence_score': 0.0
            }
    
    def _assess_data_quality(self, data: pd.DataFrame) -> float:
        """Assess data quality (0-1 score)."""
        try:
            quality_score = 1.0
            
            # Check for missing values
            missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
            quality_score *= (1 - missing_ratio)
            
            # Check data recency and completeness
            if len(data) < 100:
                quality_score *= 0.7
            elif len(data) < 50:
                quality_score *= 0.5
            
            # Check for data consistency
            if 'close' in data.columns:
                price_changes = data['close'].pct_change().abs()
                extreme_changes = (price_changes > 0.2).sum()  # >20% moves
                if extreme_changes > len(data) * 0.1:  # More than 10% extreme moves
                    quality_score *= 0.8
            
            return max(0.1, min(1.0, quality_score))
            
        except Exception as e:
            logger.error(f"Error assessing data quality: {e}")
            return 0.5  # Neutral score
    
    def _detect_market_regime(self, data: pd.DataFrame) -> str:
        """Simple market regime detection."""
        try:
            if 'close' in data.columns and len(data) >= 50:
                prices = data['close'].tail(50)
                
                # Calculate trend
                price_change = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
                
                # Calculate volatility
                volatility = prices.pct_change().std()
                
                if price_change > 0.1 and volatility < 0.05:
                    return 'bullish_stable'
                elif price_change > 0.05:
                    return 'bullish'
                elif price_change < -0.1 and volatility < 0.05:
                    return 'bearish_stable'
                elif price_change < -0.05:
                    return 'bearish'
                elif volatility > 0.08:
                    return 'volatile'
                else:
                    return 'sideways'
            
            return 'unknown'
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return 'unknown'
    
    def _get_best_performing_model(self, symbol: str) -> str:
        """Get best performing model for a symbol based on history."""
        try:
            # Simple implementation - in practice, you'd track performance per symbol
            if self.ensemble_model.is_trained:
                return 'ensemble'
            elif self.individual_models['transformer'].is_trained:
                return 'transformer'
            elif self.individual_models['lstm'].is_trained:
                return 'lstm'
            else:
                return 'xgboost'
                
        except Exception as e:
            logger.error(f"Error getting best performing model: {e}")
            return 'ensemble'
    
    async def _get_prediction_data(self, symbol: str) -> pd.DataFrame:
        """Get historical data with features for prediction."""
        try:
            # Get recent historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)  # 1 year of data for ensemble models
            
            # Retrieve historical data
            historical_data = await self.data_manager.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                limit=2000
            )
            
            if historical_data.empty:
                logger.warning(f"No historical data found for {symbol}")
                return pd.DataFrame()
            
            # Generate features if not already present
            feature_columns = [col for col in historical_data.columns 
                             if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'source', 'interval']]
            
            if len(feature_columns) < 15:  # Need sufficient features for ensemble
                logger.info(f"Generating features for {symbol}")
                historical_data = self.feature_engine.generate_all_features(historical_data)
            
            # Clean data
            historical_data = historical_data.dropna()
            
            if len(historical_data) < 200:  # Ensemble needs more data
                logger.warning(f"Insufficient clean data for {symbol}: {len(historical_data)} rows")
                return pd.DataFrame()
            
            return historical_data.sort_values('timestamp')
            
        except Exception as e:
            logger.error(f"Error getting prediction data for {symbol}: {e}")
            return pd.DataFrame()
    
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
        if len(self.prediction_cache) > 100:
            # Remove oldest cache entries
            oldest_key = min(self.cache_timestamps.keys(), 
                           key=lambda k: self.cache_timestamps[k])
            del self.prediction_cache[oldest_key]
            del self.cache_timestamps[oldest_key]
    
    async def get_ensemble_status(self) -> Dict[str, Any]:
        """Get status of ensemble predictor."""
        status = {
            'ensemble_available': self.ensemble_model.is_trained,
            'individual_models': {},
            'cache_stats': {
                'cached_predictions': len(self.prediction_cache),
                'cache_ttl_seconds': self.cache_ttl
            },
            'ensemble_config': self.ensemble_config.copy()
        }
        
        # Get individual model status
        for model_name, model in self.individual_models.items():
            status['individual_models'][model_name] = {
                'trained': model.is_trained,
                'info': model.get_model_info()
            }
        
        # Get ensemble model details if available
        if self.ensemble_model.is_trained:
            status['ensemble_details'] = self.ensemble_model.get_model_info()
        
        return status