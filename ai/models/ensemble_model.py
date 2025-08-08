# Ensemble Model for BYJY-Trader
# Phase 3.1 - Intelligent fusion of LSTM, Transformer, and XGBoost models

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import pickle
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

from .base_model import BaseModel
from .lstm_model import LSTMModel
from .transformer_model import TransformerModel
from .xgboost_model import XGBoostModel
from core.logger import get_logger

logger = get_logger(__name__)

class EnsembleModel(BaseModel):
    """
    Ensemble model that combines LSTM, Transformer, and XGBoost predictions.
    Uses intelligent weighting and meta-learning for optimal fusion.
    """
    
    def __init__(self, version: str = "3.1.0"):
        """Initialize Ensemble model."""
        super().__init__("ensemble", version)
        
        # Base models
        self.models = {
            'lstm': LSTMModel(version),
            'transformer': TransformerModel(version), 
            'xgboost': XGBoostModel(version)
        }
        
        # Ensemble configuration
        self.model_config = {
            'fusion_method': 'weighted_average',  # 'simple_average', 'weighted_average', 'stacking'
            'weight_learning': 'performance_based',  # 'equal', 'performance_based', 'adaptive'
            'meta_model': 'linear',  # 'linear', 'rf', 'xgb'
            'validation_window': 100,  # Window for performance evaluation
            'weight_update_frequency': 50,  # How often to update weights
            'min_samples_for_training': 200
        }
        
        # Model weights (initialized equally, updated based on performance)
        self.model_weights = {
            'lstm': 1/3,
            'transformer': 1/3,
            'xgboost': 1/3
        }
        
        # Meta-learner for stacking
        self.meta_model = None
        self.meta_model_trained = False
        
        # Performance tracking
        self.model_performances = {
            'lstm': {'rmse': [], 'mae': [], 'directional_accuracy': []},
            'transformer': {'rmse': [], 'mae': [], 'directional_accuracy': []},
            'xgboost': {'rmse': [], 'mae': [], 'directional_accuracy': []}
        }
        
        # Prediction history for adaptive weighting
        self.prediction_history = []
        self.performance_window = []
        
        logger.info("Ensemble Model initialized with LSTM, Transformer, and XGBoost")
    
    def build_model(self, input_shape: Tuple[int, ...], **kwargs) -> None:
        """
        Build ensemble model by initializing all base models.
        
        Args:
            input_shape: Shape of input data
            **kwargs: Additional model parameters
        """
        try:
            # Update config
            self.model_config.update(kwargs)
            
            # Build base models
            logger.info("Building base models...")
            
            # LSTM and Transformer use sequence data
            if len(input_shape) == 2:  # (sequence_length, n_features)
                self.models['lstm'].build_model(input_shape)
                self.models['transformer'].build_model(input_shape)
                
                # XGBoost uses flattened features
                xgb_input_shape = (input_shape[0] * input_shape[1],)
                self.models['xgboost'].build_model(xgb_input_shape)
            else:
                # Assume flat features for all models
                for model_name, model in self.models.items():
                    model.build_model(input_shape)
            
            # Initialize meta-model if using stacking
            if self.model_config['fusion_method'] == 'stacking':
                self._initialize_meta_model()
            
            logger.info("Ensemble model built successfully")
            
        except Exception as e:
            logger.error(f"Error building Ensemble model: {e}")
            raise
    
    def prepare_data(self, df: pd.DataFrame, target_column: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for ensemble training.
        Each base model may have different data requirements.
        
        Args:
            df: DataFrame with features and target
            target_column: Name of target column
            
        Returns:
            Tuple: (X_dict, y) where X_dict contains data for each model
        """
        try:
            logger.info("Preparing data for ensemble models...")
            
            # Prepare data for each base model
            X_dict = {}
            y_dict = {}
            
            # LSTM data preparation (sequence data)
            try:
                X_lstm, y_lstm = self.models['lstm'].prepare_data(df, target_column)
                X_dict['lstm'] = X_lstm
                y_dict['lstm'] = y_lstm
                logger.info(f"LSTM data: {X_lstm.shape}")
            except Exception as e:
                logger.warning(f"Failed to prepare LSTM data: {e}")
                X_dict['lstm'] = None
                y_dict['lstm'] = None
            
            # Transformer data preparation (sequence data)
            try:
                X_transformer, y_transformer = self.models['transformer'].prepare_data(df, target_column)
                X_dict['transformer'] = X_transformer
                y_dict['transformer'] = y_transformer
                logger.info(f"Transformer data: {X_transformer.shape}")
            except Exception as e:
                logger.warning(f"Failed to prepare Transformer data: {e}")
                X_dict['transformer'] = None
                y_dict['transformer'] = None
            
            # XGBoost data preparation (tabular data)
            try:
                X_xgb, y_xgb = self.models['xgboost'].prepare_data(df, target_column)
                X_dict['xgboost'] = X_xgb
                y_dict['xgboost'] = y_xgb
                logger.info(f"XGBoost data: {X_xgb.shape}")
            except Exception as e:
                logger.warning(f"Failed to prepare XGBoost data: {e}")
                X_dict['xgboost'] = None
                y_dict['xgboost'] = None
            
            # Use the target from any available model (they should be the same)
            y = None
            for model_name in ['lstm', 'transformer', 'xgboost']:
                if y_dict[model_name] is not None:
                    y = y_dict[model_name]
                    break
            
            if y is None:
                raise ValueError("No model could prepare data successfully")
            
            return X_dict, y
            
        except Exception as e:
            logger.error(f"Error preparing ensemble data: {e}")
            raise
    
    def train(self, X: Union[Dict, np.ndarray], y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Train the ensemble model.
        
        Args:
            X: Dictionary of model inputs or single array
            y: Training targets
            **kwargs: Training parameters
            
        Returns:
            Dict: Training results and metrics
        """
        try:
            # Update training config
            train_config = self.model_config.copy()
            train_config.update(kwargs)
            
            # Convert single array to dict if needed
            if isinstance(X, np.ndarray):
                logger.warning("Single array input provided, using for all models")
                X = {'lstm': X, 'transformer': X, 'xgboost': X}
            
            logger.info("Starting ensemble training...")
            
            # Train base models
            training_results = {}
            successful_models = []
            
            for model_name, model in self.models.items():
                if X.get(model_name) is not None:
                    try:
                        logger.info(f"Training {model_name} model...")
                        
                        # Adjust y length to match X if needed
                        X_model = X[model_name]
                        if len(X_model) != len(y):
                            # Use the shorter length
                            min_len = min(len(X_model), len(y))
                            X_model = X_model[-min_len:]
                            y_model = y[-min_len:]
                        else:
                            y_model = y
                        
                        result = model.train(X_model, y_model, **train_config)
                        training_results[model_name] = result
                        successful_models.append(model_name)
                        
                        logger.info(f"{model_name} training completed successfully")
                        
                    except Exception as e:
                        logger.error(f"Failed to train {model_name}: {e}")
                        training_results[model_name] = {'error': str(e)}
                else:
                    logger.warning(f"No data available for {model_name}")
                    training_results[model_name] = {'error': 'No data available'}
            
            if not successful_models:
                raise ValueError("No base models trained successfully")
            
            # Update model weights based on training performance
            self._update_model_weights_from_training(training_results)
            
            # Train meta-model if using stacking
            if (self.model_config['fusion_method'] == 'stacking' and 
                len(successful_models) >= 2):
                
                logger.info("Training meta-model for stacking...")
                self._train_meta_model(X, y, successful_models)
            
            self.is_trained = True
            
            # Evaluate ensemble performance
            ensemble_metrics = self._evaluate_ensemble(X, y, successful_models)
            
            # Combine results
            final_results = {
                'training_completed': True,
                'successful_models': successful_models,
                'failed_models': [name for name in self.models.keys() if name not in successful_models],
                'model_results': training_results,
                'ensemble_weights': self.model_weights.copy(),
                'ensemble_metrics': ensemble_metrics,
                'fusion_method': self.model_config['fusion_method'],
                'meta_model_trained': self.meta_model_trained
            }
            
            logger.info(f"Ensemble training completed. Successful models: {successful_models}")
            logger.info(f"Final weights: {self.model_weights}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error training ensemble model: {e}")
            raise
    
    def predict(self, X: Union[Dict, np.ndarray], **kwargs) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X: Dictionary of model inputs or single array
            **kwargs: Prediction parameters
            
        Returns:
            np.ndarray: Ensemble predictions
        """
        try:
            if not self.is_trained:
                raise ValueError("Ensemble model not trained")
            
            # Convert single array to dict if needed
            if isinstance(X, np.ndarray):
                X = {'lstm': X, 'transformer': X, 'xgboost': X}
            
            # Get predictions from each model
            predictions = {}
            successful_predictions = []
            
            for model_name, model in self.models.items():
                if (model.is_trained and X.get(model_name) is not None and 
                    self.model_weights.get(model_name, 0) > 0):
                    try:
                        pred = model.predict(X[model_name])
                        predictions[model_name] = pred
                        successful_predictions.append(model_name)
                        
                    except Exception as e:
                        logger.warning(f"Failed to get predictions from {model_name}: {e}")
            
            if not successful_predictions:
                raise ValueError("No model predictions available")
            
            # Combine predictions using selected fusion method
            ensemble_prediction = self._fuse_predictions(predictions, successful_predictions)
            
            # Update prediction history for adaptive weighting
            self._update_prediction_history(predictions, ensemble_prediction)
            
            return ensemble_prediction
            
        except Exception as e:
            logger.error(f"Error making ensemble predictions: {e}")
            raise
    
    def _fuse_predictions(self, predictions: Dict[str, np.ndarray], 
                         model_names: List[str]) -> np.ndarray:
        """
        Fuse predictions from multiple models.
        
        Args:
            predictions: Dictionary of model predictions
            model_names: List of available model names
            
        Returns:
            np.ndarray: Fused predictions
        """
        try:
            fusion_method = self.model_config['fusion_method']
            
            if fusion_method == 'simple_average':
                # Simple average
                pred_array = np.array([predictions[name] for name in model_names])
                return np.mean(pred_array, axis=0)
            
            elif fusion_method == 'weighted_average':
                # Weighted average based on model weights
                weighted_sum = np.zeros_like(predictions[model_names[0]])
                total_weight = 0
                
                for name in model_names:
                    weight = self.model_weights.get(name, 0)
                    if weight > 0:
                        weighted_sum += weight * predictions[name]
                        total_weight += weight
                
                if total_weight > 0:
                    return weighted_sum / total_weight
                else:
                    # Fallback to simple average
                    pred_array = np.array([predictions[name] for name in model_names])
                    return np.mean(pred_array, axis=0)
            
            elif fusion_method == 'stacking' and self.meta_model_trained:
                # Meta-model stacking
                # Stack predictions as features for meta-model
                stacked_features = np.column_stack([predictions[name] for name in model_names])
                return self.meta_model.predict(stacked_features)
            
            else:
                # Fallback to weighted average
                logger.warning(f"Fusion method {fusion_method} not available, using weighted average")
                return self._fuse_predictions(predictions, model_names)
            
        except Exception as e:
            logger.error(f"Error fusing predictions: {e}")
            # Fallback to simple average
            pred_array = np.array([predictions[name] for name in model_names])
            return np.mean(pred_array, axis=0)
    
    def _initialize_meta_model(self):
        """Initialize meta-model for stacking."""
        try:
            meta_type = self.model_config.get('meta_model', 'linear')
            
            if meta_type == 'linear':
                self.meta_model = LinearRegression()
            elif meta_type == 'rf':
                self.meta_model = RandomForestRegressor(
                    n_estimators=100, 
                    random_state=42,
                    n_jobs=-1
                )
            else:  # xgb
                import xgboost as xgb
                self.meta_model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=42
                )
            
            logger.info(f"Meta-model initialized: {meta_type}")
            
        except Exception as e:
            logger.error(f"Error initializing meta-model: {e}")
            self.meta_model = LinearRegression()  # Fallback
    
    def _train_meta_model(self, X: Dict, y: np.ndarray, model_names: List[str]):
        """Train meta-model for stacking."""
        try:
            if self.meta_model is None:
                self._initialize_meta_model()
            
            # Generate base model predictions for meta-training
            meta_features = []
            
            for model_name in model_names:
                if X.get(model_name) is not None:
                    # Use cross-validation predictions to avoid overfitting
                    model = self.models[model_name]
                    X_model = X[model_name]
                    
                    # Align data lengths
                    min_len = min(len(X_model), len(y))
                    X_model = X_model[-min_len:]
                    y_aligned = y[-min_len:]
                    
                    # Simple time series split for CV predictions
                    split_point = len(X_model) // 2
                    X_train, X_val = X_model[:split_point], X_model[split_point:]
                    y_train, y_val = y_aligned[:split_point], y_aligned[split_point:]
                    
                    # Train on first half, predict on second half
                    temp_model = type(model)(self.version)
                    temp_model.build_model(X_train.shape[1:] if len(X_train.shape) > 1 else (X_train.shape[1],))
                    temp_model.train(X_train, y_train, epochs=20, verbose=0)  # Quick training
                    
                    cv_predictions = temp_model.predict(X_val)
                    meta_features.append(cv_predictions)
            
            if meta_features:
                # Stack features and train meta-model
                stacked_features = np.column_stack(meta_features)
                y_meta = y[-len(stacked_features):]  # Align target
                
                self.meta_model.fit(stacked_features, y_meta)
                self.meta_model_trained = True
                
                logger.info("Meta-model trained successfully")
            else:
                logger.warning("No meta-features available for training")
                
        except Exception as e:
            logger.error(f"Error training meta-model: {e}")
            self.meta_model_trained = False
    
    def _update_model_weights_from_training(self, training_results: Dict[str, Any]):
        """Update model weights based on training performance."""
        try:
            if self.model_config['weight_learning'] == 'equal':
                return  # Keep equal weights
            
            # Extract validation metrics
            model_scores = {}
            for model_name, result in training_results.items():
                if 'error' not in result and 'validation_metrics' in result:
                    metrics = result['validation_metrics']
                    # Use inverse RMSE as score (higher is better)
                    rmse = metrics.get('rmse', float('inf'))
                    if rmse > 0:
                        model_scores[model_name] = 1.0 / rmse
                    else:
                        model_scores[model_name] = 1.0
            
            if model_scores:
                # Normalize scores to weights
                total_score = sum(model_scores.values())
                for model_name in model_scores:
                    self.model_weights[model_name] = model_scores[model_name] / total_score
                
                logger.info(f"Updated model weights based on training performance: {self.model_weights}")
            
        except Exception as e:
            logger.error(f"Error updating model weights: {e}")
    
    def _evaluate_ensemble(self, X: Dict, y: np.ndarray, model_names: List[str]) -> Dict[str, float]:
        """Evaluate ensemble performance."""
        try:
            # Get ensemble predictions
            ensemble_pred = self.predict(X)
            
            # Align data length
            min_len = min(len(ensemble_pred), len(y))
            ensemble_pred = ensemble_pred[-min_len:]
            y_eval = y[-min_len:]
            
            # Calculate metrics
            mse = mean_squared_error(y_eval, ensemble_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_eval, ensemble_pred)
            
            # Directional accuracy
            if len(y_eval) > 1:
                actual_direction = np.sign(np.diff(y_eval))
                pred_direction = np.sign(np.diff(ensemble_pred))
                directional_accuracy = np.mean(actual_direction == pred_direction)
            else:
                directional_accuracy = 0.0
            
            return {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'directional_accuracy': float(directional_accuracy)
            }
            
        except Exception as e:
            logger.error(f"Error evaluating ensemble: {e}")
            return {}
    
    def _update_prediction_history(self, predictions: Dict[str, np.ndarray], 
                                  ensemble_pred: np.ndarray):
        """Update prediction history for adaptive weighting."""
        try:
            history_entry = {
                'timestamp': datetime.now().isoformat(),
                'predictions': {name: pred[-1] if len(pred) > 0 else 0 
                              for name, pred in predictions.items()},
                'ensemble': ensemble_pred[-1] if len(ensemble_pred) > 0 else 0
            }
            
            self.prediction_history.append(history_entry)
            
            # Keep only recent history
            max_history = 1000
            if len(self.prediction_history) > max_history:
                self.prediction_history = self.prediction_history[-max_history:]
                
        except Exception as e:
            logger.error(f"Error updating prediction history: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get ensemble model information."""
        base_info = super().get_model_info()
        
        ensemble_info = {
            'base_models': {},
            'model_weights': self.model_weights.copy(),
            'fusion_method': self.model_config['fusion_method'],
            'meta_model_trained': self.meta_model_trained,
            'prediction_history_length': len(self.prediction_history)
        }
        
        # Get info from base models
        for name, model in self.models.items():
            ensemble_info['base_models'][name] = model.get_model_info()
        
        base_info.update(ensemble_info)
        return base_info
    
    def _load_model_implementation(self) -> None:
        """Load ensemble model from disk."""
        try:
            # Load base models
            for name, model in self.models.items():
                try:
                    model.load_model()
                    logger.info(f"Loaded {name} model")
                except Exception as e:
                    logger.warning(f"Failed to load {name} model: {e}")
            
            # Load ensemble-specific data
            ensemble_data_path = self.models_dir / f"{self.model_name}_{self.version}_ensemble_data.pkl"
            if ensemble_data_path.exists():
                with open(ensemble_data_path, 'rb') as f:
                    ensemble_data = pickle.load(f)
                    self.model_weights = ensemble_data.get('model_weights', self.model_weights)
                    self.prediction_history = ensemble_data.get('prediction_history', [])
                    self.meta_model_trained = ensemble_data.get('meta_model_trained', False)
            
            # Load meta-model if exists
            meta_model_path = self.models_dir / f"{self.model_name}_{self.version}_meta_model.pkl"
            if meta_model_path.exists() and self.meta_model_trained:
                with open(meta_model_path, 'rb') as f:
                    self.meta_model = pickle.load(f)
                    
        except Exception as e:
            logger.error(f"Error loading ensemble model: {e}")
            raise
    
    def save_model(self, include_optimizer: bool = False) -> bool:
        """Save ensemble model."""
        try:
            # Save base models
            success_count = 0
            for name, model in self.models.items():
                if model.save_model(include_optimizer):
                    success_count += 1
                    logger.info(f"Saved {name} model")
                else:
                    logger.warning(f"Failed to save {name} model")
            
            if success_count == 0:
                logger.error("No base models saved")
                return False
            
            # Save ensemble configuration
            success = super().save_model(include_optimizer)
            
            if success:
                # Save ensemble-specific data
                ensemble_data = {
                    'model_weights': self.model_weights,
                    'prediction_history': self.prediction_history[-100:],  # Keep last 100
                    'meta_model_trained': self.meta_model_trained
                }
                
                ensemble_data_path = self.models_dir / f"{self.model_name}_{self.version}_ensemble_data.pkl"
                with open(ensemble_data_path, 'wb') as f:
                    pickle.dump(ensemble_data, f)
                
                # Save meta-model if trained
                if self.meta_model_trained and self.meta_model is not None:
                    meta_model_path = self.models_dir / f"{self.model_name}_{self.version}_meta_model.pkl"
                    with open(meta_model_path, 'wb') as f:
                        pickle.dump(self.meta_model, f)
                
                logger.info("Ensemble model saved successfully")
            
            return success
            
        except Exception as e:
            logger.error(f"Error saving ensemble model: {e}")
            return False