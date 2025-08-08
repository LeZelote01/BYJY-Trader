# Base Model Class for BYJY-Trader AI
# Phase 2.2 - Abstract base for all AI models

import os
import json
import pickle
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
from pathlib import Path

from core.path_utils import get_project_root
from core.logger import get_logger

logger = get_logger(__name__)

class BaseModel(ABC):
    """
    Abstract base class for all AI models in BYJY-Trader.
    Provides common interface and functionality for model operations.
    """
    
    def __init__(self, model_name: str, version: str = "2.2.0"):
        """
        Initialize base model.
        
        Args:
            model_name: Name of the model
            version: Model version
        """
        self.model_name = model_name
        self.version = version
        self.model = None
        self.is_trained = False
        self.model_config = {}
        self.training_history = {}
        
        # Model storage paths
        self.models_dir = get_project_root() / "ai" / "trained_models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_path = self.models_dir / f"{model_name}_{version}.h5"
        self.config_path = self.models_dir / f"{model_name}_{version}_config.json"
        self.history_path = self.models_dir / f"{model_name}_{version}_history.pkl"
        
        logger.info(f"Initialized {model_name} model v{version}")
    
    @abstractmethod
    def build_model(self, input_shape: Tuple[int, ...], **kwargs) -> None:
        """
        Build the model architecture.
        
        Args:
            input_shape: Shape of input data
            **kwargs: Additional model parameters
        """
        pass
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X: Training features
            y: Training targets
            **kwargs: Training parameters
            
        Returns:
            Dict: Training results and metrics
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
            **kwargs: Prediction parameters
            
        Returns:
            np.ndarray: Predictions
        """
        pass
    
    def save_model(self, include_optimizer: bool = False) -> bool:
        """
        Save model to disk.
        
        Args:
            include_optimizer: Whether to include optimizer state
            
        Returns:
            bool: Success status
        """
        try:
            if self.model is None:
                logger.error("No model to save")
                return False
            
            # Save model weights/architecture
            if hasattr(self.model, 'save'):
                self.model.save(self.model_path, include_optimizer=include_optimizer)
            else:
                with open(self.model_path, 'wb') as f:
                    pickle.dump(self.model, f)
            
            # Save configuration
            config = {
                'model_name': self.model_name,
                'version': self.version,
                'is_trained': self.is_trained,
                'model_config': self.model_config,
                'saved_at': datetime.now().isoformat()
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Save training history
            if self.training_history:
                with open(self.history_path, 'wb') as f:
                    pickle.dump(self.training_history, f)
            
            logger.info(f"Model {self.model_name} saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self) -> bool:
        """
        Load model from disk.
        
        Returns:
            bool: Success status
        """
        try:
            if not self.model_path.exists():
                logger.warning(f"Model file not found: {self.model_path}")
                return False
            
            # Load configuration
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.model_config = config.get('model_config', {})
                    self.is_trained = config.get('is_trained', False)
            
            # Load training history
            if self.history_path.exists():
                with open(self.history_path, 'rb') as f:
                    self.training_history = pickle.load(f)
            
            # Load model (implementation depends on model type)
            self._load_model_implementation()
            
            logger.info(f"Model {self.model_name} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    @abstractmethod
    def _load_model_implementation(self) -> None:
        """Load model specific implementation."""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dict: Model information
        """
        return {
            'model_name': self.model_name,
            'version': self.version,
            'is_trained': self.is_trained,
            'model_exists': self.model is not None,
            'config': self.model_config,
            'model_path': str(self.model_path),
            'config_path': str(self.config_path),
            'history_available': bool(self.training_history)
        }
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dict: Evaluation metrics
        """
        if self.model is None or not self.is_trained:
            logger.error("Model not trained")
            return {}
        
        try:
            predictions = self.predict(X_test)
            
            # Calculate common metrics
            mse = np.mean((predictions - y_test) ** 2)
            mae = np.mean(np.abs(predictions - y_test))
            rmse = np.sqrt(mse)
            
            # Directional accuracy
            actual_direction = np.sign(np.diff(y_test.flatten()))
            pred_direction = np.sign(np.diff(predictions.flatten()))
            directional_accuracy = np.mean(actual_direction == pred_direction)
            
            metrics = {
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(rmse),
                'directional_accuracy': float(directional_accuracy)
            }
            
            logger.info(f"Model evaluation completed: RMSE={rmse:.4f}, DA={directional_accuracy:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {}
    
    def predict_with_confidence(self, X: np.ndarray, n_predictions: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with confidence intervals (Monte Carlo Dropout).
        
        Args:
            X: Input features
            n_predictions: Number of Monte Carlo samples
            
        Returns:
            Tuple: (predictions, confidence_intervals)
        """
        try:
            predictions = []
            
            for _ in range(n_predictions):
                pred = self.predict(X)
                predictions.append(pred)
            
            predictions = np.array(predictions)
            mean_pred = np.mean(predictions, axis=0)
            std_pred = np.std(predictions, axis=0)
            
            # 95% confidence interval
            confidence_interval = 1.96 * std_pred
            
            return mean_pred, confidence_interval
            
        except Exception as e:
            logger.error(f"Error in confidence prediction: {e}")
            return self.predict(X), np.zeros_like(self.predict(X))
    
    def get_feature_importance(self, X: np.ndarray, method: str = 'permutation') -> Dict[str, float]:
        """
        Calculate feature importance.
        
        Args:
            X: Input features
            method: Importance calculation method
            
        Returns:
            Dict: Feature importance scores
        """
        if not self.is_trained:
            return {}
        
        try:
            # Basic implementation - can be overridden by specific models
            baseline_pred = self.predict(X)
            importance_scores = {}
            
            for i in range(X.shape[-1]):  # Last dimension is features
                X_permuted = X.copy()
                np.random.shuffle(X_permuted[:, :, i])  # Permute feature i
                permuted_pred = self.predict(X_permuted)
                
                # Calculate performance degradation
                mse_baseline = np.mean((baseline_pred - X[:, -1, 0]) ** 2)  # Assume last price as target
                mse_permuted = np.mean((permuted_pred - X[:, -1, 0]) ** 2)
                
                importance_scores[f'feature_{i}'] = float(mse_permuted - mse_baseline)
            
            return importance_scores
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            return {}