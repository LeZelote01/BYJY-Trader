# XGBoost Model for BYJY-Trader
# Phase 3.1 - Gradient Boosting model for financial time series prediction

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import pickle

from .base_model import BaseModel
from core.logger import get_logger

logger = get_logger(__name__)

class XGBoostModel(BaseModel):
    """
    XGBoost model for cryptocurrency and stock price prediction.
    Optimized for capturing non-linear patterns and feature interactions.
    """
    
    def __init__(self, version: str = "3.1.0"):
        """Initialize XGBoost model."""
        super().__init__("xgboost", version)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.lookback_window = 60  # Features lookback window
        self.n_features = None
        
        # Model hyperparameters
        self.model_config = {
            'lookback_window': self.lookback_window,
            'n_estimators': 1000,
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'colsample_bylevel': 0.8,
            'min_child_weight': 3,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'gamma': 0.1,
            'random_state': 42,
            'early_stopping_rounds': 50,
            'eval_metric': 'rmse',
            'objective': 'reg:squarederror',
            'tree_method': 'hist',
            'device': 'cpu'
        }
    
    def build_model(self, input_shape: Tuple[int, ...], **kwargs) -> None:
        """
        Build XGBoost model.
        
        Args:
            input_shape: Shape of input data (n_features,)
            **kwargs: Additional model parameters
        """
        try:
            # Update config with kwargs
            self.model_config.update(kwargs)
            
            # Create XGBoost regressor
            self.model = xgb.XGBRegressor(
                n_estimators=self.model_config['n_estimators'],
                max_depth=self.model_config['max_depth'],
                learning_rate=self.model_config['learning_rate'],
                subsample=self.model_config['subsample'],
                colsample_bytree=self.model_config['colsample_bytree'],
                colsample_bylevel=self.model_config['colsample_bylevel'],
                min_child_weight=self.model_config['min_child_weight'],
                reg_alpha=self.model_config['reg_alpha'],
                reg_lambda=self.model_config['reg_lambda'],
                gamma=self.model_config['gamma'],
                random_state=self.model_config['random_state'],
                eval_metric=self.model_config['eval_metric'],
                objective=self.model_config['objective'],
                tree_method=self.model_config['tree_method'],
                device=self.model_config['device'],
                n_jobs=-1
            )
            
            logger.info(f"XGBoost model built successfully")
            
        except Exception as e:
            logger.error(f"Error building XGBoost model: {e}")
            raise
    
    def prepare_data(self, df: pd.DataFrame, target_column: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for XGBoost training.
        Creates lag features and technical indicators.
        
        Args:
            df: DataFrame with features and target
            target_column: Name of target column
            
        Returns:
            Tuple: (X, y) prepared for XGBoost
        """
        try:
            # Select base features (excluding timestamp and target)
            base_features = [col for col in df.columns 
                           if col not in ['timestamp', target_column]]
            
            if not base_features:
                raise ValueError("No feature columns found")
            
            # Create lag features and engineered features
            df_features = self._create_lag_features(df, base_features, target_column)
            
            # Extract final features and target
            feature_columns = [col for col in df_features.columns 
                             if col not in ['timestamp', target_column]]
            
            X_raw = df_features[feature_columns].values
            y_raw = df_features[target_column].values
            
            # Handle missing values
            if np.any(np.isnan(X_raw)) or np.any(np.isnan(y_raw)):
                logger.warning("Missing values detected, forward filling")
                df_clean = df_features[feature_columns + [target_column]].fillna(method='ffill')
                X_raw = df_clean[feature_columns].values
                y_raw = df_clean[target_column].values
            
            # Remove any remaining NaN rows
            mask = ~(np.isnan(X_raw).any(axis=1) | np.isnan(y_raw))
            X_raw = X_raw[mask]
            y_raw = y_raw[mask]
            
            # Scale features and target
            X_scaled = self.scaler_X.fit_transform(X_raw)
            y_scaled = self.scaler_y.fit_transform(y_raw.reshape(-1, 1)).flatten()
            
            self.n_features = X_scaled.shape[1]
            self.feature_names = feature_columns
            
            logger.info(f"XGBoost data prepared: {X_scaled.shape} X, {y_scaled.shape} y")
            logger.info(f"Features: {self.n_features} total features")
            
            return X_scaled, y_scaled
            
        except Exception as e:
            logger.error(f"Error preparing XGBoost data: {e}")
            raise
    
    def _create_lag_features(self, df: pd.DataFrame, feature_columns: List[str], target_column: str) -> pd.DataFrame:
        """
        Create lag features and additional technical indicators.
        
        Args:
            df: Original dataframe
            feature_columns: Base feature columns
            target_column: Target column name
            
        Returns:
            pd.DataFrame: DataFrame with lag features
        """
        df_features = df.copy()
        
        # Price-based features
        if target_column in df.columns:
            close_prices = df[target_column]
            
            # Price lags
            for lag in [1, 2, 3, 5, 10, 20, 30]:
                df_features[f'{target_column}_lag_{lag}'] = close_prices.shift(lag)
            
            # Price changes and returns
            df_features[f'{target_column}_return_1d'] = close_prices.pct_change(1)
            df_features[f'{target_column}_return_5d'] = close_prices.pct_change(5)
            df_features[f'{target_column}_return_20d'] = close_prices.pct_change(20)
            
            # Rolling statistics
            for window in [5, 10, 20, 50]:
                df_features[f'{target_column}_ma_{window}'] = close_prices.rolling(window).mean()
                df_features[f'{target_column}_std_{window}'] = close_prices.rolling(window).std()
                df_features[f'{target_column}_min_{window}'] = close_prices.rolling(window).min()
                df_features[f'{target_column}_max_{window}'] = close_prices.rolling(window).max()
                
                # Position relative to rolling stats
                df_features[f'{target_column}_vs_ma_{window}'] = (close_prices - df_features[f'{target_column}_ma_{window}']) / df_features[f'{target_column}_ma_{window}']
        
        # Volume features (if available)
        if 'volume' in df.columns:
            volume = df['volume']
            
            # Volume lags and changes
            for lag in [1, 2, 5, 10]:
                df_features[f'volume_lag_{lag}'] = volume.shift(lag)
            
            df_features['volume_change_1d'] = volume.pct_change(1)
            df_features['volume_change_5d'] = volume.pct_change(5)
            
            # Volume moving averages
            for window in [5, 10, 20]:
                df_features[f'volume_ma_{window}'] = volume.rolling(window).mean()
                df_features[f'volume_vs_ma_{window}'] = volume / df_features[f'volume_ma_{window}']
        
        # Technical indicator lags (if available)
        technical_indicators = ['rsi', 'macd', 'bb_upper', 'bb_lower', 'bb_middle', 
                               'ema_12', 'ema_26', 'sma_20', 'sma_50']
        
        for indicator in technical_indicators:
            if indicator in df.columns:
                # Add lags for technical indicators
                for lag in [1, 2, 5]:
                    df_features[f'{indicator}_lag_{lag}'] = df[indicator].shift(lag)
                
                # Rate of change for indicators
                df_features[f'{indicator}_roc_5'] = df[indicator].pct_change(5)
        
        # Time-based features
        if 'timestamp' in df.columns:
            timestamps = pd.to_datetime(df['timestamp'])
            
            df_features['hour'] = timestamps.dt.hour
            df_features['day_of_week'] = timestamps.dt.dayofweek
            df_features['month'] = timestamps.dt.month
            df_features['quarter'] = timestamps.dt.quarter
            
            # Cyclical encoding for time features
            df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
            df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
            df_features['dow_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
            df_features['dow_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
        
        # Market microstructure features (if OHLCV available)
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # Price ranges and spreads
            df_features['price_range'] = (df['high'] - df['low']) / df['close']
            df_features['open_close_spread'] = (df['close'] - df['open']) / df['open']
            df_features['high_close_spread'] = (df['high'] - df['close']) / df['close']
            df_features['low_close_spread'] = (df['close'] - df['low']) / df['close']
            
            # Typical price and weighted price
            df_features['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
            if 'volume' in df.columns:
                df_features['vwap'] = (df_features['typical_price'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        
        # Remove the first 60 rows to ensure all lag features are available
        df_features = df_features.iloc[60:].reset_index(drop=True)
        
        return df_features
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Train the XGBoost model with time series cross-validation.
        
        Args:
            X: Training features
            y: Training targets
            **kwargs: Training parameters
            
        Returns:
            Dict: Training results and metrics
        """
        try:
            # Update training config
            train_config = self.model_config.copy()
            train_config.update(kwargs)
            
            # Build model if not built
            if self.model is None:
                input_shape = (X.shape[1],)
                self.build_model(input_shape)
            
            # Time series split for validation
            tscv = TimeSeriesSplit(n_splits=5)
            validation_scores = []
            
            # Use last 20% as validation set
            validation_split = train_config.get('validation_split', 0.2)
            split_idx = int(len(X) * (1 - validation_split))
            
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            logger.info(f"Starting XGBoost training: {len(X_train)} train, {len(X_val)} val samples")
            
            # Train model with early stopping
            eval_set = [(X_train, y_train), (X_val, y_val)]
            eval_names = ['train', 'val']
            
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=100,
                early_stopping_rounds=train_config.get('early_stopping_rounds', 50)
            )
            
            self.is_trained = True
            
            # Store training results
            self.training_history = {
                'evals_result': self.model.evals_result(),
                'best_iteration': self.model.best_iteration,
                'best_score': self.model.best_score
            }
            
            # Evaluate on validation set
            val_predictions = self.predict(X_val)
            val_metrics = self.evaluate_model(X_val, y_val)
            
            # Feature importance
            feature_importance = self.model.feature_importances_
            if hasattr(self, 'feature_names'):
                feature_importance_dict = dict(zip(self.feature_names, feature_importance))
                # Sort by importance
                sorted_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
                top_features = dict(sorted_importance[:20])  # Top 20 features
            else:
                top_features = {}
            
            # Training results
            results = {
                'training_completed': True,
                'best_iteration': int(self.model.best_iteration),
                'best_score': float(self.model.best_score),
                'n_estimators_used': int(self.model.best_iteration + 1),
                'validation_metrics': val_metrics,
                'feature_importance': top_features,
                'model_path': str(self.model_path)
            }
            
            logger.info(f"XGBoost training completed: Best RMSE={results['best_score']:.4f}")
            logger.info(f"Top features: {list(top_features.keys())[:5]}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error training XGBoost model: {e}")
            raise
    
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Make predictions with XGBoost model.
        
        Args:
            X: Input features
            **kwargs: Prediction parameters
            
        Returns:
            np.ndarray: Predictions in original scale
        """
        try:
            if self.model is None or not self.is_trained:
                raise ValueError("XGBoost model not trained")
            
            # Make predictions
            predictions_scaled = self.model.predict(X)
            
            # Scale back to original values
            predictions = self.scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1))
            
            return predictions.flatten()
            
        except Exception as e:
            logger.error(f"Error making XGBoost predictions: {e}")
            raise
    
    def predict_with_uncertainty(self, X: np.ndarray, n_estimators: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimation using tree variance.
        
        Args:
            X: Input features
            n_estimators: Number of estimators to use (None for all)
            
        Returns:
            Tuple: (predictions, uncertainties)
        """
        try:
            if not self.is_trained:
                raise ValueError("Model not trained")
            
            # Get predictions from individual trees
            if n_estimators is None:
                n_estimators = self.model.n_estimators
            
            # Use prediction intervals based on tree variance
            predictions = self.predict(X)
            
            # Simple uncertainty estimation based on feature variance
            # In a real implementation, you might use quantile regression or other methods
            feature_std = np.std(X, axis=0)
            uncertainty = np.mean(feature_std) * np.ones(len(predictions))
            
            return predictions, uncertainty
            
        except Exception as e:
            logger.error(f"Error predicting with uncertainty: {e}")
            return self.predict(X), np.zeros(len(X))
    
    def get_feature_importance(self, X: np.ndarray = None, method: str = 'gain') -> Dict[str, float]:
        """
        Get feature importance from XGBoost model.
        
        Args:
            X: Input features (not used for XGBoost)
            method: Importance type ('gain', 'weight', 'cover')
            
        Returns:
            Dict: Feature importance scores
        """
        try:
            if not self.is_trained:
                return {}
            
            importance_values = self.model.feature_importances_
            
            if hasattr(self, 'feature_names'):
                feature_importance = dict(zip(self.feature_names, importance_values))
            else:
                feature_importance = {f'feature_{i}': float(imp) 
                                   for i, imp in enumerate(importance_values)}
            
            # Sort by importance
            sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            return dict(sorted_importance)
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}
    
    def _load_model_implementation(self) -> None:
        """Load XGBoost model from disk."""
        try:
            # Load XGBoost model
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load scalers
            scaler_X_path = self.models_dir / f"{self.model_name}_{self.version}_scaler_X.pkl"
            scaler_y_path = self.models_dir / f"{self.model_name}_{self.version}_scaler_y.pkl"
            
            if scaler_X_path.exists():
                self.scaler_X = joblib.load(scaler_X_path)
            if scaler_y_path.exists():
                self.scaler_y = joblib.load(scaler_y_path)
                
            # Load feature names
            feature_names_path = self.models_dir / f"{self.model_name}_{self.version}_features.pkl"
            if feature_names_path.exists():
                with open(feature_names_path, 'rb') as f:
                    self.feature_names = pickle.load(f)
                
        except Exception as e:
            logger.error(f"Error loading XGBoost model: {e}")
            raise
    
    def save_model(self, include_optimizer: bool = False) -> bool:
        """
        Save XGBoost model with scalers and feature names.
        
        Args:
            include_optimizer: Not used for XGBoost
            
        Returns:
            bool: Success status
        """
        try:
            if self.model is None:
                logger.error("No model to save")
                return False
            
            # Save XGBoost model
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            # Save configuration
            config = {
                'model_name': self.model_name,
                'version': self.version,
                'is_trained': self.is_trained,
                'model_config': self.model_config,
                'saved_at': pd.Timestamp.now().isoformat()
            }
            
            with open(self.config_path, 'w') as f:
                import json
                json.dump(config, f, indent=2)
            
            # Save training history
            if self.training_history:
                with open(self.history_path, 'wb') as f:
                    pickle.dump(self.training_history, f)
            
            # Save scalers
            scaler_X_path = self.models_dir / f"{self.model_name}_{self.version}_scaler_X.pkl"
            scaler_y_path = self.models_dir / f"{self.model_name}_{self.version}_scaler_y.pkl"
            
            joblib.dump(self.scaler_X, scaler_X_path)
            joblib.dump(self.scaler_y, scaler_y_path)
            
            # Save feature names
            if hasattr(self, 'feature_names'):
                feature_names_path = self.models_dir / f"{self.model_name}_{self.version}_features.pkl"
                with open(feature_names_path, 'wb') as f:
                    pickle.dump(self.feature_names, f)
            
            logger.info("XGBoost model, scalers, and features saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving XGBoost model: {e}")
            return False