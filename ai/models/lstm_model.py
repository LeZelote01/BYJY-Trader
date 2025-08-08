# LSTM Model for BYJY-Trader
# Phase 2.2 - Long Short-Term Memory model for time series prediction

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import MinMaxScaler
import joblib

from .base_model import BaseModel
from core.logger import get_logger

logger = get_logger(__name__)

class LSTMModel(BaseModel):
    """
    LSTM model for cryptocurrency and stock price prediction.
    Uses multiple LSTM layers with dropout for regularization.
    """
    
    def __init__(self, version: str = "2.2.0"):
        """Initialize LSTM model."""
        super().__init__("lstm", version)
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.sequence_length = 60  # 60 time steps
        self.n_features = None
        
        # Model hyperparameters
        self.model_config = {
            'sequence_length': self.sequence_length,
            'lstm_units': [128, 64, 32],
            'dropout_rate': 0.3,
            'dense_units': [50, 25],
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'early_stopping_patience': 15,
            'reduce_lr_patience': 7
        }
    
    def build_model(self, input_shape: Tuple[int, ...], **kwargs) -> None:
        """
        Build LSTM model architecture.
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
            **kwargs: Additional model parameters
        """
        try:
            # Update config with kwargs
            self.model_config.update(kwargs)
            
            # Input layer
            inputs = keras.Input(shape=input_shape, name='lstm_input')
            x = inputs
            
            # LSTM layers
            lstm_units = self.model_config['lstm_units']
            dropout_rate = self.model_config['dropout_rate']
            
            for i, units in enumerate(lstm_units):
                return_sequences = i < len(lstm_units) - 1  # Return sequences for all but last layer
                
                x = layers.LSTM(
                    units=units,
                    return_sequences=return_sequences,
                    dropout=dropout_rate,
                    recurrent_dropout=dropout_rate,
                    name=f'lstm_{i+1}'
                )(x)
                
                if return_sequences:
                    x = layers.BatchNormalization(name=f'batch_norm_{i+1}')(x)
            
            # Dense layers
            dense_units = self.model_config['dense_units']
            for i, units in enumerate(dense_units):
                x = layers.Dense(
                    units=units,
                    activation='relu',
                    name=f'dense_{i+1}'
                )(x)
                x = layers.Dropout(dropout_rate, name=f'dropout_{i+1}')(x)
                x = layers.BatchNormalization(name=f'dense_batch_norm_{i+1}')(x)
            
            # Output layer - predicting next price
            outputs = layers.Dense(
                units=1,
                activation='linear',
                name='price_prediction'
            )(x)
            
            # Create model
            self.model = keras.Model(inputs=inputs, outputs=outputs, name='LSTM_Predictor')
            
            # Compile model
            optimizer = keras.optimizers.Adam(
                learning_rate=self.model_config['learning_rate']
            )
            
            self.model.compile(
                optimizer=optimizer,
                loss='huber',  # Robust to outliers
                metrics=['mae', 'mse']
            )
            
            logger.info(f"LSTM model built successfully")
            logger.info(f"Model summary: {self.model.count_params()} parameters")
            
        except Exception as e:
            logger.error(f"Error building LSTM model: {e}")
            raise
    
    def prepare_data(self, df: pd.DataFrame, target_column: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM training.
        
        Args:
            df: DataFrame with features and target
            target_column: Name of target column
            
        Returns:
            Tuple: (X, y) prepared for LSTM
        """
        try:
            # Select features (excluding timestamp and target)
            feature_columns = [col for col in df.columns 
                             if col not in ['timestamp', target_column]]
            
            if not feature_columns:
                raise ValueError("No feature columns found")
            
            # Extract features and target
            X_raw = df[feature_columns].values
            y_raw = df[target_column].values
            
            # Handle missing values
            if np.any(np.isnan(X_raw)) or np.any(np.isnan(y_raw)):
                logger.warning("Missing values detected, forward filling")
                df_clean = df[feature_columns + [target_column]].fillna(method='ffill')
                X_raw = df_clean[feature_columns].values
                y_raw = df_clean[target_column].values
            
            # Scale features and target
            X_scaled = self.scaler_X.fit_transform(X_raw)
            y_scaled = self.scaler_y.fit_transform(y_raw.reshape(-1, 1)).flatten()
            
            # Create sequences
            X_sequences, y_sequences = self._create_sequences(X_scaled, y_scaled)
            
            self.n_features = X_sequences.shape[2]
            logger.info(f"Data prepared: {X_sequences.shape} X, {y_sequences.shape} y")
            
            return X_sequences, y_sequences
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            X: Features array
            y: Target array
            
        Returns:
            Tuple: (X_sequences, y_sequences)
        """
        X_sequences, y_sequences = [], []
        
        for i in range(self.sequence_length, len(X)):
            X_sequences.append(X[i-self.sequence_length:i])
            y_sequences.append(y[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Train the LSTM model.
        
        Args:
            X: Training features (samples, sequence_length, features)
            y: Training targets (samples,)
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
                input_shape = (X.shape[1], X.shape[2])  # (sequence_length, n_features)
                self.build_model(input_shape)
            
            # Split data for validation
            validation_split = train_config.get('validation_split', 0.2)
            split_idx = int(len(X) * (1 - validation_split))
            
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=train_config['early_stopping_patience'],
                    restore_best_weights=True,
                    verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=train_config['reduce_lr_patience'],
                    min_lr=1e-7,
                    verbose=1
                ),
                keras.callbacks.ModelCheckpoint(
                    filepath=str(self.model_path),
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
            ]
            
            logger.info(f"Starting LSTM training: {len(X_train)} train, {len(X_val)} val samples")
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=train_config['epochs'],
                batch_size=train_config['batch_size'],
                callbacks=callbacks,
                verbose=1
            )
            
            self.is_trained = True
            self.training_history = history.history
            
            # Evaluate on validation set
            val_predictions = self.predict(X_val)
            val_metrics = self.evaluate_model(X_val, y_val)
            
            # Training results
            results = {
                'training_completed': True,
                'epochs_trained': len(history.history['loss']),
                'final_train_loss': float(history.history['loss'][-1]),
                'final_val_loss': float(history.history['val_loss'][-1]),
                'best_val_loss': float(min(history.history['val_loss'])),
                'validation_metrics': val_metrics,
                'model_path': str(self.model_path)
            }
            
            logger.info(f"LSTM training completed: Val Loss={results['best_val_loss']:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            raise
    
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Make predictions with LSTM model.
        
        Args:
            X: Input features (samples, sequence_length, features)
            **kwargs: Prediction parameters
            
        Returns:
            np.ndarray: Predictions in original scale
        """
        try:
            if self.model is None or not self.is_trained:
                raise ValueError("Model not trained")
            
            # Make predictions
            predictions_scaled = self.model.predict(X, verbose=0)
            
            # Scale back to original values
            predictions = self.scaler_y.inverse_transform(predictions_scaled)
            
            return predictions.flatten()
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
    
    def predict_next_prices(self, 
                           df: pd.DataFrame, 
                           n_steps: int = 5,
                           target_column: str = 'close') -> np.ndarray:
        """
        Predict next n price steps.
        
        Args:
            df: DataFrame with recent data
            n_steps: Number of future steps to predict
            target_column: Target column name
            
        Returns:
            np.ndarray: Future price predictions
        """
        try:
            if len(df) < self.sequence_length:
                raise ValueError(f"Need at least {self.sequence_length} data points")
            
            # Prepare features
            feature_columns = [col for col in df.columns 
                             if col not in ['timestamp', target_column]]
            
            X_raw = df[feature_columns].tail(self.sequence_length).values
            X_scaled = self.scaler_X.transform(X_raw)
            
            predictions = []
            current_sequence = X_scaled.copy()
            
            for _ in range(n_steps):
                # Reshape for prediction
                X_input = current_sequence.reshape(1, self.sequence_length, self.n_features)
                
                # Predict next price
                next_price_scaled = self.model.predict(X_input, verbose=0)[0]
                next_price = self.scaler_y.inverse_transform([[next_price_scaled]])[0][0]
                predictions.append(next_price)
                
                # Update sequence (simplified - in reality would need all features)
                # For now, just append the prediction as the last feature
                new_features = current_sequence[-1].copy()
                new_features[0] = next_price_scaled  # Assume first feature is close price
                
                current_sequence = np.vstack([current_sequence[1:], new_features])
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"Error predicting next prices: {e}")
            return np.array([])
    
    def _load_model_implementation(self) -> None:
        """Load LSTM model from disk."""
        try:
            self.model = keras.models.load_model(self.model_path)
            
            # Load scalers
            scaler_X_path = self.models_dir / f"{self.model_name}_{self.version}_scaler_X.pkl"
            scaler_y_path = self.models_dir / f"{self.model_name}_{self.version}_scaler_y.pkl"
            
            if scaler_X_path.exists():
                self.scaler_X = joblib.load(scaler_X_path)
            if scaler_y_path.exists():
                self.scaler_y = joblib.load(scaler_y_path)
                
        except Exception as e:
            logger.error(f"Error loading LSTM model: {e}")
            raise
    
    def save_model(self, include_optimizer: bool = False) -> bool:
        """
        Save LSTM model with scalers.
        
        Args:
            include_optimizer: Whether to include optimizer state
            
        Returns:
            bool: Success status
        """
        try:
            # Save base model
            success = super().save_model(include_optimizer)
            
            if success:
                # Save scalers
                scaler_X_path = self.models_dir / f"{self.model_name}_{self.version}_scaler_X.pkl"
                scaler_y_path = self.models_dir / f"{self.model_name}_{self.version}_scaler_y.pkl"
                
                joblib.dump(self.scaler_X, scaler_X_path)
                joblib.dump(self.scaler_y, scaler_y_path)
                
                logger.info("LSTM model and scalers saved successfully")
            
            return success
            
        except Exception as e:
            logger.error(f"Error saving LSTM model: {e}")
            return False