# Transformer Model for BYJY-Trader
# Phase 3.1 - Transformer architecture for time series prediction

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

class TransformerModel(BaseModel):
    """
    Transformer model for cryptocurrency and stock price prediction.
    Uses multi-head attention to capture long-range dependencies in time series.
    """
    
    def __init__(self, version: str = "3.1.0"):
        """Initialize Transformer model."""
        super().__init__("transformer", version)
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.sequence_length = 120  # Longer sequence for transformer
        self.n_features = None
        
        # Model hyperparameters
        self.model_config = {
            'sequence_length': self.sequence_length,
            'd_model': 128,  # Embedding dimension
            'n_heads': 8,    # Number of attention heads
            'n_layers': 4,   # Number of transformer blocks
            'dff': 512,      # Feed-forward dimension
            'dropout_rate': 0.1,
            'dense_units': [64, 32],
            'learning_rate': 0.0001,
            'batch_size': 16,
            'epochs': 150,
            'early_stopping_patience': 20,
            'reduce_lr_patience': 10
        }
    
    def build_model(self, input_shape: Tuple[int, ...], **kwargs) -> None:
        """
        Build Transformer model architecture.
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
            **kwargs: Additional model parameters
        """
        try:
            # Update config with kwargs
            self.model_config.update(kwargs)
            
            sequence_length, n_features = input_shape
            d_model = self.model_config['d_model']
            
            # Input layer
            inputs = keras.Input(shape=input_shape, name='transformer_input')
            
            # Input projection to d_model dimensions
            x = layers.Dense(d_model, name='input_projection')(inputs)
            
            # Positional encoding
            x = self._add_positional_encoding(x, sequence_length, d_model)
            
            # Transformer blocks
            for i in range(self.model_config['n_layers']):
                x = self._transformer_block(
                    x, 
                    d_model=d_model,
                    n_heads=self.model_config['n_heads'],
                    dff=self.model_config['dff'],
                    dropout_rate=self.model_config['dropout_rate'],
                    name=f'transformer_block_{i+1}'
                )
            
            # Global average pooling to get fixed-size representation
            x = layers.GlobalAveragePooling1D(name='global_avg_pooling')(x)
            
            # Dense layers for final prediction
            dense_units = self.model_config['dense_units']
            dropout_rate = self.model_config['dropout_rate']
            
            for i, units in enumerate(dense_units):
                x = layers.Dense(
                    units=units,
                    activation='relu',
                    name=f'dense_{i+1}'
                )(x)
                x = layers.Dropout(dropout_rate, name=f'dropout_{i+1}')(x)
                x = layers.LayerNormalization(name=f'dense_norm_{i+1}')(x)
            
            # Output layer
            outputs = layers.Dense(
                units=1,
                activation='linear',
                name='price_prediction'
            )(x)
            
            # Create model
            self.model = keras.Model(inputs=inputs, outputs=outputs, name='Transformer_Predictor')
            
            # Compile model
            optimizer = keras.optimizers.Adam(
                learning_rate=self.model_config['learning_rate'],
                beta_1=0.9,
                beta_2=0.98,
                epsilon=1e-9
            )
            
            self.model.compile(
                optimizer=optimizer,
                loss='huber',
                metrics=['mae', 'mse']
            )
            
            logger.info(f"Transformer model built successfully")
            logger.info(f"Model summary: {self.model.count_params()} parameters")
            
        except Exception as e:
            logger.error(f"Error building Transformer model: {e}")
            raise
    
    def _add_positional_encoding(self, x, sequence_length: int, d_model: int):
        """Add positional encoding to input embeddings."""
        position = np.arange(sequence_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pos_encoding = np.zeros((sequence_length, d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        pos_encoding = tf.constant(pos_encoding[np.newaxis, :], dtype=tf.float32)
        return x + pos_encoding
    
    def _transformer_block(self, x, d_model: int, n_heads: int, dff: int, 
                          dropout_rate: float, name: str):
        """
        Single transformer block with multi-head attention and feed-forward.
        """
        # Multi-head self-attention
        attention_output = layers.MultiHeadAttention(
            num_heads=n_heads,
            key_dim=d_model // n_heads,
            name=f'{name}_attention'
        )(x, x)
        
        attention_output = layers.Dropout(dropout_rate)(attention_output)
        x1 = layers.LayerNormalization(name=f'{name}_norm1')(x + attention_output)
        
        # Feed-forward network
        ffn_output = layers.Dense(dff, activation='relu', name=f'{name}_ffn1')(x1)
        ffn_output = layers.Dropout(dropout_rate)(ffn_output)
        ffn_output = layers.Dense(d_model, name=f'{name}_ffn2')(ffn_output)
        ffn_output = layers.Dropout(dropout_rate)(ffn_output)
        
        return layers.LayerNormalization(name=f'{name}_norm2')(x1 + ffn_output)
    
    def prepare_data(self, df: pd.DataFrame, target_column: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for Transformer training.
        
        Args:
            df: DataFrame with features and target
            target_column: Name of target column
            
        Returns:
            Tuple: (X, y) prepared for Transformer
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
            logger.info(f"Transformer data prepared: {X_sequences.shape} X, {y_sequences.shape} y")
            
            return X_sequences, y_sequences
            
        except Exception as e:
            logger.error(f"Error preparing Transformer data: {e}")
            raise
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for Transformer training.
        
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
        Train the Transformer model.
        
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
                input_shape = (X.shape[1], X.shape[2])
                self.build_model(input_shape)
            
            # Split data for validation
            validation_split = train_config.get('validation_split', 0.2)
            split_idx = int(len(X) * (1 - validation_split))
            
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Custom learning rate schedule
            def lr_schedule(epoch, lr):
                if epoch < 10:
                    return lr
                elif epoch < 50:
                    return lr * 0.95
                else:
                    return lr * 0.9
            
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
                    factor=0.7,
                    patience=train_config['reduce_lr_patience'],
                    min_lr=1e-8,
                    verbose=1
                ),
                keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1),
                keras.callbacks.ModelCheckpoint(
                    filepath=str(self.model_path),
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
            ]
            
            logger.info(f"Starting Transformer training: {len(X_train)} train, {len(X_val)} val samples")
            
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
            
            logger.info(f"Transformer training completed: Val Loss={results['best_val_loss']:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error training Transformer model: {e}")
            raise
    
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Make predictions with Transformer model.
        
        Args:
            X: Input features (samples, sequence_length, features)
            **kwargs: Prediction parameters
            
        Returns:
            np.ndarray: Predictions in original scale
        """
        try:
            if self.model is None or not self.is_trained:
                raise ValueError("Transformer model not trained")
            
            # Make predictions
            predictions_scaled = self.model.predict(X, verbose=0)
            
            # Scale back to original values
            predictions = self.scaler_y.inverse_transform(predictions_scaled)
            
            return predictions.flatten()
            
        except Exception as e:
            logger.error(f"Error making Transformer predictions: {e}")
            raise
    
    def predict_next_prices(self, 
                           df: pd.DataFrame, 
                           n_steps: int = 5,
                           target_column: str = 'close') -> np.ndarray:
        """
        Predict next n price steps using Transformer.
        
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
                
                # Update sequence
                new_features = current_sequence[-1].copy()
                new_features[0] = next_price_scaled  # Assume first feature is close price
                
                current_sequence = np.vstack([current_sequence[1:], new_features])
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"Error predicting next prices with Transformer: {e}")
            return np.array([])
    
    def _load_model_implementation(self) -> None:
        """Load Transformer model from disk."""
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
            logger.error(f"Error loading Transformer model: {e}")
            raise
    
    def save_model(self, include_optimizer: bool = False) -> bool:
        """
        Save Transformer model with scalers.
        
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
                
                logger.info("Transformer model and scalers saved successfully")
            
            return success
            
        except Exception as e:
            logger.error(f"Error saving Transformer model: {e}")
            return False