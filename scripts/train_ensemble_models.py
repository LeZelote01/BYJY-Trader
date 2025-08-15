#!/usr/bin/env python3
"""
🧠 Training Script for Ensemble Models
Phase 3.1 - Train LSTM, Transformer, XGBoost, and Ensemble models
"""

import sys
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Add project root to Python path
ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

from ai.models.lstm_model import LSTMModel
from ai.models.transformer_model import TransformerModel
from ai.models.xgboost_model import XGBoostModel
from ai.models.ensemble_model import EnsembleModel
from data.storage.data_manager import DataManager
from data.processors.feature_engine import FeatureEngine
from core.logger import get_logger

logger = get_logger(__name__)

class EnsembleModelTrainer:
    """
    Trainer for ensemble models with comprehensive evaluation.
    """
    
    def __init__(self):
        """Initialize trainer components."""
        self.data_manager = DataManager()
        self.feature_engine = FeatureEngine()
        
        # Models to train
        self.models = {
            'lstm': LSTMModel("3.1.0"),
            'transformer': TransformerModel("3.1.0"),
            'xgboost': XGBoostModel("3.1.0"),
            'ensemble': EnsembleModel("3.1.0")
        }
        
        # Training configuration
        self.train_config = {
            'test_size': 0.2,
            'validation_size': 0.2,
            'min_data_points': 500,
            'symbols': ['BTCUSDT', 'ETHUSDT'],  # Test symbols
            'epochs': {
                'lstm': 50,
                'transformer': 30,
                'xgboost': 100  # XGBoost uses early stopping
            }
        }
        
        # Results storage
        self.training_results = {}
        
    async def initialize(self):
        """Initialize data manager and create tables."""
        try:
            await self.data_manager.initialize_tables()
            logger.info("Data manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize data manager: {e}")
            raise
    
    async def prepare_training_data(self, symbol: str) -> pd.DataFrame:
        """
        Prepare comprehensive training data for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            pd.DataFrame: Prepared data with features
        """
        try:
            logger.info(f"Preparing training data for {symbol}")
            
            # Get historical data (1 year)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            historical_data = await self.data_manager.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                limit=10000
            )
            
            if historical_data.empty:
                logger.warning(f"No historical data found for {symbol}")
                return pd.DataFrame()
            
            logger.info(f"Retrieved {len(historical_data)} data points for {symbol}")
            
            # Generate comprehensive features
            logger.info("Generating technical features...")
            featured_data = self.feature_engine.generate_all_features(historical_data)
            
            # Clean data
            featured_data = featured_data.dropna()
            
            if len(featured_data) < self.train_config['min_data_points']:
                logger.warning(f"Insufficient data for {symbol}: {len(featured_data)} points")
                return pd.DataFrame()
            
            logger.info(f"Prepared {len(featured_data)} data points with {len(featured_data.columns)} features")
            return featured_data
            
        except Exception as e:
            logger.error(f"Error preparing training data for {symbol}: {e}")
            return pd.DataFrame()
    
    def split_data(self, data: pd.DataFrame, target_column: str = 'close'):
        """
        Split data into train/validation/test sets with time-based splitting.
        
        Args:
            data: Full dataset
            target_column: Target column name
            
        Returns:
            Tuple: (train_data, val_data, test_data)
        """
        try:
            # Time-based split (chronological order)
            total_len = len(data)
            test_size = int(total_len * self.train_config['test_size'])
            val_size = int(total_len * self.train_config['validation_size'])
            train_size = total_len - test_size - val_size
            
            train_data = data.iloc[:train_size].copy()
            val_data = data.iloc[train_size:train_size + val_size].copy()
            test_data = data.iloc[train_size + val_size:].copy()
            
            logger.info(f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
            
            return train_data, val_data, test_data
            
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            raise
    
    async def train_individual_models(self, symbol: str, train_data: pd.DataFrame):
        """
        Train individual models (LSTM, Transformer, XGBoost).
        
        Args:
            symbol: Trading symbol
            train_data: Training data
            
        Returns:
            Dict: Training results for each model
        """
        results = {}
        
        for model_name in ['lstm', 'transformer', 'xgboost']:
            try:
                logger.info(f"Training {model_name.upper()} model for {symbol}...")
                
                model = self.models[model_name]
                
                # Prepare model-specific data
                X, y = model.prepare_data(train_data, target_column='close')
                
                if len(X) == 0:
                    logger.warning(f"No data available for {model_name} training")
                    results[model_name] = {'error': 'No training data available'}
                    continue
                
                # Train model
                epochs = self.train_config['epochs'].get(model_name, 50)
                training_result = model.train(
                    X, y,
                    epochs=epochs,
                    validation_split=0.2,
                    verbose=1
                )
                
                # Save model
                if model.save_model():
                    logger.info(f"{model_name.upper()} model saved successfully")
                else:
                    logger.warning(f"Failed to save {model_name.upper()} model")
                
                results[model_name] = training_result
                logger.info(f"{model_name.upper()} training completed successfully")
                
            except Exception as e:
                logger.error(f"Error training {model_name} model: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    async def train_ensemble_model(self, symbol: str, train_data: pd.DataFrame):
        """
        Train ensemble model using individual models.
        
        Args:
            symbol: Trading symbol
            train_data: Training data
            
        Returns:
            Dict: Ensemble training results
        """
        try:
            logger.info(f"Training Ensemble model for {symbol}...")
            
            ensemble_model = self.models['ensemble']
            
            # Prepare data for ensemble (returns dict for each base model)
            X_dict, y = ensemble_model.prepare_data(train_data, target_column='close')
            
            if not any(x is not None for x in X_dict.values()):
                logger.warning("No data available for ensemble training")
                return {'error': 'No training data available for ensemble'}
            
            # Train ensemble
            ensemble_result = ensemble_model.train(
                X_dict, y,
                epochs=30,  # Shorter epochs for ensemble meta-training
                validation_split=0.2,
                verbose=1
            )
            
            # Save ensemble model
            if ensemble_model.save_model():
                logger.info("Ensemble model saved successfully")
            else:
                logger.warning("Failed to save Ensemble model")
            
            logger.info("Ensemble training completed successfully")
            return ensemble_result
            
        except Exception as e:
            logger.error(f"Error training ensemble model: {e}")
            return {'error': str(e)}
    
    async def evaluate_models(self, symbol: str, test_data: pd.DataFrame):
        """
        Evaluate all trained models on test data.
        
        Args:
            symbol: Trading symbol
            test_data: Test data
            
        Returns:
            Dict: Evaluation results for all models
        """
        evaluation_results = {}
        
        for model_name, model in self.models.items():
            try:
                if not model.is_trained:
                    logger.warning(f"{model_name} model not trained, skipping evaluation")
                    continue
                
                logger.info(f"Evaluating {model_name.upper()} model...")
                
                if model_name == 'ensemble':
                    # Special handling for ensemble model
                    X_dict, y = model.prepare_data(test_data, target_column='close')
                    if any(x is not None for x in X_dict.values()):
                        predictions = model.predict(X_dict)
                        
                        # Align data lengths
                        min_len = min(len(predictions), len(y))
                        predictions = predictions[-min_len:]
                        y_test = y[-min_len:]
                        
                        metrics = model.evaluate_model(X_dict, y_test)
                    else:
                        metrics = {'error': 'No test data available'}
                else:
                    # Individual models
                    X, y = model.prepare_data(test_data, target_column='close')
                    if len(X) > 0:
                        metrics = model.evaluate_model(X, y)
                    else:
                        metrics = {'error': 'No test data available'}
                
                evaluation_results[model_name] = metrics
                
                if 'error' not in metrics:
                    logger.info(f"{model_name.upper()} evaluation: RMSE={metrics.get('rmse', 0):.4f}, "
                              f"DA={metrics.get('directional_accuracy', 0):.4f}")
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                evaluation_results[model_name] = {'error': str(e)}
        
        return evaluation_results
    
    async def run_comprehensive_training(self):
        """
        Run comprehensive training for all models on all symbols.
        """
        try:
            logger.info("🚀 Starting comprehensive ensemble model training...")
            
            await self.initialize()
            
            overall_results = {
                'training_started': datetime.now().isoformat(),
                'symbols': {},
                'summary': {
                    'total_symbols': len(self.train_config['symbols']),
                    'successful_trainings': 0,
                    'failed_trainings': 0
                }
            }
            
            # Train models for each symbol
            for symbol in self.train_config['symbols']:
                try:
                    logger.info(f"📈 Processing symbol: {symbol}")
                    
                    # Prepare training data
                    full_data = await self.prepare_training_data(symbol)
                    
                    if full_data.empty:
                        logger.warning(f"Skipping {symbol} due to insufficient data")
                        overall_results['symbols'][symbol] = {'error': 'Insufficient data'}
                        overall_results['summary']['failed_trainings'] += 1
                        continue
                    
                    # Split data
                    train_data, val_data, test_data = self.split_data(full_data)
                    
                    # Train individual models
                    individual_results = await self.train_individual_models(symbol, train_data)
                    
                    # Train ensemble model
                    ensemble_result = await self.train_ensemble_model(symbol, train_data)
                    
                    # Evaluate all models
                    evaluation_results = await self.evaluate_models(symbol, test_data)
                    
                    # Store results
                    symbol_results = {
                        'data_points': len(full_data),
                        'training_split': {
                            'train': len(train_data),
                            'validation': len(val_data),
                            'test': len(test_data)
                        },
                        'individual_models': individual_results,
                        'ensemble_model': ensemble_result,
                        'evaluation': evaluation_results
                    }
                    
                    overall_results['symbols'][symbol] = symbol_results
                    overall_results['summary']['successful_trainings'] += 1
                    
                    logger.info(f"✅ Completed training for {symbol}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed training for {symbol}: {e}")
                    overall_results['symbols'][symbol] = {'error': str(e)}
                    overall_results['summary']['failed_trainings'] += 1
            
            # Final summary
            overall_results['training_completed'] = datetime.now().isoformat()
            
            # Save results to file
            results_file = ROOT_DIR / "ai" / "trained_models" / "ensemble_training_results.json"
            results_file.parent.mkdir(parents=True, exist_ok=True)
            
            import json
            with open(results_file, 'w') as f:
                json.dump(overall_results, f, indent=2)
            
            logger.info(f"📊 Training completed. Results saved to: {results_file}")
            logger.info(f"📈 Summary: {overall_results['summary']['successful_trainings']} successful, "
                       f"{overall_results['summary']['failed_trainings']} failed")
            
            # Print performance comparison
            self.print_performance_summary(overall_results)
            
            return overall_results
            
        except Exception as e:
            logger.error(f"❌ Comprehensive training failed: {e}")
            raise
    
    def print_performance_summary(self, results: dict):
        """Print performance summary comparison."""
        try:
            logger.info("\n" + "="*60)
            logger.info("📊 ENSEMBLE MODEL PERFORMANCE SUMMARY")
            logger.info("="*60)
            
            for symbol, symbol_results in results['symbols'].items():
                if 'error' in symbol_results:
                    logger.info(f"\n❌ {symbol}: {symbol_results['error']}")
                    continue
                
                evaluation = symbol_results.get('evaluation', {})
                logger.info(f"\n📈 {symbol} Results:")
                logger.info("-" * 40)
                
                for model_name, metrics in evaluation.items():
                    if 'error' not in metrics:
                        rmse = metrics.get('rmse', 0)
                        da = metrics.get('directional_accuracy', 0)
                        logger.info(f"{model_name.upper():12}: RMSE={rmse:.4f}, DA={da:.3f}")
                    else:
                        logger.info(f"{model_name.upper():12}: Error - {metrics['error']}")
            
            logger.info("\n" + "="*60)
            
        except Exception as e:
            logger.error(f"Error printing performance summary: {e}")

async def main():
    """Main training function."""
    try:
        trainer = EnsembleModelTrainer()
        results = await trainer.run_comprehensive_training()
        
        print("\n🎉 Ensemble model training completed successfully!")
        print(f"📊 Check results in: {ROOT_DIR}/ai/trained_models/ensemble_training_results.json")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run training
    asyncio.run(main())