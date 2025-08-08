"""
🔧 Optimization Utils
Common utilities for optimization algorithms
"""

import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pickle
from datetime import datetime

from core.logger import get_logger

logger = get_logger(__name__)


class OptimizationUtils:
    """Utility functions for optimization algorithms."""
    
    @staticmethod
    def normalize_parameters(
        parameters: Dict[str, Any], 
        parameter_space: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Normalize parameters to [0, 1] range.
        
        Args:
            parameters: Parameter values
            parameter_space: Parameter space definition
            
        Returns:
            Dict[str, float]: Normalized parameters
        """
        normalized = {}
        
        for param_name, param_value in parameters.items():
            if param_name not in parameter_space:
                continue
                
            param_config = parameter_space[param_name]
            
            if param_config['type'] == 'float':
                min_val = param_config['min']
                max_val = param_config['max']
                if max_val > min_val:
                    normalized[param_name] = (param_value - min_val) / (max_val - min_val)
                else:
                    normalized[param_name] = 0.5
                    
            elif param_config['type'] == 'int':
                min_val = param_config['min']
                max_val = param_config['max']
                if max_val > min_val:
                    normalized[param_name] = (param_value - min_val) / (max_val - min_val)
                else:
                    normalized[param_name] = 0.5
                    
            elif param_config['type'] == 'boolean':
                normalized[param_name] = 1.0 if param_value else 0.0
                
            elif param_config['type'] == 'categorical':
                choices = param_config['choices']
                if param_value in choices:
                    normalized[param_name] = choices.index(param_value) / max(len(choices) - 1, 1)
                else:
                    normalized[param_name] = 0.0
        
        return normalized
    
    @staticmethod
    def denormalize_parameters(
        normalized_params: Dict[str, float], 
        parameter_space: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Denormalize parameters from [0, 1] range to original range.
        
        Args:
            normalized_params: Normalized parameter values
            parameter_space: Parameter space definition
            
        Returns:
            Dict[str, Any]: Denormalized parameters
        """
        denormalized = {}
        
        for param_name, norm_value in normalized_params.items():
            if param_name not in parameter_space:
                denormalized[param_name] = norm_value
                continue
                
            param_config = parameter_space[param_name]
            norm_value = np.clip(norm_value, 0, 1)
            
            if param_config['type'] == 'float':
                min_val = param_config['min']
                max_val = param_config['max']
                denormalized[param_name] = min_val + norm_value * (max_val - min_val)
                
            elif param_config['type'] == 'int':
                min_val = param_config['min']
                max_val = param_config['max']
                denormalized[param_name] = int(round(min_val + norm_value * (max_val - min_val)))
                
            elif param_config['type'] == 'boolean':
                denormalized[param_name] = norm_value > 0.5
                
            elif param_config['type'] == 'categorical':
                choices = param_config['choices']
                index = int(round(norm_value * (len(choices) - 1)))
                index = np.clip(index, 0, len(choices) - 1)
                denormalized[param_name] = choices[index]
        
        return denormalized
    
    @staticmethod
    def validate_parameter_space(parameter_space: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate parameter space configuration.
        
        Args:
            parameter_space: Parameter space to validate
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
        """
        errors = []
        
        if not parameter_space:
            return False, ["Parameter space is empty"]
        
        required_fields = ['type']
        valid_types = ['float', 'int', 'boolean', 'categorical']
        
        for param_name, param_config in parameter_space.items():
            # Check required fields
            for field in required_fields:
                if field not in param_config:
                    errors.append(f"Parameter '{param_name}' missing required field: {field}")
            
            # Check valid type
            param_type = param_config.get('type')
            if param_type not in valid_types:
                errors.append(f"Parameter '{param_name}' has invalid type: {param_type}")
                continue
            
            # Type-specific validation
            if param_type in ['float', 'int']:
                if 'min' not in param_config or 'max' not in param_config:
                    errors.append(f"Parameter '{param_name}' missing min/max values")
                else:
                    min_val = param_config['min']
                    max_val = param_config['max']
                    if min_val >= max_val:
                        errors.append(f"Parameter '{param_name}' min value must be less than max")
            
            elif param_type == 'categorical':
                if 'choices' not in param_config:
                    errors.append(f"Parameter '{param_name}' missing choices")
                else:
                    choices = param_config['choices']
                    if not choices or len(choices) < 2:
                        errors.append(f"Parameter '{param_name}' needs at least 2 choices")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def generate_parameter_combinations(
        parameter_space: Dict[str, Any],
        num_combinations: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Generate random parameter combinations.
        
        Args:
            parameter_space: Parameter space definition
            num_combinations: Number of combinations to generate
            
        Returns:
            List[Dict[str, Any]]: Parameter combinations
        """
        combinations = []
        
        for _ in range(num_combinations):
            combination = {}
            
            for param_name, param_config in parameter_space.items():
                if param_config['type'] == 'float':
                    combination[param_name] = np.random.uniform(
                        param_config['min'], param_config['max']
                    )
                elif param_config['type'] == 'int':
                    combination[param_name] = np.random.randint(
                        param_config['min'], param_config['max'] + 1
                    )
                elif param_config['type'] == 'boolean':
                    combination[param_name] = np.random.choice([True, False])
                elif param_config['type'] == 'categorical':
                    combination[param_name] = np.random.choice(param_config['choices'])
            
            combinations.append(combination)
        
        return combinations
    
    @staticmethod
    def calculate_parameter_diversity(
        parameter_sets: List[Dict[str, Any]],
        parameter_space: Dict[str, Any]
    ) -> float:
        """
        Calculate diversity of parameter sets.
        
        Args:
            parameter_sets: List of parameter dictionaries
            parameter_space: Parameter space definition
            
        Returns:
            float: Diversity score (0-1)
        """
        if len(parameter_sets) < 2:
            return 0.0
        
        # Normalize all parameter sets
        normalized_sets = []
        for params in parameter_sets:
            normalized = OptimizationUtils.normalize_parameters(params, parameter_space)
            normalized_sets.append(list(normalized.values()))
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(normalized_sets)):
            for j in range(i + 1, len(normalized_sets)):
                distance = np.linalg.norm(
                    np.array(normalized_sets[i]) - np.array(normalized_sets[j])
                )
                distances.append(distance)
        
        # Average distance as diversity measure
        if distances:
            diversity = np.mean(distances)
            # Normalize to [0, 1] range (max distance is sqrt(num_params))
            max_distance = np.sqrt(len(normalized_sets[0]))
            diversity = min(diversity / max_distance, 1.0)
        else:
            diversity = 0.0
        
        return diversity
    
    @staticmethod
    def save_optimization_results(
        results: Dict[str, Any],
        filepath: Path,
        format: str = 'json'
    ) -> bool:
        """
        Save optimization results to file.
        
        Args:
            results: Results dictionary
            filepath: File path to save to
            format: Format to save ('json' or 'pickle')
            
        Returns:
            bool: Success status
        """
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            if format == 'json':
                with open(filepath, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
            elif format == 'pickle':
                with open(filepath, 'wb') as f:
                    pickle.dump(results, f)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Optimization results saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving optimization results: {e}")
            return False
    
    @staticmethod
    def load_optimization_results(
        filepath: Path,
        format: str = 'json'
    ) -> Optional[Dict[str, Any]]:
        """
        Load optimization results from file.
        
        Args:
            filepath: File path to load from
            format: Format to load ('json' or 'pickle')
            
        Returns:
            Optional[Dict[str, Any]]: Loaded results or None
        """
        try:
            if not filepath.exists():
                logger.warning(f"Results file not found: {filepath}")
                return None
            
            if format == 'json':
                with open(filepath, 'r') as f:
                    results = json.load(f)
            elif format == 'pickle':
                with open(filepath, 'rb') as f:
                    results = pickle.load(f)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Optimization results loaded from {filepath}")
            return results
            
        except Exception as e:
            logger.error(f"Error loading optimization results: {e}")
            return None
    
    @staticmethod
    def get_default_parameter_spaces() -> Dict[str, Dict[str, Any]]:
        """
        Get default parameter spaces for different models.
        
        Returns:
            Dict[str, Dict[str, Any]]: Default parameter spaces
        """
        return {
            'lstm': {
                'layers': {'type': 'int', 'min': 1, 'max': 5},
                'neurons': {'type': 'int', 'min': 32, 'max': 512},
                'dropout': {'type': 'float', 'min': 0.1, 'max': 0.5},
                'learning_rate': {'type': 'float', 'min': 1e-5, 'max': 1e-2},
                'batch_size': {'type': 'categorical', 'choices': [16, 32, 64, 128]},
                'sequence_length': {'type': 'int', 'min': 30, 'max': 120},
                'optimizer': {'type': 'categorical', 'choices': ['adam', 'rmsprop', 'sgd']}
            },
            
            'transformer': {
                'num_heads': {'type': 'categorical', 'choices': [4, 8, 12, 16]},
                'num_layers': {'type': 'int', 'min': 2, 'max': 12},
                'd_model': {'type': 'categorical', 'choices': [128, 256, 512, 1024]},
                'dropout': {'type': 'float', 'min': 0.1, 'max': 0.3},
                'learning_rate': {'type': 'float', 'min': 1e-5, 'max': 1e-3},
                'warmup_steps': {'type': 'int', 'min': 1000, 'max': 10000}
            },
            
            'xgboost': {
                'max_depth': {'type': 'int', 'min': 3, 'max': 10},
                'learning_rate': {'type': 'float', 'min': 0.01, 'max': 0.3},
                'n_estimators': {'type': 'int', 'min': 50, 'max': 1000},
                'subsample': {'type': 'float', 'min': 0.5, 'max': 1.0},
                'colsample_bytree': {'type': 'float', 'min': 0.5, 'max': 1.0},
                'reg_alpha': {'type': 'float', 'min': 0, 'max': 1},
                'reg_lambda': {'type': 'float', 'min': 0, 'max': 1}
            },
            
            'ensemble': {
                'lstm_weight': {'type': 'float', 'min': 0, 'max': 1},
                'transformer_weight': {'type': 'float', 'min': 0, 'max': 1},
                'xgboost_weight': {'type': 'float', 'min': 0, 'max': 1},
                'voting_method': {'type': 'categorical', 'choices': ['soft', 'hard', 'weighted']},
                'meta_learner': {'type': 'boolean'}
            },
            
            'trading_strategy': {
                'stop_loss': {'type': 'float', 'min': 0.01, 'max': 0.1},
                'take_profit': {'type': 'float', 'min': 0.02, 'max': 0.2},
                'position_size': {'type': 'float', 'min': 0.01, 'max': 0.1},
                'timeframe': {'type': 'categorical', 'choices': ['5m', '15m', '1h', '4h', '1d']},
                'risk_reward_ratio': {'type': 'float', 'min': 1.0, 'max': 5.0},
                'max_positions': {'type': 'int', 'min': 1, 'max': 10}
            }
        }