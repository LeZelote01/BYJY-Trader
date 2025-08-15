"""
🎯 Objective Functions
Multi-objective functions for profit/risk optimization
"""

import numpy as np
from typing import Dict, List, Any, Optional
from enum import Enum

from core.logger import get_logger

logger = get_logger(__name__)


class ObjectiveType(Enum):
    """Enumeration of objective types."""
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class ObjectiveFunctions:
    """
    Collection of objective functions for multi-objective optimization.
    
    Handles conversion of raw performance metrics to objective values
    for profit maximization, risk minimization, speed optimization, etc.
    """
    
    def __init__(self):
        """Initialize objective functions."""
        self.objective_configs = []
        self.objective_names = []
        self.objective_types = []
        self.normalization_params = {}
        
        # Predefined objective configurations
        self.predefined_objectives = {
            'profit_sharpe': {
                'name': 'sharpe_ratio',
                'type': ObjectiveType.MAXIMIZE,
                'description': 'Maximize Sharpe ratio',
                'weight': 1.0,
                'target_range': [0, 3]
            },
            'profit_returns': {
                'name': 'annual_return',
                'type': ObjectiveType.MAXIMIZE,
                'description': 'Maximize annual returns',
                'weight': 1.0,
                'target_range': [0, 1]
            },
            'risk_drawdown': {
                'name': 'max_drawdown',
                'type': ObjectiveType.MINIMIZE,
                'description': 'Minimize maximum drawdown',
                'weight': 1.0,
                'target_range': [0, 0.5]
            },
            'risk_volatility': {
                'name': 'volatility',
                'type': ObjectiveType.MINIMIZE,
                'description': 'Minimize volatility',
                'weight': 1.0,
                'target_range': [0, 1]
            },
            'speed_training': {
                'name': 'training_time',
                'type': ObjectiveType.MINIMIZE,
                'description': 'Minimize training time',
                'weight': 1.0,
                'target_range': [1, 3600]  # 1 second to 1 hour
            },
            'speed_inference': {
                'name': 'inference_time',
                'type': ObjectiveType.MINIMIZE,
                'description': 'Minimize inference time',
                'weight': 1.0,
                'target_range': [0.001, 10]  # 1ms to 10s
            },
            'stability_consistency': {
                'name': 'consistency_score',
                'type': ObjectiveType.MAXIMIZE,
                'description': 'Maximize consistency',
                'weight': 1.0,
                'target_range': [0, 1]
            }
        }
    
    def setup_objectives(self, objective_configs: List[Dict[str, Any]]):
        """
        Setup objectives from configuration.
        
        Args:
            objective_configs: List of objective configurations
        """
        self.objective_configs = objective_configs.copy()
        self.objective_names = []
        self.objective_types = []
        
        for config in objective_configs:
            self.objective_names.append(config['name'])
            obj_type = ObjectiveType.MAXIMIZE if config.get('maximize', True) else ObjectiveType.MINIMIZE
            self.objective_types.append(obj_type)
        
        logger.info(f"Setup {len(self.objective_configs)} objectives: {self.objective_names}")
    
    def evaluate(self, metrics: Dict[str, Any]) -> List[float]:
        """
        Convert raw metrics to objective values.
        
        Args:
            metrics: Dictionary of raw performance metrics
            
        Returns:
            List[float]: Objective values
        """
        objective_values = []
        
        for i, config in enumerate(self.objective_configs):
            metric_name = config['name']
            obj_type = self.objective_types[i]
            
            if metric_name in metrics:
                raw_value = metrics[metric_name]
                objective_value = self._convert_to_objective(
                    raw_value, metric_name, obj_type, config
                )
            else:
                # Assign neutral value if metric is missing
                objective_value = 0.0
                logger.warning(f"Metric '{metric_name}' not found in evaluation results")
            
            objective_values.append(objective_value)
        
        return objective_values
    
    def _convert_to_objective(
        self,
        raw_value: float,
        metric_name: str,
        obj_type: ObjectiveType,
        config: Dict[str, Any]
    ) -> float:
        """
        Convert raw metric value to objective value.
        
        Args:
            raw_value: Raw metric value
            metric_name: Name of the metric
            obj_type: Objective type (maximize/minimize)
            config: Objective configuration
            
        Returns:
            float: Converted objective value
        """
        # Handle different metrics
        if metric_name in ['sharpe_ratio', 'calmar_ratio', 'sortino_ratio']:
            # Risk-adjusted return ratios
            return self._convert_ratio_metric(raw_value, obj_type)
        
        elif metric_name in ['annual_return', 'total_return', 'profit_factor']:
            # Return metrics
            return self._convert_return_metric(raw_value, obj_type)
        
        elif metric_name in ['max_drawdown', 'volatility', 'var', 'cvar']:
            # Risk metrics
            return self._convert_risk_metric(raw_value, obj_type)
        
        elif metric_name in ['training_time', 'inference_time', 'convergence_time']:
            # Time metrics
            return self._convert_time_metric(raw_value, obj_type, config)
        
        elif metric_name in ['win_rate', 'consistency_score', 'accuracy']:
            # Percentage/score metrics
            return self._convert_score_metric(raw_value, obj_type)
        
        elif metric_name in ['trades_count', 'positions_count']:
            # Count metrics
            return self._convert_count_metric(raw_value, obj_type, config)
        
        else:
            # Generic metric conversion
            return self._convert_generic_metric(raw_value, obj_type, config)
    
    def _convert_ratio_metric(self, raw_value: float, obj_type: ObjectiveType) -> float:
        """Convert ratio metrics (Sharpe, Calmar, etc.)."""
        # Sharpe ratio: good values are 1+, excellent 2+
        # Use tanh for smooth scaling
        normalized = np.tanh(raw_value / 2.0)  # Scale by 2 to make 2.0 -> ~0.76
        
        if obj_type == ObjectiveType.MAXIMIZE:
            return normalized
        else:
            return -normalized
    
    def _convert_return_metric(self, raw_value: float, obj_type: ObjectiveType) -> float:
        """Convert return metrics."""
        # Annual return: normalize to reasonable range
        if raw_value < 0:
            # Negative returns
            normalized = -1.0 + np.exp(raw_value)  # Maps large negative to -1
        else:
            # Positive returns: use tanh for scaling
            normalized = np.tanh(raw_value * 2.0)  # 0.5 return -> ~0.76
        
        if obj_type == ObjectiveType.MAXIMIZE:
            return normalized
        else:
            return -normalized
    
    def _convert_risk_metric(self, raw_value: float, obj_type: ObjectiveType) -> float:
        """Convert risk metrics (drawdown, volatility, VaR)."""
        # Most risk metrics should be positive (drawdown, volatility)
        abs_value = abs(raw_value)
        
        # Use exponential decay for risk metrics
        if abs_value <= 1.0:
            normalized = abs_value  # Linear for small values
        else:
            # Exponential penalty for large risk values
            normalized = 1.0 + np.log(abs_value)
        
        if obj_type == ObjectiveType.MINIMIZE:
            return -normalized  # Negative because we're maximizing objectives
        else:
            return normalized
    
    def _convert_time_metric(
        self,
        raw_value: float,
        obj_type: ObjectiveType,
        config: Dict[str, Any]
    ) -> float:
        """Convert time metrics."""
        # Time is always positive
        time_value = max(raw_value, 0.001)  # Minimum 1ms
        
        # Get target range from config
        target_range = config.get('target_range', [0.001, 3600])
        min_time, max_time = target_range
        
        # Log-scale normalization for time metrics
        log_time = np.log(time_value)
        log_min = np.log(min_time)
        log_max = np.log(max_time)
        
        if log_max > log_min:
            normalized = (log_time - log_min) / (log_max - log_min)
            normalized = np.clip(normalized, 0, 1)
        else:
            normalized = 0.5
        
        if obj_type == ObjectiveType.MINIMIZE:
            return -(normalized * 2 - 1)  # Map to [-1, 1] with lower time = higher objective
        else:
            return normalized * 2 - 1
    
    def _convert_score_metric(self, raw_value: float, obj_type: ObjectiveType) -> float:
        """Convert score metrics (accuracy, win rate, etc.)."""
        # Assume scores are in [0, 1] range
        normalized = np.clip(raw_value, 0, 1)
        
        # Map to [-1, 1] range
        mapped_value = normalized * 2 - 1
        
        if obj_type == ObjectiveType.MAXIMIZE:
            return mapped_value
        else:
            return -mapped_value
    
    def _convert_count_metric(
        self,
        raw_value: float,
        obj_type: ObjectiveType,
        config: Dict[str, Any]
    ) -> float:
        """Convert count metrics."""
        count_value = max(int(raw_value), 0)
        
        # Get target range from config
        target_range = config.get('target_range', [1, 1000])
        min_count, max_count = target_range
        
        # Linear normalization
        if max_count > min_count:
            normalized = (count_value - min_count) / (max_count - min_count)
            normalized = np.clip(normalized, 0, 1)
        else:
            normalized = 0.5
        
        # Map to [-1, 1]
        mapped_value = normalized * 2 - 1
        
        if obj_type == ObjectiveType.MAXIMIZE:
            return mapped_value
        else:
            return -mapped_value
    
    def _convert_generic_metric(
        self,
        raw_value: float,
        obj_type: ObjectiveType,
        config: Dict[str, Any]
    ) -> float:
        """Convert generic metrics using configuration."""
        # Get target range from config
        target_range = config.get('target_range', [-1, 1])
        min_val, max_val = target_range
        
        # Normalize to [0, 1]
        if max_val > min_val:
            normalized = (raw_value - min_val) / (max_val - min_val)
            normalized = np.clip(normalized, 0, 1)
        else:
            normalized = 0.5
        
        # Map to [-1, 1]
        mapped_value = normalized * 2 - 1
        
        if obj_type == ObjectiveType.MAXIMIZE:
            return mapped_value
        else:
            return -mapped_value
    
    def get_predefined_objective_configs(
        self,
        objective_names: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Get predefined objective configurations.
        
        Args:
            objective_names: List of predefined objective names
            
        Returns:
            List[Dict[str, Any]]: Objective configurations
        """
        configs = []
        
        for name in objective_names:
            if name in self.predefined_objectives:
                config = self.predefined_objectives[name].copy()
                config['maximize'] = (config['type'] == ObjectiveType.MAXIMIZE)
                configs.append(config)
            else:
                logger.warning(f"Predefined objective '{name}' not found")
        
        return configs
    
    def create_profit_risk_objectives(self) -> List[Dict[str, Any]]:
        """Create standard profit-risk objectives."""
        return [
            {
                'name': 'sharpe_ratio',
                'maximize': True,
                'weight': 0.4,
                'description': 'Maximize risk-adjusted returns (Sharpe ratio)',
                'target_range': [0, 3]
            },
            {
                'name': 'max_drawdown',
                'maximize': False,
                'weight': 0.3,
                'description': 'Minimize maximum drawdown',
                'target_range': [0, 0.5]
            },
            {
                'name': 'annual_return',
                'maximize': True,
                'weight': 0.3,
                'description': 'Maximize annual returns',
                'target_range': [-0.5, 2.0]
            }
        ]
    
    def create_speed_accuracy_objectives(self) -> List[Dict[str, Any]]:
        """Create speed-accuracy trade-off objectives."""
        return [
            {
                'name': 'accuracy',
                'maximize': True,
                'weight': 0.6,
                'description': 'Maximize prediction accuracy',
                'target_range': [0, 1]
            },
            {
                'name': 'inference_time',
                'maximize': False,
                'weight': 0.4,
                'description': 'Minimize inference time',
                'target_range': [0.001, 10]
            }
        ]
    
    def create_full_objectives(self) -> List[Dict[str, Any]]:
        """Create comprehensive objective set."""
        return [
            {
                'name': 'sharpe_ratio',
                'maximize': True,
                'weight': 0.3,
                'description': 'Maximize Sharpe ratio',
                'target_range': [0, 3]
            },
            {
                'name': 'max_drawdown',
                'maximize': False,
                'weight': 0.25,
                'description': 'Minimize maximum drawdown',
                'target_range': [0, 0.5]
            },
            {
                'name': 'training_time',
                'maximize': False,
                'weight': 0.15,
                'description': 'Minimize training time',
                'target_range': [1, 3600]
            },
            {
                'name': 'consistency_score',
                'maximize': True,
                'weight': 0.3,
                'description': 'Maximize consistency',
                'target_range': [0, 1]
            }
        ]