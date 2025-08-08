"""
⚡ Optuna Optimizer Module
Advanced hyperparameter optimization using Optuna
"""

import optuna
import logging
from typing import Dict, Any, List, Optional, Callable
import numpy as np

logger = logging.getLogger(__name__)


class OptunaOptimizer:
    """
    Advanced Optuna-based hyperparameter optimizer
    Phase 3.4 - Hyperparameter Optimization Component
    """
    
    def __init__(self, 
                 study_name: str = "byjy_optimization",
                 direction: str = "maximize",
                 n_trials: int = 100):
        """
        Initialize Optuna Optimizer
        
        Args:
            study_name: Name of the optimization study
            direction: Optimization direction (maximize/minimize)
            n_trials: Number of optimization trials
        """
        self.study_name = study_name
        self.direction = direction
        self.n_trials = n_trials
        self.study = None
        
        logger.info(f"OptunaOptimizer initialized: {study_name}, direction={direction}")
    
    def optimize(self,
                objective_function: Callable,
                parameter_space: Dict[str, Any],
                timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Run hyperparameter optimization
        
        Args:
            objective_function: Function to optimize
            parameter_space: Parameter space definition
            timeout: Timeout in seconds
            
        Returns:
            Optimization results
        """
        try:
            # Create study
            self.study = optuna.create_study(
                study_name=self.study_name,
                direction=self.direction
            )
            
            # Run optimization
            self.study.optimize(
                objective_function,
                n_trials=self.n_trials,
                timeout=timeout
            )
            
            results = {
                'best_params': self.study.best_params,
                'best_value': self.study.best_value,
                'n_trials': len(self.study.trials),
                'study_name': self.study_name
            }
            
            logger.info(f"Optimization completed: best_value={self.study.best_value}")
            return results
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return {'error': str(e)}
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history"""
        if not self.study:
            return []
            
        return [
            {
                'trial_number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name
            }
            for trial in self.study.trials
        ]