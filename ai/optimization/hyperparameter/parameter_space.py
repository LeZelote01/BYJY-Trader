"""
🎛️ Parameter Space Module
Definition of parameter spaces for optimization
"""

import optuna
from typing import Dict, Any, List, Union
import logging

logger = logging.getLogger(__name__)


class ParameterSpace:
    """
    Parameter space definition for hyperparameter optimization
    Phase 3.4 - Hyperparameter Optimization Component
    """
    
    def __init__(self):
        """Initialize Parameter Space"""
        self.spaces = {}
        logger.info("ParameterSpace initialized")
    
    def add_categorical(self, name: str, choices: List[Any]):
        """Add categorical parameter"""
        self.spaces[name] = optuna.distributions.CategoricalDistribution(choices)
    
    def add_float(self, name: str, low: float, high: float, log: bool = False):
        """Add float parameter"""
        self.spaces[name] = optuna.distributions.FloatDistribution(low, high, log=log)
    
    def add_int(self, name: str, low: int, high: int):
        """Add integer parameter"""
        self.spaces[name] = optuna.distributions.IntDistribution(low, high)
    
    def get_space_definition(self) -> Dict[str, Any]:
        """Get parameter space definition"""
        return self.spaces.copy()