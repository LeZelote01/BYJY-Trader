"""
✂️ Pruning Strategies Module
Early stopping strategies for optimization
"""

import optuna
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class PruningStrategies:
    """
    Pruning strategies for early stopping
    Phase 3.4 - Hyperparameter Optimization Component
    """
    
    def __init__(self):
        """Initialize Pruning Strategies"""
        logger.info("PruningStrategies initialized")
    
    def get_median_pruner(self, n_startup_trials: int = 5) -> optuna.pruners.BasePruner:
        """Get median pruner"""
        return optuna.pruners.MedianPruner(n_startup_trials=n_startup_trials)
    
    def get_hyperband_pruner(self) -> optuna.pruners.BasePruner:
        """Get hyperband pruner"""
        return optuna.pruners.HyperbandPruner()