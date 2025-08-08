"""
📊 Optimization History Module
Storage and analysis of optimization results
"""

import logging
from typing import Dict, Any, List
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class OptimizationHistory:
    """
    Optimization history management
    Phase 3.4 - Hyperparameter Optimization Component
    """
    
    def __init__(self, storage_path: str = "data/optimization_history.json"):
        """Initialize Optimization History"""
        self.storage_path = Path(storage_path)
        self.history = []
        logger.info(f"OptimizationHistory initialized: {storage_path}")
    
    def save_optimization_result(self, result: Dict[str, Any]):
        """Save optimization result"""
        self.history.append(result)
        
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump(self.history, f, indent=2)
                
            logger.info("Optimization result saved")
        except Exception as e:
            logger.error(f"Failed to save optimization result: {e}")
    
    def load_history(self) -> List[Dict[str, Any]]:
        """Load optimization history"""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, 'r') as f:
                    self.history = json.load(f)
                    
            logger.info(f"Loaded {len(self.history)} optimization results")
            return self.history
            
        except Exception as e:
            logger.error(f"Failed to load optimization history: {e}")
            return []
    
    def get_best_results(self, n_results: int = 10) -> List[Dict[str, Any]]:
        """Get best optimization results"""
        if not self.history:
            return []
            
        # Sort by best_value (assuming maximization)
        sorted_results = sorted(
            self.history, 
            key=lambda x: x.get('best_value', 0), 
            reverse=True
        )
        
        return sorted_results[:n_results]