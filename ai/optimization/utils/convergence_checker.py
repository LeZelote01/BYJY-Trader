"""
📊 Convergence Checker
Monitor and detect convergence in optimization algorithms
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import deque

from core.logger import get_logger

logger = get_logger(__name__)


class ConvergenceChecker:
    """
    Monitor convergence of optimization algorithms.
    
    Provides various convergence criteria and early stopping mechanisms.
    """
    
    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 1e-6,
        window_size: int = 10
    ):
        """
        Initialize convergence checker.
        
        Args:
            patience: Number of iterations without improvement for early stopping
            min_delta: Minimum change considered as improvement
            window_size: Window size for trend analysis
        """
        self.patience = patience
        self.min_delta = min_delta
        self.window_size = window_size
        
        # History tracking
        self.fitness_history = deque(maxlen=1000)
        self.improvement_count = 0
        self.best_fitness = -float('inf')
        self.iterations_without_improvement = 0
        
        # Convergence flags
        self.converged = False
        self.convergence_reason = None
    
    def update(self, fitness: float) -> bool:
        """
        Update convergence checker with new fitness value.
        
        Args:
            fitness: Current best fitness value
            
        Returns:
            bool: True if converged, False otherwise
        """
        self.fitness_history.append(fitness)
        
        # Check for improvement
        if fitness > self.best_fitness + self.min_delta:
            self.best_fitness = fitness
            self.improvement_count += 1
            self.iterations_without_improvement = 0
        else:
            self.iterations_without_improvement += 1
        
        # Check convergence criteria
        self.converged, self.convergence_reason = self._check_convergence()
        
        return self.converged
    
    def _check_convergence(self) -> Tuple[bool, Optional[str]]:
        """Check various convergence criteria."""
        
        # Early stopping based on patience
        if self.iterations_without_improvement >= self.patience:
            return True, f"Early stopping: no improvement for {self.patience} iterations"
        
        # Insufficient history
        if len(self.fitness_history) < self.window_size:
            return False, None
        
        # Fitness variance convergence
        recent_fitness = list(self.fitness_history)[-self.window_size:]
        fitness_variance = np.var(recent_fitness)
        
        if fitness_variance < self.min_delta:
            return True, f"Fitness variance convergence: variance={fitness_variance:.2e}"
        
        # Trend-based convergence
        if self._check_trend_convergence():
            return True, "Trend-based convergence: flat trend detected"
        
        # Relative improvement convergence
        if self._check_relative_improvement():
            return True, "Relative improvement convergence: improvements too small"
        
        return False, None
    
    def _check_trend_convergence(self) -> bool:
        """Check if fitness trend is flat (converged)."""
        if len(self.fitness_history) < self.window_size:
            return False
        
        recent_fitness = np.array(list(self.fitness_history)[-self.window_size:])
        
        # Linear regression to check trend
        x = np.arange(len(recent_fitness))
        slope = np.polyfit(x, recent_fitness, 1)[0]
        
        # Check if slope is effectively zero
        return abs(slope) < self.min_delta
    
    def _check_relative_improvement(self) -> bool:
        """Check if relative improvements are too small."""
        if len(self.fitness_history) < 2:
            return False
        
        current_fitness = self.fitness_history[-1]
        previous_fitness = self.fitness_history[-2]
        
        if abs(previous_fitness) < 1e-10:
            return False
        
        relative_improvement = abs(current_fitness - previous_fitness) / abs(previous_fitness)
        
        return relative_improvement < self.min_delta
    
    def get_convergence_info(self) -> Dict[str, Any]:
        """Get detailed convergence information."""
        return {
            'converged': self.converged,
            'convergence_reason': self.convergence_reason,
            'best_fitness': self.best_fitness,
            'iterations_without_improvement': self.iterations_without_improvement,
            'total_improvements': self.improvement_count,
            'fitness_history_length': len(self.fitness_history),
            'current_fitness_variance': self._get_current_variance(),
            'trend_slope': self._get_trend_slope()
        }
    
    def _get_current_variance(self) -> float:
        """Get current fitness variance."""
        if len(self.fitness_history) < self.window_size:
            return float('inf')
        
        recent_fitness = list(self.fitness_history)[-self.window_size:]
        return float(np.var(recent_fitness))
    
    def _get_trend_slope(self) -> float:
        """Get current trend slope."""
        if len(self.fitness_history) < self.window_size:
            return 0.0
        
        recent_fitness = np.array(list(self.fitness_history)[-self.window_size:])
        x = np.arange(len(recent_fitness))
        slope = np.polyfit(x, recent_fitness, 1)[0]
        
        return float(slope)
    
    def reset(self):
        """Reset convergence checker state."""
        self.fitness_history.clear()
        self.improvement_count = 0
        self.best_fitness = -float('inf')
        self.iterations_without_improvement = 0
        self.converged = False
        self.convergence_reason = None
    
    def should_stop(self) -> bool:
        """Check if optimization should stop."""
        return self.converged
    
    def get_early_stopping_info(self) -> Dict[str, Any]:
        """Get early stopping information."""
        return {
            'should_stop': self.should_stop(),
            'patience': self.patience,
            'iterations_without_improvement': self.iterations_without_improvement,
            'remaining_patience': max(0, self.patience - self.iterations_without_improvement)
        }


class MultiObjectiveConvergenceChecker:
    """
    Convergence checker for multi-objective optimization.
    
    Monitors convergence of Pareto front in multi-objective optimization.
    """
    
    def __init__(
        self,
        patience: int = 20,
        min_hypervolume_improvement: float = 1e-4,
        window_size: int = 10
    ):
        """
        Initialize multi-objective convergence checker.
        
        Args:
            patience: Patience for early stopping
            min_hypervolume_improvement: Minimum hypervolume improvement
            window_size: Window size for analysis
        """
        self.patience = patience
        self.min_hypervolume_improvement = min_hypervolume_improvement
        self.window_size = window_size
        
        # Tracking
        self.hypervolume_history = deque(maxlen=1000)
        self.front_size_history = deque(maxlen=1000)
        self.best_hypervolume = 0.0
        self.iterations_without_improvement = 0
        
        # Convergence state
        self.converged = False
        self.convergence_reason = None
    
    def update(
        self,
        pareto_front: List[Dict[str, Any]],
        hypervolume: Optional[float] = None
    ) -> bool:
        """
        Update with new Pareto front.
        
        Args:
            pareto_front: Current Pareto front
            hypervolume: Precomputed hypervolume (optional)
            
        Returns:
            bool: True if converged
        """
        # Calculate hypervolume if not provided
        if hypervolume is None:
            hypervolume = self._calculate_simple_hypervolume(pareto_front)
        
        self.hypervolume_history.append(hypervolume)
        self.front_size_history.append(len(pareto_front))
        
        # Check for improvement
        if hypervolume > self.best_hypervolume + self.min_hypervolume_improvement:
            self.best_hypervolume = hypervolume
            self.iterations_without_improvement = 0
        else:
            self.iterations_without_improvement += 1
        
        # Check convergence
        self.converged, self.convergence_reason = self._check_mo_convergence()
        
        return self.converged
    
    def _calculate_simple_hypervolume(
        self,
        pareto_front: List[Dict[str, Any]]
    ) -> float:
        """Calculate simplified hypervolume approximation."""
        if not pareto_front:
            return 0.0
        
        # Extract objectives
        objectives_matrix = np.array([sol['objectives'] for sol in pareto_front])
        
        if objectives_matrix.shape[1] == 2:
            # 2D case: calculate actual hypervolume
            return self._calculate_2d_hypervolume(objectives_matrix)
        else:
            # Higher dimensions: use approximation
            return float(np.sum(np.prod(objectives_matrix, axis=1)))
    
    def _calculate_2d_hypervolume(self, objectives_matrix: np.ndarray) -> float:
        """Calculate exact hypervolume for 2D case."""
        # Sort by first objective
        sorted_indices = np.argsort(objectives_matrix[:, 0])
        sorted_points = objectives_matrix[sorted_indices]
        
        # Reference point (assuming minimization - adjust as needed)
        ref_point = np.min(objectives_matrix, axis=0) - 1.0
        
        hypervolume = 0.0
        prev_x = ref_point[0]
        
        for point in sorted_points:
            width = point[0] - prev_x
            height = point[1] - ref_point[1]
            hypervolume += width * height
            prev_x = point[0]
        
        return float(hypervolume)
    
    def _check_mo_convergence(self) -> Tuple[bool, Optional[str]]:
        """Check multi-objective convergence criteria."""
        
        # Early stopping based on patience
        if self.iterations_without_improvement >= self.patience:
            return True, f"Early stopping: no hypervolume improvement for {self.patience} iterations"
        
        # Insufficient history
        if len(self.hypervolume_history) < self.window_size:
            return False, None
        
        # Hypervolume stability
        recent_hv = list(self.hypervolume_history)[-self.window_size:]
        hv_variance = np.var(recent_hv)
        
        if hv_variance < self.min_hypervolume_improvement:
            return True, f"Hypervolume stability: variance={hv_variance:.2e}"
        
        # Front size stability
        recent_sizes = list(self.front_size_history)[-self.window_size:]
        if len(set(recent_sizes)) == 1:  # All sizes are the same
            return True, "Pareto front size stability"
        
        return False, None
    
    def get_mo_convergence_info(self) -> Dict[str, Any]:
        """Get multi-objective convergence information."""
        return {
            'converged': self.converged,
            'convergence_reason': self.convergence_reason,
            'best_hypervolume': self.best_hypervolume,
            'current_hypervolume': self.hypervolume_history[-1] if self.hypervolume_history else 0.0,
            'iterations_without_improvement': self.iterations_without_improvement,
            'hypervolume_history_length': len(self.hypervolume_history),
            'current_front_size': self.front_size_history[-1] if self.front_size_history else 0,
            'hypervolume_trend': self._get_hypervolume_trend()
        }
    
    def _get_hypervolume_trend(self) -> float:
        """Get hypervolume trend slope."""
        if len(self.hypervolume_history) < self.window_size:
            return 0.0
        
        recent_hv = np.array(list(self.hypervolume_history)[-self.window_size:])
        x = np.arange(len(recent_hv))
        slope = np.polyfit(x, recent_hv, 1)[0]
        
        return float(slope)
    
    def reset(self):
        """Reset multi-objective convergence checker."""
        self.hypervolume_history.clear()
        self.front_size_history.clear()
        self.best_hypervolume = 0.0
        self.iterations_without_improvement = 0
        self.converged = False
        self.convergence_reason = None