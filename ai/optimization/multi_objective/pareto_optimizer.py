"""
📊 Pareto Optimizer
Multi-objective optimization using Pareto dominance
"""

import numpy as np
import asyncio
from typing import Dict, List, Any, Tuple, Optional, Callable
from datetime import datetime
import json
from pathlib import Path

from core.logger import get_logger
from core.path_utils import get_project_root
from .nsga2 import NSGA2
from .objective_functions import ObjectiveFunctions
from .pareto_front_analyzer import ParetoFrontAnalyzer

logger = get_logger(__name__)


class ParetoOptimizer:
    """
    Multi-objective Pareto optimizer for BYJY-Trader.
    
    Optimizes multiple conflicting objectives simultaneously:
    - Maximize profit (returns, Sharpe ratio)
    - Minimize risk (drawdown, volatility)
    - Minimize training time
    - Maximize stability
    """
    
    def __init__(
        self,
        population_size: int = 100,
        num_generations: int = 200,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.1,
        random_seed: Optional[int] = None
    ):
        """
        Initialize Pareto optimizer.
        
        Args:
            population_size: Size of population
            num_generations: Number of generations
            crossover_prob: Crossover probability
            mutation_prob: Mutation probability
            random_seed: Random seed for reproducibility
        """
        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        
        # Set random seed
        if random_seed:
            np.random.seed(random_seed)
        
        # Initialize components
        self.nsga2 = NSGA2(
            population_size=population_size,
            num_generations=num_generations,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob
        )
        self.objective_functions = ObjectiveFunctions()
        self.pareto_analyzer = ParetoFrontAnalyzer()
        
        # Optimization state
        self.current_population = []
        self.pareto_front = []
        self.generation_history = []
        self.is_running = False
        self.current_generation = 0
        
        # Results storage
        self.results_dir = get_project_root() / "ai" / "optimization" / "results" / "pareto"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized ParetoOptimizer with population_size={population_size}")
    
    async def optimize(
        self,
        parameter_space: Dict[str, Any],
        objective_configs: List[Dict[str, Any]],
        evaluation_function: Callable,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run multi-objective Pareto optimization.
        
        Args:
            parameter_space: Parameter space definition
            objective_configs: Configuration for each objective
            evaluation_function: Function to evaluate solutions
            **kwargs: Additional arguments
            
        Returns:
            Dict[str, Any]: Optimization results with Pareto front
        """
        self.is_running = True
        self.current_generation = 0
        
        logger.info(f"Starting Pareto optimization for {len(objective_configs)} objectives")
        
        # Setup objective functions
        self.objective_functions.setup_objectives(objective_configs)
        
        # Run NSGA-II optimization
        nsga2_results = await self.nsga2.optimize(
            parameter_space=parameter_space,
            objective_function=self._evaluate_objectives,
            num_objectives=len(objective_configs),
            evaluation_function=evaluation_function,
            **kwargs
        )
        
        self.current_population = nsga2_results['final_population']
        self.pareto_front = nsga2_results['pareto_front']
        self.generation_history = nsga2_results['generation_history']
        
        # Analyze Pareto front
        pareto_analysis = self.pareto_analyzer.analyze_front(
            self.pareto_front, objective_configs
        )
        
        # Find compromise solutions
        compromise_solutions = self._find_compromise_solutions()
        
        # Prepare results
        results = {
            'pareto_front': [self._solution_to_dict(sol) for sol in self.pareto_front],
            'compromise_solutions': [self._solution_to_dict(sol) for sol in compromise_solutions],
            'pareto_analysis': pareto_analysis,
            'final_population_size': len(self.current_population),
            'generations_completed': self.current_generation,
            'objective_configs': objective_configs,
            'parameter_space': parameter_space,
            'optimization_config': {
                'population_size': self.population_size,
                'num_generations': self.num_generations,
                'crossover_prob': self.crossover_prob,
                'mutation_prob': self.mutation_prob
            }
        }
        
        # Save results
        await self._save_results(results)
        
        self.is_running = False
        
        logger.info(f"Pareto optimization completed. Pareto front size: {len(self.pareto_front)}")
        return results
    
    async def _evaluate_objectives(
        self,
        parameters: Dict[str, Any],
        evaluation_function: Callable,
        **kwargs
    ) -> List[float]:
        """
        Evaluate all objectives for given parameters.
        
        Args:
            parameters: Parameter dictionary
            evaluation_function: Function to get raw metrics
            **kwargs: Additional arguments
            
        Returns:
            List[float]: Objective values
        """
        # Get raw performance metrics
        if asyncio.iscoroutinefunction(evaluation_function):
            metrics = await evaluation_function(parameters, **kwargs)
        else:
            loop = asyncio.get_event_loop()
            metrics = await loop.run_in_executor(
                None, lambda: evaluation_function(parameters, **kwargs)
            )
        
        # Convert to objective values
        objective_values = self.objective_functions.evaluate(metrics)
        
        return objective_values
    
    def _find_compromise_solutions(self) -> List[Dict[str, Any]]:
        """
        Find compromise solutions from Pareto front.
        
        Returns:
            List[Dict[str, Any]]: Compromise solutions
        """
        if not self.pareto_front:
            return []
        
        compromise_solutions = []
        
        # Method 1: Solution closest to ideal point
        ideal_solution = self._find_ideal_point_solution()
        if ideal_solution:
            compromise_solutions.append({
                'type': 'ideal_point',
                'solution': ideal_solution,
                'description': 'Solution closest to ideal point'
            })
        
        # Method 2: Solution with best trade-off (minimum distance to origin in normalized space)
        tradeoff_solution = self._find_tradeoff_solution()
        if tradeoff_solution:
            compromise_solutions.append({
                'type': 'best_tradeoff',
                'solution': tradeoff_solution,
                'description': 'Best balanced trade-off solution'
            })
        
        # Method 3: Knee point solution (maximum margin)
        knee_solution = self._find_knee_point_solution()
        if knee_solution:
            compromise_solutions.append({
                'type': 'knee_point',
                'solution': knee_solution,
                'description': 'Knee point with maximum margin'
            })
        
        return compromise_solutions
    
    def _find_ideal_point_solution(self) -> Optional[Dict[str, Any]]:
        """Find solution closest to ideal point."""
        if not self.pareto_front:
            return None
        
        # Extract objective values
        objective_values = np.array([sol['objectives'] for sol in self.pareto_front])
        
        # Find ideal point (best value for each objective)
        # For minimization objectives, ideal is minimum
        # For maximization objectives, ideal is maximum
        ideal_point = []
        for i in range(objective_values.shape[1]):
            obj_config = self.objective_functions.objective_configs[i]
            if obj_config.get('maximize', True):
                ideal_point.append(np.max(objective_values[:, i]))
            else:
                ideal_point.append(np.min(objective_values[:, i]))
        
        ideal_point = np.array(ideal_point)
        
        # Normalize objectives to [0, 1]
        normalized_objectives = self._normalize_objectives(objective_values)
        normalized_ideal = np.ones(len(ideal_point))  # All objectives at their best
        
        # Find solution closest to ideal point
        distances = np.linalg.norm(normalized_objectives - normalized_ideal, axis=1)
        closest_idx = np.argmin(distances)
        
        return self.pareto_front[closest_idx]
    
    def _find_tradeoff_solution(self) -> Optional[Dict[str, Any]]:
        """Find best balanced trade-off solution."""
        if not self.pareto_front:
            return None
        
        # Extract and normalize objective values
        objective_values = np.array([sol['objectives'] for sol in self.pareto_front])
        normalized_objectives = self._normalize_objectives(objective_values)
        
        # Find solution with minimum distance to origin in normalized space
        distances = np.linalg.norm(normalized_objectives, axis=1)
        best_tradeoff_idx = np.argmin(distances)
        
        return self.pareto_front[best_tradeoff_idx]
    
    def _find_knee_point_solution(self) -> Optional[Dict[str, Any]]:
        """Find knee point solution using maximum margin approach."""
        if len(self.pareto_front) < 3:
            return None
        
        # Extract and normalize objective values
        objective_values = np.array([sol['objectives'] for sol in self.pareto_front])
        
        if objective_values.shape[1] != 2:
            # Knee point detection works best for 2D, use tradeoff for higher dimensions
            return self._find_tradeoff_solution()
        
        normalized_objectives = self._normalize_objectives(objective_values)
        
        # Sort points along the front
        sorted_indices = np.lexsort((normalized_objectives[:, 1], normalized_objectives[:, 0]))
        sorted_objectives = normalized_objectives[sorted_indices]
        
        # Calculate knee point using maximum curvature
        max_curvature = 0
        knee_idx = 0
        
        for i in range(1, len(sorted_objectives) - 1):
            # Calculate curvature using three consecutive points
            p1 = sorted_objectives[i - 1]
            p2 = sorted_objectives[i]
            p3 = sorted_objectives[i + 1]
            
            # Calculate angle using dot product
            v1 = p1 - p2
            v2 = p3 - p2
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            curvature = 1 - cos_angle  # Higher value = more curved
            
            if curvature > max_curvature:
                max_curvature = curvature
                knee_idx = sorted_indices[i]
        
        return self.pareto_front[knee_idx]
    
    def _normalize_objectives(self, objective_values: np.ndarray) -> np.ndarray:
        """Normalize objective values to [0, 1] range."""
        normalized = objective_values.copy()
        
        for i in range(objective_values.shape[1]):
            obj_min = np.min(objective_values[:, i])
            obj_max = np.max(objective_values[:, i])
            
            if obj_max > obj_min:
                normalized[:, i] = (objective_values[:, i] - obj_min) / (obj_max - obj_min)
            else:
                normalized[:, i] = 0.5  # All values are the same
        
        return normalized
    
    def _solution_to_dict(self, solution: Dict[str, Any]) -> Dict[str, Any]:
        """Convert solution to serializable dictionary."""
        return {
            'parameters': solution.get('parameters', {}),
            'objectives': solution.get('objectives', []),
            'rank': solution.get('rank', 0),
            'crowding_distance': solution.get('crowding_distance', 0.0),
            'evaluation_metrics': solution.get('evaluation_metrics', {})
        }
    
    async def _save_results(self, results: Dict[str, Any]):
        """Save optimization results to disk."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = self.results_dir / f"pareto_optimization_{timestamp}.json"
            
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Pareto optimization results saved to {results_path}")
            
        except Exception as e:
            logger.error(f"Error saving Pareto results: {e}")
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status."""
        if not self.is_running:
            return {
                'status': 'idle',
                'current_generation': 0,
                'pareto_front_size': len(self.pareto_front)
            }
        
        return {
            'status': 'running',
            'current_generation': self.current_generation,
            'total_generations': self.num_generations,
            'progress_percent': (self.current_generation / self.num_generations) * 100,
            'population_size': len(self.current_population),
            'current_pareto_front_size': len(self.pareto_front)
        }
    
    def stop_optimization(self):
        """Stop optimization early."""
        self.is_running = False
        if self.nsga2:
            self.nsga2.stop_optimization()
        logger.info("Pareto optimization stopped by user request")