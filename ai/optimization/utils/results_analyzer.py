"""
📊 Results Analyzer
Analysis and reporting tools for optimization results
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

from core.logger import get_logger
from core.path_utils import get_project_root

logger = get_logger(__name__)


class ResultsAnalyzer:
    """
    Comprehensive analyzer for optimization results.
    
    Provides analysis, comparison, and visualization of optimization outcomes.
    """
    
    def __init__(self):
        """Initialize results analyzer."""
        self.results_dir = get_project_root() / "ai" / "optimization" / "results" / "analysis"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_genetic_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze genetic algorithm optimization results.
        
        Args:
            results: Genetic algorithm results
            
        Returns:
            Dict[str, Any]: Analysis report
        """
        analysis = {
            'optimization_summary': self._analyze_optimization_summary(results),
            'convergence_analysis': self._analyze_convergence(results),
            'parameter_analysis': self._analyze_best_parameters(results),
            'generation_statistics': self._analyze_generation_stats(results),
            'performance_metrics': self._calculate_performance_metrics(results)
        }
        
        return analysis
    
    def analyze_pareto_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze Pareto optimization results.
        
        Args:
            results: Pareto optimization results
            
        Returns:
            Dict[str, Any]: Analysis report
        """
        analysis = {
            'pareto_summary': self._analyze_pareto_summary(results),
            'front_quality': self._analyze_front_quality(results),
            'trade_off_analysis': self._analyze_trade_offs(results),
            'solution_diversity': self._analyze_solution_diversity(results),
            'compromise_solutions': self._analyze_compromise_solutions(results)
        }
        
        return analysis
    
    def compare_optimizations(
        self,
        results_list: List[Dict[str, Any]],
        labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple optimization results.
        
        Args:
            results_list: List of optimization results
            labels: Labels for each result set
            
        Returns:
            Dict[str, Any]: Comparison analysis
        """
        if labels is None:
            labels = [f"Optimization_{i+1}" for i in range(len(results_list))]
        
        comparison = {
            'summary_comparison': self._compare_summaries(results_list, labels),
            'convergence_comparison': self._compare_convergence(results_list, labels),
            'parameter_comparison': self._compare_parameters(results_list, labels),
            'performance_ranking': self._rank_optimizations(results_list, labels)
        }
        
        return comparison
    
    def _analyze_optimization_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze basic optimization summary."""
        return {
            'best_fitness': results.get('best_fitness', 0),
            'generations_completed': results.get('generations_completed', 0),
            'converged': results.get('converged', False),
            'total_evaluations': self._estimate_total_evaluations(results),
            'success_rate': 1.0 if results.get('best_fitness', 0) > 0 else 0.0
        }
    
    def _analyze_convergence(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze convergence behavior."""
        fitness_history = results.get('fitness_history', [])
        
        if not fitness_history:
            return {'error': 'No fitness history available'}
        
        fitness_array = np.array(fitness_history)
        
        # Find convergence point
        convergence_generation = self._find_convergence_point(fitness_array)
        
        # Calculate improvement rate
        if len(fitness_array) > 1:
            improvement_rate = (fitness_array[-1] - fitness_array[0]) / len(fitness_array)
        else:
            improvement_rate = 0.0
        
        return {
            'convergence_generation': convergence_generation,
            'improvement_rate': float(improvement_rate),
            'final_fitness': float(fitness_array[-1]),
            'fitness_variance': float(np.var(fitness_array)),
            'fitness_trend': self._calculate_fitness_trend(fitness_array),
            'plateau_analysis': self._analyze_fitness_plateau(fitness_array)
        }
    
    def _analyze_best_parameters(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze best parameters found."""
        best_params = results.get('best_parameters', {})
        
        if not best_params:
            return {'error': 'No best parameters available'}
        
        analysis = {
            'parameter_count': len(best_params),
            'parameter_types': self._categorize_parameters(best_params),
            'parameter_values': best_params,
            'parameter_ranges': self._analyze_parameter_ranges(results)
        }
        
        return analysis
    
    def _analyze_generation_stats(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze generation-by-generation statistics."""
        generation_stats = results.get('generation_stats', [])
        
        if not generation_stats:
            return {'error': 'No generation statistics available'}
        
        # Extract fitness progression
        best_fitness_per_gen = [gen.get('best_fitness', 0) for gen in generation_stats]
        avg_fitness_per_gen = [gen.get('avg_fitness', 0) for gen in generation_stats]
        
        return {
            'fitness_progression': best_fitness_per_gen,
            'average_fitness_progression': avg_fitness_per_gen,
            'diversity_over_time': [gen.get('std_fitness', 0) for gen in generation_stats],
            'improvement_generations': self._find_improvement_generations(best_fitness_per_gen),
            'stagnation_periods': self._find_stagnation_periods(best_fitness_per_gen)
        }
    
    def _calculate_performance_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance metrics."""
        config = results.get('optimization_config', {})
        
        return {
            'efficiency': self._calculate_efficiency(results),
            'convergence_speed': self._calculate_convergence_speed(results),
            'exploration_exploitation_ratio': self._calculate_exploration_ratio(results),
            'parameter_space_coverage': self._estimate_space_coverage(results),
            'algorithm_effectiveness': self._calculate_effectiveness(results)
        }
    
    def _analyze_pareto_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Pareto front summary."""
        pareto_front = results.get('pareto_front', [])
        
        return {
            'front_size': len(pareto_front),
            'num_objectives': len(pareto_front[0]['objectives']) if pareto_front else 0,
            'generations_completed': results.get('generations_completed', 0),
            'total_solutions_evaluated': self._estimate_total_evaluations(results)
        }
    
    def _analyze_front_quality(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze quality of Pareto front."""
        pareto_analysis = results.get('pareto_analysis', {})
        
        return {
            'diversity_score': pareto_analysis.get('diversity_metrics', {}).get('diversity_score', 0),
            'convergence_score': pareto_analysis.get('convergence_metrics', {}).get('convergence_score', 0),
            'hypervolume': pareto_analysis.get('convergence_metrics', {}).get('hypervolume', 0),
            'front_spread': pareto_analysis.get('diversity_metrics', {}).get('extent', 0)
        }
    
    def _analyze_trade_offs(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trade-offs in Pareto front."""
        pareto_analysis = results.get('pareto_analysis', {})
        trade_off_analysis = pareto_analysis.get('trade_off_analysis', {})
        
        return {
            'significant_trade_offs': trade_off_analysis.get('significant_trade_offs', []),
            'correlation_matrix': trade_off_analysis.get('correlation_matrix', []),
            'trade_off_strength': self._calculate_trade_off_strength(trade_off_analysis)
        }
    
    def _analyze_solution_diversity(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze diversity of solutions."""
        pareto_front = results.get('pareto_front', [])
        
        if not pareto_front:
            return {'error': 'No Pareto front available'}
        
        # Extract objectives
        objectives = [sol['objectives'] for sol in pareto_front]
        objectives_matrix = np.array(objectives)
        
        return {
            'objective_ranges': self._calculate_objective_ranges(objectives_matrix),
            'solution_spacing': self._calculate_solution_spacing(objectives_matrix),
            'coverage_area': self._calculate_coverage_area(objectives_matrix),
            'uniformity_score': self._calculate_uniformity_score(objectives_matrix)
        }
    
    def _analyze_compromise_solutions(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze compromise solutions."""
        compromise_solutions = results.get('compromise_solutions', [])
        
        analysis = {}
        for solution in compromise_solutions:
            sol_type = solution.get('type', 'unknown')
            analysis[sol_type] = {
                'parameters': solution.get('solution', {}).get('parameters', {}),
                'objectives': solution.get('solution', {}).get('objectives', []),
                'description': solution.get('description', '')
            }
        
        return analysis
    
    # Helper methods
    def _estimate_total_evaluations(self, results: Dict[str, Any]) -> int:
        """Estimate total number of fitness evaluations."""
        generations = results.get('generations_completed', 0)
        config = results.get('optimization_config', {})
        population_size = config.get('population_size', 100)
        
        return generations * population_size
    
    def _find_convergence_point(self, fitness_array: np.ndarray) -> int:
        """Find the generation where convergence occurred."""
        if len(fitness_array) < 10:
            return len(fitness_array)
        
        # Look for point where improvement becomes minimal
        improvements = np.diff(fitness_array)
        smoothed_improvements = np.convolve(improvements, np.ones(5)/5, mode='valid')
        
        # Find where improvements drop below threshold
        threshold = np.std(improvements) * 0.1
        convergence_candidates = np.where(smoothed_improvements < threshold)[0]
        
        return int(convergence_candidates[0] + 5) if len(convergence_candidates) > 0 else len(fitness_array)
    
    def _calculate_fitness_trend(self, fitness_array: np.ndarray) -> str:
        """Calculate overall fitness trend."""
        if len(fitness_array) < 2:
            return 'insufficient_data'
        
        slope = np.polyfit(range(len(fitness_array)), fitness_array, 1)[0]
        
        if slope > 0.01:
            return 'improving'
        elif slope < -0.01:
            return 'declining'
        else:
            return 'stable'
    
    def _analyze_fitness_plateau(self, fitness_array: np.ndarray) -> Dict[str, Any]:
        """Analyze fitness plateau periods."""
        if len(fitness_array) < 10:
            return {'plateau_detected': False}
        
        # Find periods of little change
        changes = np.abs(np.diff(fitness_array))
        threshold = np.std(changes) * 0.1
        
        plateau_mask = changes < threshold
        plateau_periods = []
        
        in_plateau = False
        start_idx = 0
        
        for i, is_plateau in enumerate(plateau_mask):
            if is_plateau and not in_plateau:
                in_plateau = True
                start_idx = i
            elif not is_plateau and in_plateau:
                in_plateau = False
                if i - start_idx > 5:  # Minimum plateau length
                    plateau_periods.append({'start': start_idx, 'end': i, 'length': i - start_idx})
        
        return {
            'plateau_detected': len(plateau_periods) > 0,
            'plateau_periods': plateau_periods,
            'total_plateau_generations': sum(p['length'] for p in plateau_periods),
            'plateau_ratio': sum(p['length'] for p in plateau_periods) / len(fitness_array)
        }
    
    def _categorize_parameters(self, parameters: Dict[str, Any]) -> Dict[str, int]:
        """Categorize parameters by type."""
        categories = {'numeric': 0, 'categorical': 0, 'boolean': 0}
        
        for value in parameters.values():
            if isinstance(value, bool):
                categories['boolean'] += 1
            elif isinstance(value, (int, float)):
                categories['numeric'] += 1
            else:
                categories['categorical'] += 1
        
        return categories
    
    def _calculate_efficiency(self, results: Dict[str, Any]) -> float:
        """Calculate optimization efficiency."""
        best_fitness = results.get('best_fitness', 0)
        total_evaluations = self._estimate_total_evaluations(results)
        
        if total_evaluations == 0:
            return 0.0
        
        # Normalize by typical values (this is a simplified metric)
        efficiency = best_fitness / (total_evaluations / 1000)  # Per 1000 evaluations
        
        return float(np.clip(efficiency, 0, 10))
    
    def _calculate_convergence_speed(self, results: Dict[str, Any]) -> float:
        """Calculate convergence speed."""
        convergence_info = self._analyze_convergence(results)
        convergence_gen = convergence_info.get('convergence_generation', float('inf'))
        total_generations = results.get('generations_completed', 1)
        
        if convergence_gen == float('inf'):
            return 0.0
        
        # Speed is inverse of convergence generation ratio
        speed = 1.0 - (convergence_gen / total_generations)
        
        return float(max(speed, 0.0))
    
    def _calculate_exploration_ratio(self, results: Dict[str, Any]) -> float:
        """Calculate exploration vs exploitation ratio."""
        generation_stats = results.get('generation_stats', [])
        
        if not generation_stats:
            return 0.5
        
        # Use fitness diversity as proxy for exploration
        diversities = [gen.get('std_fitness', 0) for gen in generation_stats]
        
        if not diversities:
            return 0.5
        
        # High diversity early = good exploration
        early_diversity = np.mean(diversities[:len(diversities)//3])
        late_diversity = np.mean(diversities[-len(diversities)//3:])
        
        if early_diversity + late_diversity == 0:
            return 0.5
        
        exploration_ratio = early_diversity / (early_diversity + late_diversity)
        
        return float(np.clip(exploration_ratio, 0, 1))
    
    def _estimate_space_coverage(self, results: Dict[str, Any]) -> float:
        """Estimate parameter space coverage."""
        # This is a simplified estimation
        # In practice, would need actual parameter diversity data
        generations = results.get('generations_completed', 0)
        population_size = results.get('optimization_config', {}).get('population_size', 100)
        
        # Rough estimation based on search effort
        search_effort = generations * population_size
        coverage = 1.0 - np.exp(-search_effort / 10000)  # Asymptotic approach to 1
        
        return float(np.clip(coverage, 0, 1))
    
    def _calculate_effectiveness(self, results: Dict[str, Any]) -> float:
        """Calculate overall algorithm effectiveness."""
        best_fitness = results.get('best_fitness', 0)
        converged = results.get('converged', False)
        
        # Simple effectiveness metric
        fitness_score = np.tanh(best_fitness)  # Normalize to [0, 1]
        convergence_bonus = 0.2 if converged else 0.0
        
        effectiveness = fitness_score + convergence_bonus
        
        return float(np.clip(effectiveness, 0, 1))
    
    def save_analysis(self, analysis: Dict[str, Any], filename: str = None) -> str:
        """Save analysis results."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimization_analysis_{timestamp}.json"
        
        filepath = self.results_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            
            logger.info(f"Analysis results saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving analysis: {e}")
            return ""
    
    # Additional helper methods for Pareto analysis would go here...
    def _calculate_trade_off_strength(self, trade_off_analysis: Dict[str, Any]) -> float:
        """Calculate strength of trade-offs."""
        trade_offs = trade_off_analysis.get('significant_trade_offs', [])
        
        if not trade_offs:
            return 0.0
        
        # Average correlation strength
        correlations = [abs(to.get('correlation', 0)) for to in trade_offs]
        return float(np.mean(correlations))
    
    def _calculate_objective_ranges(self, objectives_matrix: np.ndarray) -> List[float]:
        """Calculate objective value ranges."""
        if objectives_matrix.size == 0:
            return []
        
        ranges = []
        for i in range(objectives_matrix.shape[1]):
            obj_values = objectives_matrix[:, i]
            range_val = np.max(obj_values) - np.min(obj_values)
            ranges.append(float(range_val))
        
        return ranges
    
    def _calculate_solution_spacing(self, objectives_matrix: np.ndarray) -> float:
        """Calculate average spacing between solutions."""
        if len(objectives_matrix) < 2:
            return 0.0
        
        distances = []
        for i in range(len(objectives_matrix)):
            for j in range(i + 1, len(objectives_matrix)):
                dist = np.linalg.norm(objectives_matrix[i] - objectives_matrix[j])
                distances.append(dist)
        
        return float(np.mean(distances))
    
    def _calculate_coverage_area(self, objectives_matrix: np.ndarray) -> float:
        """Calculate coverage area of objectives."""
        if objectives_matrix.size == 0:
            return 0.0
        
        # Simple area calculation for 2D case
        if objectives_matrix.shape[1] == 2:
            x_range = np.max(objectives_matrix[:, 0]) - np.min(objectives_matrix[:, 0])
            y_range = np.max(objectives_matrix[:, 1]) - np.min(objectives_matrix[:, 1])
            return float(x_range * y_range)
        
        # For higher dimensions, use volume approximation
        volume = 1.0
        for i in range(objectives_matrix.shape[1]):
            range_val = np.max(objectives_matrix[:, i]) - np.min(objectives_matrix[:, i])
            volume *= range_val
        
        return float(volume)
    
    def _calculate_uniformity_score(self, objectives_matrix: np.ndarray) -> float:
        """Calculate uniformity of solution distribution."""
        if len(objectives_matrix) < 3:
            return 1.0
        
        # Calculate distances between consecutive solutions (sorted by first objective)
        sorted_indices = np.argsort(objectives_matrix[:, 0])
        sorted_objectives = objectives_matrix[sorted_indices]
        
        distances = []
        for i in range(len(sorted_objectives) - 1):
            dist = np.linalg.norm(sorted_objectives[i + 1] - sorted_objectives[i])
            distances.append(dist)
        
        # Uniformity is inverse of distance variance
        if len(distances) > 0:
            distance_std = np.std(distances)
            mean_distance = np.mean(distances)
            
            if mean_distance > 0:
                uniformity = 1.0 / (1.0 + distance_std / mean_distance)
            else:
                uniformity = 1.0
        else:
            uniformity = 1.0
        
        return float(np.clip(uniformity, 0, 1))