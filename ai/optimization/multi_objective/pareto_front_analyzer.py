"""
📊 Pareto Front Analyzer
Analysis and visualization tools for Pareto fronts
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path
import json

from core.logger import get_logger
from core.path_utils import get_project_root

logger = get_logger(__name__)


class ParetoFrontAnalyzer:
    """
    Analyzer for Pareto fronts in multi-objective optimization.
    
    Provides analysis, metrics, and visualization tools for
    understanding trade-offs in the Pareto front.
    """
    
    def __init__(self):
        """Initialize Pareto front analyzer."""
        self.results_dir = get_project_root() / "ai" / "optimization" / "results" / "analysis"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_front(
        self,
        pareto_front: List[Dict[str, Any]],
        objective_configs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of Pareto front.
        
        Args:
            pareto_front: List of Pareto-optimal solutions
            objective_configs: Configuration for objectives
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        if not pareto_front:
            return {'error': 'Empty Pareto front'}
        
        # Extract objectives matrix
        objectives_matrix = np.array([sol['objectives'] for sol in pareto_front])
        num_objectives = objectives_matrix.shape[1]
        
        analysis_results = {
            'front_size': len(pareto_front),
            'num_objectives': num_objectives,
            'objective_configs': objective_configs,
            'statistics': self._calculate_statistics(objectives_matrix),
            'diversity_metrics': self._calculate_diversity_metrics(objectives_matrix),
            'convergence_metrics': self._calculate_convergence_metrics(objectives_matrix),
            'trade_off_analysis': self._analyze_trade_offs(objectives_matrix, objective_configs),
            'extreme_points': self._find_extreme_points(pareto_front, objectives_matrix),
            'knee_points': self._find_knee_points(pareto_front, objectives_matrix)
        }
        
        # Add 2D specific analysis if applicable
        if num_objectives == 2:
            analysis_results['2d_analysis'] = self._analyze_2d_front(objectives_matrix)
        
        return analysis_results
    
    def _calculate_statistics(self, objectives_matrix: np.ndarray) -> Dict[str, Any]:
        """Calculate basic statistics for objectives."""
        return {
            'mean': objectives_matrix.mean(axis=0).tolist(),
            'std': objectives_matrix.std(axis=0).tolist(),
            'min': objectives_matrix.min(axis=0).tolist(),
            'max': objectives_matrix.max(axis=0).tolist(),
            'median': np.median(objectives_matrix, axis=0).tolist(),
            'range': (objectives_matrix.max(axis=0) - objectives_matrix.min(axis=0)).tolist()
        }
    
    def _calculate_diversity_metrics(self, objectives_matrix: np.ndarray) -> Dict[str, Any]:
        """Calculate diversity metrics for the front."""
        # Spacing metric (uniformity of distribution)
        spacing = self._calculate_spacing(objectives_matrix)
        
        # Extent (range coverage)
        extent = self._calculate_extent(objectives_matrix)
        
        # Spread metric
        spread = self._calculate_spread(objectives_matrix)
        
        return {
            'spacing': spacing,
            'extent': extent,
            'spread': spread,
            'diversity_score': self._calculate_diversity_score(spacing, extent, spread)
        }
    
    def _calculate_spacing(self, objectives_matrix: np.ndarray) -> float:
        """Calculate spacing metric (uniformity of distribution)."""
        if len(objectives_matrix) < 2:
            return 0.0
        
        distances = []
        
        for i in range(len(objectives_matrix)):
            # Find minimum distance to other solutions
            min_distance = float('inf')
            for j in range(len(objectives_matrix)):
                if i != j:
                    distance = np.linalg.norm(objectives_matrix[i] - objectives_matrix[j])
                    min_distance = min(min_distance, distance)
            distances.append(min_distance)
        
        # Calculate spacing as standard deviation of distances
        mean_distance = np.mean(distances)
        spacing = np.sqrt(np.mean([(d - mean_distance) ** 2 for d in distances]))
        
        return float(spacing)
    
    def _calculate_extent(self, objectives_matrix: np.ndarray) -> float:
        """Calculate extent (range coverage) metric."""
        # Euclidean distance between extreme points
        min_point = objectives_matrix.min(axis=0)
        max_point = objectives_matrix.max(axis=0)
        extent = np.linalg.norm(max_point - min_point)
        
        return float(extent)
    
    def _calculate_spread(self, objectives_matrix: np.ndarray) -> float:
        """Calculate spread metric."""
        if len(objectives_matrix) < 3:
            return 0.0
        
        # Find extreme solutions for each objective
        extreme_indices = []
        for obj_idx in range(objectives_matrix.shape[1]):
            min_idx = np.argmin(objectives_matrix[:, obj_idx])
            max_idx = np.argmax(objectives_matrix[:, obj_idx])
            extreme_indices.extend([min_idx, max_idx])
        
        extreme_indices = list(set(extreme_indices))  # Remove duplicates
        
        if len(extreme_indices) < 2:
            return 0.0
        
        # Calculate distances between consecutive solutions (sorted)
        sorted_indices = np.argsort(objectives_matrix[:, 0])  # Sort by first objective
        consecutive_distances = []
        
        for i in range(len(sorted_indices) - 1):
            curr_point = objectives_matrix[sorted_indices[i]]
            next_point = objectives_matrix[sorted_indices[i + 1]]
            distance = np.linalg.norm(next_point - curr_point)
            consecutive_distances.append(distance)
        
        if not consecutive_distances:
            return 0.0
        
        # Spread calculation
        mean_distance = np.mean(consecutive_distances)
        spread = np.sum([abs(d - mean_distance) for d in consecutive_distances])
        
        return float(spread)
    
    def _calculate_diversity_score(
        self,
        spacing: float,
        extent: float,
        spread: float
    ) -> float:
        """Calculate overall diversity score."""
        # Normalize metrics to [0, 1] and combine
        # Lower spacing = better diversity
        # Higher extent = better diversity
        # Lower spread = better diversity
        
        # Simple weighted combination (can be improved)
        diversity_score = 0.4 * (1.0 / (1.0 + spacing)) + 0.3 * np.tanh(extent) + 0.3 * (1.0 / (1.0 + spread))
        
        return float(diversity_score)
    
    def _calculate_convergence_metrics(self, objectives_matrix: np.ndarray) -> Dict[str, Any]:
        """Calculate convergence metrics."""
        # Hypervolume indicator (simplified)
        hypervolume = self._calculate_hypervolume(objectives_matrix)
        
        # Generational Distance (requires reference front - using ideal point)
        generational_distance = self._calculate_generational_distance(objectives_matrix)
        
        return {
            'hypervolume': hypervolume,
            'generational_distance': generational_distance,
            'convergence_score': self._calculate_convergence_score(hypervolume, generational_distance)
        }
    
    def _calculate_hypervolume(self, objectives_matrix: np.ndarray) -> float:
        """Calculate hypervolume indicator (simplified for 2D/3D)."""
        if objectives_matrix.shape[1] == 2:
            return self._calculate_2d_hypervolume(objectives_matrix)
        elif objectives_matrix.shape[1] == 3:
            return self._calculate_3d_hypervolume(objectives_matrix)
        else:
            # For higher dimensions, use approximation
            return self._approximate_hypervolume(objectives_matrix)
    
    def _calculate_2d_hypervolume(self, objectives_matrix: np.ndarray) -> float:
        """Calculate exact hypervolume for 2D case."""
        # Sort by first objective
        sorted_indices = np.argsort(objectives_matrix[:, 0])
        sorted_points = objectives_matrix[sorted_indices]
        
        # Reference point (worst values)
        ref_point = objectives_matrix.min(axis=0) - 1.0
        
        hypervolume = 0.0
        prev_x = ref_point[0]
        
        for point in sorted_points:
            width = point[0] - prev_x
            height = point[1] - ref_point[1]
            hypervolume += width * height
            prev_x = point[0]
        
        return float(hypervolume)
    
    def _calculate_3d_hypervolume(self, objectives_matrix: np.ndarray) -> float:
        """Calculate approximate hypervolume for 3D case."""
        # Simplified calculation for 3D
        ref_point = objectives_matrix.min(axis=0) - 1.0
        
        total_volume = 0.0
        for point in objectives_matrix:
            volume = np.prod(point - ref_point)
            total_volume += volume
        
        return float(total_volume)
    
    def _approximate_hypervolume(self, objectives_matrix: np.ndarray) -> float:
        """Approximate hypervolume for higher dimensions."""
        # Monte Carlo approximation
        ref_point = objectives_matrix.min(axis=0) - 1.0
        max_point = objectives_matrix.max(axis=0) + 1.0
        
        n_samples = 10000
        dominated_count = 0
        
        for _ in range(n_samples):
            # Generate random point
            random_point = np.random.uniform(ref_point, max_point)
            
            # Check if dominated by any solution in the front
            for solution in objectives_matrix:
                if np.all(solution >= random_point):
                    dominated_count += 1
                    break
        
        # Approximate hypervolume
        total_volume = np.prod(max_point - ref_point)
        hypervolume = (dominated_count / n_samples) * total_volume
        
        return float(hypervolume)
    
    def _calculate_generational_distance(self, objectives_matrix: np.ndarray) -> float:
        """Calculate generational distance to ideal point."""
        # Use ideal point as reference
        ideal_point = objectives_matrix.max(axis=0)  # Best values for each objective
        
        distances = []
        for point in objectives_matrix:
            distance = np.linalg.norm(point - ideal_point)
            distances.append(distance)
        
        generational_distance = np.mean(distances)
        
        return float(generational_distance)
    
    def _calculate_convergence_score(
        self,
        hypervolume: float,
        generational_distance: float
    ) -> float:
        """Calculate overall convergence score."""
        # Higher hypervolume = better convergence
        # Lower generational distance = better convergence
        
        hv_score = np.tanh(hypervolume)
        gd_score = 1.0 / (1.0 + generational_distance)
        
        convergence_score = 0.6 * hv_score + 0.4 * gd_score
        
        return float(convergence_score)
    
    def _analyze_trade_offs(
        self,
        objectives_matrix: np.ndarray,
        objective_configs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze trade-offs between objectives."""
        num_objectives = objectives_matrix.shape[1]
        
        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(objectives_matrix.T)
        
        # Identify strong trade-offs (negative correlations)
        trade_offs = []
        
        for i in range(num_objectives):
            for j in range(i + 1, num_objectives):
                correlation = correlation_matrix[i, j]
                
                if correlation < -0.5:  # Strong negative correlation
                    trade_offs.append({
                        'objective1': objective_configs[i]['name'],
                        'objective2': objective_configs[j]['name'],
                        'correlation': float(correlation),
                        'strength': 'strong' if correlation < -0.7 else 'moderate'
                    })
        
        return {
            'correlation_matrix': correlation_matrix.tolist(),
            'significant_trade_offs': trade_offs,
            'num_trade_offs': len(trade_offs)
        }
    
    def _find_extreme_points(
        self,
        pareto_front: List[Dict[str, Any]],
        objectives_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """Find extreme points in the Pareto front."""
        extreme_points = {}
        
        for obj_idx in range(objectives_matrix.shape[1]):
            # Find minimum and maximum for each objective
            min_idx = np.argmin(objectives_matrix[:, obj_idx])
            max_idx = np.argmax(objectives_matrix[:, obj_idx])
            
            obj_name = f'objective_{obj_idx}'
            extreme_points[f'{obj_name}_min'] = {
                'index': int(min_idx),
                'solution': pareto_front[min_idx],
                'objective_value': float(objectives_matrix[min_idx, obj_idx])
            }
            extreme_points[f'{obj_name}_max'] = {
                'index': int(max_idx),
                'solution': pareto_front[max_idx],
                'objective_value': float(objectives_matrix[max_idx, obj_idx])
            }
        
        return extreme_points
    
    def _find_knee_points(
        self,
        pareto_front: List[Dict[str, Any]],
        objectives_matrix: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Find knee points in the Pareto front."""
        if objectives_matrix.shape[1] != 2 or len(pareto_front) < 3:
            return []  # Knee point detection works best for 2D
        
        # Sort points by first objective
        sorted_indices = np.argsort(objectives_matrix[:, 0])
        sorted_objectives = objectives_matrix[sorted_indices]
        
        knee_points = []
        
        # Calculate curvature for each point
        for i in range(1, len(sorted_objectives) - 1):
            p1 = sorted_objectives[i - 1]
            p2 = sorted_objectives[i]
            p3 = sorted_objectives[i + 1]
            
            # Calculate angle using vectors
            v1 = p1 - p2
            v2 = p3 - p2
            
            # Calculate angle
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            curvature = np.pi - angle  # Higher curvature = more bent
            
            if curvature > np.pi / 4:  # Significant curvature threshold
                original_idx = sorted_indices[i]
                knee_points.append({
                    'index': int(original_idx),
                    'solution': pareto_front[original_idx],
                    'curvature': float(curvature),
                    'angle_degrees': float(np.degrees(angle))
                })
        
        # Sort by curvature (descending)
        knee_points.sort(key=lambda x: x['curvature'], reverse=True)
        
        return knee_points
    
    def _analyze_2d_front(self, objectives_matrix: np.ndarray) -> Dict[str, Any]:
        """Special analysis for 2D Pareto fronts."""
        # Sort by first objective
        sorted_indices = np.argsort(objectives_matrix[:, 0])
        sorted_objectives = objectives_matrix[sorted_indices]
        
        # Calculate front length
        front_length = 0.0
        for i in range(len(sorted_objectives) - 1):
            distance = np.linalg.norm(sorted_objectives[i + 1] - sorted_objectives[i])
            front_length += distance
        
        # Calculate average slope
        if len(sorted_objectives) >= 2:
            total_slope = (sorted_objectives[-1, 1] - sorted_objectives[0, 1]) / \
                         (sorted_objectives[-1, 0] - sorted_objectives[0, 0] + 1e-10)
        else:
            total_slope = 0.0
        
        return {
            'front_length': float(front_length),
            'average_slope': float(total_slope),
            'convexity': self._calculate_convexity(sorted_objectives)
        }
    
    def _calculate_convexity(self, sorted_objectives: np.ndarray) -> float:
        """Calculate convexity measure for 2D front."""
        if len(sorted_objectives) < 3:
            return 0.0
        
        # Check if front is convex by calculating cross products
        convexity_violations = 0
        
        for i in range(1, len(sorted_objectives) - 1):
            p1 = sorted_objectives[i - 1]
            p2 = sorted_objectives[i]
            p3 = sorted_objectives[i + 1]
            
            # Cross product to check orientation
            cross_product = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
            
            if cross_product < 0:  # Non-convex point
                convexity_violations += 1
        
        # Convexity score (1 = perfectly convex, 0 = many violations)
        convexity_score = 1.0 - (convexity_violations / max(len(sorted_objectives) - 2, 1))
        
        return float(convexity_score)
    
    def save_analysis(self, analysis_results: Dict[str, Any], filename: str = None) -> str:
        """Save analysis results to file."""
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pareto_analysis_{timestamp}.json"
        
        filepath = self.results_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            
            logger.info(f"Pareto front analysis saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving analysis results: {e}")
            return ""
    
    def create_summary_report(self, analysis_results: Dict[str, Any]) -> str:
        """Create human-readable summary report."""
        if 'error' in analysis_results:
            return f"Analysis Error: {analysis_results['error']}"
        
        report = []
        report.append("=== PARETO FRONT ANALYSIS REPORT ===\n")
        
        # Basic information
        report.append(f"Pareto Front Size: {analysis_results['front_size']}")
        report.append(f"Number of Objectives: {analysis_results['num_objectives']}")
        
        # Statistics
        stats = analysis_results['statistics']
        report.append(f"\nObjective Statistics:")
        for i, (mean, std) in enumerate(zip(stats['mean'], stats['std'])):
            report.append(f"  Objective {i+1}: Mean = {mean:.4f}, Std = {std:.4f}")
        
        # Diversity metrics
        diversity = analysis_results['diversity_metrics']
        report.append(f"\nDiversity Metrics:")
        report.append(f"  Diversity Score: {diversity['diversity_score']:.4f}")
        report.append(f"  Spacing: {diversity['spacing']:.4f}")
        report.append(f"  Extent: {diversity['extent']:.4f}")
        
        # Convergence metrics
        convergence = analysis_results['convergence_metrics']
        report.append(f"\nConvergence Metrics:")
        report.append(f"  Convergence Score: {convergence['convergence_score']:.4f}")
        report.append(f"  Hypervolume: {convergence['hypervolume']:.4f}")
        
        # Trade-offs
        trade_offs = analysis_results['trade_off_analysis']
        if trade_offs['significant_trade_offs']:
            report.append(f"\nSignificant Trade-offs:")
            for trade_off in trade_offs['significant_trade_offs']:
                report.append(f"  {trade_off['objective1']} vs {trade_off['objective2']}: "
                            f"Correlation = {trade_off['correlation']:.3f} ({trade_off['strength']})")
        
        # Knee points
        knee_points = analysis_results.get('knee_points', [])
        if knee_points:
            report.append(f"\nKnee Points Found: {len(knee_points)}")
            for i, knee in enumerate(knee_points[:3]):  # Show top 3
                report.append(f"  Knee Point {i+1}: Curvature = {knee['curvature']:.3f}")
        
        return "\n".join(report)