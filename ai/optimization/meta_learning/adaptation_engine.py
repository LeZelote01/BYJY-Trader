"""
🔄 Adaptation Engine
Real-time adaptation system for optimization algorithms
"""

import numpy as np
from typing import Dict, List, Any, Optional, Callable
from enum import Enum

from core.logger import get_logger

logger = get_logger(__name__)


class AdaptationStrategy(Enum):
    """Available adaptation strategies."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"


class AdaptationEngine:
    """
    Real-time adaptation engine for optimization parameters.
    
    Monitors optimization progress and adapts parameters dynamically
    to improve convergence and performance.
    """
    
    def __init__(
        self,
        strategy: AdaptationStrategy = AdaptationStrategy.BALANCED,
        adaptation_frequency: int = 10,
        min_generations_before_adaptation: int = 20
    ):
        """
        Initialize adaptation engine.
        
        Args:
            strategy: Adaptation strategy to use
            adaptation_frequency: How often to check for adaptations (in generations)
            min_generations_before_adaptation: Minimum generations before first adaptation
        """
        self.strategy = strategy
        self.adaptation_frequency = adaptation_frequency
        self.min_generations = min_generations_before_adaptation
        
        # Monitoring state
        self.generation_count = 0
        self.fitness_history = []
        self.adaptation_history = []
        self.last_adaptation_generation = 0
        
        # Current parameters
        self.current_parameters = {}
        
        # Adaptation triggers
        self.stagnation_threshold = 5
        self.improvement_threshold = 0.001
        
        logger.info(f"AdaptationEngine initialized with {strategy.value} strategy")
    
    def update(
        self,
        generation: int,
        best_fitness: float,
        avg_fitness: float,
        population_diversity: float,
        current_params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Update adaptation engine with current optimization state.
        
        Args:
            generation: Current generation number
            best_fitness: Best fitness in current generation
            avg_fitness: Average fitness in current generation
            population_diversity: Population diversity measure
            current_params: Current optimization parameters
            
        Returns:
            Optional[Dict[str, Any]]: New parameters if adaptation is needed
        """
        self.generation_count = generation
        self.fitness_history.append(best_fitness)
        self.current_parameters = current_params.copy()
        
        # Check if adaptation is due
        if not self._should_adapt():
            return None
        
        # Analyze optimization state
        analysis = self._analyze_optimization_state(best_fitness, avg_fitness, population_diversity)
        
        # Determine adaptation actions
        adaptations = self._determine_adaptations(analysis)
        
        if adaptations:
            self.last_adaptation_generation = generation
            self.adaptation_history.append({
                'generation': generation,
                'analysis': analysis,
                'adaptations': adaptations
            })
            
            logger.info(f"Generation {generation}: Adapting parameters - {list(adaptations.keys())}")
            return adaptations
        
        return None
    
    def _should_adapt(self) -> bool:
        """Check if adaptation should be performed."""
        # Minimum generations check
        if self.generation_count < self.min_generations:
            return False
        
        # Frequency check
        generations_since_last = self.generation_count - self.last_adaptation_generation
        if generations_since_last < self.adaptation_frequency:
            return False
        
        return True
    
    def _analyze_optimization_state(
        self,
        best_fitness: float,
        avg_fitness: float,
        population_diversity: float
    ) -> Dict[str, Any]:
        """Analyze current optimization state."""
        analysis = {
            'current_fitness': best_fitness,
            'fitness_trend': self._analyze_fitness_trend(),
            'convergence_state': self._analyze_convergence_state(),
            'diversity_level': self._classify_diversity_level(population_diversity),
            'stagnation_detected': self._detect_stagnation(),
            'exploration_exploitation_balance': self._analyze_exploration_exploitation()
        }
        
        return analysis
    
    def _analyze_fitness_trend(self) -> str:
        """Analyze fitness trend over recent generations."""
        if len(self.fitness_history) < 10:
            return 'insufficient_data'
        
        recent_fitness = self.fitness_history[-10:]
        
        # Calculate trend using linear regression
        x = np.arange(len(recent_fitness))
        slope = np.polyfit(x, recent_fitness, 1)[0]
        
        if slope > self.improvement_threshold:
            return 'improving'
        elif slope < -self.improvement_threshold:
            return 'declining'
        else:
            return 'stagnant'
    
    def _analyze_convergence_state(self) -> str:
        """Analyze convergence state."""
        if len(self.fitness_history) < 20:
            return 'early'
        
        # Check fitness variance over recent generations
        recent_fitness = self.fitness_history[-20:]
        fitness_variance = np.var(recent_fitness)
        
        if fitness_variance < 1e-6:
            return 'converged'
        elif fitness_variance < 1e-3:
            return 'converging'
        else:
            return 'exploring'
    
    def _classify_diversity_level(self, diversity: float) -> str:
        """Classify population diversity level."""
        if diversity > 0.7:
            return 'high'
        elif diversity > 0.3:
            return 'medium'
        else:
            return 'low'
    
    def _detect_stagnation(self) -> bool:
        """Detect if optimization is stagnating."""
        if len(self.fitness_history) < self.stagnation_threshold * 2:
            return False
        
        recent_best = max(self.fitness_history[-self.stagnation_threshold:])
        previous_best = max(self.fitness_history[-self.stagnation_threshold*2:-self.stagnation_threshold])
        
        improvement = recent_best - previous_best
        
        return improvement < self.improvement_threshold
    
    def _analyze_exploration_exploitation(self) -> str:
        """Analyze exploration vs exploitation balance."""
        if len(self.fitness_history) < 20:
            return 'unknown'
        
        # Simple heuristic: if fitness is improving steadily, we're exploiting well
        # If fitness is highly variable, we're exploring more
        
        recent_fitness = self.fitness_history[-20:]
        fitness_std = np.std(recent_fitness)
        fitness_trend = self._analyze_fitness_trend()
        
        if fitness_trend == 'improving' and fitness_std < 0.1:
            return 'exploitation_heavy'
        elif fitness_std > 0.5:
            return 'exploration_heavy'
        else:
            return 'balanced'
    
    def _determine_adaptations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Determine what parameters to adapt based on analysis."""
        adaptations = {}
        
        # Adapt based on strategy
        if self.strategy == AdaptationStrategy.CONSERVATIVE:
            adaptations.update(self._conservative_adaptations(analysis))
        elif self.strategy == AdaptationStrategy.BALANCED:
            adaptations.update(self._balanced_adaptations(analysis))
        elif self.strategy == AdaptationStrategy.AGGRESSIVE:
            adaptations.update(self._aggressive_adaptations(analysis))
        
        return adaptations
    
    def _conservative_adaptations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Conservative adaptation strategy - small, safe changes."""
        adaptations = {}
        
        # Only adapt if clear issues are detected
        if analysis['stagnation_detected']:
            # Slightly increase mutation rate
            current_mutation = self.current_parameters.get('mutation_prob', 0.1)
            adaptations['mutation_prob'] = min(current_mutation * 1.1, 0.3)
        
        if analysis['diversity_level'] == 'low':
            # Slightly increase population size
            current_pop = self.current_parameters.get('population_size', 100)
            adaptations['population_size'] = min(int(current_pop * 1.05), 150)
        
        return adaptations
    
    def _balanced_adaptations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Balanced adaptation strategy - moderate changes based on state."""
        adaptations = {}
        
        # Mutation rate adaptation
        if analysis['stagnation_detected']:
            current_mutation = self.current_parameters.get('mutation_prob', 0.1)
            adaptations['mutation_prob'] = min(current_mutation * 1.2, 0.4)
        elif analysis['fitness_trend'] == 'improving' and analysis['diversity_level'] == 'high':
            current_mutation = self.current_parameters.get('mutation_prob', 0.1)
            adaptations['mutation_prob'] = max(current_mutation * 0.9, 0.05)
        
        # Crossover rate adaptation
        if analysis['diversity_level'] == 'low':
            current_crossover = self.current_parameters.get('crossover_prob', 0.8)
            adaptations['crossover_prob'] = min(current_crossover * 1.1, 0.95)
        
        # Population size adaptation
        if analysis['convergence_state'] == 'converged' and analysis['diversity_level'] == 'low':
            current_pop = self.current_parameters.get('population_size', 100)
            adaptations['population_size'] = min(int(current_pop * 1.2), 200)
        
        # Early stopping adaptation
        if analysis['fitness_trend'] == 'stagnant':
            current_patience = self.current_parameters.get('early_stopping_patience', 20)
            adaptations['early_stopping_patience'] = max(int(current_patience * 0.8), 10)
        
        return adaptations
    
    def _aggressive_adaptations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Aggressive adaptation strategy - large changes for fast optimization."""
        adaptations = {}
        
        # Aggressive mutation rate changes
        if analysis['stagnation_detected'] or analysis['diversity_level'] == 'low':
            adaptations['mutation_prob'] = min(
                self.current_parameters.get('mutation_prob', 0.1) * 1.5, 0.5
            )
        
        # Dynamic population size
        if analysis['convergence_state'] == 'exploring':
            current_pop = self.current_parameters.get('population_size', 100)
            adaptations['population_size'] = min(int(current_pop * 1.3), 250)
        elif analysis['convergence_state'] == 'converged':
            # Restart with larger population
            adaptations['population_size'] = 200
            adaptations['mutation_prob'] = 0.3
        
        # Adaptive crossover
        if analysis['exploration_exploitation_balance'] == 'exploitation_heavy':
            adaptations['crossover_prob'] = min(
                self.current_parameters.get('crossover_prob', 0.8) * 1.2, 0.95
            )
        
        return adaptations
    
    def get_adaptation_recommendations(
        self,
        current_analysis: Dict[str, Any]
    ) -> List[str]:
        """Get human-readable adaptation recommendations."""
        recommendations = []
        
        if current_analysis['stagnation_detected']:
            recommendations.append("🔄 Increase mutation rate to escape local optima")
            recommendations.append("📈 Consider increasing population size for more diversity")
        
        if current_analysis['diversity_level'] == 'low':
            recommendations.append("🌟 Boost population diversity through parameter changes")
            recommendations.append("🔀 Increase crossover rate for better mixing")
        
        if current_analysis['fitness_trend'] == 'declining':
            recommendations.append("⚠️ Fitness declining - review parameter choices")
            recommendations.append("🎯 Consider reverting recent parameter changes")
        
        if current_analysis['convergence_state'] == 'converged':
            recommendations.append("🏁 Convergence detected - consider restart with new parameters")
            recommendations.append("🚀 Increase exploration through higher mutation/crossover")
        
        return recommendations
    
    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """Get history of adaptations performed."""
        return self.adaptation_history.copy()
    
    def reset(self):
        """Reset adaptation engine state."""
        self.generation_count = 0
        self.fitness_history.clear()
        self.adaptation_history.clear()
        self.last_adaptation_generation = 0
        self.current_parameters.clear()
        
        logger.info("AdaptationEngine reset")
    
    def set_strategy(self, strategy: AdaptationStrategy):
        """Change adaptation strategy."""
        self.strategy = strategy
        logger.info(f"Adaptation strategy changed to {strategy.value}")
    
    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get adaptation statistics."""
        if not self.adaptation_history:
            return {
                'total_adaptations': 0,
                'average_adaptation_frequency': 0,
                'most_adapted_parameter': None
            }
        
        # Count parameter adaptations
        param_counts = {}
        for adaptation in self.adaptation_history:
            for param in adaptation['adaptations'].keys():
                param_counts[param] = param_counts.get(param, 0) + 1
        
        most_adapted = max(param_counts, key=param_counts.get) if param_counts else None
        
        # Calculate average frequency
        if len(self.adaptation_history) > 1:
            generations = [a['generation'] for a in self.adaptation_history]
            avg_frequency = np.mean(np.diff(generations))
        else:
            avg_frequency = 0
        
        return {
            'total_adaptations': len(self.adaptation_history),
            'average_adaptation_frequency': avg_frequency,
            'most_adapted_parameter': most_adapted,
            'parameter_adaptation_counts': param_counts,
            'strategy': self.strategy.value
        }