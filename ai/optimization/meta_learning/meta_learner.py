"""
🧠 Meta-Learner
Advanced meta-learning system for optimization adaptation
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path

from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LearningPattern:
    """Represents a learning pattern detected in optimization."""
    pattern_type: str
    parameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    context: Dict[str, Any]
    confidence: float
    frequency: int = 1


class MetaLearner:
    """
    Meta-learning system for optimization algorithms.
    
    Learns patterns from optimization history to improve future optimizations.
    """
    
    def __init__(self):
        """Initialize meta-learner."""
        self.learned_patterns = []
        self.optimization_history = []
        self.pattern_database = {}
        
        # Learning parameters
        self.min_pattern_confidence = 0.7
        self.min_pattern_frequency = 3
        self.max_patterns = 1000
        
        logger.info("MetaLearner initialized")
    
    def learn_from_optimization(
        self,
        optimization_result: Dict[str, Any],
        context: Dict[str, Any] = None
    ):
        """
        Learn patterns from completed optimization.
        
        Args:
            optimization_result: Results from optimization run
            context: Additional context information
        """
        if context is None:
            context = {}
        
        # Store optimization history
        self.optimization_history.append({
            'result': optimization_result,
            'context': context,
            'timestamp': np.datetime64('now')
        })
        
        # Extract patterns
        patterns = self._extract_patterns(optimization_result, context)
        
        # Update pattern database
        for pattern in patterns:
            self._update_pattern_database(pattern)
        
        # Cleanup old patterns if necessary
        if len(self.learned_patterns) > self.max_patterns:
            self._cleanup_patterns()
        
        logger.info(f"Learned {len(patterns)} new patterns from optimization")
    
    def recommend_parameters(
        self,
        target_context: Dict[str, Any],
        model_type: str = None
    ) -> Dict[str, Any]:
        """
        Recommend optimization parameters based on learned patterns.
        
        Args:
            target_context: Context for the new optimization
            model_type: Type of model being optimized
            
        Returns:
            Dict[str, Any]: Recommended parameters
        """
        # Find relevant patterns
        relevant_patterns = self._find_relevant_patterns(target_context, model_type)
        
        if not relevant_patterns:
            return self._get_default_recommendations(model_type)
        
        # Aggregate recommendations from patterns
        recommendations = self._aggregate_recommendations(relevant_patterns)
        
        logger.info(f"Generated recommendations based on {len(relevant_patterns)} relevant patterns")
        return recommendations
    
    def predict_performance(
        self,
        parameters: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, float]:
        """
        Predict optimization performance based on parameters and context.
        
        Args:
            parameters: Optimization parameters
            context: Optimization context
            
        Returns:
            Dict[str, float]: Predicted performance metrics
        """
        if context is None:
            context = {}
        
        # Find similar historical cases
        similar_cases = self._find_similar_cases(parameters, context)
        
        if not similar_cases:
            return self._get_default_predictions()
        
        # Aggregate performance predictions
        predictions = self._aggregate_performance_predictions(similar_cases)
        
        return predictions
    
    def _extract_patterns(
        self,
        optimization_result: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[LearningPattern]:
        """Extract learning patterns from optimization result."""
        patterns = []
        
        # Pattern 1: Convergence patterns
        convergence_pattern = self._extract_convergence_pattern(optimization_result, context)
        if convergence_pattern:
            patterns.append(convergence_pattern)
        
        # Pattern 2: Parameter effectiveness patterns
        param_patterns = self._extract_parameter_patterns(optimization_result, context)
        patterns.extend(param_patterns)
        
        # Pattern 3: Performance correlation patterns
        correlation_pattern = self._extract_correlation_pattern(optimization_result, context)
        if correlation_pattern:
            patterns.append(correlation_pattern)
        
        return patterns
    
    def _extract_convergence_pattern(
        self,
        result: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Optional[LearningPattern]:
        """Extract convergence behavior patterns."""
        if 'converged' not in result or 'generations_completed' not in result:
            return None
        
        converged = result['converged']
        generations = result['generations_completed']
        best_fitness = result.get('best_fitness', 0)
        
        # Define convergence pattern
        pattern_params = {
            'converged': converged,
            'generations_to_converge': generations,
            'final_fitness': best_fitness,
            'convergence_speed': best_fitness / max(generations, 1)
        }
        
        performance_metrics = {
            'fitness': best_fitness,
            'efficiency': best_fitness / max(generations, 1),
            'success_rate': 1.0 if converged else 0.0
        }
        
        confidence = 0.8 if converged else 0.6
        
        return LearningPattern(
            pattern_type='convergence',
            parameters=pattern_params,
            performance_metrics=performance_metrics,
            context=context,
            confidence=confidence
        )
    
    def _extract_parameter_patterns(
        self,
        result: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[LearningPattern]:
        """Extract parameter effectiveness patterns."""
        patterns = []
        
        best_params = result.get('best_parameters', {})
        best_fitness = result.get('best_fitness', 0)
        
        if not best_params:
            return patterns
        
        # Categorize parameters by effectiveness
        for param_name, param_value in best_params.items():
            pattern_params = {
                'parameter_name': param_name,
                'effective_value': param_value,
                'parameter_type': type(param_value).__name__,
                'context_hash': hash(str(sorted(context.items())))
            }
            
            performance_metrics = {
                'fitness': best_fitness,
                'parameter_contribution': best_fitness * 0.1  # Simplified contribution
            }
            
            confidence = min(best_fitness, 1.0) if best_fitness > 0 else 0.5
            
            pattern = LearningPattern(
                pattern_type='parameter_effectiveness',
                parameters=pattern_params,
                performance_metrics=performance_metrics,
                context=context,
                confidence=confidence
            )
            
            patterns.append(pattern)
        
        return patterns
    
    def _extract_correlation_pattern(
        self,
        result: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Optional[LearningPattern]:
        """Extract parameter correlation patterns."""
        generation_stats = result.get('generation_stats', [])
        
        if len(generation_stats) < 10:
            return None
        
        # Analyze fitness progression
        fitness_progression = [gen.get('best_fitness', 0) for gen in generation_stats]
        
        # Calculate correlation metrics
        correlation_strength = self._calculate_fitness_correlation(fitness_progression)
        
        pattern_params = {
            'correlation_type': 'fitness_progression',
            'correlation_strength': correlation_strength,
            'progression_pattern': 'improving' if correlation_strength > 0 else 'declining'
        }
        
        performance_metrics = {
            'correlation_strength': abs(correlation_strength),
            'predictability': min(abs(correlation_strength) * 2, 1.0)
        }
        
        confidence = min(abs(correlation_strength) + 0.3, 1.0)
        
        return LearningPattern(
            pattern_type='correlation',
            parameters=pattern_params,
            performance_metrics=performance_metrics,
            context=context,
            confidence=confidence
        )
    
    def _calculate_fitness_correlation(self, fitness_progression: List[float]) -> float:
        """Calculate correlation in fitness progression."""
        if len(fitness_progression) < 2:
            return 0.0
        
        x = np.arange(len(fitness_progression))
        y = np.array(fitness_progression)
        
        correlation = np.corrcoef(x, y)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    
    def _update_pattern_database(self, pattern: LearningPattern):
        """Update pattern database with new pattern."""
        pattern_key = f"{pattern.pattern_type}_{hash(str(pattern.parameters))}"
        
        if pattern_key in self.pattern_database:
            # Update existing pattern
            existing_pattern = self.pattern_database[pattern_key]
            existing_pattern.frequency += 1
            
            # Update confidence based on frequency
            existing_pattern.confidence = min(
                existing_pattern.confidence + 0.1,
                1.0
            )
            
            # Update performance metrics (weighted average)
            for metric, value in pattern.performance_metrics.items():
                if metric in existing_pattern.performance_metrics:
                    existing_pattern.performance_metrics[metric] = (
                        existing_pattern.performance_metrics[metric] * 0.8 + value * 0.2
                    )
        else:
            # Add new pattern
            self.pattern_database[pattern_key] = pattern
            self.learned_patterns.append(pattern)
    
    def _find_relevant_patterns(
        self,
        target_context: Dict[str, Any],
        model_type: str = None
    ) -> List[LearningPattern]:
        """Find patterns relevant to the target context."""
        relevant_patterns = []
        
        for pattern in self.learned_patterns:
            if pattern.confidence < self.min_pattern_confidence:
                continue
            
            if pattern.frequency < self.min_pattern_frequency:
                continue
            
            # Check context similarity
            similarity = self._calculate_context_similarity(pattern.context, target_context)
            
            if similarity > 0.5:  # Threshold for relevance
                relevant_patterns.append((pattern, similarity))
        
        # Sort by relevance (similarity * confidence)
        relevant_patterns.sort(key=lambda x: x[1] * x[0].confidence, reverse=True)
        
        return [pattern for pattern, similarity in relevant_patterns[:10]]  # Top 10
    
    def _calculate_context_similarity(
        self,
        context1: Dict[str, Any],
        context2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between contexts."""
        if not context1 or not context2:
            return 0.0
        
        common_keys = set(context1.keys()) & set(context2.keys())
        
        if not common_keys:
            return 0.0
        
        similarities = []
        
        for key in common_keys:
            val1 = context1[key]
            val2 = context2[key]
            
            if val1 == val2:
                similarities.append(1.0)
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numerical similarity
                max_val = max(abs(val1), abs(val2), 1e-10)
                similarity = 1.0 - abs(val1 - val2) / max_val
                similarities.append(max(similarity, 0.0))
            else:
                similarities.append(0.0)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _aggregate_recommendations(
        self,
        relevant_patterns: List[LearningPattern]
    ) -> Dict[str, Any]:
        """Aggregate recommendations from relevant patterns."""
        recommendations = {
            'population_size': [],
            'num_generations': [],
            'crossover_prob': [],
            'mutation_prob': [],
            'early_stopping_patience': []
        }
        
        for pattern in relevant_patterns:
            weight = pattern.confidence * pattern.frequency
            
            if pattern.pattern_type == 'convergence':
                # Recommend based on convergence patterns
                if pattern.parameters.get('converged', False):
                    recommendations['early_stopping_patience'].append((20, weight))
                    recommendations['num_generations'].append((150, weight))
                else:
                    recommendations['num_generations'].append((300, weight))
            
            elif pattern.pattern_type == 'parameter_effectiveness':
                # Use effective parameters as recommendations
                param_name = pattern.parameters.get('parameter_name')
                effective_value = pattern.parameters.get('effective_value')
                
                if param_name in recommendations:
                    recommendations[param_name].append((effective_value, weight))
        
        # Calculate weighted averages
        final_recommendations = {}
        
        for param_name, values_weights in recommendations.items():
            if values_weights:
                weighted_sum = sum(value * weight for value, weight in values_weights)
                total_weight = sum(weight for value, weight in values_weights)
                
                if total_weight > 0:
                    final_recommendations[param_name] = weighted_sum / total_weight
        
        return final_recommendations
    
    def _find_similar_cases(
        self,
        parameters: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find similar historical optimization cases."""
        similar_cases = []
        
        for history_entry in self.optimization_history:
            # Calculate parameter similarity
            historical_params = history_entry['result'].get('best_parameters', {})
            param_similarity = self._calculate_parameter_similarity(parameters, historical_params)
            
            # Calculate context similarity
            historical_context = history_entry.get('context', {})
            context_similarity = self._calculate_context_similarity(context, historical_context)
            
            # Overall similarity
            overall_similarity = (param_similarity + context_similarity) / 2
            
            if overall_similarity > 0.3:  # Threshold for similarity
                similar_cases.append({
                    'result': history_entry['result'],
                    'similarity': overall_similarity
                })
        
        # Sort by similarity
        similar_cases.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similar_cases[:5]  # Top 5 similar cases
    
    def _calculate_parameter_similarity(
        self,
        params1: Dict[str, Any],
        params2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between parameter sets."""
        if not params1 or not params2:
            return 0.0
        
        common_keys = set(params1.keys()) & set(params2.keys())
        
        if not common_keys:
            return 0.0
        
        similarities = []
        
        for key in common_keys:
            val1 = params1[key]
            val2 = params2[key]
            
            if val1 == val2:
                similarities.append(1.0)
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                max_val = max(abs(val1), abs(val2), 1e-10)
                similarity = 1.0 - abs(val1 - val2) / max_val
                similarities.append(max(similarity, 0.0))
            else:
                similarities.append(0.0)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _aggregate_performance_predictions(
        self,
        similar_cases: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Aggregate performance predictions from similar cases."""
        predictions = {
            'expected_fitness': 0.0,
            'convergence_probability': 0.0,
            'expected_generations': 0.0,
            'confidence': 0.0
        }
        
        total_weight = 0.0
        
        for case in similar_cases:
            result = case['result']
            similarity = case['similarity']
            
            weight = similarity
            
            predictions['expected_fitness'] += result.get('best_fitness', 0) * weight
            predictions['convergence_probability'] += (1.0 if result.get('converged', False) else 0.0) * weight
            predictions['expected_generations'] += result.get('generations_completed', 100) * weight
            
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            for key in predictions:
                if key != 'confidence':
                    predictions[key] /= total_weight
            
            predictions['confidence'] = min(total_weight / len(similar_cases), 1.0)
        
        return predictions
    
    def _get_default_recommendations(self, model_type: str = None) -> Dict[str, Any]:
        """Get default recommendations when no patterns are available."""
        defaults = {
            'population_size': 100,
            'num_generations': 200,
            'crossover_prob': 0.8,
            'mutation_prob': 0.1,
            'early_stopping_patience': 20
        }
        
        # Adjust based on model type
        if model_type == 'lstm':
            defaults['num_generations'] = 150
            defaults['mutation_prob'] = 0.15
        elif model_type == 'transformer':
            defaults['num_generations'] = 250
            defaults['population_size'] = 80
        elif model_type == 'ensemble':
            defaults['num_generations'] = 300
            defaults['population_size'] = 120
        
        return defaults
    
    def _get_default_predictions(self) -> Dict[str, float]:
        """Get default performance predictions."""
        return {
            'expected_fitness': 0.7,
            'convergence_probability': 0.6,
            'expected_generations': 150,
            'confidence': 0.3
        }
    
    def _cleanup_patterns(self):
        """Remove low-quality patterns to maintain database size."""
        # Sort patterns by quality (confidence * frequency)
        self.learned_patterns.sort(
            key=lambda p: p.confidence * p.frequency,
            reverse=True
        )
        
        # Keep only the best patterns
        self.learned_patterns = self.learned_patterns[:self.max_patterns]
        
        # Rebuild pattern database
        self.pattern_database = {
            f"{p.pattern_type}_{hash(str(p.parameters))}": p
            for p in self.learned_patterns
        }
        
        logger.info(f"Cleaned up pattern database, kept {len(self.learned_patterns)} patterns")
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get meta-learning statistics."""
        pattern_types = {}
        for pattern in self.learned_patterns:
            pattern_types[pattern.pattern_type] = pattern_types.get(pattern.pattern_type, 0) + 1
        
        return {
            'total_patterns': len(self.learned_patterns),
            'pattern_types': pattern_types,
            'optimization_history_size': len(self.optimization_history),
            'average_pattern_confidence': np.mean([p.confidence for p in self.learned_patterns]) if self.learned_patterns else 0.0,
            'high_confidence_patterns': len([p for p in self.learned_patterns if p.confidence > 0.8])
        }
    
    def save_patterns(self, filepath: Path):
        """Save learned patterns to file."""
        try:
            pattern_data = {
                'patterns': [
                    {
                        'pattern_type': p.pattern_type,
                        'parameters': p.parameters,
                        'performance_metrics': p.performance_metrics,
                        'context': p.context,
                        'confidence': p.confidence,
                        'frequency': p.frequency
                    }
                    for p in self.learned_patterns
                ]
            }
            
            with open(filepath, 'w') as f:
                json.dump(pattern_data, f, indent=2, default=str)
            
            logger.info(f"Saved {len(self.learned_patterns)} patterns to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving patterns: {e}")
    
    def load_patterns(self, filepath: Path):
        """Load patterns from file."""
        try:
            if not filepath.exists():
                logger.warning(f"Pattern file not found: {filepath}")
                return
            
            with open(filepath, 'r') as f:
                pattern_data = json.load(f)
            
            loaded_patterns = []
            for p_data in pattern_data.get('patterns', []):
                pattern = LearningPattern(
                    pattern_type=p_data['pattern_type'],
                    parameters=p_data['parameters'],
                    performance_metrics=p_data['performance_metrics'],
                    context=p_data['context'],
                    confidence=p_data['confidence'],
                    frequency=p_data.get('frequency', 1)
                )
                loaded_patterns.append(pattern)
            
            self.learned_patterns = loaded_patterns
            
            # Rebuild pattern database
            self.pattern_database = {
                f"{p.pattern_type}_{hash(str(p.parameters))}": p
                for p in self.learned_patterns
            }
            
            logger.info(f"Loaded {len(self.learned_patterns)} patterns from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading patterns: {e}")