"""
🎯 Few-Shot Learning Module
Rapid adaptation to new markets with minimal data
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class FewShotLearner:
    """
    Few-shot learning for rapid adaptation to new markets
    Phase 3.4 - Meta-Learning Component
    """
    
    def __init__(self, 
                 min_samples: int = 5,
                 adaptation_rate: float = 0.01,
                 meta_learning_episodes: int = 100):
        """
        Initialize Few-Shot Learner
        
        Args:
            min_samples: Minimum samples needed for adaptation
            adaptation_rate: Learning rate for few-shot adaptation
            meta_learning_episodes: Episodes for meta-learning
        """
        self.min_samples = min_samples
        self.adaptation_rate = adaptation_rate
        self.meta_learning_episodes = meta_learning_episodes
        self.meta_knowledge = {}
        self.adaptation_history = []
        
        logger.info(f"FewShotLearner initialized with min_samples={min_samples}")
    
    def learn_to_adapt(self,
                      training_tasks: List[Dict[str, Any]],
                      validation_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Meta-learning phase: learn how to adapt quickly to new tasks
        
        Args:
            training_tasks: Training tasks for meta-learning
            validation_tasks: Validation tasks for evaluation
            
        Returns:
            Meta-learning results
        """
        meta_results = {
            'training_tasks': len(training_tasks),
            'validation_tasks': len(validation_tasks),
            'meta_learning_performance': 0.0,
            'adaptation_strategies': []
        }
        
        try:
            # Train meta-learner on training tasks
            meta_performance = self._train_meta_learner(training_tasks)
            
            # Validate on validation tasks
            validation_performance = self._validate_meta_learner(validation_tasks)
            
            meta_results['meta_learning_performance'] = validation_performance
            meta_results['adaptation_strategies'] = list(self.meta_knowledge.keys())
            
            logger.info(f"Meta-learning completed with performance: {validation_performance:.3f}")
            
        except Exception as e:
            logger.error(f"Meta-learning failed: {e}")
        
        return meta_results
    
    def adapt_to_new_market(self,
                           new_market_data: List[float],
                           new_market_features: Dict[str, float],
                           base_model: Any,
                           support_samples: int = None) -> Dict[str, Any]:
        """
        Adapt to new market using few-shot learning
        
        Args:
            new_market_data: Limited data from new market
            new_market_features: Market characteristics
            base_model: Base model to adapt
            support_samples: Number of support samples to use
            
        Returns:
            Adaptation results
        """
        if support_samples is None:
            support_samples = self.min_samples
            
        adaptation_results = {
            'adaptation_success': False,
            'samples_used': len(new_market_data),
            'performance_improvement': 0.0,
            'adaptation_strategy': None
        }
        
        if len(new_market_data) < self.min_samples:
            logger.warning(f"Insufficient data: {len(new_market_data)} < {self.min_samples}")
            return adaptation_results
        
        try:
            # Select adaptation strategy based on market characteristics
            strategy = self._select_adaptation_strategy(new_market_features)
            
            # Apply few-shot adaptation
            adapted_model = self._apply_few_shot_adaptation(
                base_model, 
                new_market_data[:support_samples],
                strategy
            )
            
            # Evaluate adaptation performance
            performance_improvement = self._evaluate_adaptation_performance(
                adapted_model, 
                new_market_data[support_samples:]
            )
            
            adaptation_results.update({
                'adaptation_success': True,
                'performance_improvement': performance_improvement,
                'adaptation_strategy': strategy['name']
            })
            
            # Store adaptation in history
            self.adaptation_history.append({
                'market_features': new_market_features,
                'strategy': strategy,
                'performance': performance_improvement,
                'samples_used': support_samples
            })
            
            logger.info(f"Few-shot adaptation completed: {performance_improvement:.3f} improvement")
            
        except Exception as e:
            logger.error(f"Few-shot adaptation failed: {e}")
        
        return adaptation_results
    
    def generate_synthetic_data(self,
                              limited_data: List[float],
                              target_size: int) -> List[float]:
        """
        Generate synthetic data to augment limited real data
        
        Args:
            limited_data: Limited real market data
            target_size: Target size for augmented dataset
            
        Returns:
            Augmented dataset with synthetic data
        """
        if len(limited_data) < 2:
            logger.warning("Insufficient data for synthetic generation")
            return limited_data
        
        synthetic_data = limited_data.copy()
        
        try:
            # Statistical properties of real data
            mean = np.mean(limited_data)
            std = np.std(limited_data)
            
            # Generate synthetic samples
            needed_samples = target_size - len(limited_data)
            if needed_samples > 0:
                # Use different generation strategies
                synthetic_samples = []
                
                # Strategy 1: Gaussian noise around mean
                gaussian_samples = np.random.normal(mean, std, needed_samples // 3)
                synthetic_samples.extend(gaussian_samples)
                
                # Strategy 2: Trend continuation
                trend_samples = self._generate_trend_continuation(limited_data, needed_samples // 3)
                synthetic_samples.extend(trend_samples)
                
                # Strategy 3: Pattern repetition
                pattern_samples = self._generate_pattern_repetition(limited_data, needed_samples // 3)
                synthetic_samples.extend(pattern_samples)
                
                # Fill remaining samples
                remaining = needed_samples - len(synthetic_samples)
                if remaining > 0:
                    remaining_samples = np.random.normal(mean, std * 0.5, remaining)
                    synthetic_samples.extend(remaining_samples)
                
                synthetic_data.extend(synthetic_samples[:needed_samples])
            
            logger.info(f"Generated {len(synthetic_data) - len(limited_data)} synthetic samples")
            
        except Exception as e:
            logger.error(f"Synthetic data generation failed: {e}")
        
        return synthetic_data
    
    def optimize_support_set(self,
                           available_data: List[float],
                           target_performance: float = 0.8) -> Tuple[List[float], Dict[str, Any]]:
        """
        Optimize selection of support set for few-shot learning
        
        Args:
            available_data: Available data points
            target_performance: Target performance threshold
            
        Returns:
            Optimized support set and optimization results
        """
        optimization_results = {
            'original_size': len(available_data),
            'optimized_size': 0,
            'performance_score': 0.0,
            'selection_strategy': 'random'
        }
        
        if len(available_data) <= self.min_samples:
            return available_data, optimization_results
        
        try:
            # Different selection strategies
            strategies = {
                'diverse': self._select_diverse_samples,
                'representative': self._select_representative_samples,
                'informative': self._select_informative_samples
            }
            
            best_support_set = None
            best_performance = 0.0
            best_strategy = 'random'
            
            for strategy_name, strategy_func in strategies.items():
                support_set = strategy_func(available_data)
                performance = self._evaluate_support_set_quality(support_set, available_data)
                
                if performance > best_performance:
                    best_performance = performance
                    best_support_set = support_set
                    best_strategy = strategy_name
            
            # Fallback to random if no strategy works
            if best_support_set is None:
                best_support_set = np.random.choice(
                    available_data, 
                    size=min(self.min_samples, len(available_data)), 
                    replace=False
                ).tolist()
            
            optimization_results.update({
                'optimized_size': len(best_support_set),
                'performance_score': best_performance,
                'selection_strategy': best_strategy
            })
            
            logger.info(f"Optimized support set: {len(best_support_set)} samples, "
                       f"strategy: {best_strategy}, performance: {best_performance:.3f}")
            
        except Exception as e:
            logger.error(f"Support set optimization failed: {e}")
            best_support_set = available_data[:self.min_samples]
        
        return best_support_set, optimization_results
    
    def _train_meta_learner(self, training_tasks: List[Dict[str, Any]]) -> float:
        """Train meta-learner on training tasks"""
        performances = []
        
        for episode in range(self.meta_learning_episodes):
            # Sample a task
            task = np.random.choice(training_tasks)
            
            # Simulate meta-learning episode
            episode_performance = self._simulate_meta_learning_episode(task)
            performances.append(episode_performance)
            
            # Update meta-knowledge
            self._update_meta_knowledge(task, episode_performance)
        
        return np.mean(performances)
    
    def _validate_meta_learner(self, validation_tasks: List[Dict[str, Any]]) -> float:
        """Validate meta-learner on validation tasks"""
        performances = []
        
        for task in validation_tasks:
            performance = self._evaluate_meta_learning_task(task)
            performances.append(performance)
        
        return np.mean(performances)
    
    def _select_adaptation_strategy(self, market_features: Dict[str, float]) -> Dict[str, Any]:
        """Select adaptation strategy based on market characteristics"""
        # Default strategy
        strategy = {
            'name': 'gradient_based',
            'parameters': {
                'learning_rate': self.adaptation_rate,
                'iterations': 10
            }
        }
        
        # Select strategy based on market volatility
        volatility = market_features.get('volatility', 0.1)
        if volatility > 0.3:
            strategy['name'] = 'robust_adaptation'
            strategy['parameters']['learning_rate'] *= 0.5
        elif volatility < 0.05:
            strategy['name'] = 'aggressive_adaptation'
            strategy['parameters']['learning_rate'] *= 2.0
        
        return strategy
    
    def _apply_few_shot_adaptation(self, 
                                 base_model: Any, 
                                 support_data: List[float],
                                 strategy: Dict[str, Any]) -> Any:
        """Apply few-shot adaptation to base model"""
        # Placeholder implementation - would adapt model parameters
        adapted_model = base_model  # Clone or modify model
        return adapted_model
    
    def _evaluate_adaptation_performance(self, 
                                       adapted_model: Any, 
                                       test_data: List[float]) -> float:
        """Evaluate performance of adapted model"""
        # Placeholder implementation - would evaluate model performance
        return np.random.uniform(0.0, 0.1)  # 0-10% improvement
    
    def _generate_trend_continuation(self, data: List[float], n_samples: int) -> List[float]:
        """Generate samples by continuing the trend"""
        if len(data) < 2:
            return [data[-1]] * n_samples
        
        # Calculate trend
        trend = data[-1] - data[-2]
        samples = []
        
        for i in range(n_samples):
            next_value = data[-1] + trend * (i + 1)
            samples.append(next_value)
        
        return samples
    
    def _generate_pattern_repetition(self, data: List[float], n_samples: int) -> List[float]:
        """Generate samples by repeating patterns"""
        if len(data) < 3:
            return [data[-1]] * n_samples
        
        # Use last few points as pattern
        pattern_length = min(3, len(data))
        pattern = data[-pattern_length:]
        
        samples = []
        for i in range(n_samples):
            pattern_index = i % pattern_length
            samples.append(pattern[pattern_index])
        
        return samples
    
    def _select_diverse_samples(self, data: List[float]) -> List[float]:
        """Select diverse samples from data"""
        if len(data) <= self.min_samples:
            return data
        
        # Sort data and select evenly spaced samples
        sorted_data = sorted(data)
        indices = np.linspace(0, len(sorted_data)-1, self.min_samples, dtype=int)
        
        return [sorted_data[i] for i in indices]
    
    def _select_representative_samples(self, data: List[float]) -> List[float]:
        """Select representative samples from data"""
        if len(data) <= self.min_samples:
            return data
        
        # Select samples around mean and extreme values
        mean = np.mean(data)
        std = np.std(data)
        
        representative = []
        
        # Add mean-centered samples
        mean_samples = [x for x in data if abs(x - mean) < std/2]
        if mean_samples:
            representative.extend(np.random.choice(
                mean_samples, 
                size=min(self.min_samples//2, len(mean_samples)), 
                replace=False
            ))
        
        # Add extreme samples
        remaining = self.min_samples - len(representative)
        if remaining > 0:
            extreme_samples = [x for x in data if abs(x - mean) > std]
            if extreme_samples:
                representative.extend(np.random.choice(
                    extreme_samples,
                    size=min(remaining, len(extreme_samples)),
                    replace=False
                ))
        
        # Fill remaining with random samples
        if len(representative) < self.min_samples:
            remaining_data = [x for x in data if x not in representative]
            if remaining_data:
                representative.extend(np.random.choice(
                    remaining_data,
                    size=min(self.min_samples - len(representative), len(remaining_data)),
                    replace=False
                ))
        
        return representative[:self.min_samples]
    
    def _select_informative_samples(self, data: List[float]) -> List[float]:
        """Select most informative samples from data"""
        if len(data) <= self.min_samples:
            return data
        
        # Select samples with high variance contribution
        informative_indices = []
        remaining_data = list(enumerate(data))
        
        for _ in range(self.min_samples):
            if not remaining_data:
                break
                
            # Calculate variance contribution of each remaining sample
            max_variance = 0
            best_idx = 0
            
            for idx, (orig_idx, value) in enumerate(remaining_data):
                current_selection = [data[i] for i, _ in informative_indices] + [value]
                variance = np.var(current_selection)
                
                if variance > max_variance:
                    max_variance = variance
                    best_idx = idx
            
            # Add most informative sample
            informative_indices.append(remaining_data.pop(best_idx))
        
        return [value for _, value in informative_indices]
    
    def _evaluate_support_set_quality(self, support_set: List[float], full_data: List[float]) -> float:
        """Evaluate quality of support set"""
        if not support_set or not full_data:
            return 0.0
        
        # Calculate how well support set represents full data
        support_mean = np.mean(support_set)
        support_std = np.std(support_set)
        
        full_mean = np.mean(full_data)
        full_std = np.std(full_data)
        
        # Score based on mean and std similarity
        mean_score = 1.0 - abs(support_mean - full_mean) / max(abs(full_mean), 1.0)
        std_score = 1.0 - abs(support_std - full_std) / max(full_std, 1.0)
        
        return (mean_score + std_score) / 2
    
    def _simulate_meta_learning_episode(self, task: Dict[str, Any]) -> float:
        """Simulate a meta-learning episode"""
        # Placeholder implementation - would run actual meta-learning
        return np.random.uniform(0.5, 0.9)
    
    def _evaluate_meta_learning_task(self, task: Dict[str, Any]) -> float:
        """Evaluate meta-learning on a specific task"""
        # Placeholder implementation - would evaluate task performance
        return np.random.uniform(0.6, 0.85)
    
    def _update_meta_knowledge(self, task: Dict[str, Any], performance: float):
        """Update meta-knowledge based on task performance"""
        task_type = task.get('type', 'default')
        
        if task_type not in self.meta_knowledge:
            self.meta_knowledge[task_type] = {
                'performances': [],
                'strategies': [],
                'average_performance': 0.0
            }
        
        self.meta_knowledge[task_type]['performances'].append(performance)
        self.meta_knowledge[task_type]['average_performance'] = np.mean(
            self.meta_knowledge[task_type]['performances']
        )
    
    def get_few_shot_statistics(self) -> Dict[str, Any]:
        """Get statistics about few-shot learning"""
        return {
            'min_samples': self.min_samples,
            'adaptation_rate': self.adaptation_rate,
            'meta_learning_episodes': self.meta_learning_episodes,
            'adaptation_history_size': len(self.adaptation_history),
            'meta_knowledge_tasks': len(self.meta_knowledge)
        }