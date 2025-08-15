"""
🎯 Fitness Evaluator
Multi-criteria fitness evaluation for genetic algorithm optimization
"""

import asyncio
import numpy as np
from typing import Dict, Any, Callable, Optional, List
from datetime import datetime
import time

from core.logger import get_logger
from .chromosome import Chromosome

logger = get_logger(__name__)


class FitnessEvaluator:
    """
    Multi-criteria fitness evaluator for genetic algorithm optimization.
    
    Evaluates chromosomes based on multiple objectives:
    - Profit maximization (Sharpe ratio, returns)
    - Risk minimization (drawdown, volatility)
    - Speed (training/inference time)
    - Stability (consistency across periods)
    """
    
    def __init__(self):
        """Initialize fitness evaluator."""
        self.evaluation_cache = {}
        self.evaluation_count = 0
        self.total_evaluation_time = 0.0
        
        # Default weights for multi-objective fitness
        self.default_weights = {
            'profit': 0.4,      # Profit maximization
            'risk': 0.3,        # Risk minimization (inverted)
            'speed': 0.15,      # Speed optimization (inverted time)
            'stability': 0.15   # Stability/consistency
        }
    
    async def evaluate(
        self,
        chromosome: Chromosome,
        fitness_function: Callable,
        weights: Optional[Dict[str, float]] = None,
        cache_results: bool = True,
        **kwargs
    ) -> float:
        """
        Evaluate fitness of chromosome.
        
        Args:
            chromosome: Chromosome to evaluate
            fitness_function: Function to compute fitness metrics
            weights: Weights for multi-objective fitness
            cache_results: Whether to cache evaluation results
            **kwargs: Additional arguments for fitness function
            
        Returns:
            float: Fitness value
        """
        start_time = time.time()
        
        # Check cache
        chromosome_hash = hash(chromosome)
        if cache_results and chromosome_hash in self.evaluation_cache:
            cached_fitness = self.evaluation_cache[chromosome_hash]
            chromosome.fitness = cached_fitness
            chromosome.increment_evaluation_count()
            return cached_fitness
        
        try:
            # Call fitness function with chromosome parameters
            metrics = await self._call_fitness_function(
                fitness_function, chromosome.genes, **kwargs
            )
            
            # Calculate multi-objective fitness
            fitness = self._calculate_multi_objective_fitness(metrics, weights)
            
            # Store fitness in chromosome
            chromosome.fitness = fitness
            chromosome.increment_evaluation_count()
            
            # Cache result
            if cache_results:
                self.evaluation_cache[chromosome_hash] = fitness
            
            # Update statistics
            self.evaluation_count += 1
            self.total_evaluation_time += time.time() - start_time
            
            return fitness
            
        except Exception as e:
            logger.error(f"Error evaluating chromosome: {e}")
            # Assign very low fitness for failed evaluations
            chromosome.fitness = -float('inf')
            return -float('inf')
    
    async def _call_fitness_function(
        self,
        fitness_function: Callable,
        parameters: Dict[str, Any],
        **kwargs
    ) -> Dict[str, float]:
        """
        Call fitness function with proper async handling.
        
        Args:
            fitness_function: Function to evaluate parameters
            parameters: Parameter dictionary
            **kwargs: Additional arguments
            
        Returns:
            Dict[str, float]: Fitness metrics
        """
        if asyncio.iscoroutinefunction(fitness_function):
            return await fitness_function(parameters, **kwargs)
        else:
            # Run in thread pool for sync functions
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, 
                lambda: fitness_function(parameters, **kwargs)
            )
    
    def _calculate_multi_objective_fitness(
        self,
        metrics: Dict[str, float],
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate multi-objective fitness from individual metrics.
        
        Args:
            metrics: Dictionary of fitness metrics
            weights: Weights for different objectives
            
        Returns:
            float: Combined fitness value
        """
        if weights is None:
            weights = self.default_weights.copy()
        
        fitness_components = {}
        
        # Profit component (higher is better)
        profit_metrics = ['sharpe_ratio', 'returns', 'profit_factor', 'win_rate']
        profit_score = 0
        profit_count = 0
        
        for metric in profit_metrics:
            if metric in metrics:
                # Normalize to [0, 1] range
                if metric == 'sharpe_ratio':
                    # Sharpe ratio: good values are 1+, excellent 2+
                    normalized = np.tanh(metrics[metric] / 2.0) * 0.5 + 0.5
                elif metric == 'returns':
                    # Returns: normalize annual returns
                    normalized = np.tanh(metrics[metric] * 5.0) * 0.5 + 0.5
                elif metric == 'profit_factor':
                    # Profit factor: 1 = break-even, 2+ = good
                    normalized = np.tanh((metrics[metric] - 1.0) * 2.0) * 0.5 + 0.5
                elif metric == 'win_rate':
                    # Win rate: 0.5 = random, 0.6+ = good
                    normalized = metrics[metric]
                else:
                    normalized = np.tanh(metrics[metric]) * 0.5 + 0.5
                
                profit_score += normalized
                profit_count += 1
        
        if profit_count > 0:
            fitness_components['profit'] = profit_score / profit_count
        else:
            fitness_components['profit'] = 0.0
        
        # Risk component (lower is better, so invert)
        risk_metrics = ['max_drawdown', 'volatility', 'var']
        risk_score = 0
        risk_count = 0
        
        for metric in risk_metrics:
            if metric in metrics:
                if metric == 'max_drawdown':
                    # Max drawdown: 0 = perfect, 0.1 = 10% drawdown
                    normalized = 1.0 - np.clip(metrics[metric], 0, 1)
                elif metric == 'volatility':
                    # Volatility: normalize to reasonable range
                    normalized = 1.0 - np.tanh(metrics[metric] * 10.0)
                elif metric == 'var':
                    # VaR: typically negative, so invert and normalize
                    normalized = 1.0 - np.tanh(abs(metrics[metric]) * 10.0)
                else:
                    # General risk metric (assume higher = worse)
                    normalized = 1.0 - np.tanh(metrics[metric])
                
                risk_score += normalized
                risk_count += 1
        
        if risk_count > 0:
            fitness_components['risk'] = risk_score / risk_count
        else:
            fitness_components['risk'] = 0.5  # Neutral if no risk metrics
        
        # Speed component (lower time is better, so invert)
        speed_metrics = ['training_time', 'inference_time', 'convergence_time']
        speed_score = 0
        speed_count = 0
        
        for metric in speed_metrics:
            if metric in metrics:
                # Normalize time metrics (assume seconds)
                if metric in ['training_time', 'convergence_time']:
                    # Training time: 60s = average, normalize accordingly
                    normalized = 1.0 - np.tanh(metrics[metric] / 60.0)
                elif metric == 'inference_time':
                    # Inference time: 1s = slow, normalize accordingly
                    normalized = 1.0 - np.tanh(metrics[metric])
                else:
                    normalized = 1.0 - np.tanh(metrics[metric])
                
                speed_score += normalized
                speed_count += 1
        
        if speed_count > 0:
            fitness_components['speed'] = speed_score / speed_count
        else:
            fitness_components['speed'] = 0.5  # Neutral if no speed metrics
        
        # Stability component (higher consistency is better)
        stability_metrics = ['consistency', 'std_returns', 'confidence']
        stability_score = 0
        stability_count = 0
        
        for metric in stability_metrics:
            if metric in metrics:
                if metric == 'consistency':
                    # Consistency: already 0-1 range
                    normalized = metrics[metric]
                elif metric == 'std_returns':
                    # Standard deviation of returns (lower is better)
                    normalized = 1.0 - np.tanh(metrics[metric] * 5.0)
                elif metric == 'confidence':
                    # Confidence score: already 0-1 range
                    normalized = metrics[metric]
                else:
                    normalized = np.tanh(metrics[metric]) * 0.5 + 0.5
                
                stability_score += normalized
                stability_count += 1
        
        if stability_count > 0:
            fitness_components['stability'] = stability_score / stability_count
        else:
            fitness_components['stability'] = 0.5  # Neutral if no stability metrics
        
        # Combine components with weights
        total_fitness = 0.0
        total_weight = 0.0
        
        for component, score in fitness_components.items():
            if component in weights:
                total_fitness += weights[component] * score
                total_weight += weights[component]
        
        # Normalize by total weight
        if total_weight > 0:
            final_fitness = total_fitness / total_weight
        else:
            final_fitness = 0.0
        
        return final_fitness
    
    def evaluate_batch(
        self,
        chromosomes: List[Chromosome],
        fitness_function: Callable,
        **kwargs
    ) -> List[float]:
        """
        Evaluate batch of chromosomes synchronously.
        
        Args:
            chromosomes: List of chromosomes to evaluate
            fitness_function: Function to compute fitness
            **kwargs: Additional arguments
            
        Returns:
            List[float]: Fitness values
        """
        fitness_values = []
        
        for chromosome in chromosomes:
            try:
                metrics = fitness_function(chromosome.genes, **kwargs)
                fitness = self._calculate_multi_objective_fitness(metrics)
                chromosome.fitness = fitness
                chromosome.increment_evaluation_count()
                fitness_values.append(fitness)
                
            except Exception as e:
                logger.error(f"Error evaluating chromosome in batch: {e}")
                chromosome.fitness = -float('inf')
                fitness_values.append(-float('inf'))
        
        self.evaluation_count += len(chromosomes)
        return fitness_values
    
    def clear_cache(self):
        """Clear evaluation cache."""
        self.evaluation_cache.clear()
        logger.info("Evaluation cache cleared")
    
    def get_evaluation_stats(self) -> Dict[str, Any]:
        """
        Get evaluation statistics.
        
        Returns:
            Dict[str, Any]: Evaluation statistics
        """
        avg_eval_time = (self.total_evaluation_time / self.evaluation_count 
                        if self.evaluation_count > 0 else 0)
        
        return {
            'total_evaluations': self.evaluation_count,
            'cache_size': len(self.evaluation_cache),
            'total_evaluation_time': self.total_evaluation_time,
            'average_evaluation_time': avg_eval_time
        }