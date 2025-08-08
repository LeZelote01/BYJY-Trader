"""
🧬 Genetic Algorithm Optimizer
Advanced genetic algorithm for parameter optimization
"""

import numpy as np
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime
import json
from pathlib import Path
import random

from core.logger import get_logger
from core.path_utils import get_project_root
from .chromosome import Chromosome
from .crossover import CrossoverOperator
from .mutation import MutationOperator
from .selection import SelectionOperator
from .fitness_evaluator import FitnessEvaluator

logger = get_logger(__name__)


class GeneticOptimizer:
    """
    Advanced Genetic Algorithm Optimizer for BYJY-Trader
    
    Optimizes parameters for LSTM, Transformer, XGBoost, Ensemble models,
    and trading strategies using genetic algorithms.
    """
    
    def __init__(
        self,
        population_size: int = 100,
        num_generations: int = 200,
        crossover_prob: float = 0.8,
        mutation_prob: float = 0.1,
        elitism_ratio: float = 0.1,
        early_stopping_patience: int = 20,
        random_seed: Optional[int] = None
    ):
        """
        Initialize Genetic Optimizer.
        
        Args:
            population_size: Number of chromosomes in population
            num_generations: Maximum number of generations
            crossover_prob: Probability of crossover
            mutation_prob: Probability of mutation
            elitism_ratio: Ratio of elite individuals to preserve
            early_stopping_patience: Generations without improvement to stop
            random_seed: Random seed for reproducibility
        """
        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.elitism_ratio = elitism_ratio
        self.early_stopping_patience = early_stopping_patience
        
        # Set random seed
        if random_seed:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # Initialize operators
        self.crossover_operator = CrossoverOperator()
        self.mutation_operator = MutationOperator()
        self.selection_operator = SelectionOperator()
        self.fitness_evaluator = FitnessEvaluator()
        
        # Optimization state
        self.population: List[Chromosome] = []
        self.best_chromosome: Optional[Chromosome] = None
        self.generation_history: List[Dict[str, Any]] = []
        self.is_running = False
        self.current_generation = 0
        
        # Results storage
        self.results_dir = get_project_root() / "ai" / "optimization" / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized GeneticOptimizer with population_size={population_size}, "
                   f"generations={num_generations}")
    
    def initialize_population(self, parameter_space: Dict[str, Any]) -> List[Chromosome]:
        """
        Initialize population with random chromosomes.
        
        Args:
            parameter_space: Dictionary defining parameter ranges
            
        Returns:
            List[Chromosome]: Initial population
        """
        population = []
        
        for _ in range(self.population_size):
            # Generate random parameters within bounds
            genes = {}
            for param_name, param_config in parameter_space.items():
                if param_config['type'] == 'float':
                    genes[param_name] = np.random.uniform(
                        param_config['min'], param_config['max']
                    )
                elif param_config['type'] == 'int':
                    genes[param_name] = np.random.randint(
                        param_config['min'], param_config['max'] + 1
                    )
                elif param_config['type'] == 'categorical':
                    genes[param_name] = np.random.choice(param_config['choices'])
                elif param_config['type'] == 'boolean':
                    genes[param_name] = np.random.choice([True, False])
            
            chromosome = Chromosome(genes)
            population.append(chromosome)
        
        logger.info(f"Initialized population of {len(population)} chromosomes")
        return population
    
    async def optimize(
        self,
        parameter_space: Dict[str, Any],
        fitness_function: Callable,
        fitness_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run genetic algorithm optimization.
        
        Args:
            parameter_space: Parameter space definition
            fitness_function: Function to evaluate fitness
            fitness_kwargs: Additional arguments for fitness function
            
        Returns:
            Dict: Optimization results
        """
        self.is_running = True
        self.current_generation = 0
        
        if fitness_kwargs is None:
            fitness_kwargs = {}
        
        # Initialize population
        self.population = self.initialize_population(parameter_space)
        
        # Evaluate initial population
        logger.info("Evaluating initial population...")
        await self._evaluate_population(fitness_function, fitness_kwargs)
        
        # Track best fitness for early stopping
        best_fitness_history = []
        generations_without_improvement = 0
        
        logger.info(f"Starting genetic algorithm optimization for {self.num_generations} generations")
        
        for generation in range(self.num_generations):
            self.current_generation = generation + 1
            
            # Create new generation
            new_population = await self._create_new_generation()
            
            # Evaluate new population
            await self._evaluate_population_subset(new_population, fitness_function, fitness_kwargs)
            
            # Replace old population
            self.population = new_population
            
            # Track best chromosome
            current_best = max(self.population, key=lambda x: x.fitness)
            if self.best_chromosome is None or current_best.fitness > self.best_chromosome.fitness:
                self.best_chromosome = current_best.copy()
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
            
            # Log progress
            avg_fitness = np.mean([c.fitness for c in self.population])
            best_fitness_history.append(current_best.fitness)
            
            generation_stats = {
                'generation': self.current_generation,
                'best_fitness': current_best.fitness,
                'avg_fitness': avg_fitness,
                'std_fitness': np.std([c.fitness for c in self.population]),
                'best_genes': current_best.genes.copy(),
                'timestamp': datetime.now().isoformat()
            }
            self.generation_history.append(generation_stats)
            
            logger.info(f"Generation {self.current_generation}: "
                       f"Best={current_best.fitness:.6f}, Avg={avg_fitness:.6f}")
            
            # Early stopping check
            if generations_without_improvement >= self.early_stopping_patience:
                logger.info(f"Early stopping at generation {self.current_generation} "
                           f"(no improvement for {generations_without_improvement} generations)")
                break
        
        self.is_running = False
        
        # Prepare results
        results = {
            'best_parameters': self.best_chromosome.genes,
            'best_fitness': self.best_chromosome.fitness,
            'generations_completed': self.current_generation,
            'converged': generations_without_improvement >= self.early_stopping_patience,
            'fitness_history': best_fitness_history,
            'generation_stats': self.generation_history,
            'optimization_config': {
                'population_size': self.population_size,
                'crossover_prob': self.crossover_prob,
                'mutation_prob': self.mutation_prob,
                'elitism_ratio': self.elitism_ratio
            },
            'parameter_space': parameter_space
        }
        
        # Save results
        await self._save_results(results)
        
        logger.info(f"Genetic optimization completed. Best fitness: {self.best_chromosome.fitness:.6f}")
        return results
    
    async def _create_new_generation(self) -> List[Chromosome]:
        """Create new generation using selection, crossover, and mutation."""
        new_population = []
        
        # Elitism - preserve best individuals
        num_elites = int(self.population_size * self.elitism_ratio)
        elite_chromosomes = sorted(self.population, key=lambda x: x.fitness, reverse=True)[:num_elites]
        new_population.extend([c.copy() for c in elite_chromosomes])
        
        # Generate rest of population
        while len(new_population) < self.population_size:
            # Selection
            parent1 = self.selection_operator.tournament_selection(self.population)
            parent2 = self.selection_operator.tournament_selection(self.population)
            
            # Crossover
            if np.random.random() < self.crossover_prob:
                child1, child2 = self.crossover_operator.uniform_crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            if np.random.random() < self.mutation_prob:
                child1 = self.mutation_operator.gaussian_mutation(child1)
            if np.random.random() < self.mutation_prob:
                child2 = self.mutation_operator.gaussian_mutation(child2)
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        return new_population[:self.population_size]
    
    async def _evaluate_population(self, fitness_function: Callable, fitness_kwargs: Dict[str, Any]):
        """Evaluate fitness for entire population."""
        tasks = []
        for chromosome in self.population:
            task = self.fitness_evaluator.evaluate(
                chromosome, fitness_function, **fitness_kwargs
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks)
    
    async def _evaluate_population_subset(
        self, 
        population_subset: List[Chromosome], 
        fitness_function: Callable, 
        fitness_kwargs: Dict[str, Any]
    ):
        """Evaluate fitness for population subset (new individuals only)."""
        tasks = []
        for chromosome in population_subset:
            if chromosome.fitness is None:  # Only evaluate if not already evaluated
                task = self.fitness_evaluator.evaluate(
                    chromosome, fitness_function, **fitness_kwargs
                )
                tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks)
    
    async def _save_results(self, results: Dict[str, Any]):
        """Save optimization results to disk."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = self.results_dir / f"genetic_optimization_{timestamp}.json"
            
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Results saved to {results_path}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status."""
        if not self.is_running:
            return {'status': 'idle', 'current_generation': 0}
        
        current_best = max(self.population, key=lambda x: x.fitness) if self.population else None
        avg_fitness = np.mean([c.fitness for c in self.population]) if self.population else 0
        
        return {
            'status': 'running',
            'current_generation': self.current_generation,
            'total_generations': self.num_generations,
            'progress_percent': (self.current_generation / self.num_generations) * 100,
            'population_size': len(self.population),
            'current_best_fitness': current_best.fitness if current_best else None,
            'current_avg_fitness': avg_fitness,
            'best_overall_fitness': self.best_chromosome.fitness if self.best_chromosome else None
        }
    
    def stop_optimization(self):
        """Stop optimization early."""
        self.is_running = False
        logger.info("Optimization stopped by user request")