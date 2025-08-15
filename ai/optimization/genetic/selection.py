"""
🎯 Selection Operators
Various selection strategies for genetic algorithms
"""

import numpy as np
import random
from typing import List, Tuple
from .chromosome import Chromosome
from core.logger import get_logger

logger = get_logger(__name__)


class SelectionOperator:
    """
    Collection of selection operators for genetic algorithms.
    """
    
    def __init__(self):
        """Initialize selection operator."""
        self.selection_methods = {
            'tournament': self.tournament_selection,
            'roulette': self.roulette_wheel_selection,
            'rank': self.rank_based_selection,
            'sus': self.stochastic_universal_sampling,
            'elite': self.elite_selection
        }
    
    def tournament_selection(
        self, 
        population: List[Chromosome],
        tournament_size: int = 3
    ) -> Chromosome:
        """
        Tournament selection - select best individual from random tournament.
        
        Args:
            population: Population of chromosomes
            tournament_size: Size of tournament
            
        Returns:
            Chromosome: Selected chromosome
        """
        if tournament_size > len(population):
            tournament_size = len(population)
        
        # Select random individuals for tournament
        tournament = random.sample(population, tournament_size)
        
        # Return best individual from tournament
        return max(tournament, key=lambda x: x.fitness if x.fitness is not None else -float('inf'))
    
    def roulette_wheel_selection(
        self, 
        population: List[Chromosome]
    ) -> Chromosome:
        """
        Roulette wheel selection - probability proportional to fitness.
        
        Args:
            population: Population of chromosomes
            
        Returns:
            Chromosome: Selected chromosome
        """
        # Handle negative fitness values by shifting
        fitnesses = [c.fitness if c.fitness is not None else 0 for c in population]
        min_fitness = min(fitnesses)
        
        if min_fitness < 0:
            # Shift all fitnesses to be positive
            shifted_fitnesses = [f - min_fitness + 1e-10 for f in fitnesses]
        else:
            shifted_fitnesses = [f + 1e-10 for f in fitnesses]  # Add small value to avoid zero
        
        total_fitness = sum(shifted_fitnesses)
        
        if total_fitness == 0:
            return random.choice(population)
        
        # Generate random value
        r = random.uniform(0, total_fitness)
        
        # Find selected individual
        cumulative_fitness = 0
        for i, fitness in enumerate(shifted_fitnesses):
            cumulative_fitness += fitness
            if cumulative_fitness >= r:
                return population[i]
        
        # Fallback (shouldn't happen)
        return population[-1]
    
    def rank_based_selection(
        self, 
        population: List[Chromosome],
        selection_pressure: float = 1.5
    ) -> Chromosome:
        """
        Rank-based selection - probability based on rank, not fitness value.
        
        Args:
            population: Population of chromosomes
            selection_pressure: Selection pressure (1.0 = uniform, 2.0 = linear)
            
        Returns:
            Chromosome: Selected chromosome
        """
        # Sort population by fitness
        sorted_population = sorted(
            population, 
            key=lambda x: x.fitness if x.fitness is not None else -float('inf')
        )
        
        n = len(population)
        
        # Calculate selection probabilities based on rank
        probabilities = []
        for rank in range(n):
            # Linear ranking
            prob = (2 - selection_pressure) / n + (2 * rank * (selection_pressure - 1)) / (n * (n - 1))
            probabilities.append(prob)
        
        # Roulette wheel selection on probabilities
        r = random.random()
        cumulative_prob = 0
        
        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if cumulative_prob >= r:
                return sorted_population[i]
        
        return sorted_population[-1]
    
    def stochastic_universal_sampling(
        self, 
        population: List[Chromosome],
        num_selections: int = 1
    ) -> List[Chromosome]:
        """
        Stochastic Universal Sampling - evenly spaced selection.
        
        Args:
            population: Population of chromosomes
            num_selections: Number of individuals to select
            
        Returns:
            List[Chromosome]: Selected chromosomes
        """
        # Handle negative fitness values
        fitnesses = [c.fitness if c.fitness is not None else 0 for c in population]
        min_fitness = min(fitnesses)
        
        if min_fitness < 0:
            shifted_fitnesses = [f - min_fitness + 1e-10 for f in fitnesses]
        else:
            shifted_fitnesses = [f + 1e-10 for f in fitnesses]
        
        total_fitness = sum(shifted_fitnesses)
        
        if total_fitness == 0:
            return random.choices(population, k=num_selections)
        
        # Calculate pointer spacing
        pointer_spacing = total_fitness / num_selections
        start = random.uniform(0, pointer_spacing)
        
        # Generate evenly spaced pointers
        pointers = [start + i * pointer_spacing for i in range(num_selections)]
        
        # Select individuals
        selected = []
        cumulative_fitness = 0
        population_index = 0
        
        for pointer in pointers:
            while cumulative_fitness < pointer and population_index < len(population):
                cumulative_fitness += shifted_fitnesses[population_index]
                population_index += 1
            
            if population_index > 0:
                selected.append(population[population_index - 1])
            else:
                selected.append(population[0])
        
        return selected if num_selections > 1 else selected[0]
    
    def elite_selection(
        self, 
        population: List[Chromosome],
        num_elites: int = 1
    ) -> List[Chromosome]:
        """
        Elite selection - select best individuals.
        
        Args:
            population: Population of chromosomes
            num_elites: Number of elite individuals to select
            
        Returns:
            List[Chromosome]: Elite chromosomes
        """
        sorted_population = sorted(
            population,
            key=lambda x: x.fitness if x.fitness is not None else -float('inf'),
            reverse=True
        )
        
        elites = sorted_population[:num_elites]
        return elites if num_elites > 1 else elites[0]
    
    def diversity_selection(
        self, 
        population: List[Chromosome],
        diversity_weight: float = 0.3
    ) -> Chromosome:
        """
        Diversity-preserving selection - balance fitness and diversity.
        
        Args:
            population: Population of chromosomes
            diversity_weight: Weight for diversity component
            
        Returns:
            Chromosome: Selected chromosome
        """
        if len(population) < 2:
            return random.choice(population)
        
        # Calculate diversity scores
        diversity_scores = []
        
        for i, chromosome in enumerate(population):
            # Calculate average distance to other chromosomes
            distances = []
            for j, other in enumerate(population):
                if i != j:
                    try:
                        distance = chromosome.distance_to(other)
                        distances.append(distance)
                    except ValueError:
                        # Skip if distance calculation fails
                        continue
            
            avg_distance = np.mean(distances) if distances else 0
            diversity_scores.append(avg_distance)
        
        # Normalize diversity scores
        if max(diversity_scores) > 0:
            diversity_scores = [score / max(diversity_scores) for score in diversity_scores]
        
        # Normalize fitness scores
        fitnesses = [c.fitness if c.fitness is not None else 0 for c in population]
        if max(fitnesses) > min(fitnesses):
            normalized_fitnesses = [(f - min(fitnesses)) / (max(fitnesses) - min(fitnesses)) 
                                   for f in fitnesses]
        else:
            normalized_fitnesses = [1.0] * len(fitnesses)
        
        # Combined scores
        combined_scores = [
            (1 - diversity_weight) * fitness + diversity_weight * diversity
            for fitness, diversity in zip(normalized_fitnesses, diversity_scores)
        ]
        
        # Select based on combined scores
        total_score = sum(combined_scores)
        if total_score == 0:
            return random.choice(population)
        
        r = random.uniform(0, total_score)
        cumulative_score = 0
        
        for i, score in enumerate(combined_scores):
            cumulative_score += score
            if cumulative_score >= r:
                return population[i]
        
        return population[-1]
    
    def select(
        self, 
        population: List[Chromosome],
        method: str = 'tournament',
        **kwargs
    ) -> Chromosome:
        """
        General selection method that dispatches to specific operators.
        
        Args:
            population: Population of chromosomes
            method: Selection method name
            **kwargs: Additional parameters for selection method
            
        Returns:
            Chromosome: Selected chromosome
        """
        if method not in self.selection_methods:
            logger.warning(f"Unknown selection method: {method}. Using tournament selection.")
            method = 'tournament'
        
        selection_func = self.selection_methods[method]
        return selection_func(population, **kwargs)