"""
🔀 Crossover Operators
Various crossover strategies for genetic algorithms
"""

import numpy as np
import random
from typing import Tuple, List, Dict, Any
from .chromosome import Chromosome
from core.logger import get_logger

logger = get_logger(__name__)


class CrossoverOperator:
    """
    Collection of crossover operators for genetic algorithms.
    """
    
    def __init__(self):
        """Initialize crossover operator."""
        self.crossover_methods = {
            'uniform': self.uniform_crossover,
            'single_point': self.single_point_crossover,
            'two_point': self.two_point_crossover,
            'blend': self.blend_crossover,
            'arithmetic': self.arithmetic_crossover
        }
    
    def uniform_crossover(
        self, 
        parent1: Chromosome, 
        parent2: Chromosome,
        prob: float = 0.5
    ) -> Tuple[Chromosome, Chromosome]:
        """
        Uniform crossover - each gene independently chosen from either parent.
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            prob: Probability of choosing gene from parent1
            
        Returns:
            Tuple[Chromosome, Chromosome]: Two offspring chromosomes
        """
        genes1 = parent1.genes.copy()
        genes2 = parent2.genes.copy()
        
        for gene_name in genes1.keys():
            if gene_name in genes2:
                if random.random() < prob:
                    # Swap genes
                    genes1[gene_name], genes2[gene_name] = genes2[gene_name], genes1[gene_name]
        
        child1 = Chromosome(genes1)
        child2 = Chromosome(genes2)
        
        return child1, child2
    
    def single_point_crossover(
        self, 
        parent1: Chromosome, 
        parent2: Chromosome
    ) -> Tuple[Chromosome, Chromosome]:
        """
        Single-point crossover for ordered genes.
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            
        Returns:
            Tuple[Chromosome, Chromosome]: Two offspring chromosomes
        """
        gene_names = list(parent1.genes.keys())
        crossover_point = random.randint(1, len(gene_names) - 1)
        
        genes1 = parent1.genes.copy()
        genes2 = parent2.genes.copy()
        
        # Swap genes after crossover point
        for i in range(crossover_point, len(gene_names)):
            gene_name = gene_names[i]
            if gene_name in genes2:
                genes1[gene_name], genes2[gene_name] = genes2[gene_name], genes1[gene_name]
        
        child1 = Chromosome(genes1)
        child2 = Chromosome(genes2)
        
        return child1, child2
    
    def two_point_crossover(
        self, 
        parent1: Chromosome, 
        parent2: Chromosome
    ) -> Tuple[Chromosome, Chromosome]:
        """
        Two-point crossover for ordered genes.
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            
        Returns:
            Tuple[Chromosome, Chromosome]: Two offspring chromosomes
        """
        gene_names = list(parent1.genes.keys())
        if len(gene_names) < 3:
            return self.single_point_crossover(parent1, parent2)
        
        # Choose two crossover points
        point1 = random.randint(1, len(gene_names) - 2)
        point2 = random.randint(point1 + 1, len(gene_names) - 1)
        
        genes1 = parent1.genes.copy()
        genes2 = parent2.genes.copy()
        
        # Swap genes between crossover points
        for i in range(point1, point2):
            gene_name = gene_names[i]
            if gene_name in genes2:
                genes1[gene_name], genes2[gene_name] = genes2[gene_name], genes1[gene_name]
        
        child1 = Chromosome(genes1)
        child2 = Chromosome(genes2)
        
        return child1, child2
    
    def blend_crossover(
        self, 
        parent1: Chromosome, 
        parent2: Chromosome,
        alpha: float = 0.5
    ) -> Tuple[Chromosome, Chromosome]:
        """
        Blend crossover for numeric genes (BLX-α).
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            alpha: Blending parameter
            
        Returns:
            Tuple[Chromosome, Chromosome]: Two offspring chromosomes
        """
        genes1 = parent1.genes.copy()
        genes2 = parent2.genes.copy()
        
        for gene_name in genes1.keys():
            if gene_name in genes2:
                val1 = genes1[gene_name]
                val2 = genes2[gene_name]
                
                # Only apply blend crossover to numeric values
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    min_val = min(val1, val2)
                    max_val = max(val1, val2)
                    range_val = max_val - min_val
                    
                    # Extended range for blending
                    low = min_val - alpha * range_val
                    high = max_val + alpha * range_val
                    
                    # Generate new values
                    new_val1 = random.uniform(low, high)
                    new_val2 = random.uniform(low, high)
                    
                    # Preserve type (int or float)
                    if isinstance(val1, int):
                        new_val1 = int(round(new_val1))
                        new_val2 = int(round(new_val2))
                    
                    genes1[gene_name] = new_val1
                    genes2[gene_name] = new_val2
        
        child1 = Chromosome(genes1)
        child2 = Chromosome(genes2)
        
        return child1, child2
    
    def arithmetic_crossover(
        self, 
        parent1: Chromosome, 
        parent2: Chromosome,
        weight: float = 0.5
    ) -> Tuple[Chromosome, Chromosome]:
        """
        Arithmetic crossover - weighted average of parent genes.
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            weight: Weight for parent1 (1-weight for parent2)
            
        Returns:
            Tuple[Chromosome, Chromosome]: Two offspring chromosomes
        """
        genes1 = parent1.genes.copy()
        genes2 = parent2.genes.copy()
        
        for gene_name in genes1.keys():
            if gene_name in genes2:
                val1 = genes1[gene_name]
                val2 = genes2[gene_name]
                
                # Only apply arithmetic crossover to numeric values
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # Weighted average
                    new_val1 = weight * val1 + (1 - weight) * val2
                    new_val2 = (1 - weight) * val1 + weight * val2
                    
                    # Preserve type (int or float)
                    if isinstance(val1, int):
                        new_val1 = int(round(new_val1))
                        new_val2 = int(round(new_val2))
                    
                    genes1[gene_name] = new_val1
                    genes2[gene_name] = new_val2
        
        child1 = Chromosome(genes1)
        child2 = Chromosome(genes2)
        
        return child1, child2
    
    def adaptive_crossover(
        self, 
        parent1: Chromosome, 
        parent2: Chromosome,
        generation: int,
        max_generations: int
    ) -> Tuple[Chromosome, Chromosome]:
        """
        Adaptive crossover that changes strategy based on generation.
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            generation: Current generation
            max_generations: Maximum generations
            
        Returns:
            Tuple[Chromosome, Chromosome]: Two offspring chromosomes
        """
        # Progress ratio (0 to 1)
        progress = generation / max_generations
        
        if progress < 0.3:
            # Early generations: use blend crossover for exploration
            return self.blend_crossover(parent1, parent2, alpha=0.5)
        elif progress < 0.7:
            # Middle generations: use uniform crossover
            return self.uniform_crossover(parent1, parent2)
        else:
            # Late generations: use arithmetic crossover for exploitation
            return self.arithmetic_crossover(parent1, parent2)
    
    def crossover(
        self, 
        parent1: Chromosome, 
        parent2: Chromosome,
        method: str = 'uniform',
        **kwargs
    ) -> Tuple[Chromosome, Chromosome]:
        """
        General crossover method that dispatches to specific operators.
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            method: Crossover method name
            **kwargs: Additional parameters for crossover method
            
        Returns:
            Tuple[Chromosome, Chromosome]: Two offspring chromosomes
        """
        if method not in self.crossover_methods:
            logger.warning(f"Unknown crossover method: {method}. Using uniform crossover.")
            method = 'uniform'
        
        crossover_func = self.crossover_methods[method]
        return crossover_func(parent1, parent2, **kwargs)