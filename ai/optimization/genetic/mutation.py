"""
🧬 Mutation Operators
Various mutation strategies for genetic algorithms
"""

import numpy as np
import random
from typing import Dict, Any, List
from .chromosome import Chromosome
from core.logger import get_logger

logger = get_logger(__name__)


class MutationOperator:
    """
    Collection of mutation operators for genetic algorithms.
    """
    
    def __init__(self):
        """Initialize mutation operator."""
        self.mutation_methods = {
            'gaussian': self.gaussian_mutation,
            'uniform': self.uniform_mutation,
            'polynomial': self.polynomial_mutation,
            'adaptive': self.adaptive_mutation,
            'gene_swap': self.gene_swap_mutation
        }
    
    def gaussian_mutation(
        self, 
        chromosome: Chromosome,
        mutation_strength: float = 0.1,
        parameter_space: Dict[str, Any] = None
    ) -> Chromosome:
        """
        Gaussian mutation - add Gaussian noise to numeric genes.
        
        Args:
            chromosome: Chromosome to mutate
            mutation_strength: Standard deviation of Gaussian noise
            parameter_space: Parameter constraints
            
        Returns:
            Chromosome: Mutated chromosome
        """
        mutated_genes = chromosome.genes.copy()
        
        for gene_name, gene_value in mutated_genes.items():
            if isinstance(gene_value, (int, float)):
                # Add Gaussian noise
                noise = np.random.normal(0, mutation_strength)
                
                if isinstance(gene_value, int):
                    # For integers, scale noise and round
                    if parameter_space and gene_name in parameter_space:
                        param_range = parameter_space[gene_name]['max'] - parameter_space[gene_name]['min']
                        scaled_noise = noise * param_range
                    else:
                        scaled_noise = noise * abs(gene_value) if gene_value != 0 else noise
                    
                    new_value = int(round(gene_value + scaled_noise))
                else:
                    # For floats
                    if parameter_space and gene_name in parameter_space:
                        param_range = parameter_space[gene_name]['max'] - parameter_space[gene_name]['min']
                        scaled_noise = noise * param_range
                    else:
                        scaled_noise = noise * abs(gene_value) if gene_value != 0 else noise
                    
                    new_value = gene_value + scaled_noise
                
                mutated_genes[gene_name] = new_value
            
            elif isinstance(gene_value, bool):
                # Boolean mutation with low probability
                if random.random() < 0.1:
                    mutated_genes[gene_name] = not gene_value
        
        mutated_chromosome = Chromosome(mutated_genes)
        
        # Repair if parameter space provided
        if parameter_space:
            mutated_chromosome.repair(parameter_space)
        
        return mutated_chromosome
    
    def uniform_mutation(
        self, 
        chromosome: Chromosome,
        mutation_rate: float = 0.1,
        parameter_space: Dict[str, Any] = None
    ) -> Chromosome:
        """
        Uniform mutation - replace genes with random values from parameter space.
        
        Args:
            chromosome: Chromosome to mutate
            mutation_rate: Probability of mutating each gene
            parameter_space: Parameter space definition
            
        Returns:
            Chromosome: Mutated chromosome
        """
        if not parameter_space:
            logger.warning("Uniform mutation requires parameter space")
            return chromosome.copy()
        
        mutated_genes = chromosome.genes.copy()
        
        for gene_name, gene_value in mutated_genes.items():
            if gene_name in parameter_space and random.random() < mutation_rate:
                param_config = parameter_space[gene_name]
                
                if param_config['type'] == 'float':
                    mutated_genes[gene_name] = random.uniform(
                        param_config['min'], param_config['max']
                    )
                elif param_config['type'] == 'int':
                    mutated_genes[gene_name] = random.randint(
                        param_config['min'], param_config['max']
                    )
                elif param_config['type'] == 'categorical':
                    mutated_genes[gene_name] = random.choice(param_config['choices'])
                elif param_config['type'] == 'boolean':
                    mutated_genes[gene_name] = random.choice([True, False])
        
        return Chromosome(mutated_genes)
    
    def polynomial_mutation(
        self, 
        chromosome: Chromosome,
        eta: float = 20.0,
        parameter_space: Dict[str, Any] = None
    ) -> Chromosome:
        """
        Polynomial mutation - bounded mutation with controllable distribution.
        
        Args:
            chromosome: Chromosome to mutate
            eta: Distribution parameter (higher = more concentrated near original)
            parameter_space: Parameter constraints
            
        Returns:
            Chromosome: Mutated chromosome
        """
        if not parameter_space:
            return self.gaussian_mutation(chromosome)
        
        mutated_genes = chromosome.genes.copy()
        
        for gene_name, gene_value in mutated_genes.items():
            if gene_name in parameter_space and isinstance(gene_value, (int, float)):
                param_config = parameter_space[gene_name]
                
                if param_config['type'] in ['float', 'int']:
                    xl = param_config['min']
                    xu = param_config['max']
                    
                    # Normalize gene value
                    if xu - xl == 0:
                        continue
                    
                    y = (gene_value - xl) / (xu - xl)
                    y = np.clip(y, 0, 1)  # Ensure bounds
                    
                    # Generate random number
                    u = random.random()
                    
                    # Calculate delta
                    if u < 0.5:
                        delta = (2 * u) ** (1.0 / (eta + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                    
                    # Apply mutation
                    y_new = y + delta
                    y_new = np.clip(y_new, 0, 1)
                    
                    # Denormalize
                    new_value = xl + y_new * (xu - xl)
                    
                    if param_config['type'] == 'int':
                        new_value = int(round(new_value))
                    
                    mutated_genes[gene_name] = new_value
        
        return Chromosome(mutated_genes)
    
    def adaptive_mutation(
        self, 
        chromosome: Chromosome,
        generation: int,
        max_generations: int,
        parameter_space: Dict[str, Any] = None
    ) -> Chromosome:
        """
        Adaptive mutation that changes strength based on generation.
        
        Args:
            chromosome: Chromosome to mutate
            generation: Current generation
            max_generations: Maximum generations
            parameter_space: Parameter constraints
            
        Returns:
            Chromosome: Mutated chromosome
        """
        # Adaptive mutation strength
        progress = generation / max_generations
        initial_strength = 0.3
        final_strength = 0.05
        mutation_strength = initial_strength * (1 - progress) + final_strength * progress
        
        return self.gaussian_mutation(chromosome, mutation_strength, parameter_space)
    
    def gene_swap_mutation(
        self, 
        chromosome: Chromosome,
        swap_probability: float = 0.1
    ) -> Chromosome:
        """
        Gene swap mutation - swap values between genes of same type.
        
        Args:
            chromosome: Chromosome to mutate
            swap_probability: Probability of performing swap
            
        Returns:
            Chromosome: Mutated chromosome
        """
        if random.random() > swap_probability:
            return chromosome.copy()
        
        mutated_genes = chromosome.genes.copy()
        gene_names = list(mutated_genes.keys())
        
        # Group genes by type
        numeric_genes = []
        boolean_genes = []
        categorical_genes = []
        
        for gene_name in gene_names:
            gene_value = mutated_genes[gene_name]
            if isinstance(gene_value, (int, float)):
                numeric_genes.append(gene_name)
            elif isinstance(gene_value, bool):
                boolean_genes.append(gene_name)
            else:
                categorical_genes.append(gene_name)
        
        # Perform swaps within each type
        for gene_group in [numeric_genes, boolean_genes, categorical_genes]:
            if len(gene_group) >= 2:
                gene1, gene2 = random.sample(gene_group, 2)
                mutated_genes[gene1], mutated_genes[gene2] = mutated_genes[gene2], mutated_genes[gene1]
        
        return Chromosome(mutated_genes)
    
    def creep_mutation(
        self, 
        chromosome: Chromosome,
        creep_rate: float = 0.01,
        parameter_space: Dict[str, Any] = None
    ) -> Chromosome:
        """
        Creep mutation - small incremental changes to numeric genes.
        
        Args:
            chromosome: Chromosome to mutate
            creep_rate: Rate of creeping mutation
            parameter_space: Parameter constraints
            
        Returns:
            Chromosome: Mutated chromosome
        """
        mutated_genes = chromosome.genes.copy()
        
        for gene_name, gene_value in mutated_genes.items():
            if isinstance(gene_value, (int, float)):
                if parameter_space and gene_name in parameter_space:
                    param_config = parameter_space[gene_name]
                    param_range = param_config['max'] - param_config['min']
                    max_change = creep_rate * param_range
                else:
                    max_change = creep_rate * abs(gene_value) if gene_value != 0 else creep_rate
                
                # Random change in [-max_change, max_change]
                change = random.uniform(-max_change, max_change)
                new_value = gene_value + change
                
                if isinstance(gene_value, int):
                    new_value = int(round(new_value))
                
                mutated_genes[gene_name] = new_value
        
        mutated_chromosome = Chromosome(mutated_genes)
        
        # Repair if parameter space provided
        if parameter_space:
            mutated_chromosome.repair(parameter_space)
        
        return mutated_chromosome
    
    def mutate(
        self, 
        chromosome: Chromosome,
        method: str = 'gaussian',
        **kwargs
    ) -> Chromosome:
        """
        General mutation method that dispatches to specific operators.
        
        Args:
            chromosome: Chromosome to mutate
            method: Mutation method name
            **kwargs: Additional parameters for mutation method
            
        Returns:
            Chromosome: Mutated chromosome
        """
        if method not in self.mutation_methods:
            logger.warning(f"Unknown mutation method: {method}. Using gaussian mutation.")
            method = 'gaussian'
        
        mutation_func = self.mutation_methods[method]
        return mutation_func(chromosome, **kwargs)