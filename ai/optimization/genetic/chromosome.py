"""
🧬 Chromosome Class
Representation of solution candidates for genetic algorithm
"""

import copy
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Chromosome:
    """
    Chromosome class representing a solution candidate in genetic algorithm.
    
    Contains genes (parameters) and associated fitness value.
    """
    
    def __init__(self, genes: Dict[str, Any]):
        """
        Initialize chromosome with genes.
        
        Args:
            genes: Dictionary of parameter names and values
        """
        self.genes = genes.copy()
        self.fitness: Optional[float] = None
        self.age = 0  # Number of generations this chromosome has survived
        self.evaluation_count = 0  # Number of times this chromosome was evaluated
    
    def copy(self) -> 'Chromosome':
        """Create a deep copy of the chromosome."""
        new_chromosome = Chromosome(self.genes.copy())
        new_chromosome.fitness = self.fitness
        new_chromosome.age = self.age
        new_chromosome.evaluation_count = self.evaluation_count
        return new_chromosome
    
    def get_gene(self, gene_name: str) -> Any:
        """Get value of specific gene."""
        return self.genes.get(gene_name)
    
    def set_gene(self, gene_name: str, value: Any):
        """Set value of specific gene."""
        self.genes[gene_name] = value
        # Reset fitness when genes change
        self.fitness = None
    
    def get_genes_as_array(self) -> np.ndarray:
        """Convert genes to numpy array for numeric operations."""
        numeric_genes = []
        for value in self.genes.values():
            if isinstance(value, (int, float)):
                numeric_genes.append(float(value))
            elif isinstance(value, bool):
                numeric_genes.append(float(value))
            # Skip non-numeric values
        
        return np.array(numeric_genes)
    
    def set_genes_from_array(self, gene_array: np.ndarray, gene_names: list):
        """Set genes from numpy array."""
        if len(gene_array) != len(gene_names):
            raise ValueError("Array length must match number of gene names")
        
        for i, name in enumerate(gene_names):
            # Determine original type and convert back
            original_type = type(self.genes.get(name, float()))
            if original_type == bool:
                self.genes[name] = bool(gene_array[i] > 0.5)
            elif original_type == int:
                self.genes[name] = int(round(gene_array[i]))
            else:
                self.genes[name] = float(gene_array[i])
        
        # Reset fitness when genes change
        self.fitness = None
    
    def distance_to(self, other: 'Chromosome') -> float:
        """
        Calculate Euclidean distance to another chromosome.
        Only considers numeric genes.
        """
        self_array = self.get_genes_as_array()
        other_array = other.get_genes_as_array()
        
        if len(self_array) != len(other_array):
            raise ValueError("Chromosomes must have same number of numeric genes")
        
        return np.linalg.norm(self_array - other_array)
    
    def is_valid(self, parameter_space: Dict[str, Any]) -> bool:
        """
        Check if chromosome genes are within valid parameter space.
        
        Args:
            parameter_space: Dictionary defining parameter constraints
            
        Returns:
            bool: True if all genes are valid
        """
        for gene_name, gene_value in self.genes.items():
            if gene_name not in parameter_space:
                continue  # Skip genes not in parameter space
            
            param_config = parameter_space[gene_name]
            
            if param_config['type'] == 'float':
                if not (param_config['min'] <= gene_value <= param_config['max']):
                    return False
            elif param_config['type'] == 'int':
                if not (param_config['min'] <= gene_value <= param_config['max']):
                    return False
                if not isinstance(gene_value, int):
                    return False
            elif param_config['type'] == 'categorical':
                if gene_value not in param_config['choices']:
                    return False
            elif param_config['type'] == 'boolean':
                if not isinstance(gene_value, bool):
                    return False
        
        return True
    
    def repair(self, parameter_space: Dict[str, Any]):
        """
        Repair chromosome by clipping genes to valid ranges.
        
        Args:
            parameter_space: Dictionary defining parameter constraints
        """
        for gene_name, gene_value in self.genes.items():
            if gene_name not in parameter_space:
                continue
            
            param_config = parameter_space[gene_name]
            
            if param_config['type'] == 'float':
                self.genes[gene_name] = np.clip(
                    gene_value, param_config['min'], param_config['max']
                )
            elif param_config['type'] == 'int':
                clipped_value = np.clip(
                    gene_value, param_config['min'], param_config['max']
                )
                self.genes[gene_name] = int(round(clipped_value))
            elif param_config['type'] == 'categorical':
                if gene_value not in param_config['choices']:
                    self.genes[gene_name] = np.random.choice(param_config['choices'])
            elif param_config['type'] == 'boolean':
                if not isinstance(gene_value, bool):
                    self.genes[gene_name] = bool(gene_value)
        
        # Reset fitness after repair
        self.fitness = None
    
    def increment_age(self):
        """Increment age of chromosome."""
        self.age += 1
    
    def increment_evaluation_count(self):
        """Increment evaluation count."""
        self.evaluation_count += 1
    
    def __str__(self) -> str:
        """String representation of chromosome."""
        fitness_str = f"{self.fitness:.6f}" if self.fitness is not None else "None"
        return f"Chromosome(genes={len(self.genes)}, fitness={fitness_str}, age={self.age})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __eq__(self, other: 'Chromosome') -> bool:
        """Check equality based on genes."""
        if not isinstance(other, Chromosome):
            return False
        return self.genes == other.genes
    
    def __hash__(self) -> int:
        """Hash based on genes for use in sets."""
        # Convert genes to hashable format
        hashable_genes = []
        for key, value in sorted(self.genes.items()):
            if isinstance(value, (list, dict)):
                hashable_genes.append((key, str(value)))
            else:
                hashable_genes.append((key, value))
        
        return hash(tuple(hashable_genes))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chromosome to dictionary for serialization."""
        return {
            'genes': self.genes,
            'fitness': self.fitness,
            'age': self.age,
            'evaluation_count': self.evaluation_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Chromosome':
        """Create chromosome from dictionary."""
        chromosome = cls(data['genes'])
        chromosome.fitness = data.get('fitness')
        chromosome.age = data.get('age', 0)
        chromosome.evaluation_count = data.get('evaluation_count', 0)
        return chromosome