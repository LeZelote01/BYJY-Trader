"""
🧬 NSGA-II Algorithm
Non-dominated Sorting Genetic Algorithm II for multi-objective optimization
"""

import numpy as np
import asyncio
from typing import Dict, List, Any, Tuple, Optional, Callable
from datetime import datetime
import random

from core.logger import get_logger

logger = get_logger(__name__)


class NSGA2:
    """
    NSGA-II (Non-dominated Sorting Genetic Algorithm II) implementation.
    
    Multi-objective evolutionary algorithm that maintains diversity through
    crowding distance and uses non-dominated sorting for selection.
    """
    
    def __init__(
        self,
        population_size: int = 100,
        num_generations: int = 200,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.1,
        tournament_size: int = 2,
        random_seed: Optional[int] = None
    ):
        """
        Initialize NSGA-II algorithm.
        
        Args:
            population_size: Size of population (should be even)
            num_generations: Number of generations
            crossover_prob: Crossover probability
            mutation_prob: Mutation probability
            tournament_size: Tournament size for selection
            random_seed: Random seed for reproducibility
        """
        # Ensure even population size for NSGA-II
        self.population_size = population_size if population_size % 2 == 0 else population_size + 1
        self.num_generations = num_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        
        # Set random seed
        if random_seed:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # Algorithm state
        self.population = []
        self.pareto_fronts = []
        self.crowding_distances = []
        self.is_running = False
        self.current_generation = 0
        
        logger.info(f"Initialized NSGA-II with population_size={self.population_size}")
    
    async def optimize(
        self,
        parameter_space: Dict[str, Any],
        objective_function: Callable,
        num_objectives: int,
        evaluation_function: Callable = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run NSGA-II optimization.
        
        Args:
            parameter_space: Parameter space definition
            objective_function: Function to evaluate objectives
            num_objectives: Number of objectives
            evaluation_function: Base evaluation function
            **kwargs: Additional arguments
            
        Returns:
            Dict[str, Any]: Optimization results
        """
        self.is_running = True
        self.current_generation = 0
        generation_history = []
        
        logger.info(f"Starting NSGA-II optimization with {num_objectives} objectives")
        
        # Initialize population
        self.population = await self._initialize_population(
            parameter_space, objective_function, evaluation_function, **kwargs
        )
        
        # Main evolution loop
        for generation in range(self.num_generations):
            self.current_generation = generation + 1
            
            # Create offspring population
            offspring_population = await self._create_offspring(
                parameter_space, objective_function, evaluation_function, **kwargs
            )
            
            # Combine parent and offspring populations
            combined_population = self.population + offspring_population
            
            # Environmental selection using non-dominated sorting and crowding distance
            self.population = self._environmental_selection(combined_population)
            
            # Update Pareto fronts and crowding distances
            self._update_pareto_fronts()
            self._calculate_crowding_distances()
            
            # Record generation statistics
            gen_stats = self._calculate_generation_stats(generation + 1)
            generation_history.append(gen_stats)
            
            # Log progress
            logger.info(f"Generation {self.current_generation}: "
                       f"Fronts={len(self.pareto_fronts)}, "
                       f"Front0_size={len(self.pareto_fronts[0]) if self.pareto_fronts else 0}")
            
            # Early stopping check (optional - based on convergence)
            if self._check_convergence(generation_history):
                logger.info(f"Convergence detected at generation {self.current_generation}")
                break
        
        # Extract final Pareto front
        final_pareto_front = self._extract_pareto_front()
        
        # Prepare results
        results = {
            'final_population': [self._solution_to_dict(sol) for sol in self.population],
            'pareto_front': [self._solution_to_dict(sol) for sol in final_pareto_front],
            'pareto_fronts': [[self._solution_to_dict(sol) for sol in front] 
                             for front in self.pareto_fronts],
            'generation_history': generation_history,
            'generations_completed': self.current_generation,
            'num_objectives': num_objectives,
            'population_size': self.population_size,
            'algorithm_config': {
                'crossover_prob': self.crossover_prob,
                'mutation_prob': self.mutation_prob,
                'tournament_size': self.tournament_size
            }
        }
        
        self.is_running = False
        
        logger.info(f"NSGA-II optimization completed. Final Pareto front size: {len(final_pareto_front)}")
        return results
    
    async def _initialize_population(
        self,
        parameter_space: Dict[str, Any],
        objective_function: Callable,
        evaluation_function: Callable = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Initialize population with random solutions."""
        population = []
        
        for _ in range(self.population_size):
            # Generate random parameters
            parameters = {}
            for param_name, param_config in parameter_space.items():
                if param_config['type'] == 'float':
                    parameters[param_name] = np.random.uniform(
                        param_config['min'], param_config['max']
                    )
                elif param_config['type'] == 'int':
                    parameters[param_name] = np.random.randint(
                        param_config['min'], param_config['max'] + 1
                    )
                elif param_config['type'] == 'categorical':
                    parameters[param_name] = np.random.choice(param_config['choices'])
                elif param_config['type'] == 'boolean':
                    parameters[param_name] = np.random.choice([True, False])
            
            # Evaluate objectives
            if evaluation_function:
                objectives = await objective_function(
                    parameters, evaluation_function, **kwargs
                )
            else:
                objectives = await objective_function(parameters, **kwargs)
            
            solution = {
                'parameters': parameters,
                'objectives': objectives,
                'rank': 0,
                'crowding_distance': 0.0
            }
            
            population.append(solution)
        
        logger.info(f"Initialized population of {len(population)} solutions")
        return population
    
    async def _create_offspring(
        self,
        parameter_space: Dict[str, Any],
        objective_function: Callable,
        evaluation_function: Callable = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Create offspring population through selection, crossover, and mutation."""
        offspring = []
        
        while len(offspring) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if random.random() < self.crossover_prob:
                child1_params, child2_params = self._crossover(
                    parent1['parameters'], parent2['parameters'], parameter_space
                )
            else:
                child1_params = parent1['parameters'].copy()
                child2_params = parent2['parameters'].copy()
            
            # Mutation
            if random.random() < self.mutation_prob:
                child1_params = self._mutate(child1_params, parameter_space)
            if random.random() < self.mutation_prob:
                child2_params = self._mutate(child2_params, parameter_space)
            
            # Evaluate offspring
            for child_params in [child1_params, child2_params]:
                if len(offspring) >= self.population_size:
                    break
                
                try:
                    if evaluation_function:
                        objectives = await objective_function(
                            child_params, evaluation_function, **kwargs
                        )
                    else:
                        objectives = await objective_function(child_params, **kwargs)
                    
                    child_solution = {
                        'parameters': child_params,
                        'objectives': objectives,
                        'rank': 0,
                        'crowding_distance': 0.0
                    }
                    
                    offspring.append(child_solution)
                    
                except Exception as e:
                    logger.warning(f"Failed to evaluate offspring: {e}")
                    # Skip this offspring
                    continue
        
        return offspring[:self.population_size]
    
    def _tournament_selection(self) -> Dict[str, Any]:
        """Tournament selection based on dominance and crowding distance."""
        tournament = random.sample(self.population, min(self.tournament_size, len(self.population)))
        
        # Sort by rank first, then by crowding distance
        tournament.sort(key=lambda x: (x['rank'], -x['crowding_distance']))
        
        return tournament[0]
    
    def _crossover(
        self,
        parent1_params: Dict[str, Any],
        parent2_params: Dict[str, Any],
        parameter_space: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Simulated binary crossover (SBX) for real parameters."""
        child1_params = parent1_params.copy()
        child2_params = parent2_params.copy()
        eta_c = 20.0  # Distribution index for crossover
        
        for param_name in parent1_params.keys():
            if param_name not in parameter_space:
                continue
            
            param_config = parameter_space[param_name]
            
            if param_config['type'] == 'float':
                # SBX crossover for float parameters
                x1 = parent1_params[param_name]
                x2 = parent2_params[param_name]
                
                if abs(x1 - x2) > 1e-14:
                    xl = param_config['min']
                    xu = param_config['max']
                    
                    if x1 > x2:
                        x1, x2 = x2, x1
                    
                    # Calculate beta
                    rand = random.random()
                    
                    if rand <= 0.5:
                        beta = (2 * rand) ** (1.0 / (eta_c + 1))
                    else:
                        beta = (1.0 / (2 * (1 - rand))) ** (1.0 / (eta_c + 1))
                    
                    # Generate offspring
                    c1 = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
                    c2 = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
                    
                    # Apply bounds
                    c1 = np.clip(c1, xl, xu)
                    c2 = np.clip(c2, xl, xu)
                    
                    child1_params[param_name] = c1
                    child2_params[param_name] = c2
            
            elif param_config['type'] == 'int':
                # Integer crossover
                x1 = parent1_params[param_name]
                x2 = parent2_params[param_name]
                
                if random.random() < 0.5:
                    child1_params[param_name] = x1
                    child2_params[param_name] = x2
                else:
                    child1_params[param_name] = x2
                    child2_params[param_name] = x1
            
            elif param_config['type'] == 'categorical':
                # Uniform crossover for categorical
                if random.random() < 0.5:
                    child1_params[param_name] = parent2_params[param_name]
                    child2_params[param_name] = parent1_params[param_name]
            
            elif param_config['type'] == 'boolean':
                # Boolean crossover
                if random.random() < 0.5:
                    child1_params[param_name] = parent2_params[param_name]
                    child2_params[param_name] = parent1_params[param_name]
        
        return child1_params, child2_params
    
    def _mutate(
        self,
        parameters: Dict[str, Any],
        parameter_space: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Polynomial mutation for real parameters."""
        mutated_params = parameters.copy()
        eta_m = 20.0  # Distribution index for mutation
        
        for param_name, param_value in parameters.items():
            if param_name not in parameter_space:
                continue
            
            param_config = parameter_space[param_name]
            
            if param_config['type'] == 'float':
                # Polynomial mutation for float parameters
                xl = param_config['min']
                xu = param_config['max']
                
                if xu - xl == 0:
                    continue
                
                delta1 = (param_value - xl) / (xu - xl)
                delta2 = (xu - param_value) / (xu - xl)
                
                rand = random.random()
                mut_pow = 1.0 / (eta_m + 1)
                
                if rand < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta_m + 1)
                    delta_q = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta_m + 1)
                    delta_q = 1.0 - val ** mut_pow
                
                mutated_value = param_value + delta_q * (xu - xl)
                mutated_value = np.clip(mutated_value, xl, xu)
                mutated_params[param_name] = mutated_value
            
            elif param_config['type'] == 'int':
                # Integer mutation
                if random.random() < 0.1:  # 10% mutation probability for integers
                    mutated_params[param_name] = random.randint(
                        param_config['min'], param_config['max']
                    )
            
            elif param_config['type'] == 'categorical':
                # Categorical mutation
                if random.random() < 0.1:  # 10% mutation probability
                    mutated_params[param_name] = random.choice(param_config['choices'])
            
            elif param_config['type'] == 'boolean':
                # Boolean mutation
                if random.random() < 0.1:  # 10% mutation probability
                    mutated_params[param_name] = not param_value
        
        return mutated_params
    
    def _environmental_selection(self, combined_population: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Environmental selection using non-dominated sorting and crowding distance."""
        # Non-dominated sorting
        fronts = self._non_dominated_sorting(combined_population)
        
        # Build new population
        new_population = []
        front_index = 0
        
        while len(new_population) + len(fronts[front_index]) <= self.population_size:
            # Add entire front
            new_population.extend(fronts[front_index])
            front_index += 1
            
            if front_index >= len(fronts):
                break
        
        # If we need to add partial front, use crowding distance
        if len(new_population) < self.population_size and front_index < len(fronts):
            remaining_slots = self.population_size - len(new_population)
            last_front = fronts[front_index]
            
            # Calculate crowding distances for last front
            self._calculate_crowding_distance_for_front(last_front)
            
            # Sort by crowding distance (descending) and add best individuals
            last_front.sort(key=lambda x: x['crowding_distance'], reverse=True)
            new_population.extend(last_front[:remaining_slots])
        
        return new_population
    
    def _non_dominated_sorting(self, population: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Perform non-dominated sorting on population."""
        fronts = [[]]
        domination_counts = {}
        dominated_solutions = {}
        
        # Initialize domination structures
        for p in population:
            p_id = id(p)
            domination_counts[p_id] = 0
            dominated_solutions[p_id] = []
        
        # Calculate domination relationships
        for i, p in enumerate(population):
            for j, q in enumerate(population):
                if i == j:
                    continue
                
                p_id = id(p)
                q_id = id(q)
                
                if self._dominates(p, q):
                    dominated_solutions[p_id].append(q)
                elif self._dominates(q, p):
                    domination_counts[p_id] += 1
            
            # If not dominated by anyone, belongs to first front
            if domination_counts[id(p)] == 0:
                p['rank'] = 0
                fronts[0].append(p)
        
        # Generate subsequent fronts
        front_index = 0
        while fronts[front_index]:
            next_front = []
            
            for p in fronts[front_index]:
                p_id = id(p)
                
                for q in dominated_solutions[p_id]:
                    q_id = id(q)
                    domination_counts[q_id] -= 1
                    
                    if domination_counts[q_id] == 0:
                        q['rank'] = front_index + 1
                        next_front.append(q)
            
            front_index += 1
            if next_front:
                fronts.append(next_front)
            else:
                break
        
        return fronts
    
    def _dominates(self, solution1: Dict[str, Any], solution2: Dict[str, Any]) -> bool:
        """Check if solution1 dominates solution2."""
        objectives1 = solution1['objectives']
        objectives2 = solution2['objectives']
        
        if len(objectives1) != len(objectives2):
            return False
        
        # At least one objective is better, and none are worse
        at_least_one_better = False
        
        for i in range(len(objectives1)):
            if objectives1[i] > objectives2[i]:  # Assuming maximization
                at_least_one_better = True
            elif objectives1[i] < objectives2[i]:
                return False  # solution1 is worse in this objective
        
        return at_least_one_better
    
    def _update_pareto_fronts(self):
        """Update Pareto fronts for current population."""
        self.pareto_fronts = self._non_dominated_sorting(self.population)
    
    def _calculate_crowding_distances(self):
        """Calculate crowding distances for all fronts."""
        for front in self.pareto_fronts:
            self._calculate_crowding_distance_for_front(front)
    
    def _calculate_crowding_distance_for_front(self, front: List[Dict[str, Any]]):
        """Calculate crowding distance for solutions in a front."""
        if len(front) <= 2:
            for solution in front:
                solution['crowding_distance'] = float('inf')
            return
        
        # Initialize distances to 0
        for solution in front:
            solution['crowding_distance'] = 0.0
        
        num_objectives = len(front[0]['objectives'])
        
        for obj_index in range(num_objectives):
            # Sort by objective value
            front.sort(key=lambda x: x['objectives'][obj_index])
            
            # Set boundary solutions to infinite distance
            front[0]['crowding_distance'] = float('inf')
            front[-1]['crowding_distance'] = float('inf')
            
            # Calculate range
            obj_min = front[0]['objectives'][obj_index]
            obj_max = front[-1]['objectives'][obj_index]
            obj_range = obj_max - obj_min
            
            if obj_range == 0:
                continue
            
            # Calculate distances for intermediate solutions
            for i in range(1, len(front) - 1):
                if front[i]['crowding_distance'] != float('inf'):
                    distance = (front[i + 1]['objectives'][obj_index] - 
                               front[i - 1]['objectives'][obj_index]) / obj_range
                    front[i]['crowding_distance'] += distance
    
    def _extract_pareto_front(self) -> List[Dict[str, Any]]:
        """Extract the first Pareto front (best solutions)."""
        if self.pareto_fronts:
            return self.pareto_fronts[0].copy()
        return []
    
    def _calculate_generation_stats(self, generation: int) -> Dict[str, Any]:
        """Calculate statistics for current generation."""
        if not self.pareto_fronts:
            return {'generation': generation, 'num_fronts': 0}
        
        # Calculate hypervolume (simplified)
        pareto_front = self.pareto_fronts[0]
        if pareto_front:
            objectives_matrix = np.array([sol['objectives'] for sol in pareto_front])
            
            # Simple statistics
            mean_objectives = np.mean(objectives_matrix, axis=0).tolist()
            std_objectives = np.std(objectives_matrix, axis=0).tolist()
            
            return {
                'generation': generation,
                'num_fronts': len(self.pareto_fronts),
                'pareto_front_size': len(pareto_front),
                'mean_objectives': mean_objectives,
                'std_objectives': std_objectives,
                'total_population': len(self.population)
            }
        
        return {
            'generation': generation,
            'num_fronts': len(self.pareto_fronts),
            'pareto_front_size': 0
        }
    
    def _check_convergence(self, generation_history: List[Dict[str, Any]]) -> bool:
        """Check if algorithm has converged."""
        if len(generation_history) < 10:
            return False
        
        # Simple convergence check: Pareto front size stability
        recent_front_sizes = [gen['pareto_front_size'] for gen in generation_history[-5:]]
        
        if len(set(recent_front_sizes)) == 1 and recent_front_sizes[0] > 0:
            # Front size has been stable for 5 generations
            return True
        
        return False
    
    def _solution_to_dict(self, solution: Dict[str, Any]) -> Dict[str, Any]:
        """Convert solution to serializable dictionary."""
        return {
            'parameters': solution.get('parameters', {}),
            'objectives': solution.get('objectives', []),
            'rank': solution.get('rank', 0),
            'crowding_distance': solution.get('crowding_distance', 0.0)
        }
    
    def stop_optimization(self):
        """Stop optimization early."""
        self.is_running = False
        logger.info("NSGA-II optimization stopped by user request")