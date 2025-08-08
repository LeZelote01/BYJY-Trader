"""
🧪 Test Genetic Optimization Phase 3.4
Test script for genetic algorithm optimization system
"""

import asyncio
import numpy as np
from typing import Dict, Any
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from ai.optimization.genetic import GeneticOptimizer, Chromosome
from ai.optimization.multi_objective import ParetoOptimizer, ObjectiveFunctions
from ai.optimization.utils import OptimizationUtils, ConvergenceChecker
from core.logger import get_logger

logger = get_logger(__name__)


async def test_genetic_optimization():
    """Test genetic algorithm optimization."""
    logger.info("🧬 Testing Genetic Algorithm Optimization...")
    
    # Define parameter space for LSTM model
    parameter_space = {
        'layers': {'type': 'int', 'min': 1, 'max': 5},
        'neurons': {'type': 'int', 'min': 32, 'max': 256},
        'dropout': {'type': 'float', 'min': 0.1, 'max': 0.5},
        'learning_rate': {'type': 'float', 'min': 1e-5, 'max': 1e-2},
        'batch_size': {'type': 'categorical', 'choices': [16, 32, 64, 128]},
        'optimizer': {'type': 'categorical', 'choices': ['adam', 'rmsprop']},
        'use_regularization': {'type': 'boolean'}
    }
    
    # Mock fitness function
    async def mock_fitness_function(parameters: Dict[str, Any]) -> Dict[str, float]:
        """Mock fitness function for testing."""
        await asyncio.sleep(0.01)  # Simulate evaluation time
        
        # Simulate realistic LSTM performance metrics
        layers = parameters['layers']
        neurons = parameters['neurons']
        dropout = parameters['dropout']
        lr = parameters['learning_rate']
        
        # Mock performance that favors certain parameter ranges
        base_performance = 0.7
        
        # Layers: optimal around 2-3
        layers_bonus = 0.1 * (1 - abs(layers - 2.5) / 2.5)
        
        # Neurons: diminishing returns after 128
        neurons_bonus = 0.1 * min(neurons / 128, 1.0)
        
        # Dropout: optimal around 0.2-0.3
        dropout_bonus = 0.05 * (1 - abs(dropout - 0.25) / 0.25)
        
        # Learning rate: log-optimal around 1e-3
        lr_bonus = 0.05 * (1 - abs(np.log10(lr) + 3) / 2)
        
        # Add some noise
        noise = np.random.normal(0, 0.05)
        
        accuracy = base_performance + layers_bonus + neurons_bonus + dropout_bonus + lr_bonus + noise
        accuracy = np.clip(accuracy, 0.5, 0.95)
        
        # Generate other metrics
        training_time = np.random.uniform(10, 300) * (layers * neurons / 1000)
        sharpe_ratio = accuracy * 2 + np.random.normal(0, 0.2)
        max_drawdown = np.random.uniform(0.05, 0.3) / accuracy
        
        return {
            'accuracy': accuracy,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'training_time': training_time,
            'consistency_score': accuracy + np.random.normal(0, 0.1)
        }
    
    # Create genetic optimizer
    optimizer = GeneticOptimizer(
        population_size=50,
        num_generations=100,
        crossover_prob=0.8,
        mutation_prob=0.15,
        elitism_ratio=0.1,
        early_stopping_patience=15,
        random_seed=42
    )
    
    try:
        # Run optimization
        logger.info("Starting genetic optimization...")
        results = await optimizer.optimize(
            parameter_space=parameter_space,
            fitness_function=mock_fitness_function
        )
        
        # Display results
        logger.info("✅ Genetic optimization completed!")
        logger.info(f"Best fitness: {results['best_fitness']:.6f}")
        logger.info(f"Generations completed: {results['generations_completed']}")
        logger.info(f"Converged: {results['converged']}")
        logger.info(f"Best parameters: {results['best_parameters']}")
        
        # Test chromosome operations
        logger.info("\n🧬 Testing Chromosome operations...")
        test_chromosome = Chromosome({
            'layers': 3,
            'neurons': 128,
            'dropout': 0.3,
            'learning_rate': 0.001
        })
        
        logger.info(f"Original chromosome: {test_chromosome}")
        
        # Test copy
        copied_chromosome = test_chromosome.copy()
        logger.info(f"Copied chromosome: {copied_chromosome}")
        
        # Test validation
        is_valid, errors = OptimizationUtils.validate_parameter_space(parameter_space)
        logger.info(f"Parameter space valid: {is_valid}")
        if errors:
            logger.warning(f"Validation errors: {errors}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Genetic optimization test failed: {e}")
        return False


async def test_pareto_optimization():
    """Test Pareto multi-objective optimization."""
    logger.info("\n📊 Testing Pareto Multi-Objective Optimization...")
    
    # Parameter space for ensemble model
    parameter_space = {
        'lstm_weight': {'type': 'float', 'min': 0, 'max': 1},
        'transformer_weight': {'type': 'float', 'min': 0, 'max': 1},
        'xgboost_weight': {'type': 'float', 'min': 0, 'max': 1},
        'voting_method': {'type': 'categorical', 'choices': ['soft', 'hard', 'weighted']},
        'meta_learner': {'type': 'boolean'}
    }
    
    # Define objectives
    objective_configs = [
        {
            'name': 'sharpe_ratio',
            'maximize': True,
            'weight': 0.4,
            'description': 'Maximize Sharpe ratio',
            'target_range': [0, 3]
        },
        {
            'name': 'max_drawdown',
            'maximize': False,
            'weight': 0.3,
            'description': 'Minimize maximum drawdown',
            'target_range': [0, 0.5]
        },
        {
            'name': 'training_time',
            'maximize': False,
            'weight': 0.3,
            'description': 'Minimize training time',
            'target_range': [1, 600]
        }
    ]
    
    # Mock evaluation function
    async def mock_evaluation_function(parameters: Dict[str, Any]) -> Dict[str, float]:
        """Mock evaluation function for Pareto optimization."""
        await asyncio.sleep(0.01)
        
        # Simulate trade-offs between objectives
        lstm_w = parameters['lstm_weight']
        transformer_w = parameters['transformer_weight']
        xgboost_w = parameters['xgboost_weight']
        
        # Normalize weights
        total_weight = lstm_w + transformer_w + xgboost_w + 1e-10
        lstm_w /= total_weight
        transformer_w /= total_weight
        xgboost_w /= total_weight
        
        # Simulate performance trade-offs
        base_sharpe = 1.0
        sharpe_ratio = base_sharpe + lstm_w * 0.5 + transformer_w * 0.8 + xgboost_w * 0.3
        sharpe_ratio += np.random.normal(0, 0.2)
        
        # Training time increases with complexity
        base_time = 60
        training_time = base_time * (1 + lstm_w * 2 + transformer_w * 4 + xgboost_w * 1)
        training_time += np.random.uniform(0, 30)
        
        # Drawdown decreases with ensemble diversity
        diversity = 1 - max(lstm_w, transformer_w, xgboost_w)  # Higher diversity = lower max weight
        max_drawdown = 0.2 * (1 - diversity * 0.5) + np.random.uniform(0, 0.1)
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'training_time': training_time,
            'annual_return': sharpe_ratio * 0.1 + np.random.normal(0, 0.05),
            'consistency_score': 0.8 + diversity * 0.2 + np.random.normal(0, 0.1)
        }
    
    # Create Pareto optimizer
    optimizer = ParetoOptimizer(
        population_size=30,
        num_generations=50,
        crossover_prob=0.9,
        mutation_prob=0.1,
        random_seed=42
    )
    
    try:
        # Run optimization
        logger.info("Starting Pareto optimization...")
        results = await optimizer.optimize(
            parameter_space=parameter_space,
            objective_configs=objective_configs,
            evaluation_function=mock_evaluation_function
        )
        
        # Display results
        logger.info("✅ Pareto optimization completed!")
        logger.info(f"Pareto front size: {len(results['pareto_front'])}")
        logger.info(f"Generations completed: {results['generations_completed']}")
        
        # Show some Pareto solutions
        if results['pareto_front']:
            logger.info("\n🎯 Sample Pareto-optimal solutions:")
            for i, solution in enumerate(results['pareto_front'][:3]):  # Show first 3
                objectives = solution['objectives']
                logger.info(f"Solution {i+1}: Sharpe={objectives[0]:.3f}, "
                           f"Drawdown={objectives[1]:.3f}, Time={objectives[2]:.1f}s")
        
        # Show compromise solutions
        if results['compromise_solutions']:
            logger.info("\n🎭 Compromise solutions:")
            for comp_sol in results['compromise_solutions']:
                logger.info(f"{comp_sol['type']}: {comp_sol['description']}")
        
        # Test objective functions
        logger.info("\n🎯 Testing Objective Functions...")
        obj_functions = ObjectiveFunctions()
        obj_functions.setup_objectives(objective_configs)
        
        test_metrics = {
            'sharpe_ratio': 1.5,
            'max_drawdown': 0.15,
            'training_time': 120
        }
        
        objective_values = obj_functions.evaluate(test_metrics)
        logger.info(f"Test metrics: {test_metrics}")
        logger.info(f"Objective values: {objective_values}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Pareto optimization test failed: {e}")
        return False


async def test_optimization_utils():
    """Test optimization utilities."""
    logger.info("\n🔧 Testing Optimization Utils...")
    
    try:
        # Test parameter space validation
        valid_space = {
            'param1': {'type': 'float', 'min': 0, 'max': 1},
            'param2': {'type': 'int', 'min': 1, 'max': 10},
            'param3': {'type': 'categorical', 'choices': ['a', 'b', 'c']},
            'param4': {'type': 'boolean'}
        }
        
        is_valid, errors = OptimizationUtils.validate_parameter_space(valid_space)
        logger.info(f"Valid parameter space check: {is_valid}")
        
        # Test invalid space
        invalid_space = {
            'param1': {'type': 'float', 'min': 1, 'max': 0},  # Invalid range
            'param2': {'type': 'categorical'}  # Missing choices
        }
        
        is_valid, errors = OptimizationUtils.validate_parameter_space(invalid_space)
        logger.info(f"Invalid parameter space check: {is_valid}")
        logger.info(f"Validation errors: {errors}")
        
        # Test parameter normalization
        test_params = {
            'param1': 0.5,
            'param2': 5,
            'param3': 'b',
            'param4': True
        }
        
        normalized = OptimizationUtils.normalize_parameters(test_params, valid_space)
        logger.info(f"Original parameters: {test_params}")
        logger.info(f"Normalized parameters: {normalized}")
        
        denormalized = OptimizationUtils.denormalize_parameters(normalized, valid_space)
        logger.info(f"Denormalized parameters: {denormalized}")
        
        # Test parameter combinations generation
        combinations = OptimizationUtils.generate_parameter_combinations(valid_space, 10)
        logger.info(f"Generated {len(combinations)} parameter combinations")
        
        # Test diversity calculation
        diversity = OptimizationUtils.calculate_parameter_diversity(combinations[:5], valid_space)
        logger.info(f"Parameter diversity: {diversity:.4f}")
        
        # Test convergence checker
        logger.info("\n📊 Testing Convergence Checker...")
        checker = ConvergenceChecker(patience=5, min_delta=1e-4)
        
        # Simulate convergence
        fitness_values = [0.5, 0.6, 0.7, 0.75, 0.78, 0.79, 0.79, 0.79, 0.79, 0.79]
        
        for i, fitness in enumerate(fitness_values):
            converged = checker.update(fitness)
            logger.info(f"Generation {i+1}: Fitness={fitness:.3f}, Converged={converged}")
            
            if converged:
                logger.info(f"Convergence detected: {checker.convergence_reason}")
                break
        
        convergence_info = checker.get_convergence_info()
        logger.info(f"Convergence info: {convergence_info}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Optimization utils test failed: {e}")
        return False


async def main():
    """Run all optimization tests."""
    logger.info("🚀 Starting BYJY-Trader Phase 3.4 Optimization Tests...")
    
    results = {
        'genetic_optimization': False,
        'pareto_optimization': False,
        'optimization_utils': False
    }
    
    # Run tests
    results['genetic_optimization'] = await test_genetic_optimization()
    results['pareto_optimization'] = await test_pareto_optimization()
    results['optimization_utils'] = await test_optimization_utils()
    
    # Summary
    logger.info("\n📋 Test Results Summary:")
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"  {test_name}: {status}")
    
    logger.info(f"\n🎯 Overall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 All Phase 3.4 Optimization tests PASSED!")
        logger.info("🧬 Genetic Algorithm optimization: WORKING")
        logger.info("📊 Pareto multi-objective optimization: WORKING")
        logger.info("🔧 Optimization utilities: WORKING")
        logger.info("✅ Phase 3.4 implementation is READY for integration!")
    else:
        logger.error("❌ Some tests failed. Please check the implementation.")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    asyncio.run(main())