"""
🧬 BYJY-Trader Optimisation Module
Phase 3.4 - Genetic Algorithms, Multi-objective Optimization, Meta-Learning
"""

from .genetic import GeneticOptimizer, Chromosome, CrossoverOperator, MutationOperator, SelectionOperator, FitnessEvaluator
from .multi_objective import ParetoOptimizer, NSGA2, ObjectiveFunctions, ParetoFrontAnalyzer
from .meta_learning import MetaLearner, AdaptationEngine, PatternRecognizer, TransferLearner, FewShotLearner
from .hyperparameter import OptunaOptimizer, ParameterSpace, PruningStrategies, OptimizationHistory, ParallelOptimizer
from .adaptive import AdaptiveStrategyManager, MarketRegimeDetector, StrategySelector, DynamicRebalancer, PerformanceMonitor

__version__ = "3.4.0"
__all__ = [
    # Genetic Algorithm
    'GeneticOptimizer', 'Chromosome', 'CrossoverOperator', 'MutationOperator', 
    'SelectionOperator', 'FitnessEvaluator',
    
    # Multi-objective Optimization
    'ParetoOptimizer', 'NSGA2', 'ObjectiveFunctions', 'ParetoFrontAnalyzer',
    
    # Meta-learning
    'MetaLearner', 'AdaptationEngine', 'PatternRecognizer', 'TransferLearner', 'FewShotLearner',
    
    # Hyperparameter Optimization
    'OptunaOptimizer', 'ParameterSpace', 'PruningStrategies', 'OptimizationHistory', 'ParallelOptimizer',
    
    # Adaptive Strategies
    'AdaptiveStrategyManager', 'MarketRegimeDetector', 'StrategySelector', 'DynamicRebalancer', 'PerformanceMonitor'
]