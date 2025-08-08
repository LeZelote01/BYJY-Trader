"""
📊 Multi-Objective Optimization Module  
NSGA-II and Pareto optimization for profit/risk trade-offs
"""

from .pareto_optimizer import ParetoOptimizer
from .nsga2 import NSGA2
from .objective_functions import ObjectiveFunctions
from .pareto_front_analyzer import ParetoFrontAnalyzer

__all__ = [
    'ParetoOptimizer',
    'NSGA2',
    'ObjectiveFunctions', 
    'ParetoFrontAnalyzer'
]