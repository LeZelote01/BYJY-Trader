"""
🔄 Adaptive Strategy Manager Module
Main manager for adaptive trading strategies
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class AdaptiveStrategyManager:
    """
    Main manager for adaptive trading strategies
    Phase 3.4 - Adaptive Strategies Component
    """
    
    def __init__(self, 
                 adaptation_threshold: float = 0.1,
                 min_performance_samples: int = 10):
        """
        Initialize Adaptive Strategy Manager
        
        Args:
            adaptation_threshold: Threshold for strategy adaptation
            min_performance_samples: Minimum samples for performance evaluation
        """
        self.adaptation_threshold = adaptation_threshold
        self.min_performance_samples = min_performance_samples
        self.active_strategies = {}
        self.performance_history = {}
        
        logger.info("AdaptiveStrategyManager initialized")
    
    def manage_strategies(self,
                         market_data: Dict[str, Any],
                         available_strategies: List[str]) -> Dict[str, Any]:
        """
        Manage and adapt trading strategies
        
        Args:
            market_data: Current market data
            available_strategies: List of available strategies
            
        Returns:
            Strategy management results
        """
        results = {
            'current_strategy': None,
            'adaptation_applied': False,
            'performance_metrics': {},
            'recommendations': []
        }
        
        try:
            # Evaluate current strategy performance
            current_performance = self._evaluate_current_performance()
            
            # Check if adaptation is needed
            if self._should_adapt_strategy(current_performance):
                # Select best strategy for current conditions
                best_strategy = self._select_optimal_strategy(market_data, available_strategies)
                
                if best_strategy:
                    results['current_strategy'] = best_strategy
                    results['adaptation_applied'] = True
                    
                    logger.info(f"Strategy adapted to: {best_strategy}")
            
            results['performance_metrics'] = current_performance
            
        except Exception as e:
            logger.error(f"Strategy management failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def add_strategy_performance(self, strategy_name: str, performance_data: Dict[str, float]):
        """Add performance data for strategy"""
        if strategy_name not in self.performance_history:
            self.performance_history[strategy_name] = []
        
        self.performance_history[strategy_name].append(performance_data)
        
        # Keep only recent performance data
        if len(self.performance_history[strategy_name]) > 100:
            self.performance_history[strategy_name] = self.performance_history[strategy_name][-100:]
    
    def get_strategy_rankings(self) -> List[Dict[str, Any]]:
        """Get current strategy rankings"""
        rankings = []
        
        for strategy_name, history in self.performance_history.items():
            if len(history) >= self.min_performance_samples:
                avg_performance = np.mean([h.get('return', 0) for h in history])
                risk_score = np.std([h.get('return', 0) for h in history])
                
                rankings.append({
                    'strategy': strategy_name,
                    'avg_performance': float(avg_performance),
                    'risk_score': float(risk_score),
                    'sharpe_ratio': float(avg_performance / max(risk_score, 0.001)),
                    'sample_size': len(history)
                })
        
        # Sort by Sharpe ratio
        rankings.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
        
        return rankings
    
    def _evaluate_current_performance(self) -> Dict[str, float]:
        """Evaluate current strategy performance"""
        # Placeholder implementation
        return {
            'return': np.random.uniform(-0.05, 0.05),
            'volatility': np.random.uniform(0.01, 0.03),
            'sharpe_ratio': np.random.uniform(0.5, 2.0),
            'max_drawdown': np.random.uniform(0.01, 0.10)
        }
    
    def _should_adapt_strategy(self, current_performance: Dict[str, float]) -> bool:
        """Check if strategy adaptation is needed"""
        # Simple adaptation logic based on performance decline
        return current_performance.get('return', 0) < -self.adaptation_threshold
    
    def _select_optimal_strategy(self, market_data: Dict[str, Any], available_strategies: List[str]) -> Optional[str]:
        """Select optimal strategy for current market conditions"""
        if not available_strategies:
            return None
        
        # Get strategy rankings
        rankings = self.get_strategy_rankings()
        
        # Return best performing strategy
        if rankings:
            return rankings[0]['strategy']
        
        # Fallback to first available strategy
        return available_strategies[0] if available_strategies else None