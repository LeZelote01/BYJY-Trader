"""
🎯 Strategy Selector Module
Intelligent selection of optimal trading strategies
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


class StrategySelector:
    """
    Intelligent strategy selection based on market conditions
    Phase 3.4 - Adaptive Strategies Component
    """
    
    def __init__(self, 
                 selection_window: int = 30,
                 min_performance_threshold: float = 0.01):
        """
        Initialize Strategy Selector
        
        Args:
            selection_window: Window for performance evaluation
            min_performance_threshold: Minimum performance threshold
        """
        self.selection_window = selection_window
        self.min_performance_threshold = min_performance_threshold
        self.strategy_performance = {}
        self.selection_history = []
        
        logger.info("StrategySelector initialized")
    
    def select_optimal_strategy(self,
                               market_conditions: Dict[str, Any],
                               available_strategies: List[str],
                               strategy_configs: Dict[str, Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Select optimal strategy for current market conditions
        
        Args:
            market_conditions: Current market conditions
            available_strategies: List of available strategies
            strategy_configs: Configuration for each strategy
            
        Returns:
            Strategy selection results
        """
        selection_result = {
            'selected_strategy': None,
            'confidence': 0.0,
            'reasoning': [],
            'alternative_strategies': [],
            'performance_forecast': 0.0
        }
        
        if not available_strategies:
            logger.warning("No strategies available for selection")
            return selection_result
        
        try:
            # Evaluate each strategy for current conditions
            strategy_scores = self._evaluate_strategies(market_conditions, available_strategies)
            
            if not strategy_scores:
                # Fallback to first available strategy
                selection_result['selected_strategy'] = available_strategies[0]
                selection_result['reasoning'].append("Fallback selection - insufficient data")
                return selection_result
            
            # Sort strategies by score
            sorted_strategies = sorted(strategy_scores.items(), key=lambda x: x[1]['score'], reverse=True)
            
            best_strategy, best_score = sorted_strategies[0]
            
            selection_result.update({
                'selected_strategy': best_strategy,
                'confidence': best_score['confidence'],
                'reasoning': best_score.get('reasons', []),
                'alternative_strategies': [s[0] for s in sorted_strategies[1:3]],  # Top 2 alternatives
                'performance_forecast': best_score.get('forecast', 0.0)
            })
            
            # Record selection
            self.selection_history.append({
                'strategy': best_strategy,
                'market_conditions': market_conditions,
                'confidence': best_score['confidence'],
                'timestamp': len(self.selection_history)
            })
            
            logger.info(f"Selected strategy: {best_strategy} (confidence: {best_score['confidence']:.3f})")
            
        except Exception as e:
            logger.error(f"Strategy selection failed: {e}")
            selection_result['error'] = str(e)
            # Fallback
            if available_strategies:
                selection_result['selected_strategy'] = available_strategies[0]
        
        return selection_result
    
    def update_strategy_performance(self,
                                   strategy_name: str,
                                   performance_metrics: Dict[str, float],
                                   market_conditions: Dict[str, Any]):
        """Update performance data for strategy"""
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = []
        
        performance_entry = {
            'metrics': performance_metrics,
            'market_conditions': market_conditions,
            'timestamp': len(self.strategy_performance[strategy_name])
        }
        
        self.strategy_performance[strategy_name].append(performance_entry)
        
        # Keep only recent performance data
        if len(self.strategy_performance[strategy_name]) > 100:
            self.strategy_performance[strategy_name] = self.strategy_performance[strategy_name][-100:]
    
    def get_strategy_recommendations(self,
                                   market_forecast: Dict[str, Any],
                                   risk_tolerance: float = 0.5) -> List[Dict[str, Any]]:
        """
        Get strategy recommendations based on market forecast
        
        Args:
            market_forecast: Predicted market conditions
            risk_tolerance: Risk tolerance level (0.0 to 1.0)
            
        Returns:
            List of strategy recommendations
        """
        recommendations = []
        
        try:
            for strategy_name, performance_history in self.strategy_performance.items():
                if len(performance_history) < 5:  # Minimum data required
                    continue
                
                # Calculate strategy characteristics
                returns = [p['metrics'].get('return', 0) for p in performance_history[-self.selection_window:]]
                volatility = np.std(returns) if len(returns) > 1 else 0.0
                avg_return = np.mean(returns) if returns else 0.0
                
                # Risk-adjusted score
                sharpe_ratio = avg_return / max(volatility, 0.001)
                
                # Adjust for risk tolerance
                risk_adjusted_score = avg_return - (1 - risk_tolerance) * volatility
                
                # Market condition suitability
                suitability_score = self._calculate_market_suitability(
                    strategy_name, market_forecast, performance_history
                )
                
                # Combined recommendation score
                overall_score = (risk_adjusted_score + suitability_score) / 2
                
                recommendations.append({
                    'strategy': strategy_name,
                    'overall_score': float(overall_score),
                    'expected_return': float(avg_return),
                    'expected_volatility': float(volatility),
                    'sharpe_ratio': float(sharpe_ratio),
                    'market_suitability': float(suitability_score),
                    'recommendation': self._get_recommendation_level(overall_score)
                })
            
            # Sort by overall score
            recommendations.sort(key=lambda x: x['overall_score'], reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
        
        return recommendations
    
    def analyze_strategy_switching_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in strategy switching"""
        if len(self.selection_history) < 10:
            return {'insufficient_data': True}
        
        switching_analysis = {
            'total_switches': 0,
            'switch_frequency': 0.0,
            'most_stable_periods': [],
            'common_switches': {},
            'performance_after_switch': []
        }
        
        try:
            # Count switches
            switches = []
            for i in range(1, len(self.selection_history)):
                prev_strategy = self.selection_history[i-1]['strategy']
                curr_strategy = self.selection_history[i]['strategy']
                
                if prev_strategy != curr_strategy:
                    switches.append({
                        'from': prev_strategy,
                        'to': curr_strategy,
                        'timestamp': i
                    })
            
            switching_analysis['total_switches'] = len(switches)
            switching_analysis['switch_frequency'] = len(switches) / len(self.selection_history)
            
            # Analyze common switching patterns
            switch_patterns = {}
            for switch in switches:
                pattern = f"{switch['from']} -> {switch['to']}"
                switch_patterns[pattern] = switch_patterns.get(pattern, 0) + 1
            
            switching_analysis['common_switches'] = switch_patterns
            
            logger.info(f"Strategy switching analysis completed: {len(switches)} switches detected")
            
        except Exception as e:
            logger.error(f"Strategy switching analysis failed: {e}")
            switching_analysis['error'] = str(e)
        
        return switching_analysis
    
    def _evaluate_strategies(self,
                           market_conditions: Dict[str, Any],
                           available_strategies: List[str]) -> Dict[str, Dict[str, Any]]:
        """Evaluate strategies for current market conditions"""
        strategy_scores = {}
        
        for strategy_name in available_strategies:
            score_info = {
                'score': 0.5,  # Default neutral score
                'confidence': 0.5,
                'reasons': [],
                'forecast': 0.0
            }
            
            # Get historical performance
            if strategy_name in self.strategy_performance:
                performance_history = self.strategy_performance[strategy_name]
                
                if len(performance_history) >= 5:
                    # Calculate performance metrics
                    recent_performance = performance_history[-self.selection_window:]
                    
                    returns = [p['metrics'].get('return', 0) for p in recent_performance]
                    avg_return = np.mean(returns) if returns else 0.0
                    volatility = np.std(returns) if len(returns) > 1 else 0.0
                    
                    # Performance-based score
                    performance_score = max(0.0, min(1.0, (avg_return + 0.1) / 0.2))  # Normalize to 0-1
                    
                    # Market condition matching
                    condition_score = self._calculate_market_suitability(
                        strategy_name, market_conditions, performance_history
                    )
                    
                    # Combined score
                    combined_score = (performance_score * 0.6) + (condition_score * 0.4)
                    
                    score_info.update({
                        'score': combined_score,
                        'confidence': min(1.0, len(recent_performance) / self.selection_window),
                        'reasons': [
                            f"Average return: {avg_return:.3f}",
                            f"Volatility: {volatility:.3f}",
                            f"Market suitability: {condition_score:.3f}"
                        ],
                        'forecast': avg_return
                    })
            
            strategy_scores[strategy_name] = score_info
        
        return strategy_scores
    
    def _calculate_market_suitability(self,
                                    strategy_name: str,
                                    current_conditions: Dict[str, Any],
                                    performance_history: List[Dict[str, Any]]) -> float:
        """Calculate how suitable strategy is for current market conditions"""
        if not performance_history:
            return 0.5
        
        # Find similar market conditions in history
        similar_conditions = []
        
        current_volatility = current_conditions.get('volatility', 0.1)
        current_trend = current_conditions.get('trend_direction', 'sideways')
        
        for entry in performance_history[-self.selection_window:]:
            hist_conditions = entry['market_conditions']
            hist_volatility = hist_conditions.get('volatility', 0.1)
            hist_trend = hist_conditions.get('trend_direction', 'sideways')
            
            # Calculate similarity
            vol_similarity = 1.0 - abs(current_volatility - hist_volatility) / max(current_volatility, hist_volatility, 0.001)
            trend_similarity = 1.0 if current_trend == hist_trend else 0.5
            
            overall_similarity = (vol_similarity + trend_similarity) / 2
            
            if overall_similarity > 0.7:  # Threshold for similarity
                performance = entry['metrics'].get('return', 0.0)
                similar_conditions.append(performance)
        
        if similar_conditions:
            # Average performance in similar conditions
            avg_performance = np.mean(similar_conditions)
            # Normalize to 0-1 scale
            return max(0.0, min(1.0, (avg_performance + 0.1) / 0.2))
        
        return 0.5  # Neutral score if no similar conditions found
    
    def _get_recommendation_level(self, score: float) -> str:
        """Get recommendation level based on score"""
        if score >= 0.8:
            return "Highly Recommended"
        elif score >= 0.6:
            return "Recommended"
        elif score >= 0.4:
            return "Neutral"
        elif score >= 0.2:
            return "Not Recommended"
        else:
            return "Strongly Not Recommended"
    
    def get_selection_statistics(self) -> Dict[str, Any]:
        """Get statistics about strategy selection"""
        if not self.selection_history:
            return {'no_data': True}
        
        # Strategy usage frequency
        strategy_counts = {}
        for selection in self.selection_history:
            strategy = selection['strategy']
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        # Average confidence
        confidences = [s['confidence'] for s in self.selection_history if 'confidence' in s]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return {
            'total_selections': len(self.selection_history),
            'strategy_usage': strategy_counts,
            'average_confidence': float(avg_confidence),
            'strategies_tracked': len(self.strategy_performance)
        }