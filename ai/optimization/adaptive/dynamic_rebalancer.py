"""
⚖️ Dynamic Rebalancer Module
Dynamic portfolio rebalancing based on market conditions
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DynamicRebalancer:
    """
    Dynamic portfolio rebalancing system
    Phase 3.4 - Adaptive Strategies Component
    """
    
    def __init__(self, 
                 rebalance_threshold: float = 0.05,
                 max_rebalance_frequency: int = 24,  # hours
                 risk_target: float = 0.15):
        """
        Initialize Dynamic Rebalancer
        
        Args:
            rebalance_threshold: Threshold for triggering rebalance
            max_rebalance_frequency: Maximum rebalance frequency in hours
            risk_target: Target portfolio risk level
        """
        self.rebalance_threshold = rebalance_threshold
        self.max_rebalance_frequency = max_rebalance_frequency
        self.risk_target = risk_target
        self.last_rebalance = None
        self.rebalance_history = []
        self.current_allocations = {}
        self.target_allocations = {}
        
        logger.info("DynamicRebalancer initialized")
    
    def check_rebalance_need(self,
                           current_portfolio: Dict[str, float],
                           market_conditions: Dict[str, Any],
                           performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Check if portfolio rebalancing is needed
        
        Args:
            current_portfolio: Current portfolio allocations
            market_conditions: Current market conditions
            performance_metrics: Recent performance metrics
            
        Returns:
            Rebalance recommendation
        """
        rebalance_info = {
            'rebalance_needed': False,
            'reason': None,
            'urgency': 'low',
            'recommended_actions': [],
            'risk_level': 'normal'
        }
        
        try:
            self.current_allocations = current_portfolio.copy()
            
            # Check if enough time has passed since last rebalance
            if not self._can_rebalance():
                rebalance_info['reason'] = 'Rebalance frequency limit reached'
                return rebalance_info
            
            # Calculate optimal allocations for current conditions
            optimal_allocations = self._calculate_optimal_allocations(
                market_conditions, performance_metrics
            )
            
            # Check deviation from optimal allocations
            max_deviation = self._calculate_allocation_deviation(
                current_portfolio, optimal_allocations
            )
            
            # Check risk level
            current_risk = self._estimate_portfolio_risk(current_portfolio, market_conditions)
            
            # Determine if rebalancing is needed
            if max_deviation > self.rebalance_threshold:
                rebalance_info.update({
                    'rebalance_needed': True,
                    'reason': f'Allocation deviation: {max_deviation:.3f}',
                    'urgency': self._determine_urgency(max_deviation),
                    'recommended_actions': self._generate_rebalance_actions(
                        current_portfolio, optimal_allocations
                    )
                })
            
            elif abs(current_risk - self.risk_target) > 0.05:
                rebalance_info.update({
                    'rebalance_needed': True,
                    'reason': f'Risk deviation: current={current_risk:.3f}, target={self.risk_target:.3f}',
                    'urgency': 'medium',
                    'recommended_actions': self._generate_risk_adjustment_actions(
                        current_portfolio, current_risk
                    )
                })
            
            rebalance_info['risk_level'] = self._categorize_risk_level(current_risk)
            self.target_allocations = optimal_allocations
            
            logger.info(f"Rebalance check: needed={rebalance_info['rebalance_needed']}, "
                       f"deviation={max_deviation:.3f}, risk={current_risk:.3f}")
            
        except Exception as e:
            logger.error(f"Rebalance check failed: {e}")
            rebalance_info['error'] = str(e)
        
        return rebalance_info
    
    def execute_rebalance(self,
                         current_portfolio: Dict[str, float],
                         rebalance_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute portfolio rebalancing
        
        Args:
            current_portfolio: Current portfolio allocations
            rebalance_actions: Rebalancing actions to execute
            
        Returns:
            Rebalancing results
        """
        rebalance_result = {
            'success': False,
            'new_allocations': {},
            'executed_actions': [],
            'performance_impact': 0.0,
            'execution_cost': 0.0
        }
        
        try:
            new_allocations = current_portfolio.copy()
            total_cost = 0.0
            executed_actions = []
            
            for action in rebalance_actions:
                # Simulate action execution
                execution_result = self._execute_single_action(new_allocations, action)
                
                if execution_result['success']:
                    new_allocations.update(execution_result['new_allocations'])
                    total_cost += execution_result.get('cost', 0.0)
                    executed_actions.append(action)
            
            # Update rebalance history
            self.rebalance_history.append({
                'timestamp': datetime.now(),
                'old_allocations': current_portfolio,
                'new_allocations': new_allocations,
                'actions': executed_actions,
                'cost': total_cost
            })
            
            self.last_rebalance = datetime.now()
            
            rebalance_result.update({
                'success': True,
                'new_allocations': new_allocations,
                'executed_actions': executed_actions,
                'execution_cost': total_cost
            })
            
            logger.info(f"Rebalance executed: {len(executed_actions)} actions, cost: {total_cost:.4f}")
            
        except Exception as e:
            logger.error(f"Rebalance execution failed: {e}")
            rebalance_result['error'] = str(e)
        
        return rebalance_result
    
    def optimize_allocations(self,
                           available_assets: List[str],
                           expected_returns: Dict[str, float],
                           risk_estimates: Dict[str, float],
                           correlations: Dict[str, Dict[str, float]] = None) -> Dict[str, float]:
        """
        Optimize portfolio allocations using modern portfolio theory
        
        Args:
            available_assets: List of available assets
            expected_returns: Expected returns for each asset
            risk_estimates: Risk estimates for each asset
            correlations: Asset correlation matrix
            
        Returns:
            Optimal allocations
        """
        if not available_assets:
            return {}
        
        try:
            # Simple equal-weight allocation as baseline
            equal_weight = 1.0 / len(available_assets)
            optimal_allocations = {asset: equal_weight for asset in available_assets}
            
            # Risk-based adjustment
            total_risk_score = sum(1.0 / max(risk_estimates.get(asset, 0.1), 0.01) for asset in available_assets)
            
            for asset in available_assets:
                risk = risk_estimates.get(asset, 0.1)
                expected_return = expected_returns.get(asset, 0.0)
                
                # Inverse risk weighting with return adjustment
                risk_weight = (1.0 / max(risk, 0.01)) / total_risk_score
                return_adjustment = 1.0 + expected_return  # Simple return-based adjustment
                
                optimal_allocations[asset] = risk_weight * return_adjustment
            
            # Normalize to sum to 1.0
            total_weight = sum(optimal_allocations.values())
            if total_weight > 0:
                optimal_allocations = {
                    asset: weight / total_weight 
                    for asset, weight in optimal_allocations.items()
                }
            
            logger.info(f"Optimized allocations for {len(available_assets)} assets")
            return optimal_allocations
            
        except Exception as e:
            logger.error(f"Allocation optimization failed: {e}")
            # Fallback to equal weights
            return {asset: 1.0 / len(available_assets) for asset in available_assets}
    
    def get_rebalance_performance(self) -> Dict[str, Any]:
        """Get performance metrics of rebalancing strategy"""
        if len(self.rebalance_history) < 2:
            return {'insufficient_data': True}
        
        performance_metrics = {
            'total_rebalances': len(self.rebalance_history),
            'average_cost': 0.0,
            'cost_efficiency': 0.0,
            'frequency': 0.0,
            'recent_performance': []
        }
        
        try:
            # Calculate average cost
            costs = [r.get('cost', 0.0) for r in self.rebalance_history]
            performance_metrics['average_cost'] = float(np.mean(costs))
            
            # Calculate frequency (rebalances per day)
            if len(self.rebalance_history) >= 2:
                time_span = (
                    self.rebalance_history[-1]['timestamp'] - 
                    self.rebalance_history[0]['timestamp']
                ).total_seconds() / (24 * 3600)  # days
                
                if time_span > 0:
                    performance_metrics['frequency'] = len(self.rebalance_history) / time_span
            
            logger.info("Rebalance performance analysis completed")
            
        except Exception as e:
            logger.error(f"Rebalance performance analysis failed: {e}")
            performance_metrics['error'] = str(e)
        
        return performance_metrics
    
    def _can_rebalance(self) -> bool:
        """Check if rebalancing is allowed based on frequency limits"""
        if self.last_rebalance is None:
            return True
        
        time_since_last = datetime.now() - self.last_rebalance
        hours_since_last = time_since_last.total_seconds() / 3600
        
        return hours_since_last >= self.max_rebalance_frequency
    
    def _calculate_optimal_allocations(self,
                                     market_conditions: Dict[str, Any],
                                     performance_metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate optimal allocations for current conditions"""
        # Simple heuristic-based allocation
        
        # Default equal allocation
        if not self.current_allocations:
            return {}
        
        assets = list(self.current_allocations.keys())
        optimal = {}
        
        # Market regime-based adjustments
        volatility = market_conditions.get('volatility', 0.1)
        trend = market_conditions.get('trend_direction', 'sideways')
        
        for asset in assets:
            base_allocation = 1.0 / len(assets)  # Equal weight baseline
            
            # Volatility adjustment
            if volatility > 0.2:  # High volatility - reduce risk
                base_allocation *= 0.8
            elif volatility < 0.05:  # Low volatility - increase allocation
                base_allocation *= 1.1
            
            # Trend adjustment (simplified)
            if trend == 'bullish':
                base_allocation *= 1.05  # Slightly increase in bull market
            elif trend == 'bearish':
                base_allocation *= 0.9   # Reduce in bear market
            
            optimal[asset] = base_allocation
        
        # Normalize
        total = sum(optimal.values())
        if total > 0:
            optimal = {asset: weight / total for asset, weight in optimal.items()}
        
        return optimal
    
    def _calculate_allocation_deviation(self,
                                      current: Dict[str, float],
                                      optimal: Dict[str, float]) -> float:
        """Calculate maximum deviation between current and optimal allocations"""
        if not current or not optimal:
            return 0.0
        
        max_deviation = 0.0
        
        for asset in set(current.keys()) | set(optimal.keys()):
            current_weight = current.get(asset, 0.0)
            optimal_weight = optimal.get(asset, 0.0)
            deviation = abs(current_weight - optimal_weight)
            max_deviation = max(max_deviation, deviation)
        
        return max_deviation
    
    def _estimate_portfolio_risk(self,
                               portfolio: Dict[str, float],
                               market_conditions: Dict[str, Any]) -> float:
        """Estimate portfolio risk level"""
        if not portfolio:
            return 0.0
        
        # Simple risk estimation based on market conditions
        base_risk = market_conditions.get('volatility', 0.1)
        
        # Diversification adjustment
        n_assets = len(portfolio)
        diversification_factor = max(0.5, 1.0 - (n_assets - 1) * 0.1)
        
        # Concentration risk
        max_weight = max(portfolio.values()) if portfolio else 0.0
        concentration_factor = 1.0 + max(0.0, max_weight - 0.5)
        
        portfolio_risk = base_risk * diversification_factor * concentration_factor
        
        return min(1.0, max(0.0, portfolio_risk))
    
    def _determine_urgency(self, deviation: float) -> str:
        """Determine urgency level based on deviation"""
        if deviation > 0.15:
            return 'high'
        elif deviation > 0.10:
            return 'medium'
        else:
            return 'low'
    
    def _generate_rebalance_actions(self,
                                   current: Dict[str, float],
                                   target: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate rebalancing actions"""
        actions = []
        
        for asset in set(current.keys()) | set(target.keys()):
            current_weight = current.get(asset, 0.0)
            target_weight = target.get(asset, 0.0)
            
            if abs(current_weight - target_weight) > 0.01:  # Minimum threshold
                action = {
                    'asset': asset,
                    'current_weight': current_weight,
                    'target_weight': target_weight,
                    'adjustment': target_weight - current_weight,
                    'action_type': 'increase' if target_weight > current_weight else 'decrease'
                }
                actions.append(action)
        
        return actions
    
    def _generate_risk_adjustment_actions(self,
                                        current: Dict[str, float],
                                        current_risk: float) -> List[Dict[str, Any]]:
        """Generate actions to adjust portfolio risk"""
        actions = []
        
        if current_risk > self.risk_target:
            # Reduce risk - move to less risky assets
            actions.append({
                'action_type': 'risk_reduction',
                'description': 'Reduce allocation to high-risk assets',
                'current_risk': current_risk,
                'target_risk': self.risk_target
            })
        else:
            # Increase risk - move to more risky assets
            actions.append({
                'action_type': 'risk_increase', 
                'description': 'Increase allocation to growth assets',
                'current_risk': current_risk,
                'target_risk': self.risk_target
            })
        
        return actions
    
    def _categorize_risk_level(self, risk: float) -> str:
        """Categorize risk level"""
        if risk > 0.25:
            return 'high'
        elif risk > 0.15:
            return 'medium'
        elif risk > 0.05:
            return 'normal'
        else:
            return 'low'
    
    def _execute_single_action(self,
                             current_allocations: Dict[str, float],
                             action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single rebalancing action"""
        result = {
            'success': False,
            'new_allocations': {},
            'cost': 0.0
        }
        
        try:
            asset = action.get('asset')
            target_weight = action.get('target_weight', 0.0)
            
            if asset and asset in current_allocations:
                new_allocations = current_allocations.copy()
                new_allocations[asset] = target_weight
                
                # Simple cost model (0.1% transaction cost)
                cost = abs(target_weight - current_allocations[asset]) * 0.001
                
                result.update({
                    'success': True,
                    'new_allocations': new_allocations,
                    'cost': cost
                })
        
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            result['error'] = str(e)
        
        return result