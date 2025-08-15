"""
Reward Functions for Trading RL Agents

Implements various sophisticated reward functions for trading reinforcement learning,
including risk-adjusted returns, drawdown penalties, and multi-objective optimization.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import logging
from collections import deque

from core.logger import get_logger

logger = get_logger(__name__)

class BaseRewardFunction(ABC):
    """
    Abstract Base Class for Reward Functions
    """
    
    def __init__(self, name: str):
        self.name = name
        self.reset()
    
    @abstractmethod
    def calculate_reward(self, portfolio_info: Dict, market_info: Dict) -> float:
        """
        Calculate reward given portfolio and market information
        
        Args:
            portfolio_info: Current portfolio state
            market_info: Current market state
            
        Returns:
            Calculated reward
        """
        pass
    
    def reset(self):
        """Reset reward function state"""
        pass
    
    def get_info(self) -> Dict:
        """Get additional information about reward calculation"""
        return {'name': self.name}


class ProfitRiskReward(BaseRewardFunction):
    """
    Profit-Risk Balanced Reward Function
    
    Balances profit maximization with risk minimization using
    a combination of returns and drawdown penalties.
    """
    
    def __init__(self, 
                 profit_weight: float = 1.0,
                 risk_weight: float = 0.5,
                 drawdown_penalty: float = 2.0,
                 volatility_penalty: float = 0.1,
                 transaction_cost_penalty: float = 0.1):
        """
        Initialize Profit-Risk Reward Function
        
        Args:
            profit_weight: Weight for profit component
            risk_weight: Weight for risk penalty
            drawdown_penalty: Penalty multiplier for drawdowns
            volatility_penalty: Penalty for portfolio volatility
            transaction_cost_penalty: Penalty for transaction costs
        """
        super().__init__("ProfitRiskReward")
        
        self.profit_weight = profit_weight
        self.risk_weight = risk_weight
        self.drawdown_penalty = drawdown_penalty
        self.volatility_penalty = volatility_penalty
        self.transaction_cost_penalty = transaction_cost_penalty
        
        # State tracking
        self.previous_value = None
        self.peak_value = 0
        self.returns_history = deque(maxlen=100)
        
        logger.info(f"Initialized ProfitRiskReward with profit_weight={profit_weight}, risk_weight={risk_weight}")
    
    def calculate_reward(self, portfolio_info: Dict, market_info: Dict) -> float:
        """
        Calculate profit-risk balanced reward
        
        Args:
            portfolio_info: Portfolio information with keys:
                - value: Current portfolio value
                - position: Current position
                - cash: Available cash
                - transaction_costs: Recent transaction costs
            market_info: Market information with keys:
                - price: Current price
                - volatility: Current market volatility
                
        Returns:
            Calculated reward
        """
        current_value = portfolio_info.get('value', 0.0)
        initial_value = portfolio_info.get('initial_value', current_value)
        transaction_costs = portfolio_info.get('transaction_costs', 0.0)
        
        # Initialize if first step
        if self.previous_value is None:
            self.previous_value = current_value
            self.peak_value = current_value
            return 0.0
        
        # Calculate return
        step_return = (current_value - self.previous_value) / self.previous_value if self.previous_value > 0 else 0.0
        self.returns_history.append(step_return)
        
        # Update peak value
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        # 1. Profit Component
        profit_reward = self.profit_weight * step_return
        
        # 2. Risk Penalties
        risk_penalty = 0.0
        
        # Drawdown penalty
        if self.peak_value > 0:
            current_drawdown = (self.peak_value - current_value) / self.peak_value
            if current_drawdown > 0.05:  # Penalize drawdowns > 5%
                risk_penalty += self.drawdown_penalty * current_drawdown**2
        
        # Volatility penalty
        if len(self.returns_history) >= 10:
            recent_volatility = np.std(list(self.returns_history)[-10:])
            if recent_volatility > 0.02:  # Penalize high volatility
                risk_penalty += self.volatility_penalty * recent_volatility
        
        # Transaction cost penalty
        if transaction_costs > 0:
            cost_ratio = transaction_costs / current_value if current_value > 0 else 0
            risk_penalty += self.transaction_cost_penalty * cost_ratio
        
        # 3. Combine components
        reward = profit_reward - self.risk_weight * risk_penalty
        
        # Update state
        self.previous_value = current_value
        
        return reward
    
    def reset(self):
        """Reset reward function state"""
        self.previous_value = None
        self.peak_value = 0
        self.returns_history.clear()
    
    def get_info(self) -> Dict:
        """Get reward function information"""
        return {
            'name': self.name,
            'profit_weight': self.profit_weight,
            'risk_weight': self.risk_weight,
            'peak_value': self.peak_value,
            'current_volatility': np.std(list(self.returns_history)) if len(self.returns_history) >= 2 else 0.0,
            'returns_history_length': len(self.returns_history)
        }


class SharpeReward(BaseRewardFunction):
    """
    Sharpe Ratio Based Reward Function
    
    Rewards agents based on risk-adjusted returns using Sharpe ratio calculation.
    """
    
    def __init__(self, 
                 lookback_window: int = 50,
                 risk_free_rate: float = 0.02,
                 annualization_factor: float = 252.0):
        """
        Initialize Sharpe Reward Function
        
        Args:
            lookback_window: Window for Sharpe calculation
            risk_free_rate: Annual risk-free rate
            annualization_factor: Factor to annualize returns
        """
        super().__init__("SharpeReward")
        
        self.lookback_window = lookback_window
        self.risk_free_rate = risk_free_rate
        self.annualization_factor = annualization_factor
        
        self.returns_history = deque(maxlen=lookback_window)
        self.previous_value = None
        
        logger.info(f"Initialized SharpeReward with lookback_window={lookback_window}")
    
    def calculate_reward(self, portfolio_info: Dict, market_info: Dict) -> float:
        """Calculate Sharpe ratio based reward"""
        current_value = portfolio_info.get('value', 0.0)
        
        # Initialize if first step
        if self.previous_value is None:
            self.previous_value = current_value
            return 0.0
        
        # Calculate return
        if self.previous_value > 0:
            step_return = (current_value - self.previous_value) / self.previous_value
            self.returns_history.append(step_return)
        
        # Calculate Sharpe ratio if enough history
        if len(self.returns_history) < 10:
            reward = 0.0
        else:
            returns = np.array(list(self.returns_history))
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            # Annualized metrics
            annual_return = mean_return * self.annualization_factor
            annual_volatility = std_return * np.sqrt(self.annualization_factor)
            
            # Sharpe ratio
            if annual_volatility > 0:
                sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility
                reward = sharpe_ratio / 10.0  # Scale for RL
            else:
                reward = 0.0
        
        self.previous_value = current_value
        return reward
    
    def reset(self):
        """Reset reward function state"""
        self.previous_value = None
        self.returns_history.clear()


class MaxDrawdownReward(BaseRewardFunction):
    """
    Maximum Drawdown Penalized Reward Function
    
    Heavily penalizes large drawdowns while rewarding profits.
    """
    
    def __init__(self, 
                 profit_weight: float = 1.0,
                 drawdown_penalty: float = 5.0,
                 max_acceptable_drawdown: float = 0.15):
        """
        Initialize Max Drawdown Reward Function
        
        Args:
            profit_weight: Weight for profit component
            drawdown_penalty: Penalty multiplier for drawdowns
            max_acceptable_drawdown: Maximum acceptable drawdown threshold
        """
        super().__init__("MaxDrawdownReward")
        
        self.profit_weight = profit_weight
        self.drawdown_penalty = drawdown_penalty
        self.max_acceptable_drawdown = max_acceptable_drawdown
        
        self.previous_value = None
        self.peak_value = 0
        self.max_drawdown = 0
        
    def calculate_reward(self, portfolio_info: Dict, market_info: Dict) -> float:
        """Calculate max drawdown penalized reward"""
        current_value = portfolio_info.get('value', 0.0)
        
        # Initialize if first step
        if self.previous_value is None:
            self.previous_value = current_value
            self.peak_value = current_value
            return 0.0
        
        # Calculate step return
        step_return = (current_value - self.previous_value) / self.previous_value if self.previous_value > 0 else 0.0
        
        # Update peak value
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        # Calculate current drawdown
        current_drawdown = (self.peak_value - current_value) / self.peak_value if self.peak_value > 0 else 0.0
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Profit component
        profit_reward = self.profit_weight * step_return
        
        # Drawdown penalty (exponential penalty for large drawdowns)
        if current_drawdown > self.max_acceptable_drawdown:
            drawdown_penalty = self.drawdown_penalty * (current_drawdown / self.max_acceptable_drawdown)**2
        else:
            drawdown_penalty = self.drawdown_penalty * (current_drawdown**2)
        
        reward = profit_reward - drawdown_penalty
        
        self.previous_value = current_value
        return reward
    
    def reset(self):
        """Reset reward function state"""
        self.previous_value = None
        self.peak_value = 0
        self.max_drawdown = 0


class CalmarReward(BaseRewardFunction):
    """
    Calmar Ratio Based Reward Function
    
    Uses Calmar ratio (annual return / max drawdown) as reward signal.
    """
    
    def __init__(self, 
                 lookback_window: int = 100,
                 annualization_factor: float = 252.0):
        """
        Initialize Calmar Reward Function
        
        Args:
            lookback_window: Window for Calmar calculation
            annualization_factor: Factor to annualize returns
        """
        super().__init__("CalmarReward")
        
        self.lookback_window = lookback_window
        self.annualization_factor = annualization_factor
        
        self.value_history = deque(maxlen=lookback_window)
        self.previous_value = None
        
    def calculate_reward(self, portfolio_info: Dict, market_info: Dict) -> float:
        """Calculate Calmar ratio based reward"""
        current_value = portfolio_info.get('value', 0.0)
        
        # Initialize if first step
        if self.previous_value is None:
            self.previous_value = current_value
            self.value_history.append(current_value)
            return 0.0
        
        self.value_history.append(current_value)
        
        # Calculate Calmar ratio if enough history
        if len(self.value_history) < 20:
            reward = 0.0
        else:
            values = np.array(list(self.value_history))
            
            # Calculate returns
            returns = np.diff(values) / values[:-1]
            annual_return = np.mean(returns) * self.annualization_factor
            
            # Calculate max drawdown
            peak = np.maximum.accumulate(values)
            drawdown = (peak - values) / peak
            max_drawdown = np.max(drawdown)
            
            # Calmar ratio
            if max_drawdown > 0.01:  # Avoid division by zero
                calmar_ratio = annual_return / max_drawdown
                reward = calmar_ratio / 100.0  # Scale for RL
            else:
                reward = annual_return
        
        self.previous_value = current_value
        return reward
    
    def reset(self):
        """Reset reward function state"""
        self.previous_value = None
        self.value_history.clear()


class MultiObjectiveReward(BaseRewardFunction):
    """
    Multi-Objective Reward Function
    
    Combines multiple objectives with configurable weights for
    comprehensive trading performance evaluation.
    """
    
    def __init__(self, 
                 objectives: Dict[str, float] = None,
                 lookback_window: int = 50):
        """
        Initialize Multi-Objective Reward Function
        
        Args:
            objectives: Dictionary of objective weights
            lookback_window: Window for calculations
        """
        super().__init__("MultiObjectiveReward")
        
        if objectives is None:
            objectives = {
                'profit': 1.0,
                'sharpe': 0.5,
                'drawdown': -1.0,
                'volatility': -0.3,
                'win_rate': 0.3
            }
        
        self.objectives = objectives
        self.lookback_window = lookback_window
        
        # State tracking
        self.previous_value = None
        self.peak_value = 0
        self.returns_history = deque(maxlen=lookback_window)
        self.trades = []
        
        logger.info(f"Initialized MultiObjectiveReward with objectives: {objectives}")
    
    def calculate_reward(self, portfolio_info: Dict, market_info: Dict) -> float:
        """Calculate multi-objective reward"""
        current_value = portfolio_info.get('value', 0.0)
        position = portfolio_info.get('position', 0.0)
        previous_position = portfolio_info.get('previous_position', 0.0)
        
        # Initialize if first step
        if self.previous_value is None:
            self.previous_value = current_value
            self.peak_value = current_value
            return 0.0
        
        # Calculate step return
        step_return = (current_value - self.previous_value) / self.previous_value if self.previous_value > 0 else 0.0
        self.returns_history.append(step_return)
        
        # Update peak value
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        # Track trades
        if abs(position - previous_position) > 0.01:
            self.trades.append(step_return)
        
        # Calculate individual objectives
        objectives_scores = {}
        
        # 1. Profit objective
        objectives_scores['profit'] = step_return
        
        # 2. Sharpe ratio objective
        if len(self.returns_history) >= 10:
            returns = np.array(list(self.returns_history))
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            objectives_scores['sharpe'] = mean_return / (std_return + 1e-8)
        else:
            objectives_scores['sharpe'] = 0.0
        
        # 3. Drawdown objective (negative penalty)
        current_drawdown = (self.peak_value - current_value) / self.peak_value if self.peak_value > 0 else 0.0
        objectives_scores['drawdown'] = current_drawdown
        
        # 4. Volatility objective (negative penalty)
        if len(self.returns_history) >= 10:
            recent_volatility = np.std(list(self.returns_history)[-10:])
            objectives_scores['volatility'] = recent_volatility
        else:
            objectives_scores['volatility'] = 0.0
        
        # 5. Win rate objective
        if len(self.trades) >= 10:
            winning_trades = sum(1 for trade in self.trades[-20:] if trade > 0)
            win_rate = winning_trades / min(len(self.trades[-20:]), 20)
            objectives_scores['win_rate'] = win_rate - 0.5  # Center around 0
        else:
            objectives_scores['win_rate'] = 0.0
        
        # Combine objectives
        reward = 0.0
        for objective, weight in self.objectives.items():
            if objective in objectives_scores:
                reward += weight * objectives_scores[objective]
        
        self.previous_value = current_value
        return reward
    
    def reset(self):
        """Reset reward function state"""
        self.previous_value = None
        self.peak_value = 0
        self.returns_history.clear()
        self.trades.clear()
    
    def get_info(self) -> Dict:
        """Get reward function information"""
        info = super().get_info()
        info.update({
            'objectives': self.objectives,
            'peak_value': self.peak_value,
            'num_trades': len(self.trades),
            'win_rate': sum(1 for trade in self.trades if trade > 0) / len(self.trades) if self.trades else 0.0
        })
        return info


class AdaptiveReward(BaseRewardFunction):
    """
    Adaptive Reward Function
    
    Automatically adjusts reward components based on market conditions
    and agent performance.
    """
    
    def __init__(self, 
                 adaptation_period: int = 100,
                 base_components: Dict[str, float] = None):
        """
        Initialize Adaptive Reward Function
        
        Args:
            adaptation_period: Period for adaptation
            base_components: Base reward components
        """
        super().__init__("AdaptiveReward")
        
        if base_components is None:
            base_components = {
                'profit': 1.0,
                'risk': 0.5,
                'consistency': 0.3
            }
        
        self.adaptation_period = adaptation_period
        self.base_components = base_components.copy()
        self.current_components = base_components.copy()
        
        # Performance tracking
        self.performance_history = deque(maxlen=adaptation_period)
        self.adaptation_counter = 0
        
        # State tracking
        self.previous_value = None
        self.returns_history = deque(maxlen=adaptation_period)
        
    def calculate_reward(self, portfolio_info: Dict, market_info: Dict) -> float:
        """Calculate adaptive reward"""
        current_value = portfolio_info.get('value', 0.0)
        
        # Initialize if first step
        if self.previous_value is None:
            self.previous_value = current_value
            return 0.0
        
        # Calculate step return
        step_return = (current_value - self.previous_value) / self.previous_value if self.previous_value > 0 else 0.0
        self.returns_history.append(step_return)
        
        # Calculate reward components
        profit_component = step_return
        
        risk_component = 0.0
        if len(self.returns_history) >= 10:
            volatility = np.std(list(self.returns_history)[-10:])
            risk_component = -volatility
        
        consistency_component = 0.0
        if len(self.returns_history) >= 20:
            recent_returns = list(self.returns_history)[-20:]
            consistency = -np.std(recent_returns) / (abs(np.mean(recent_returns)) + 1e-8)
            consistency_component = consistency
        
        # Calculate weighted reward
        reward = (self.current_components['profit'] * profit_component +
                 self.current_components['risk'] * risk_component +
                 self.current_components['consistency'] * consistency_component)
        
        # Store performance
        self.performance_history.append(reward)
        self.adaptation_counter += 1
        
        # Adapt weights periodically
        if self.adaptation_counter >= self.adaptation_period and len(self.performance_history) >= self.adaptation_period:
            self._adapt_components()
            self.adaptation_counter = 0
        
        self.previous_value = current_value
        return reward
    
    def _adapt_components(self):
        """Adapt reward components based on recent performance"""
        recent_performance = list(self.performance_history)
        avg_performance = np.mean(recent_performance)
        
        # Increase emphasis on components that led to good performance
        if avg_performance > 0:
            # Good performance - maintain current weights
            adaptation_factor = 0.95  # Slight decay toward base
        else:
            # Poor performance - adjust weights
            adaptation_factor = 0.8  # Stronger adjustment
            
            # Increase risk weight if performance is poor
            self.current_components['risk'] = min(2.0, self.current_components['risk'] * 1.2)
            
            # Decrease profit weight slightly
            self.current_components['profit'] = max(0.5, self.current_components['profit'] * 0.9)
        
        # Decay toward base components to prevent drift
        for component in self.current_components:
            self.current_components[component] = (
                adaptation_factor * self.current_components[component] +
                (1 - adaptation_factor) * self.base_components[component]
            )
        
        logger.debug(f"Adapted reward components: {self.current_components}")
    
    def reset(self):
        """Reset reward function state"""
        self.previous_value = None
        self.returns_history.clear()
        self.performance_history.clear()
        self.adaptation_counter = 0
        self.current_components = self.base_components.copy()
    
    def get_info(self) -> Dict:
        """Get reward function information"""
        return {
            'name': self.name,
            'current_components': self.current_components,
            'base_components': self.base_components,
            'adaptation_counter': self.adaptation_counter,
            'avg_recent_performance': np.mean(list(self.performance_history)) if self.performance_history else 0.0
        }