"""
Market Simulator for Reinforcement Learning

Advanced market simulation with realistic market dynamics,
volatility regimes, and multi-asset support.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

from core.logger import get_logger

logger = get_logger(__name__)

class MarketSimulator:
    """
    Advanced Market Simulator for RL Training
    
    Features:
    - Realistic price movements with volatility clustering
    - Multiple volatility regimes
    - Correlation between assets
    - Market impact simulation
    - News event simulation
    """
    
    def __init__(self,
                 base_price: float = 50000.0,
                 volatility: float = 0.02,
                 drift: float = 0.0001,
                 regime_switch_prob: float = 0.01):
        """
        Initialize Market Simulator
        
        Args:
            base_price: Starting price
            volatility: Base volatility
            drift: Trend component
            regime_switch_prob: Probability of volatility regime switch
        """
        self.base_price = base_price
        self.volatility = volatility
        self.drift = drift
        self.regime_switch_prob = regime_switch_prob
        
        # Volatility regimes
        self.volatility_regimes = {
            'low': 0.01,
            'normal': 0.02,
            'high': 0.05,
            'extreme': 0.10
        }
        
        self.current_regime = 'normal'
        self.current_price = base_price
        self.price_history = [base_price]
        self.volume_history = []
        
        # Market microstructure
        self.bid_ask_spread = 0.0001  # 0.01%
        self.market_impact_factor = 0.0001
        
        logger.info(f"MarketSimulator initialized with base_price={base_price}")
    
    def simulate_step(self, volume: float = 0.0) -> Dict:
        """
        Simulate one market step
        
        Args:
            volume: Trading volume (for market impact)
            
        Returns:
            Dict with OHLCV data
        """
        # Check for regime switch
        if np.random.random() < self.regime_switch_prob:
            self._switch_regime()
        
        # Get current volatility
        current_vol = self.volatility_regimes[self.current_regime]
        
        # Generate return with regime-dependent volatility
        base_return = np.random.normal(self.drift, current_vol)
        
        # Add volatility clustering (GARCH-like effect)
        if len(self.price_history) > 1:
            prev_return = (self.price_history[-1] - self.price_history[-2]) / self.price_history[-2]
            volatility_adjustment = abs(prev_return) * 0.5
            base_return += np.random.normal(0, volatility_adjustment)
        
        # Apply market impact if trading volume provided
        if volume > 0:
            market_impact = np.sign(volume) * min(abs(volume) * self.market_impact_factor, 0.01)
            base_return += market_impact
        
        # Calculate new price
        new_price = self.current_price * (1 + base_return)
        
        # Generate OHLC from current price move
        open_price = self.current_price
        close_price = new_price
        
        # High and low with realistic intrabar movement
        high_low_range = abs(base_return) * np.random.uniform(1.5, 3.0)
        high_price = max(open_price, close_price) * (1 + high_low_range / 2)
        low_price = min(open_price, close_price) * (1 - high_low_range / 2)
        
        # Generate realistic volume
        base_volume = np.random.lognormal(15, 1)  # Log-normal distribution
        
        # Volume spikes during high volatility
        if current_vol > 0.03:
            base_volume *= np.random.uniform(2, 5)
        
        # Update state
        self.current_price = close_price
        self.price_history.append(close_price)
        self.volume_history.append(base_volume)
        
        # Keep history manageable
        if len(self.price_history) > 10000:
            self.price_history = self.price_history[-5000:]
            self.volume_history = self.volume_history[-5000:]
        
        return {
            'timestamp': datetime.now(),
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': base_volume,
            'regime': self.current_regime,
            'volatility': current_vol
        }
    
    def _switch_regime(self):
        """Switch volatility regime"""
        regimes = list(self.volatility_regimes.keys())
        current_idx = regimes.index(self.current_regime)
        
        # Bias toward neighboring regimes
        if self.current_regime == 'low':
            weights = [0.3, 0.6, 0.1, 0.0]
        elif self.current_regime == 'normal':
            weights = [0.2, 0.3, 0.4, 0.1]
        elif self.current_regime == 'high':
            weights = [0.1, 0.4, 0.3, 0.2]
        else:  # extreme
            weights = [0.0, 0.2, 0.5, 0.3]
        
        self.current_regime = np.random.choice(regimes, p=weights)
        logger.debug(f"Regime switched to: {self.current_regime}")
    
    def simulate_market_crash(self, crash_magnitude: float = -0.3):
        """Simulate market crash event"""
        crash_return = crash_magnitude + np.random.normal(0, 0.1)
        self.current_price *= (1 + crash_return)
        self.price_history.append(self.current_price)
        
        # Switch to extreme volatility regime
        self.current_regime = 'extreme'
        
        logger.warning(f"Market crash simulated: {crash_return:.2%} drop")
    
    def simulate_bull_run(self, run_magnitude: float = 0.5, duration: int = 100):
        """Simulate sustained bull run"""
        daily_return = (1 + run_magnitude) ** (1/duration) - 1
        
        for _ in range(duration):
            bull_return = daily_return + np.random.normal(0, 0.01)
            self.current_price *= (1 + bull_return)
            self.price_history.append(self.current_price)
        
        logger.info(f"Bull run simulated: {run_magnitude:.1%} over {duration} periods")
    
    def get_market_summary(self) -> Dict:
        """Get current market summary statistics"""
        if len(self.price_history) < 2:
            return {}
        
        returns = np.diff(self.price_history) / self.price_history[:-1]
        
        return {
            'current_price': self.current_price,
            'regime': self.current_regime,
            'current_volatility': self.volatility_regimes[self.current_regime],
            'realized_volatility': np.std(returns[-100:]) if len(returns) >= 100 else np.std(returns),
            'price_change_24h': (self.current_price - self.price_history[-24]) / self.price_history[-24] if len(self.price_history) >= 24 else 0,
            'price_change_7d': (self.current_price - self.price_history[-168]) / self.price_history[-168] if len(self.price_history) >= 168 else 0,
            'volume_24h_avg': np.mean(self.volume_history[-24:]) if len(self.volume_history) >= 24 else 0
        }