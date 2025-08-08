"""
Trading Environment for Reinforcement Learning

Implements a comprehensive trading environment that simulates realistic
market conditions for training RL agents.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from core.database import DatabaseManager
from core.logger import get_logger
from data.collectors.binance_collector import BinanceCollector
from data.processors.feature_engine import FeatureEngine

logger = get_logger(__name__)

class TradingEnvironment(gym.Env):
    """
    Trading Environment for RL Agents
    
    Simulates realistic trading conditions with:
    - Historical market data
    - Transaction costs
    - Slippage simulation  
    - Position management
    - Risk constraints
    """
    
    def __init__(self, 
                 symbol: str = "BTCUSDT",
                 initial_balance: float = 10000.0,
                 lookback_window: int = 50,
                 transaction_cost: float = 0.001,
                 max_position_size: float = 1.0,
                 start_date: str = None,
                 end_date: str = None):
        """
        Initialize Trading Environment
        
        Args:
            symbol: Trading symbol
            initial_balance: Starting balance
            lookback_window: Number of historical observations
            transaction_cost: Trading fee percentage
            max_position_size: Maximum position as fraction of balance
            start_date: Start date for training data
            end_date: End date for training data
        """
        super().__init__()
        
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.lookback_window = lookback_window
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        
        # Initialize components
        self.db_manager = DatabaseManager()
        self.data_collector = BinanceCollector()
        self.feature_engine = FeatureEngine()
        
        # Load and prepare data
        self._load_market_data(start_date, end_date)
        self._prepare_features()
        
        # Trading state
        self.current_step = 0
        self.position = 0.0  # Current position size (-1 to 1)
        self.entry_price = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.max_drawdown = 0.0
        self.peak_balance = initial_balance
        
        # Define action and observation spaces
        self._setup_spaces()
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_balances = []
        
        logger.info(f"TradingEnvironment initialized for {symbol} with {len(self.data)} data points")
    
    def _load_market_data(self, start_date: str = None, end_date: str = None):
        """Load historical market data"""
        try:
            # Use data collector to get historical data
            if start_date and end_date:
                self.data = self.data_collector.collect_historical_data(
                    self.symbol, 
                    start_date=start_date,
                    end_date=end_date,
                    interval="1h"
                )
            else:
                # Default to last 10000 hours (~1.1 years)
                end_time = datetime.now()
                start_time = end_time - timedelta(hours=10000)
                self.data = self.data_collector.collect_historical_data(
                    self.symbol,
                    start_date=start_time.strftime("%Y-%m-%d"),
                    end_date=end_time.strftime("%Y-%m-%d"),
                    interval="1h"
                )
                
            if self.data is None or len(self.data) < self.lookback_window:
                raise ValueError(f"Insufficient data loaded: {len(self.data) if self.data is not None else 0}")
                
            logger.info(f"Loaded {len(self.data)} data points for {self.symbol}")
            
        except Exception as e:
            logger.error(f"Failed to load market data: {e}")
            # Generate synthetic data as fallback
            self._generate_synthetic_data()
    
    def _generate_synthetic_data(self):
        """Generate synthetic market data for testing"""
        logger.warning("Generating synthetic data for testing")
        
        # Generate realistic price movements
        np.random.seed(42)
        n_points = 5000
        base_price = 50000.0
        
        returns = np.random.normal(0.0001, 0.02, n_points)  # Small positive drift with volatility
        prices = [base_price]
        
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        # Create DataFrame similar to real data
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(hours=n_points),
            periods=n_points,
            freq='1H'
        )
        
        self.data = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices[:-1],
            'close': prices[1:],
            'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices[1:]],
            'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices[1:]],
            'volume': np.random.lognormal(15, 1, n_points)
        })
        
        logger.info(f"Generated {len(self.data)} synthetic data points")
    
    def _prepare_features(self):
        """Prepare technical features for the environment"""
        try:
            # Use feature engine to create technical indicators
            self.features_df = self.feature_engine.create_features(self.data)
            
            # Select key features for RL state
            self.feature_columns = [
                'returns', 'returns_1h', 'returns_4h', 'returns_24h',
                'rsi_14', 'bb_position', 'macd_signal', 'atr_normalized',
                'volume_sma_ratio', 'price_sma_20', 'price_sma_50', 'price_ema_12',
                'bollinger_squeeze', 'trend_strength', 'volatility_regime'
            ]
            
            # Fill missing values and normalize
            self.features_df = self.features_df[self.feature_columns].fillna(0)
            self.features_normalized = self._normalize_features(self.features_df)
            
            logger.info(f"Prepared {len(self.feature_columns)} features for RL environment")
            
        except Exception as e:
            logger.error(f"Failed to prepare features: {e}")
            # Create basic features as fallback
            self._create_basic_features()
    
    def _create_basic_features(self):
        """Create basic features if feature engine fails"""
        logger.warning("Creating basic features as fallback")
        
        # Basic price features
        self.data['returns'] = self.data['close'].pct_change().fillna(0)
        self.data['sma_20'] = self.data['close'].rolling(20).mean()
        self.data['sma_50'] = self.data['close'].rolling(50).mean()
        self.data['rsi'] = self._calculate_rsi(self.data['close'], 14)
        
        self.feature_columns = ['returns', 'sma_20', 'sma_50', 'rsi']
        self.features_df = self.data[self.feature_columns].fillna(method='bfill').fillna(0)
        self.features_normalized = self._normalize_features(self.features_df)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _normalize_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """Normalize features to [-1, 1] range"""
        # Use rolling statistics for normalization
        means = features_df.rolling(window=200, min_periods=50).mean()
        stds = features_df.rolling(window=200, min_periods=50).std()
        
        normalized = (features_df - means) / (stds + 1e-8)
        return np.clip(normalized.fillna(0).values, -3, 3)
    
    def _setup_spaces(self):
        """Setup action and observation spaces"""
        # Action space: [position_change, confidence]
        # position_change: -1 (sell all) to +1 (buy all)
        # confidence: 0 to 1 (how confident in the action)
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        # Observation space: [market_features, portfolio_state]
        market_features_dim = len(self.feature_columns) * self.lookback_window
        portfolio_features_dim = 8  # position, balance, drawdown, etc.
        
        obs_dim = market_features_dim + portfolio_features_dim
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        logger.info(f"Action space: {self.action_space}")
        logger.info(f"Observation space: {self.observation_space}")
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = self.lookback_window
        self.current_balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.max_drawdown = 0.0
        self.peak_balance = self.initial_balance
        
        # Reset tracking
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_balances = [self.initial_balance]
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment"""
        if self.current_step >= len(self.features_normalized) - 1:
            return self._get_observation(), 0.0, True, self._get_info()
        
        # Parse action
        position_target = float(action[0])  # Target position
        confidence = float(action[1])       # Action confidence
        
        # Apply confidence scaling
        position_target *= confidence
        
        # Clip to maximum position size
        position_target = np.clip(position_target, -self.max_position_size, self.max_position_size)
        
        # Execute trade
        reward = self._execute_trade(position_target)
        
        # Update state
        self.current_step += 1
        self.episode_rewards.append(reward)
        self.episode_actions.append(action)
        self.episode_balances.append(self.current_balance)
        
        # Check if episode is done
        done = (self.current_step >= len(self.features_normalized) - 1 or 
                self.current_balance <= self.initial_balance * 0.5)  # 50% drawdown limit
        
        return self._get_observation(), reward, done, self._get_info()
    
    def _execute_trade(self, target_position: float) -> float:
        """Execute trading action and calculate reward"""
        current_price = self.data.iloc[self.current_step]['close']
        
        # Calculate position change
        position_change = target_position - self.position
        
        if abs(position_change) > 0.01:  # Only execute if significant change
            # Calculate transaction cost
            cost = abs(position_change) * self.transaction_cost * self.current_balance
            
            # Update position
            old_position = self.position
            self.position = target_position
            
            # Track trades
            if old_position * self.position <= 0:  # Position direction change
                self.total_trades += 1
                
                # Calculate P&L from previous position if any
                if abs(old_position) > 0.01 and self.entry_price > 0:
                    pnl = old_position * (current_price - self.entry_price) * self.current_balance
                    self.current_balance += pnl
                    
                    if pnl > 0:
                        self.winning_trades += 1
                
                # Set new entry price
                self.entry_price = current_price
            
            # Apply transaction cost
            self.current_balance -= cost
            
        # Calculate unrealized P&L
        unrealized_pnl = 0.0
        if abs(self.position) > 0.01 and self.entry_price > 0:
            unrealized_pnl = self.position * (current_price - self.entry_price) * self.current_balance
        
        # Update portfolio value
        portfolio_value = self.current_balance + unrealized_pnl
        
        # Update drawdown
        if portfolio_value > self.peak_balance:
            self.peak_balance = portfolio_value
        
        current_drawdown = (self.peak_balance - portfolio_value) / self.peak_balance
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Calculate reward
        reward = self._calculate_reward(portfolio_value, current_drawdown)
        
        return reward
    
    def _calculate_reward(self, portfolio_value: float, drawdown: float) -> float:
        """Calculate step reward"""
        # Base reward from portfolio return
        portfolio_return = (portfolio_value - self.initial_balance) / self.initial_balance
        
        # Risk-adjusted reward
        reward = portfolio_return
        
        # Penalize high drawdown
        if drawdown > 0.1:  # More than 10% drawdown
            reward -= drawdown * 2
        
        # Bonus for maintaining profitable trades
        if self.total_trades > 10:
            win_rate = self.winning_trades / self.total_trades
            if win_rate > 0.6:
                reward += 0.1 * (win_rate - 0.5)
            else:
                reward -= 0.1 * (0.5 - win_rate)
        
        # Scale reward
        return reward * 100  # Scale for better learning
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        # Market features (lookback window)
        if self.current_step < self.lookback_window:
            start_idx = 0
            market_features = np.zeros((self.lookback_window, len(self.feature_columns)))
            available_steps = self.current_step + 1
            market_features[-available_steps:] = self.features_normalized[start_idx:self.current_step + 1]
        else:
            start_idx = self.current_step - self.lookback_window + 1
            market_features = self.features_normalized[start_idx:self.current_step + 1]
        
        # Flatten market features
        market_obs = market_features.flatten()
        
        # Portfolio state
        current_price = self.data.iloc[self.current_step]['close']
        unrealized_pnl = 0.0
        if abs(self.position) > 0.01 and self.entry_price > 0:
            unrealized_pnl = self.position * (current_price - self.entry_price)
        
        portfolio_value = self.current_balance + unrealized_pnl * self.current_balance
        
        portfolio_obs = np.array([
            self.position,  # Current position
            (self.current_balance - self.initial_balance) / self.initial_balance,  # Balance change
            unrealized_pnl,  # Unrealized P&L
            self.max_drawdown,  # Maximum drawdown
            self.total_trades / 100.0,  # Number of trades (normalized)
            self.winning_trades / max(self.total_trades, 1),  # Win rate
            (portfolio_value - self.initial_balance) / self.initial_balance,  # Total return
            min(self.current_step / 1000.0, 1.0)  # Episode progress
        ], dtype=np.float32)
        
        # Combine observations
        observation = np.concatenate([market_obs, portfolio_obs])
        
        return observation.astype(np.float32)
    
    def _get_info(self) -> Dict:
        """Get additional information"""
        current_price = self.data.iloc[self.current_step]['close']
        unrealized_pnl = 0.0
        if abs(self.position) > 0.01 and self.entry_price > 0:
            unrealized_pnl = self.position * (current_price - self.entry_price)
        
        portfolio_value = self.current_balance + unrealized_pnl * self.current_balance
        
        return {
            'step': self.current_step,
            'balance': self.current_balance,
            'position': self.position,
            'portfolio_value': portfolio_value,
            'total_return': (portfolio_value - self.initial_balance) / self.initial_balance,
            'max_drawdown': self.max_drawdown,
            'total_trades': self.total_trades,
            'win_rate': self.winning_trades / max(self.total_trades, 1),
            'price': current_price
        }
    
    def render(self, mode='human'):
        """Render environment state"""
        info = self._get_info()
        if mode == 'human':
            print(f"Step: {info['step']}")
            print(f"Balance: ${info['balance']:.2f}")
            print(f"Position: {info['position']:.3f}")
            print(f"Portfolio Value: ${info['portfolio_value']:.2f}")
            print(f"Total Return: {info['total_return']:.2%}")
            print(f"Max Drawdown: {info['max_drawdown']:.2%}")
            print(f"Trades: {info['total_trades']} (Win Rate: {info['win_rate']:.1%})")
            print("-" * 50)