"""
RL Portfolio Manager

Advanced portfolio management system for reinforcement learning trading agents
with position sizing, risk management, and performance tracking.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass
from collections import deque
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from core.logger import get_logger
from trading.risk_management.risk_manager import RiskManager

logger = get_logger(__name__)

@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    size: float  # Positive for long, negative for short
    entry_price: float
    entry_time: datetime
    current_price: float
    unrealized_pnl: float = 0.0
    
    def update_price(self, new_price: float):
        """Update current price and unrealized P&L"""
        self.current_price = new_price
        self.unrealized_pnl = self.size * (new_price - self.entry_price)

@dataclass  
class Trade:
    """Represents a completed trade"""
    symbol: str
    side: str  # 'buy' or 'sell'
    size: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    commission: float
    
    @property
    def return_pct(self) -> float:
        """Return percentage of the trade"""
        if self.entry_price != 0:
            return (self.exit_price - self.entry_price) / self.entry_price
        return 0.0

class Portfolio:
    """
    Portfolio state for RL trading
    """
    
    def __init__(self, 
                 initial_cash: float = 10000.0,
                 commission_rate: float = 0.001):
        """
        Initialize Portfolio
        
        Args:
            initial_cash: Starting cash amount
            commission_rate: Commission rate for trades
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.commission_rate = commission_rate
        
        # Positions and trades
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        
        # Performance tracking
        self.equity_curve = [initial_cash]
        self.timestamps = [datetime.now()]
        self.peak_value = initial_cash
        self.max_drawdown = 0.0
        
        # Statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_commission = 0.0
        
    @property
    def market_value(self) -> float:
        """Total market value of positions"""
        return sum(pos.size * pos.current_price for pos in self.positions.values())
    
    @property
    def unrealized_pnl(self) -> float:
        """Total unrealized P&L"""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    @property
    def total_value(self) -> float:
        """Total portfolio value (cash + positions)"""
        return self.cash + self.market_value
    
    @property
    def total_return(self) -> float:
        """Total return percentage"""
        if self.initial_cash != 0:
            return (self.total_value - self.initial_cash) / self.initial_cash
        return 0.0
    
    @property
    def current_drawdown(self) -> float:
        """Current drawdown from peak"""
        if self.peak_value != 0:
            return (self.peak_value - self.total_value) / self.peak_value
        return 0.0
    
    @property
    def win_rate(self) -> float:
        """Win rate of completed trades"""
        if self.total_trades > 0:
            return self.winning_trades / self.total_trades
        return 0.0
    
    def update_position_prices(self, prices: Dict[str, float]):
        """Update position prices"""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].update_price(price)
        
        # Update equity curve
        current_value = self.total_value
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        self.equity_curve.append(current_value)
        self.timestamps.append(datetime.now())
    
    def get_position_size(self, symbol: str) -> float:
        """Get current position size for symbol"""
        return self.positions.get(symbol, Position(symbol, 0, 0, datetime.now(), 0)).size
    
    def get_info(self) -> Dict:
        """Get portfolio information"""
        return {
            'total_value': self.total_value,
            'cash': self.cash,
            'market_value': self.market_value,
            'unrealized_pnl': self.unrealized_pnl,
            'total_return': self.total_return,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'win_rate': self.win_rate,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'total_commission': self.total_commission,
            'positions': {symbol: pos.size for symbol, pos in self.positions.items()}
        }


class RLPortfolioManager:
    """
    Reinforcement Learning Portfolio Manager
    
    Manages portfolio state and executes trades for RL agents with
    sophisticated risk management and performance tracking.
    """
    
    def __init__(self,
                 initial_cash: float = 10000.0,
                 commission_rate: float = 0.001,
                 max_position_size: float = 1.0,
                 max_total_leverage: float = 2.0,
                 risk_free_rate: float = 0.02):
        """
        Initialize RL Portfolio Manager
        
        Args:
            initial_cash: Starting cash amount
            commission_rate: Commission rate for trades
            max_position_size: Maximum position size as fraction of portfolio
            max_total_leverage: Maximum total leverage
            risk_free_rate: Annual risk-free rate
        """
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.max_position_size = max_position_size
        self.max_total_leverage = max_total_leverage
        self.risk_free_rate = risk_free_rate
        
        # Initialize portfolio
        self.portfolio = Portfolio(initial_cash, commission_rate)
        
        # Risk management
        try:
            self.risk_manager = RiskManager()
        except Exception as e:
            logger.warning(f"Could not initialize RiskManager: {e}")
            self.risk_manager = None
        
        # Performance tracking
        self.returns_history = deque(maxlen=252)  # 1 year of daily returns
        self.sharpe_history = deque(maxlen=100)
        self.volatility_history = deque(maxlen=100)
        
        # Action history for analysis
        self.action_history = deque(maxlen=1000)
        
        logger.info(f"RLPortfolioManager initialized with initial_cash={initial_cash}")
    
    def execute_action(self, 
                      symbol: str, 
                      target_position: float, 
                      current_price: float, 
                      confidence: float = 1.0) -> Dict:
        """
        Execute RL agent action
        
        Args:
            symbol: Trading symbol
            target_position: Target position (-1 to 1)
            current_price: Current market price
            confidence: Action confidence (0 to 1)
            
        Returns:
            Dictionary with execution results
        """
        # Apply confidence scaling
        target_position *= confidence
        
        # Clip to maximum position size
        target_position = np.clip(target_position, -self.max_position_size, self.max_position_size)
        
        # Get current position
        current_position = self.portfolio.get_position_size(symbol)
        
        # Calculate position change
        position_change = target_position - current_position
        
        execution_info = {
            'symbol': symbol,
            'current_position': current_position,
            'target_position': target_position,
            'position_change': position_change,
            'executed': False,
            'commission': 0.0,
            'error': None
        }
        
        # Execute trade if significant change
        if abs(position_change) > 0.01:  # Minimum trade size threshold
            try:
                # Risk checks
                if not self._risk_check(symbol, target_position, current_price):
                    execution_info['error'] = 'Risk check failed'
                    return execution_info
                
                # Execute trade
                success = self._execute_trade(symbol, position_change, current_price)
                
                if success:
                    execution_info['executed'] = True
                    execution_info['commission'] = abs(position_change) * current_price * self.commission_rate
                    
                    # Record action
                    self.action_history.append({
                        'timestamp': datetime.now(),
                        'symbol': symbol,
                        'action': target_position,
                        'confidence': confidence,
                        'price': current_price,
                        'executed': True
                    })
                else:
                    execution_info['error'] = 'Trade execution failed'
                    
            except Exception as e:
                execution_info['error'] = f'Execution error: {str(e)}'
                logger.error(f"Trade execution error: {e}")
        
        # Update portfolio prices
        self.portfolio.update_position_prices({symbol: current_price})
        
        # Update performance metrics
        self._update_performance_metrics()
        
        return execution_info
    
    def _risk_check(self, symbol: str, target_position: float, price: float) -> bool:
        """
        Perform risk checks before trade execution
        
        Args:
            symbol: Trading symbol
            target_position: Target position size
            price: Current price
            
        Returns:
            True if risk checks pass
        """
        # Check maximum position size
        if abs(target_position) > self.max_position_size:
            logger.warning(f"Position size {target_position} exceeds maximum {self.max_position_size}")
            return False
        
        # Check total leverage
        total_exposure = sum(abs(pos.size * pos.current_price) for pos in self.portfolio.positions.values())
        total_exposure += abs(target_position * price)
        
        leverage = total_exposure / max(self.portfolio.total_value, 1000)  # Avoid division by zero
        
        if leverage > self.max_total_leverage:
            logger.warning(f"Total leverage {leverage:.2f} exceeds maximum {self.max_total_leverage}")
            return False
        
        # Check cash availability for long positions
        if target_position > 0:
            required_cash = target_position * price * (1 + self.commission_rate)
            if required_cash > self.portfolio.cash:
                logger.warning(f"Insufficient cash: required {required_cash}, available {self.portfolio.cash}")
                return False
        
        # Use external risk manager if available
        if self.risk_manager:
            try:
                risk_check = self.risk_manager.check_trade_risk(
                    symbol=symbol,
                    size=target_position,
                    price=price,
                    portfolio_value=self.portfolio.total_value
                )
                if not risk_check.get('approved', True):
                    logger.warning(f"Risk manager rejected trade: {risk_check.get('reason', 'Unknown')}")
                    return False
            except Exception as e:
                logger.warning(f"Risk manager check failed: {e}")
        
        return True
    
    def _execute_trade(self, symbol: str, position_change: float, price: float) -> bool:
        """
        Execute trade and update portfolio
        
        Args:
            symbol: Trading symbol
            position_change: Position change amount
            price: Execution price
            
        Returns:
            True if trade executed successfully
        """
        try:
            current_position = self.portfolio.get_position_size(symbol)
            new_position_size = current_position + position_change
            
            # Calculate commission
            commission = abs(position_change) * price * self.commission_rate
            
            # Update cash
            cash_change = -position_change * price - commission
            self.portfolio.cash += cash_change
            self.portfolio.total_commission += commission
            
            # Update position
            if abs(new_position_size) < 1e-8:  # Close position
                if symbol in self.portfolio.positions:
                    # Record completed trade
                    pos = self.portfolio.positions[symbol]
                    trade = Trade(
                        symbol=symbol,
                        side='sell' if pos.size > 0 else 'buy',
                        size=abs(pos.size),
                        entry_price=pos.entry_price,
                        exit_price=price,
                        entry_time=pos.entry_time,
                        exit_time=datetime.now(),
                        pnl=pos.size * (price - pos.entry_price),
                        commission=commission
                    )
                    
                    self.portfolio.trades.append(trade)
                    self.portfolio.total_trades += 1
                    
                    if trade.pnl > 0:
                        self.portfolio.winning_trades += 1
                    
                    # Remove position
                    del self.portfolio.positions[symbol]
            else:
                # Update or create position
                if symbol in self.portfolio.positions:
                    # Update existing position
                    pos = self.portfolio.positions[symbol]
                    
                    # Check if position direction is changing
                    if pos.size * new_position_size < 0:
                        # Direction change - record partial trade
                        exit_size = -pos.size
                        trade = Trade(
                            symbol=symbol,
                            side='sell' if pos.size > 0 else 'buy',
                            size=abs(exit_size),
                            entry_price=pos.entry_price,
                            exit_price=price,
                            entry_time=pos.entry_time,
                            exit_time=datetime.now(),
                            pnl=exit_size * (price - pos.entry_price),
                            commission=commission * abs(exit_size) / abs(position_change)
                        )
                        
                        self.portfolio.trades.append(trade)
                        self.portfolio.total_trades += 1
                        
                        if trade.pnl > 0:
                            self.portfolio.winning_trades += 1
                        
                        # Create new position
                        self.portfolio.positions[symbol] = Position(
                            symbol=symbol,
                            size=new_position_size,
                            entry_price=price,
                            entry_time=datetime.now(),
                            current_price=price
                        )
                    else:
                        # Same direction - average entry price
                        total_value = pos.size * pos.entry_price + position_change * price
                        pos.size = new_position_size
                        pos.entry_price = total_value / new_position_size if new_position_size != 0 else price
                        pos.current_price = price
                else:
                    # Create new position
                    self.portfolio.positions[symbol] = Position(
                        symbol=symbol,
                        size=new_position_size,
                        entry_price=price,
                        entry_time=datetime.now(),
                        current_price=price
                    )
            
            return True
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return False
    
    def _update_performance_metrics(self):
        """Update performance metrics"""
        if len(self.portfolio.equity_curve) < 2:
            return
        
        # Calculate returns
        current_return = (self.portfolio.equity_curve[-1] - self.portfolio.equity_curve[-2]) / self.portfolio.equity_curve[-2]
        self.returns_history.append(current_return)
        
        # Calculate Sharpe ratio if enough history
        if len(self.returns_history) >= 30:
            returns = np.array(list(self.returns_history))
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return > 0:
                # Annualized Sharpe ratio
                annual_return = mean_return * 252
                annual_volatility = std_return * np.sqrt(252)
                sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility
                self.sharpe_history.append(sharpe_ratio)
            
            # Track volatility
            self.volatility_history.append(std_return * np.sqrt(252))
    
    def get_portfolio_state(self) -> Dict:
        """Get current portfolio state for RL agent"""
        portfolio_info = self.portfolio.get_info()
        
        # Add performance metrics
        portfolio_info.update({
            'sharpe_ratio': np.mean(list(self.sharpe_history)) if self.sharpe_history else 0.0,
            'volatility': np.mean(list(self.volatility_history)) if self.volatility_history else 0.0,
            'current_return': self.returns_history[-1] if self.returns_history else 0.0,
            'avg_return': np.mean(list(self.returns_history)) if self.returns_history else 0.0,
            'return_std': np.std(list(self.returns_history)) if len(self.returns_history) > 1 else 0.0
        })
        
        return portfolio_info
    
    def reset(self):
        """Reset portfolio to initial state"""
        self.portfolio = Portfolio(self.initial_cash, self.commission_rate)
        self.returns_history.clear()
        self.sharpe_history.clear()
        self.volatility_history.clear()
        self.action_history.clear()
        
        logger.info("Portfolio reset to initial state")
    
    def get_performance_summary(self) -> Dict:
        """Get detailed performance summary"""
        if not self.portfolio.trades:
            return {'no_trades': True}
        
        # Calculate trade statistics
        trade_pnls = [trade.pnl for trade in self.portfolio.trades]
        winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
        losing_trades = [pnl for pnl in trade_pnls if pnl < 0]
        
        # Risk metrics
        returns = list(self.returns_history) if self.returns_history else [0]
        
        summary = {
            'total_return': self.portfolio.total_return,
            'total_trades': len(self.portfolio.trades),
            'win_rate': len(winning_trades) / len(self.portfolio.trades),
            'avg_win': np.mean(winning_trades) if winning_trades else 0,
            'avg_loss': np.mean(losing_trades) if losing_trades else 0,
            'profit_factor': abs(sum(winning_trades) / sum(losing_trades)) if losing_trades and sum(losing_trades) != 0 else float('inf'),
            'max_drawdown': self.portfolio.max_drawdown,
            'sharpe_ratio': np.mean(list(self.sharpe_history)) if self.sharpe_history else 0,
            'volatility': np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0,
            'calmar_ratio': self.portfolio.total_return / max(self.portfolio.max_drawdown, 0.01),
            'total_commission': self.portfolio.total_commission,
            'final_value': self.portfolio.total_value
        }
        
        return summary
    
    def export_trades(self) -> pd.DataFrame:
        """Export trades to DataFrame"""
        if not self.portfolio.trades:
            return pd.DataFrame()
        
        trades_data = []
        for trade in self.portfolio.trades:
            trades_data.append({
                'symbol': trade.symbol,
                'side': trade.side,
                'size': trade.size,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'pnl': trade.pnl,
                'return_pct': trade.return_pct,
                'commission': trade.commission
            })
        
        return pd.DataFrame(trades_data)
    
    def export_equity_curve(self) -> pd.DataFrame:
        """Export equity curve to DataFrame"""
        return pd.DataFrame({
            'timestamp': self.portfolio.timestamps,
            'equity': self.portfolio.equity_curve,
            'drawdown': [(self.portfolio.peak_value - eq) / self.portfolio.peak_value for eq in self.portfolio.equity_curve]
        })