# Feature Engineering Engine for BYJY-Trader
# Phase 2.1 - Automatic Technical Analysis Features

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import asyncio
from datetime import datetime, timedelta
from core.logger import get_logger

logger = get_logger(__name__)

class FeatureEngine:
    """
    Automatic feature engineering for trading data.
    Generates technical indicators and derived features.
    """
    
    def __init__(self):
        """Initialize the Feature Engineering Engine."""
        self.name = "Feature Engine"
        self.supported_features = [
            'sma', 'ema', 'rsi', 'macd', 'bollinger_bands', 'stochastic',
            'williams_r', 'cci', 'atr', 'adx', 'obv', 'vwap',
            'price_changes', 'volatility', 'momentum', 'volume_features'
        ]
        logger.info(f"Initialized Feature Engine with {len(self.supported_features)} feature types")
    
    def generate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all available technical features for the dataset.
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: Enhanced DataFrame with technical features
        """
        if df.empty or len(df) < 50:  # Need minimum data for calculations
            logger.warning("Insufficient data for feature generation")
            return df
        
        logger.info(f"Generating features for {len(df)} data points")
        
        # Make a copy to avoid modifying original data
        enhanced_df = df.copy()
        
        try:
            # Price-based features
            enhanced_df = self._add_price_features(enhanced_df)
            
            # Moving averages
            enhanced_df = self._add_moving_averages(enhanced_df)
            
            # Momentum indicators
            enhanced_df = self._add_momentum_indicators(enhanced_df)
            
            # Volatility indicators
            enhanced_df = self._add_volatility_indicators(enhanced_df)
            
            # Volume indicators
            enhanced_df = self._add_volume_indicators(enhanced_df)
            
            # Trend indicators
            enhanced_df = self._add_trend_indicators(enhanced_df)
            
            # Statistical features
            enhanced_df = self._add_statistical_features(enhanced_df)
            
            logger.info(f"Generated {enhanced_df.shape[1] - df.shape[1]} new features")
            return enhanced_df
            
        except Exception as e:
            logger.error(f"Error generating features: {e}")
            return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic price-based features."""
        # Price changes
        df['price_change'] = df['close'].pct_change()
        df['price_change_abs'] = df['close'].diff()
        
        # Log returns
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # OHLC ratios
        df['hl_ratio'] = (df['high'] - df['low']) / df['close']
        df['oc_ratio'] = (df['open'] - df['close']) / df['close']
        df['body_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
        
        # Gaps
        df['gap_up'] = (df['open'] > df['close'].shift(1)).astype(int)
        df['gap_down'] = (df['open'] < df['close'].shift(1)).astype(int)
        
        return df
    
    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add moving average features."""
        periods = [5, 10, 20, 50, 100, 200]
        
        for period in periods:
            if len(df) >= period:
                # Simple Moving Average
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
                
                # Exponential Moving Average
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
                
                # Price position relative to MA
                df[f'price_vs_sma_{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}']
                df[f'price_vs_ema_{period}'] = (df['close'] - df[f'ema_{period}']) / df[f'ema_{period}']
        
        # Moving average crossovers
        if 'sma_20' in df.columns and 'sma_50' in df.columns:
            df['sma_20_50_cross'] = (df['sma_20'] > df['sma_50']).astype(int)
        
        return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based indicators."""
        # RSI (Relative Strength Index)
        df = self._calculate_rsi(df)
        
        # MACD
        df = self._calculate_macd(df)
        
        # Stochastic Oscillator
        df = self._calculate_stochastic(df)
        
        # Williams %R
        df = self._calculate_williams_r(df)
        
        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            if len(df) > period:
                df[f'roc_{period}'] = df['close'].pct_change(periods=period)
        
        return df
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Relative Strength Index."""
        if len(df) < period + 1:
            return df
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # RSI-based signals
        df[f'rsi_{period}_overbought'] = (df[f'rsi_{period}'] > 70).astype(int)
        df[f'rsi_{period}_oversold'] = (df[f'rsi_{period}'] < 30).astype(int)
        
        return df
    
    def _calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        if len(df) < slow + signal:
            return df
        
        ema_fast = df['close'].ewm(span=fast).mean()
        ema_slow = df['close'].ewm(span=slow).mean()
        
        df['macd_line'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd_line'].ewm(span=signal).mean()
        df['macd_histogram'] = df['macd_line'] - df['macd_signal']
        
        # MACD crossover signals
        df['macd_bullish_cross'] = ((df['macd_line'] > df['macd_signal']) & 
                                    (df['macd_line'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
        df['macd_bearish_cross'] = ((df['macd_line'] < df['macd_signal']) & 
                                    (df['macd_line'].shift(1) >= df['macd_signal'].shift(1))).astype(int)
        
        return df
    
    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Calculate Stochastic Oscillator."""
        if len(df) < k_period:
            return df
        
        lowest_low = df['low'].rolling(window=k_period).min()
        highest_high = df['high'].rolling(window=k_period).max()
        
        df['stoch_k'] = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
        
        # Stochastic signals
        df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)
        df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)
        
        return df
    
    def _calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Williams %R."""
        if len(df) < period:
            return df
        
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()
        
        df[f'williams_r_{period}'] = -100 * (highest_high - df['close']) / (highest_high - lowest_low)
        
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based indicators."""
        # Bollinger Bands
        df = self._calculate_bollinger_bands(df)
        
        # Average True Range (ATR)
        df = self._calculate_atr(df)
        
        # Historical Volatility
        for period in [10, 20, 30]:
            if len(df) > period:
                df[f'volatility_{period}'] = df['log_return'].rolling(window=period).std() * np.sqrt(252)
        
        return df
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        if len(df) < period:
            return df
        
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        
        df[f'bb_upper_{period}'] = sma + (std * std_dev)
        df[f'bb_middle_{period}'] = sma
        df[f'bb_lower_{period}'] = sma - (std * std_dev)
        
        # Bollinger Band position
        df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
        
        # Bollinger Band squeeze
        df[f'bb_squeeze_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / df[f'bb_middle_{period}']
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average True Range."""
        if len(df) < 2:
            return df
        
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df[f'atr_{period}'] = true_range.rolling(window=period).mean()
        
        # Normalized ATR
        df[f'atr_normalized_{period}'] = df[f'atr_{period}'] / df['close']
        
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""
        if 'volume' not in df.columns or df['volume'].isna().all():
            return df
        
        # Volume moving averages
        for period in [10, 20, 50]:
            if len(df) >= period:
                df[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()
                df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_sma_{period}']
        
        # On-Balance Volume (OBV)
        df['obv'] = self._calculate_obv(df)
        
        # Volume Price Trend (VPT)
        df['vpt'] = self._calculate_vpt(df)
        
        # VWAP (Volume Weighted Average Price)
        df['vwap'] = self._calculate_vwap(df)
        
        return df
    
    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume."""
        obv = pd.Series(index=df.index, dtype='float64')
        obv.iloc[0] = df['volume'].iloc[0]
        
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def _calculate_vpt(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Price Trend."""
        vpt = pd.Series(index=df.index, dtype='float64')
        vpt.iloc[0] = 0
        
        for i in range(1, len(df)):
            price_change = (df['close'].iloc[i] - df['close'].iloc[i-1]) / df['close'].iloc[i-1]
            vpt.iloc[i] = vpt.iloc[i-1] + df['volume'].iloc[i] * price_change
        
        return vpt
    
    def _calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        cumulative_volume = df['volume'].cumsum()
        cumulative_tp_volume = (typical_price * df['volume']).cumsum()
        
        vwap = cumulative_tp_volume / cumulative_volume
        return vwap.fillna(df['close'])
    
    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-following indicators."""
        # ADX (Average Directional Index)
        df = self._calculate_adx(df)
        
        # Commodity Channel Index (CCI)
        df = self._calculate_cci(df)
        
        # Parabolic SAR approximation
        df = self._calculate_parabolic_sar(df)
        
        return df
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average Directional Index (simplified version)."""
        if len(df) < period + 1:
            return df
        
        # Calculate directional movement
        df['dm_plus'] = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
                                 np.maximum(df['high'] - df['high'].shift(1), 0), 0)
        df['dm_minus'] = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
                                  np.maximum(df['low'].shift(1) - df['low'], 0), 0)
        
        # True Range
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Smoothed values
        atr = tr.rolling(window=period).mean()
        di_plus = 100 * (df['dm_plus'].rolling(window=period).mean() / atr)
        di_minus = 100 * (df['dm_minus'].rolling(window=period).mean() / atr)
        
        # ADX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        df[f'adx_{period}'] = dx.rolling(window=period).mean()
        
        return df
    
    def _calculate_cci(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Calculate Commodity Channel Index."""
        if len(df) < period:
            return df
        
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        
        df[f'cci_{period}'] = (typical_price - sma_tp) / (0.015 * mad)
        
        return df
    
    def _calculate_parabolic_sar(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Parabolic SAR (simplified version)."""
        if len(df) < 5:
            return df
        
        # Simplified SAR calculation
        sar = pd.Series(index=df.index, dtype='float64')
        af = 0.02  # Acceleration factor
        
        # Initialize
        sar.iloc[0] = df['low'].iloc[0]
        
        for i in range(1, len(df)):
            if i == 1:
                sar.iloc[i] = df['low'].iloc[0]
            else:
                # Simplified calculation (proper SAR is more complex)
                sar.iloc[i] = sar.iloc[i-1] + af * (df['high'].iloc[i-1] - sar.iloc[i-1])
        
        df['parabolic_sar'] = sar
        df['sar_signal'] = (df['close'] > df['parabolic_sar']).astype(int)
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features."""
        # Price quantiles
        for window in [20, 50]:
            if len(df) >= window:
                df[f'price_rank_{window}'] = df['close'].rolling(window=window).rank(pct=True)
        
        # Z-scores
        for window in [20, 50]:
            if len(df) >= window:
                rolling_mean = df['close'].rolling(window=window).mean()
                rolling_std = df['close'].rolling(window=window).std()
                df[f'price_zscore_{window}'] = (df['close'] - rolling_mean) / rolling_std
        
        # Fractal patterns (simplified)
        df['local_high'] = ((df['high'] > df['high'].shift(1)) & 
                           (df['high'] > df['high'].shift(-1))).astype(int)
        df['local_low'] = ((df['low'] < df['low'].shift(1)) & 
                          (df['low'] < df['low'].shift(-1))).astype(int)
        
        return df
    
    def get_feature_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics of generated features.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Dict: Feature summary statistics
        """
        feature_columns = [col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        summary = {
            'total_features': len(feature_columns),
            'feature_categories': {
                'price_features': len([col for col in feature_columns if 'price' in col or 'return' in col]),
                'moving_averages': len([col for col in feature_columns if 'sma_' in col or 'ema_' in col]),
                'momentum': len([col for col in feature_columns if any(x in col for x in ['rsi', 'macd', 'stoch', 'roc'])]),
                'volatility': len([col for col in feature_columns if any(x in col for x in ['bb_', 'atr_', 'volatility'])]),
                'volume': len([col for col in feature_columns if 'volume' in col or col in ['obv', 'vpt', 'vwap']]),
                'trend': len([col for col in feature_columns if any(x in col for x in ['adx', 'cci', 'sar'])]),
                'statistical': len([col for col in feature_columns if any(x in col for x in ['rank', 'zscore', 'local'])])
            },
            'data_quality': {
                'total_rows': len(df),
                'missing_values': df[feature_columns].isnull().sum().sum(),
                'complete_rows': len(df.dropna())
            }
        }
        
        return summary