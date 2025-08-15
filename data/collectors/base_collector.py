# Base Collector Class for BYJY-Trader
# Phase 2.1 - Historical Data Collection System

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from core.path_utils import get_project_root
from core.logger import get_logger

logger = get_logger(__name__)

class BaseCollector(ABC):
    """
    Base class for all data collectors.
    Provides common functionality for historical data collection.
    """
    
    def __init__(self, name: str, rate_limit: float = 1.0):
        """
        Initialize base collector.
        
        Args:
            name: Collector name
            rate_limit: Rate limit in seconds between requests
        """
        self.name = name
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.is_connected = False
        self.error_count = 0
        self.max_retries = 3
        self.timeout = 30
        
        logger.info(f"Initialized {self.name} collector with rate limit {rate_limit}s")
    
    async def _rate_limit_check(self):
        """Enforce rate limiting between requests."""
        current_time = datetime.now().timestamp()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit:
            sleep_time = self.rate_limit - time_since_last
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = datetime.now().timestamp()
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to data source.
        
        Returns:
            bool: True if connection successful
        """
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from data source."""
        pass
    
    @abstractmethod
    async def get_historical_data(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch historical data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            interval: Time interval ('1m', '5m', '1h', '1d', etc.)
            start_date: Start date for data collection
            end_date: End date for data collection (optional)
            limit: Maximum number of records (optional)
            
        Returns:
            pd.DataFrame: OHLCV data with standardized columns
        """
        pass
    
    @abstractmethod
    async def get_available_symbols(self) -> List[str]:
        """
        Get list of available trading symbols.
        
        Returns:
            List[str]: Available symbols
        """
        pass
    
    async def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if symbol is available.
        
        Args:
            symbol: Symbol to validate
            
        Returns:
            bool: True if symbol is valid
        """
        try:
            available_symbols = await self.get_available_symbols()
            return symbol.upper() in [s.upper() for s in available_symbols]
        except Exception as e:
            logger.error(f"Error validating symbol {symbol}: {e}")
            return False
    
    def standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize dataframe columns and format.
        
        Args:
            df: Raw dataframe from API
            
        Returns:
            pd.DataFrame: Standardized dataframe
        """
        if df.empty:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Ensure standard column names
        column_mapping = {
            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume',
            'open_time': 'timestamp', 'close_time': 'timestamp',
            'Open Time': 'timestamp', 'Close Time': 'timestamp'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Ensure required columns exist
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                if col == 'timestamp' and df.index.name in ['date', 'datetime', 'time']:
                    df.reset_index(inplace=True)
                    df.rename(columns={df.columns[0]: 'timestamp'}, inplace=True)
                else:
                    logger.warning(f"Missing column {col} in dataframe")
                    df[col] = np.nan
        
        # Convert data types
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception as e:
            logger.error(f"Error converting data types: {e}")
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df[required_columns]
    
    async def collect_with_retry(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Collect data with retry logic.
        
        Args:
            symbol: Trading symbol
            interval: Time interval
            start_date: Start date
            end_date: End date (optional)
            limit: Record limit (optional)
            
        Returns:
            pd.DataFrame: Collected data
        """
        for attempt in range(self.max_retries):
            try:
                await self._rate_limit_check()
                data = await self.get_historical_data(symbol, interval, start_date, end_date, limit)
                self.error_count = 0  # Reset error count on success
                return data
            except Exception as e:
                self.error_count += 1
                logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {e}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Max retries reached for {symbol}")
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return pd.DataFrame()
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get collector status information.
        
        Returns:
            Dict: Status information
        """
        return {
            'name': self.name,
            'connected': self.is_connected,
            'error_count': self.error_count,
            'rate_limit': self.rate_limit,
            'last_request': self.last_request_time
        }