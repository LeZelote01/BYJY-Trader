# Binance Data Collector for BYJY-Trader
# Phase 2.1 - Historical Data Collection System

import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import json
from .base_collector import BaseCollector
from core.logger import get_logger

logger = get_logger(__name__)

class BinanceCollector(BaseCollector):
    """
    Binance data collector for historical cryptocurrency data.
    Uses Binance public API (no API key required for historical data).
    """
    
    def __init__(self, rate_limit: float = 1.0):
        """
        Initialize Binance collector.
        
        Args:
            rate_limit: Rate limit in seconds (Binance allows 1200 requests/minute)
        """
        super().__init__("Binance", rate_limit)
        self.base_url = "https://api.binance.com"
        self.session = None
        
        # Binance interval mapping
        self.interval_mapping = {
            '1m': '1m', '3m': '3m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '8h', '12h': '12h',
            '1d': '1d', '3d': '3d', '1w': '1w', '1M': '1M'
        }
    
    async def connect(self) -> bool:
        """
        Establish connection to Binance API.
        
        Returns:
            bool: True if connection successful
        """
        try:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
            
            # Test connection with server time
            url = f"{self.base_url}/api/v3/time"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    server_time = data['serverTime']
                    self.is_connected = True
                    logger.info(f"Connected to Binance API. Server time: {server_time}")
                    return True
                else:
                    logger.error(f"Failed to connect to Binance API: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error connecting to Binance API: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from Binance API."""
        if self.session:
            await self.session.close()
            self.session = None
        self.is_connected = False
        logger.info("Disconnected from Binance API")
    
    async def get_available_symbols(self) -> List[str]:
        """
        Get list of available trading symbols from Binance.
        
        Returns:
            List[str]: Available symbols
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to Binance API")
        
        try:
            url = f"{self.base_url}/api/v3/exchangeInfo"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    symbols = [symbol['symbol'] for symbol in data['symbols'] 
                             if symbol['status'] == 'TRADING']
                    logger.info(f"Retrieved {len(symbols)} active symbols from Binance")
                    return symbols
                else:
                    raise Exception(f"API error: {response.status}")
                    
        except Exception as e:
            logger.error(f"Error getting symbols from Binance: {e}")
            raise
    
    async def get_historical_data(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch historical kline data from Binance.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            interval: Time interval
            start_date: Start date
            end_date: End date (optional)
            limit: Record limit (1-1000, default 500)
            
        Returns:
            pd.DataFrame: OHLCV data
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to Binance API")
        
        if interval not in self.interval_mapping:
            raise ValueError(f"Unsupported interval: {interval}")
        
        try:
            # Prepare parameters
            params = {
                'symbol': symbol.upper(),
                'interval': self.interval_mapping[interval],
                'startTime': int(start_date.timestamp() * 1000),
            }
            
            if end_date:
                params['endTime'] = int(end_date.timestamp() * 1000)
            
            if limit:
                params['limit'] = min(limit, 1000)  # Binance max is 1000
            else:
                params['limit'] = 500
            
            url = f"{self.base_url}/api/v3/klines"
            
            logger.info(f"Fetching Binance data: {symbol} {interval} from {start_date}")
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if not data:
                        logger.warning(f"No data returned for {symbol}")
                        return pd.DataFrame()
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(data, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_volume', 'count', 'taker_buy_volume',
                        'taker_buy_quote_volume', 'ignore'
                    ])
                    
                    # Convert timestamp and data types
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col])
                    
                    # Return standardized format
                    result_df = self.standardize_dataframe(df)
                    logger.info(f"Collected {len(result_df)} records for {symbol}")
                    return result_df
                    
                else:
                    error_text = await response.text()
                    raise Exception(f"API error {response.status}: {error_text}")
                    
        except Exception as e:
            logger.error(f"Error fetching Binance data for {symbol}: {e}")
            raise
    
    async def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get detailed information about a trading symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict: Symbol information
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to Binance API")
        
        try:
            url = f"{self.base_url}/api/v3/exchangeInfo"
            params = {'symbol': symbol.upper()}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['symbols'][0] if data['symbols'] else {}
                else:
                    raise Exception(f"API error: {response.status}")
                    
        except Exception as e:
            logger.error(f"Error getting symbol info for {symbol}: {e}")
            raise
    
    async def get_24h_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get 24h ticker statistics for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict: 24h ticker data
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to Binance API")
        
        try:
            url = f"{self.base_url}/api/v3/ticker/24hr"
            params = {'symbol': symbol.upper()}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    raise Exception(f"API error: {response.status}")
                    
        except Exception as e:
            logger.error(f"Error getting 24h ticker for {symbol}: {e}")
            raise