# Yahoo Finance Data Collector for BYJY-Trader
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

class YahooCollector(BaseCollector):
    """
    Yahoo Finance data collector for historical stock and ETF data.
    Uses Yahoo Finance public API for free historical data.
    """
    
    def __init__(self, rate_limit: float = 2.0):
        """
        Initialize Yahoo Finance collector.
        
        Args:
            rate_limit: Rate limit in seconds (conservative for public API)
        """
        super().__init__("Yahoo Finance", rate_limit)
        self.base_url = "https://query1.finance.yahoo.com"
        self.session = None
        
        # Yahoo interval mapping
        self.interval_mapping = {
            '1m': '1m', '2m': '2m', '5m': '5m', '15m': '15m', '30m': '30m', '60m': '60m',
            '90m': '90m', '1h': '1h', '1d': '1d', '5d': '5d', '1wk': '1wk',
            '1mo': '1mo', '3mo': '3mo'
        }
        
        # Common symbol suffixes for different markets
        self.market_suffixes = {
            'US': '',  # No suffix for US markets
            'CA': '.TO',  # Toronto Stock Exchange
            'UK': '.L',   # London Stock Exchange
            'DE': '.DE',  # Frankfurt Stock Exchange
            'FR': '.PA',  # Paris Stock Exchange
            'JP': '.T',   # Tokyo Stock Exchange
        }
    
    async def connect(self) -> bool:
        """
        Establish connection to Yahoo Finance API.
        
        Returns:
            bool: True if connection successful
        """
        try:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            )
            
            # Test connection with a simple query
            test_symbol = 'AAPL'
            url = f"{self.base_url}/v8/finance/chart/{test_symbol}"
            params = {
                'interval': '1d',
                'range': '1d'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'chart' in data and data['chart']['result']:
                        self.is_connected = True
                        logger.info("Connected to Yahoo Finance API")
                        return True
                    else:
                        logger.error("Yahoo Finance API returned invalid response")
                        return False
                else:
                    logger.error(f"Failed to connect to Yahoo Finance API: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error connecting to Yahoo Finance API: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from Yahoo Finance API."""
        if self.session:
            await self.session.close()
            self.session = None
        self.is_connected = False
        logger.info("Disconnected from Yahoo Finance API")
    
    async def get_available_symbols(self) -> List[str]:
        """
        Get list of popular symbols (Yahoo Finance doesn't provide full symbol list).
        
        Returns:
            List[str]: Popular symbols
        """
        # Return popular stocks, ETFs, and indices
        popular_symbols = [
            # Major US Stocks
            'AAPL', 'GOOGL', 'GOOG', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'BABA',
            'SPY', 'QQQ', 'IWM', 'GLD', 'SLV', 'VTI', 'VOO', 'BTC-USD', 'ETH-USD',
            # Major Indices
            '^GSPC', '^DJI', '^IXIC', '^RUT', '^VIX',
            # Major Forex
            'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X',
            # Commodities
            'GC=F', 'SI=F', 'CL=F', 'NG=F'
        ]
        
        logger.info(f"Retrieved {len(popular_symbols)} popular symbols for Yahoo Finance")
        return popular_symbols
    
    async def search_symbol(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for symbols using Yahoo Finance search API.
        
        Args:
            query: Search query
            
        Returns:
            List[Dict]: Search results
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to Yahoo Finance API")
        
        try:
            url = f"{self.base_url}/v1/finance/search"
            params = {
                'q': query,
                'quotes_count': 10,
                'news_count': 0
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('quotes', [])
                else:
                    raise Exception(f"Search API error: {response.status}")
                    
        except Exception as e:
            logger.error(f"Error searching symbols: {e}")
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
        Fetch historical data from Yahoo Finance.
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'BTC-USD')
            interval: Time interval
            start_date: Start date
            end_date: End date (optional)
            limit: Record limit (not used by Yahoo API)
            
        Returns:
            pd.DataFrame: OHLCV data
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to Yahoo Finance API")
        
        if interval not in self.interval_mapping:
            raise ValueError(f"Unsupported interval: {interval}")
        
        try:
            # Calculate period parameters
            end_date = end_date or datetime.now()
            period1 = int(start_date.timestamp())
            period2 = int(end_date.timestamp())
            
            url = f"{self.base_url}/v8/finance/chart/{symbol.upper()}"
            params = {
                'period1': period1,
                'period2': period2,
                'interval': self.interval_mapping[interval],
                'events': 'div,splits',
                'includeAdjustedClose': 'true'
            }
            
            logger.info(f"Fetching Yahoo Finance data: {symbol} {interval} from {start_date}")
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if not data.get('chart', {}).get('result'):
                        logger.warning(f"No data returned for {symbol}")
                        return pd.DataFrame()
                    
                    result = data['chart']['result'][0]
                    
                    if not result.get('timestamp'):
                        logger.warning(f"No timestamp data for {symbol}")
                        return pd.DataFrame()
                    
                    # Extract OHLCV data
                    timestamps = result['timestamp']
                    quotes = result.get('indicators', {}).get('quote', [{}])[0]
                    
                    df_data = {
                        'timestamp': [datetime.fromtimestamp(ts) for ts in timestamps],
                        'open': quotes.get('open', []),
                        'high': quotes.get('high', []),
                        'low': quotes.get('low', []),
                        'close': quotes.get('close', []),
                        'volume': quotes.get('volume', [])
                    }
                    
                    df = pd.DataFrame(df_data)
                    
                    # Remove rows with all NaN values
                    df = df.dropna(subset=['open', 'high', 'low', 'close'])
                    
                    if df.empty:
                        logger.warning(f"No valid data for {symbol}")
                        return pd.DataFrame()
                    
                    # Return standardized format
                    result_df = self.standardize_dataframe(df)
                    logger.info(f"Collected {len(result_df)} records for {symbol}")
                    return result_df
                    
                else:
                    error_text = await response.text()
                    raise Exception(f"API error {response.status}: {error_text}")
                    
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance data for {symbol}: {e}")
            raise
    
    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get current quote for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict: Quote data
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to Yahoo Finance API")
        
        try:
            url = f"{self.base_url}/v7/finance/quote"
            params = {'symbols': symbol.upper()}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('quoteResponse', {}).get('result'):
                        return data['quoteResponse']['result'][0]
                    else:
                        return {}
                else:
                    raise Exception(f"Quote API error: {response.status}")
                    
        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {e}")
            raise
    
    async def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if symbol exists by attempting to get its quote.
        
        Args:
            symbol: Symbol to validate
            
        Returns:
            bool: True if symbol is valid
        """
        try:
            quote = await self.get_quote(symbol)
            return bool(quote and 'symbol' in quote)
        except Exception as e:
            logger.error(f"Error validating symbol {symbol}: {e}")
            return False