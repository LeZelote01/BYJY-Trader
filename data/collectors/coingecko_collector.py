# CoinGecko Data Collector for BYJY-Trader  
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

class CoinGeckoCollector(BaseCollector):
    """
    CoinGecko data collector for historical cryptocurrency data.
    Uses CoinGecko public API for free crypto market data.
    """
    
    def __init__(self, rate_limit: float = 1.5):
        """
        Initialize CoinGecko collector.
        
        Args:
            rate_limit: Rate limit in seconds (CoinGecko free tier: 50 calls/minute)
        """
        super().__init__("CoinGecko", rate_limit)
        self.base_url = "https://api.coingecko.com/api/v3"
        self.session = None
        self.coins_list = {}  # Cache for coin ID to symbol mapping
        
        # CoinGecko interval mapping (days parameter)
        self.interval_mapping = {
            '1d': 1,
            '7d': 7,
            '14d': 14,
            '30d': 30,
            '90d': 90,
            '180d': 180,
            '365d': 365,
            'max': 'max'
        }
    
    async def connect(self) -> bool:
        """
        Establish connection to CoinGecko API.
        
        Returns:
            bool: True if connection successful
        """
        try:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers={'User-Agent': 'BYJY-Trader/2.1.0'}
            )
            
            # Test connection with ping endpoint
            url = f"{self.base_url}/ping"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('gecko_says') == '(V3) To the Moon!':
                        self.is_connected = True
                        logger.info("Connected to CoinGecko API")
                        
                        # Load coins list for symbol mapping
                        await self._load_coins_list()
                        return True
                    else:
                        logger.error("CoinGecko API returned unexpected response")
                        return False
                else:
                    logger.error(f"Failed to connect to CoinGecko API: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error connecting to CoinGecko API: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from CoinGecko API."""
        if self.session:
            await self.session.close()
            self.session = None
        self.is_connected = False
        logger.info("Disconnected from CoinGecko API")
    
    async def _load_coins_list(self):
        """Load coins list for symbol to ID mapping."""
        try:
            url = f"{self.base_url}/coins/list"
            async with self.session.get(url) as response:
                if response.status == 200:
                    coins = await response.json()
                    self.coins_list = {
                        coin['symbol'].upper(): coin['id'] for coin in coins
                    }
                    logger.info(f"Loaded {len(self.coins_list)} coins from CoinGecko")
                else:
                    logger.error(f"Error loading coins list: {response.status}")
        except Exception as e:
            logger.error(f"Error loading coins list: {e}")
    
    def _symbol_to_id(self, symbol: str) -> str:
        """
        Convert symbol to CoinGecko coin ID.
        
        Args:
            symbol: Coin symbol (e.g., 'BTC')
            
        Returns:
            str: CoinGecko coin ID
        """
        symbol_upper = symbol.upper()
        
        # Remove common suffixes (e.g., 'USDT' from 'BTCUSDT')
        if symbol_upper.endswith('USDT'):
            base_symbol = symbol_upper[:-4]
        elif symbol_upper.endswith('USD'):
            base_symbol = symbol_upper[:-3]
        elif symbol_upper.endswith('BTC'):
            base_symbol = symbol_upper[:-3]
        elif symbol_upper.endswith('ETH'):
            base_symbol = symbol_upper[:-3]
        else:
            base_symbol = symbol_upper
        
        return self.coins_list.get(base_symbol, symbol.lower())
    
    async def get_available_symbols(self) -> List[str]:
        """
        Get list of available cryptocurrency symbols.
        
        Returns:
            List[str]: Available symbols
        """
        if not self.coins_list:
            await self._load_coins_list()
        
        # Return top 100 by market cap
        try:
            url = f"{self.base_url}/coins/markets"
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': 100,
                'page': 1,
                'sparkline': False
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    symbols = [coin['symbol'].upper() for coin in data]
                    logger.info(f"Retrieved {len(symbols)} top symbols from CoinGecko")
                    return symbols
                else:
                    raise Exception(f"API error: {response.status}")
                    
        except Exception as e:
            logger.error(f"Error getting symbols from CoinGecko: {e}")
            # Return cached symbols if API fails
            return list(self.coins_list.keys())[:100]
    
    async def get_historical_data(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch historical data from CoinGecko.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC', 'BTCUSDT')
            interval: Time interval (1d, 7d, 30d, etc.)
            start_date: Start date
            end_date: End date (optional, calculated from interval)
            limit: Record limit (not used by CoinGecko)
            
        Returns:
            pd.DataFrame: Price data (CoinGecko only provides price, not full OHLCV)
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to CoinGecko API")
        
        try:
            coin_id = self._symbol_to_id(symbol)
            
            # Determine days parameter
            if interval in self.interval_mapping:
                days = self.interval_mapping[interval]
            else:
                # Calculate days from start_date to end_date
                end_date = end_date or datetime.now()
                days = (end_date - start_date).days
                days = max(1, days)  # Minimum 1 day
            
            url = f"{self.base_url}/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'daily' if days > 90 else 'hourly'
            }
            
            logger.info(f"Fetching CoinGecko data: {symbol} ({coin_id}) for {days} days")
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if not data.get('prices'):
                        logger.warning(f"No price data returned for {symbol}")
                        return pd.DataFrame()
                    
                    # Extract price data
                    prices = data['prices']
                    volumes = data.get('total_volumes', [])
                    
                    # Create DataFrame
                    df_data = []
                    for i, (timestamp, price) in enumerate(prices):
                        volume = volumes[i][1] if i < len(volumes) else 0
                        df_data.append({
                            'timestamp': datetime.fromtimestamp(timestamp / 1000),
                            'price': price,
                            'volume': volume
                        })
                    
                    df = pd.DataFrame(df_data)
                    
                    if df.empty:
                        logger.warning(f"No valid data for {symbol}")
                        return pd.DataFrame()
                    
                    # CoinGecko doesn't provide OHLC data, so we create approximate OHLC
                    # This is a limitation - for real OHLC data, use Binance or other exchanges
                    df['open'] = df['price']
                    df['high'] = df['price']
                    df['low'] = df['price']
                    df['close'] = df['price']
                    
                    # Return standardized format
                    result_df = self.standardize_dataframe(df)
                    logger.info(f"Collected {len(result_df)} records for {symbol}")
                    return result_df
                    
                else:
                    error_text = await response.text()
                    raise Exception(f"API error {response.status}: {error_text}")
                    
        except Exception as e:
            logger.error(f"Error fetching CoinGecko data for {symbol}: {e}")
            raise
    
    async def get_coin_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get detailed information about a cryptocurrency.
        
        Args:
            symbol: Crypto symbol
            
        Returns:
            Dict: Coin information
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to CoinGecko API")
        
        try:
            coin_id = self._symbol_to_id(symbol)
            url = f"{self.base_url}/coins/{coin_id}"
            params = {
                'localization': False,
                'tickers': False,
                'market_data': True,
                'community_data': False,
                'developer_data': False,
                'sparkline': False
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    raise Exception(f"API error: {response.status}")
                    
        except Exception as e:
            logger.error(f"Error getting coin info for {symbol}: {e}")
            raise
    
    async def get_trending_coins(self) -> List[Dict[str, Any]]:
        """
        Get trending cryptocurrencies.
        
        Returns:
            List[Dict]: Trending coins data
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to CoinGecko API")
        
        try:
            url = f"{self.base_url}/search/trending"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('coins', [])
                else:
                    raise Exception(f"API error: {response.status}")
                    
        except Exception as e:
            logger.error(f"Error getting trending coins: {e}")
            raise
    
    async def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if cryptocurrency symbol exists.
        
        Args:
            symbol: Symbol to validate
            
        Returns:
            bool: True if symbol is valid
        """
        try:
            coin_id = self._symbol_to_id(symbol)
            coin_info = await self.get_coin_info(symbol)
            return bool(coin_info and 'id' in coin_info)
        except Exception as e:
            logger.error(f"Error validating symbol {symbol}: {e}")
            return False