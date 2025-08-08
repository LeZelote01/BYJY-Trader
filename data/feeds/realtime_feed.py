# Real-time Data Feed for BYJY-Trader
# Phase 2.1 - Live Market Data Streaming

import asyncio
import websockets
import json
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from dataclasses import dataclass
import aiohttp

from core.logger import get_logger

logger = get_logger(__name__)

@dataclass
class MarketData:
    """Market data structure."""
    symbol: str
    price: float
    volume: float
    timestamp: datetime
    source: str
    bid: Optional[float] = None
    ask: Optional[float] = None
    change_24h: Optional[float] = None
    change_percent_24h: Optional[float] = None

class RealtimeFeed:
    """
    Real-time market data feed aggregator.
    Connects to multiple exchanges for live price feeds.
    """
    
    def __init__(self):
        """Initialize the real-time feed system."""
        self.name = "Real-time Feed"
        self.active_connections = {}
        self.subscribers = {}
        self.is_running = False
        
        # WebSocket URLs for different exchanges
        self.ws_urls = {
            'binance': 'wss://stream.binance.com:9443/ws',
            'coinbase': 'wss://ws-feed.exchange.coinbase.com',
            'kraken': 'wss://ws.kraken.com'
        }
        
        logger.info("Initialized Real-time Feed system")
    
    async def start_feed(self, symbols: List[str], sources: List[str] = None):
        """
        Start real-time data feeds for specified symbols.
        
        Args:
            symbols: List of trading symbols to monitor
            sources: List of data sources (default: ['binance'])
        """
        if not symbols:
            raise ValueError("At least one symbol must be specified")
        
        sources = sources or ['binance']
        self.is_running = True
        
        try:
            # Start feed for each source
            tasks = []
            for source in sources:
                if source in self.ws_urls:
                    task = asyncio.create_task(
                        self._start_source_feed(source, symbols)
                    )
                    tasks.append(task)
                else:
                    logger.warning(f"Unsupported real-time source: {source}")
            
            if tasks:
                logger.info(f"Starting real-time feeds for {len(symbols)} symbols from {sources}")
                await asyncio.gather(*tasks, return_exceptions=True)
            else:
                logger.error("No valid sources specified for real-time feed")
                
        except Exception as e:
            logger.error(f"Error starting real-time feed: {e}")
            raise
    
    async def stop_feed(self):
        """Stop all real-time data feeds."""
        self.is_running = False
        
        # Close all active WebSocket connections
        for source, ws in self.active_connections.items():
            try:
                if not ws.closed:
                    await ws.close()
                logger.info(f"Closed {source} WebSocket connection")
            except Exception as e:
                logger.error(f"Error closing {source} connection: {e}")
        
        self.active_connections.clear()
        logger.info("All real-time feeds stopped")
    
    async def _start_source_feed(self, source: str, symbols: List[str]):
        """Start real-time feed for a specific source."""
        while self.is_running:
            try:
                if source == 'binance':
                    await self._binance_feed(symbols)
                elif source == 'coinbase':
                    await self._coinbase_feed(symbols)
                elif source == 'kraken':
                    await self._kraken_feed(symbols)
                else:
                    logger.error(f"Feed handler not implemented for {source}")
                    break
                    
            except Exception as e:
                logger.error(f"Error in {source} feed: {e}")
                await asyncio.sleep(5)  # Wait before reconnecting
    
    async def _binance_feed(self, symbols: List[str]):
        """Handle Binance WebSocket feed."""
        try:
            # Create stream names for symbols
            streams = []
            for symbol in symbols:
                symbol_lower = symbol.lower()
                streams.append(f"{symbol_lower}@ticker")  # 24hr ticker
                streams.append(f"{symbol_lower}@trade")   # Individual trades
            
            stream_url = f"{self.ws_urls['binance']}/{'/'.join(streams)}"
            
            async with websockets.connect(stream_url) as websocket:
                self.active_connections['binance'] = websocket
                logger.info(f"Connected to Binance WebSocket for {len(symbols)} symbols")
                
                async for message in websocket:
                    if not self.is_running:
                        break
                    
                    try:
                        data = json.loads(message)
                        await self._process_binance_message(data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to decode Binance message: {e}")
                    except Exception as e:
                        logger.error(f"Error processing Binance message: {e}")
                        
        except websockets.exceptions.ConnectionClosedError:
            logger.warning("Binance WebSocket connection closed")
        except Exception as e:
            logger.error(f"Binance WebSocket error: {e}")
            raise
    
    async def _process_binance_message(self, data: dict):
        """Process incoming Binance WebSocket message."""
        try:
            if 'stream' in data and 'data' in data:
                stream = data['stream']
                msg_data = data['data']
                
                if '@ticker' in stream:
                    # 24hr ticker data
                    market_data = MarketData(
                        symbol=msg_data['s'],
                        price=float(msg_data['c']),  # Last price
                        volume=float(msg_data['v']), # Volume
                        timestamp=datetime.fromtimestamp(msg_data['C'] / 1000),
                        source='binance',
                        bid=float(msg_data['b']),    # Best bid
                        ask=float(msg_data['a']),    # Best ask
                        change_24h=float(msg_data['P']), # 24hr change %
                        change_percent_24h=float(msg_data['P'])
                    )
                    
                    await self._notify_subscribers(market_data)
                    
                elif '@trade' in stream:
                    # Individual trade data
                    market_data = MarketData(
                        symbol=msg_data['s'],
                        price=float(msg_data['p']),
                        volume=float(msg_data['q']),
                        timestamp=datetime.fromtimestamp(msg_data['T'] / 1000),
                        source='binance'
                    )
                    
                    await self._notify_subscribers(market_data)
                    
        except Exception as e:
            logger.error(f"Error processing Binance message: {e}")
    
    async def _coinbase_feed(self, symbols: List[str]):
        """Handle Coinbase WebSocket feed (placeholder)."""
        logger.info("Coinbase feed not implemented yet")
        await asyncio.sleep(60)  # Placeholder delay
    
    async def _kraken_feed(self, symbols: List[str]):
        """Handle Kraken WebSocket feed (placeholder).""" 
        logger.info("Kraken feed not implemented yet")
        await asyncio.sleep(60)  # Placeholder delay
    
    async def subscribe(self, callback: Callable[[MarketData], None], symbols: List[str] = None):
        """
        Subscribe to market data updates.
        
        Args:
            callback: Function to call when new data arrives
            symbols: Specific symbols to monitor (None for all)
        """
        subscriber_id = id(callback)
        self.subscribers[subscriber_id] = {
            'callback': callback,
            'symbols': symbols or []
        }
        
        logger.info(f"Added subscriber {subscriber_id} for {len(symbols or [])} symbols")
        return subscriber_id
    
    async def unsubscribe(self, subscriber_id: int):
        """Remove a subscriber."""
        if subscriber_id in self.subscribers:
            del self.subscribers[subscriber_id]
            logger.info(f"Removed subscriber {subscriber_id}")
    
    async def _notify_subscribers(self, market_data: MarketData):
        """Notify all relevant subscribers of new market data."""
        try:
            for subscriber_id, subscriber in self.subscribers.items():
                # Check if subscriber wants this symbol
                symbols = subscriber['symbols']
                if not symbols or market_data.symbol in symbols:
                    try:
                        callback = subscriber['callback']
                        if asyncio.iscoroutinefunction(callback):
                            await callback(market_data)
                        else:
                            callback(market_data)
                    except Exception as e:
                        logger.error(f"Error notifying subscriber {subscriber_id}: {e}")
                        
        except Exception as e:
            logger.error(f"Error notifying subscribers: {e}")
    
    async def get_current_prices(self, symbols: List[str], source: str = 'binance') -> Dict[str, MarketData]:
        """
        Get current prices via REST API (fallback when WebSocket not available).
        
        Args:
            symbols: List of symbols to get prices for
            source: Data source to use
            
        Returns:
            Dict[str, MarketData]: Current market data for symbols
        """
        try:
            if source == 'binance':
                return await self._get_binance_prices(symbols)
            else:
                raise ValueError(f"REST API not implemented for {source}")
                
        except Exception as e:
            logger.error(f"Error getting current prices: {e}")
            return {}
    
    async def _get_binance_prices(self, symbols: List[str]) -> Dict[str, MarketData]:
        """Get current prices from Binance REST API."""
        try:
            async with aiohttp.ClientSession() as session:
                prices = {}
                
                # Get 24hr ticker data for all symbols at once
                url = "https://api.binance.com/api/v3/ticker/24hr"
                
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Filter for requested symbols
                        for ticker in data:
                            symbol = ticker['symbol']
                            if symbol in symbols:
                                market_data = MarketData(
                                    symbol=symbol,
                                    price=float(ticker['lastPrice']),
                                    volume=float(ticker['volume']),
                                    timestamp=datetime.fromtimestamp(ticker['closeTime'] / 1000),
                                    source='binance',
                                    bid=float(ticker.get('bidPrice', 0)),
                                    ask=float(ticker.get('askPrice', 0)),
                                    change_24h=float(ticker['priceChange']),
                                    change_percent_24h=float(ticker['priceChangePercent'])
                                )
                                prices[symbol] = market_data
                    else:
                        logger.error(f"Binance API error: {response.status}")
                
                return prices
                
        except Exception as e:
            logger.error(f"Error getting Binance prices: {e}")
            return {}
    
    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get status of all real-time connections.
        
        Returns:
            Dict: Connection status information
        """
        status = {
            'is_running': self.is_running,
            'active_connections': len(self.active_connections),
            'subscribers': len(self.subscribers),
            'connections': {}
        }
        
        for source, ws in self.active_connections.items():
            status['connections'][source] = {
                'connected': not ws.closed if ws else False,
                'url': self.ws_urls.get(source, 'unknown')
            }
        
        return status
    
    async def test_connection(self, source: str) -> bool:
        """
        Test connection to a specific data source.
        
        Args:
            source: Data source to test
            
        Returns:
            bool: True if connection successful
        """
        try:
            if source == 'binance':
                # Test REST API connection
                async with aiohttp.ClientSession() as session:
                    url = "https://api.binance.com/api/v3/time"
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        return response.status == 200
            else:
                logger.warning(f"Test connection not implemented for {source}")
                return False
                
        except Exception as e:
            logger.error(f"Error testing {source} connection: {e}")
            return False