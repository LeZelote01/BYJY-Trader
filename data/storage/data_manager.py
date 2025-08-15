# Data Manager for BYJY-Trader
# Phase 2.1 - Optimized Data Storage and Retrieval

import asyncio
import sqlite3
import pandas as pd
import numpy as np
import json
import pickle
import gzip
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import aiosqlite

from core.path_utils import get_project_root
from core.logger import get_logger
from core.database import DatabaseManager

logger = get_logger(__name__)

class DataManager:
    """
    Optimized data storage and retrieval system for trading data.
    Uses SQLite with compression and indexing for performance.
    """
    
    def __init__(self):
        """Initialize the Data Manager."""
        self.db_manager = DatabaseManager()
        self.data_dir = get_project_root() / "data" / "storage"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for frequently accessed data
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes TTL
        self.cache_timestamps = {}
        
        logger.info(f"Initialized Data Manager with storage at {self.data_dir}")
    
    async def initialize_tables(self):
        """Initialize data storage tables."""
        try:
            # Historical data table
            await self.db_manager.execute_query('''
                CREATE TABLE IF NOT EXISTS historical_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    source TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    data_hash TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, source, interval, timestamp)
                )
            ''')
            
            # Features table for computed technical indicators
            await self.db_manager.execute_query('''
                CREATE TABLE IF NOT EXISTS feature_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    features_json TEXT NOT NULL,
                    features_version TEXT DEFAULT '2.1.0',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, interval, timestamp, features_version)
                )
            ''')
            
            # Data collection metadata
            await self.db_manager.execute_query('''
                CREATE TABLE IF NOT EXISTS collection_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    source TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    start_date DATETIME,
                    end_date DATETIME,
                    total_records INTEGER,
                    last_update DATETIME DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'active',
                    UNIQUE(symbol, source, interval)
                )
            ''')
            
            # Create indexes for performance
            indexes = [
                'CREATE INDEX IF NOT EXISTS idx_historical_symbol_time ON historical_data(symbol, timestamp)',
                'CREATE INDEX IF NOT EXISTS idx_historical_source_interval ON historical_data(source, interval)',
                'CREATE INDEX IF NOT EXISTS idx_feature_symbol_time ON feature_data(symbol, timestamp)',
                'CREATE INDEX IF NOT EXISTS idx_metadata_symbol ON collection_metadata(symbol)'
            ]
            
            for index_sql in indexes:
                await self.db_manager.execute_query(index_sql)
            
            logger.info("Data storage tables initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing data tables: {e}")
            raise
    
    async def store_historical_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        source: str,
        interval: str,
        batch_size: int = 1000
    ) -> int:
        """
        Store historical data efficiently with batch processing.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            source: Data source (Binance, Yahoo, CoinGecko)
            interval: Time interval
            batch_size: Batch size for insertions
            
        Returns:
            int: Number of records stored
        """
        if df.empty:
            return 0
        
        try:
            # Prepare data for insertion
            df_clean = df.copy()
            df_clean['symbol'] = symbol.upper()
            df_clean['source'] = source
            df_clean['interval'] = interval
            df_clean['data_hash'] = df_clean.apply(
                lambda row: hash(f"{row['open']}{row['high']}{row['low']}{row['close']}{row['volume']}"), 
                axis=1
            )
            
            # Convert to records for batch insertion
            records = df_clean[['symbol', 'source', 'interval', 'timestamp', 
                               'open', 'high', 'low', 'close', 'volume', 'data_hash']].to_dict('records')
            
            inserted_count = 0
            
            # Process in batches using simple record-by-record insertion
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                
                try:
                    # Simple record-by-record insertion with proper error handling
                    for record in batch:
                        try:
                            sql = '''
                                INSERT OR IGNORE INTO historical_data 
                                (symbol, source, interval, timestamp, open, high, low, close, volume, data_hash)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            '''
                            
                            values = [
                                record['symbol'], record['source'], record['interval'],
                                record['timestamp'], record['open'], record['high'],
                                record['low'], record['close'], record['volume'], record['data_hash']
                            ]
                            
                            result = await self.db_manager.execute_query(sql, values)
                            if result:
                                inserted_count += 1
                            
                        except Exception as record_error:
                            logger.warning(f"Record insertion error: {record_error}")
                            continue
                    
                except Exception as batch_error:
                    logger.warning(f"Batch processing error: {batch_error}")
                    continue
            
            # Update metadata
            await self._update_collection_metadata(symbol, source, interval, len(df))
            
            logger.info(f"Stored {inserted_count} records for {symbol} {source} {interval}")
            return inserted_count
            
        except Exception as e:
            logger.error(f"Error storing historical data: {e}")
            raise
    
    async def get_historical_data(
        self,
        symbol: str,
        source: Optional[str] = None,
        interval: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Retrieve historical data with flexible filtering.
        
        Args:
            symbol: Trading symbol
            source: Data source filter (optional)
            interval: Time interval filter (optional)
            start_date: Start date filter (optional)
            end_date: End date filter (optional)
            limit: Record limit (optional)
            
        Returns:
            pd.DataFrame: Historical data
        """
        try:
            # Build query with dynamic filters
            conditions = ["symbol = ?"]
            params = [symbol.upper()]
            
            if source:
                conditions.append("source = ?")
                params.append(source)
            
            if interval:
                conditions.append("interval = ?")
                params.append(interval)
            
            if start_date:
                conditions.append("timestamp >= ?")
                params.append(start_date)
            
            if end_date:
                conditions.append("timestamp <= ?")
                params.append(end_date)
            
            where_clause = " AND ".join(conditions)
            
            sql = f'''
                SELECT timestamp, open, high, low, close, volume, source, interval
                FROM historical_data
                WHERE {where_clause}
                ORDER BY timestamp ASC
            '''
            
            if limit:
                sql += f" LIMIT {limit}"
            
            # Check cache first
            cache_key = f"{symbol}_{source}_{interval}_{start_date}_{end_date}_{limit}"
            if self._is_cached(cache_key):
                logger.debug(f"Returning cached data for {cache_key}")
                return self.cache[cache_key]
            
            # Fetch from database
            result = await self.db_manager.fetch_all(sql, params)
            
            if result:
                df = pd.DataFrame(result)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Cache the result
                self._cache_data(cache_key, df)
                
                logger.info(f"Retrieved {len(df)} records for {symbol}")
                return df
            else:
                logger.info(f"No data found for {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error retrieving historical data: {e}")
            return pd.DataFrame()
    
    async def store_features(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str,
        features_version: str = "2.1.0"
    ) -> int:
        """
        Store computed features efficiently.
        
        Args:
            df: DataFrame with features
            symbol: Trading symbol
            interval: Time interval
            features_version: Features version
            
        Returns:
            int: Number of feature records stored
        """
        try:
            # Extract feature columns (exclude OHLCV and timestamp)
            excluded_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            feature_cols = [col for col in df.columns if col not in excluded_cols]
            
            if not feature_cols:
                logger.warning("No features found in DataFrame")
                return 0
            
            records = []
            for _, row in df.iterrows():
                features = {col: row[col] for col in feature_cols if pd.notna(row[col])}
                
                if features:  # Only store if features exist
                    records.append({
                        'symbol': symbol.upper(),
                        'interval': interval,
                        'timestamp': row['timestamp'],
                        'features_json': json.dumps(features, default=str),
                        'features_version': features_version
                    })
            
            if not records:
                return 0
            
            # Batch insert
            placeholders = ', '.join(['(?, ?, ?, ?, ?)'] * len(records))
            values = []
            for record in records:
                values.extend([
                    record['symbol'], record['interval'], record['timestamp'],
                    record['features_json'], record['features_version']
                ])
            
            sql = f'''
                INSERT OR REPLACE INTO feature_data 
                (symbol, interval, timestamp, features_json, features_version)
                VALUES {placeholders}
            '''
            
            result = await self.db_manager.execute_query(sql, values)
            logger.info(f"Stored {result} feature records for {symbol}")
            return result
            
        except Exception as e:
            logger.error(f"Error storing features: {e}")
            raise
    
    async def get_features(
        self,
        symbol: str,
        interval: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        features_version: str = "2.1.0"
    ) -> pd.DataFrame:
        """
        Retrieve computed features.
        
        Args:
            symbol: Trading symbol
            interval: Time interval
            start_date: Start date filter (optional)
            end_date: End date filter (optional)
            features_version: Features version
            
        Returns:
            pd.DataFrame: Features data
        """
        try:
            conditions = ["symbol = ?", "interval = ?", "features_version = ?"]
            params = [symbol.upper(), interval, features_version]
            
            if start_date:
                conditions.append("timestamp >= ?")
                params.append(start_date)
            
            if end_date:
                conditions.append("timestamp <= ?")
                params.append(end_date)
            
            where_clause = " AND ".join(conditions)
            
            sql = f'''
                SELECT timestamp, features_json
                FROM feature_data
                WHERE {where_clause}
                ORDER BY timestamp ASC
            '''
            
            result = await self.db_manager.fetch_all(sql, params)
            
            if result:
                # Reconstruct DataFrame with features
                data_records = []
                for row in result:
                    timestamp = pd.to_datetime(row[0])
                    features = json.loads(row[1])
                    features['timestamp'] = timestamp
                    data_records.append(features)
                
                df = pd.DataFrame(data_records)
                logger.info(f"Retrieved {len(df)} feature records for {symbol}")
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error retrieving features: {e}")
            return pd.DataFrame()
    
    async def get_data_summary(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get summary statistics of stored data.
        
        Args:
            symbol: Specific symbol filter (optional)
            
        Returns:
            Dict: Data summary statistics
        """
        try:
            summary = {}
            
            # Historical data summary
            if symbol:
                hist_sql = '''
                    SELECT source, interval, COUNT(*) as count,
                           MIN(timestamp) as start_date,
                           MAX(timestamp) as end_date
                    FROM historical_data 
                    WHERE symbol = ?
                    GROUP BY source, interval
                '''
                hist_params = [symbol.upper()]
            else:
                hist_sql = '''
                    SELECT symbol, source, interval, COUNT(*) as count,
                           MIN(timestamp) as start_date,
                           MAX(timestamp) as end_date
                    FROM historical_data 
                    GROUP BY symbol, source, interval
                '''
                hist_params = []
            
            hist_result = await self.db_manager.fetch_all(hist_sql, hist_params)
            
            # Features data summary
            if symbol:
                feat_sql = '''
                    SELECT interval, features_version, COUNT(*) as count
                    FROM feature_data 
                    WHERE symbol = ?
                    GROUP BY interval, features_version
                '''
                feat_params = [symbol.upper()]
            else:
                feat_sql = '''
                    SELECT symbol, interval, features_version, COUNT(*) as count
                    FROM feature_data 
                    GROUP BY symbol, interval, features_version
                '''
                feat_params = []
            
            feat_result = await self.db_manager.fetch_all(feat_sql, feat_params)
            
            summary = {
                'historical_data': [dict(row) for row in hist_result] if hist_result else [],
                'feature_data': [dict(row) for row in feat_result] if feat_result else [],
                'total_historical_records': sum(row['count'] for row in hist_result) if hist_result else 0,
                'total_feature_records': sum(row['count'] for row in feat_result) if feat_result else 0,
                'cache_stats': {
                    'cached_items': len(self.cache),
                    'cache_hit_potential': len(self.cache_timestamps)
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting data summary: {e}")
            return {}
    
    async def _update_collection_metadata(
        self,
        symbol: str,
        source: str,
        interval: str,
        record_count: int
    ):
        """Update collection metadata for tracking purposes."""
        try:
            sql = '''
                INSERT OR REPLACE INTO collection_metadata
                (symbol, source, interval, total_records, last_update)
                VALUES (:symbol, :source, :interval, :record_count, CURRENT_TIMESTAMP)
            '''
            
            await self.db_manager.execute_query(sql, {
                'symbol': symbol.upper(),
                'source': source,
                'interval': interval,
                'record_count': record_count
            })
            
        except Exception as e:
            logger.error(f"Error updating metadata: {e}")
    
    def _is_cached(self, cache_key: str) -> bool:
        """Check if data is cached and still valid."""
        if cache_key not in self.cache:
            return False
        
        if cache_key not in self.cache_timestamps:
            return False
        
        # Check TTL
        cached_time = self.cache_timestamps[cache_key]
        if datetime.now().timestamp() - cached_time > self.cache_ttl:
            # Remove expired cache
            del self.cache[cache_key]
            del self.cache_timestamps[cache_key]
            return False
        
        return True
    
    def _cache_data(self, cache_key: str, data: pd.DataFrame):
        """Cache data with timestamp."""
        self.cache[cache_key] = data.copy()
        self.cache_timestamps[cache_key] = datetime.now().timestamp()
        
        # Limit cache size (remove oldest if too large)
        if len(self.cache) > 100:  # Max 100 cached items
            oldest_key = min(self.cache_timestamps.keys(), 
                           key=lambda k: self.cache_timestamps[k])
            del self.cache[oldest_key]
            del self.cache_timestamps[oldest_key]
    
    async def clear_cache(self):
        """Clear all cached data."""
        self.cache.clear()
        self.cache_timestamps.clear()
        logger.info("Data cache cleared")
    
    async def cleanup_old_data(self, days_to_keep: int = 365):
        """
        Clean up old data to manage storage size.
        
        Args:
            days_to_keep: Number of days to keep
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Clean historical data
            hist_sql = "DELETE FROM historical_data WHERE timestamp < ?"
            hist_result = await self.db_manager.execute_query(hist_sql, [cutoff_date])
            
            # Clean feature data
            feat_sql = "DELETE FROM feature_data WHERE timestamp < ?"
            feat_result = await self.db_manager.execute_query(feat_sql, [cutoff_date])
            
            logger.info(f"Cleaned up {hist_result} historical and {feat_result} feature records older than {days_to_keep} days")
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")