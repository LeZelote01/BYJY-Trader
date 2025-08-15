"""
📊 Correlation Analyzer - Phase 3.2
Analyseur corrélation sentiment→prix avec métriques temps réel
Objectif: >0.6 corrélation sentiment→prix sur 30 jours
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

from core.logger import get_logger
from core.database import get_database_manager

logger = get_logger("byjy.ai.sentiment.correlation")


@dataclass
class CorrelationResult:
    """Résultat analyse corrélation"""
    symbol: str
    period_days: int
    correlation: float
    p_value: float
    sample_size: int
    sentiment_avg: float
    price_change: float
    calculated_at: datetime
    is_significant: bool


class CorrelationAnalyzer:
    """Analyseur corrélation sentiment-prix avancé"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.db_manager = get_database_manager()
        self.min_correlation_threshold = config.get('min_correlation', 0.6)
        self.significance_level = config.get('significance_level', 0.05)
        self.is_initialized = False
        
        # Cache corrélations
        self.correlation_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        self.analysis_count = 0
        
    async def initialize(self) -> bool:
        """Initialisation analyseur corrélation"""
        try:
            # Créer tables si nécessaire
            await self._create_correlation_tables()
            
            self.is_initialized = True
            logger.info("✅ Correlation analyzer initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Correlation analyzer initialization failed: {e}")
            return False
    
    async def _create_correlation_tables(self):
        """Créer tables pour stocker corrélations"""
        
        # Table historique corrélations
        correlation_history_sql = """
        CREATE TABLE IF NOT EXISTS sentiment_correlations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            period_days INTEGER NOT NULL,
            correlation REAL NOT NULL,
            p_value REAL NOT NULL,
            sample_size INTEGER NOT NULL,
            sentiment_avg REAL NOT NULL,
            price_change REAL NOT NULL,
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_significant BOOLEAN NOT NULL,
            UNIQUE(symbol, period_days, calculated_at)
        )
        """
        
        # Table métriques sentiment
        sentiment_metrics_sql = """
        CREATE TABLE IF NOT EXISTS sentiment_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            date DATE NOT NULL,
            sentiment_score REAL NOT NULL,
            sentiment_volume INTEGER NOT NULL,
            price_open REAL,
            price_close REAL,
            price_change_pct REAL,
            volume REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, date)
        )
        """
        
        await self.db_manager.execute_query(correlation_history_sql)
        await self.db_manager.execute_query(sentiment_metrics_sql)
        
        # Index pour performance
        await self.db_manager.execute_query(
            "CREATE INDEX IF NOT EXISTS idx_sentiment_correlations_symbol_date ON sentiment_correlations(symbol, calculated_at)"
        )
        await self.db_manager.execute_query(
            "CREATE INDEX IF NOT EXISTS idx_sentiment_metrics_symbol_date ON sentiment_metrics(symbol, date)"
        )
    
    async def calculate_correlation(self, symbol: str, period_days: int = 30) -> CorrelationResult:
        """
        Calculer corrélation sentiment-prix pour un symbole
        
        Args:
            symbol: Symbole à analyser (BTC, ETH, etc.)
            period_days: Période d'analyse en jours
            
        Returns:
            CorrelationResult avec métadonnées
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Vérifier cache
            cache_key = f"{symbol}_{period_days}"
            if self._is_cache_valid(cache_key):
                logger.debug(f"📈 Using cached correlation for {symbol} ({period_days}d)")
                return self.correlation_cache[cache_key]['result']
            
            # Récupérer données sentiment et prix
            sentiment_data = await self._get_sentiment_data(symbol, period_days)
            price_data = await self._get_price_data(symbol, period_days)
            
            if len(sentiment_data) < 5 or len(price_data) < 5:
                logger.warning(f"⚠️ Insufficient data for {symbol}: sentiment={len(sentiment_data)}, price={len(price_data)}")
                return CorrelationResult(
                    symbol=symbol,
                    period_days=period_days,
                    correlation=0.0,
                    p_value=1.0,
                    sample_size=0,
                    sentiment_avg=0.0,
                    price_change=0.0,
                    calculated_at=datetime.now(),
                    is_significant=False
                )
            
            # Aligner données par date
            aligned_data = self._align_data(sentiment_data, price_data)
            
            if len(aligned_data) < 5:
                logger.warning(f"⚠️ Insufficient aligned data for {symbol}: {len(aligned_data)} points")
                return CorrelationResult(
                    symbol=symbol,
                    period_days=period_days,
                    correlation=0.0,
                    p_value=1.0,
                    sample_size=len(aligned_data),
                    sentiment_avg=0.0,
                    price_change=0.0,
                    calculated_at=datetime.now(),
                    is_significant=False
                )
            
            # Calcul corrélation
            result = await self._compute_correlation(symbol, period_days, aligned_data)
            
            # Stocker en cache
            self.correlation_cache[cache_key] = {
                'result': result,
                'timestamp': datetime.now()
            }
            
            # Sauvegarder historique
            await self._save_correlation_result(result)
            
            self.analysis_count += 1
            logger.info(f"📊 Correlation calculated for {symbol} ({period_days}d): {result.correlation:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Correlation calculation failed for {symbol}: {e}")
            return CorrelationResult(
                symbol=symbol,
                period_days=period_days,
                correlation=0.0,
                p_value=1.0,
                sample_size=0,
                sentiment_avg=0.0,
                price_change=0.0,
                calculated_at=datetime.now(),
                is_significant=False
            )
    
    async def _get_sentiment_data(self, symbol: str, period_days: int) -> List[Dict[str, Any]]:
        """Récupérer données sentiment depuis base"""
        
        start_date = datetime.now() - timedelta(days=period_days)
        
        query = """
        SELECT date, sentiment_score, sentiment_volume
        FROM sentiment_metrics 
        WHERE symbol = ? AND date >= ?
        ORDER BY date
        """
        
        try:
            results = await self.db_manager.fetch_all(query, (symbol, start_date.date()))
            return [
                {
                    'date': result[0],
                    'sentiment_score': result[1],
                    'volume': result[2]
                }
                for result in results
            ]
        except Exception as e:
            logger.error(f"❌ Failed to fetch sentiment data for {symbol}: {e}")
            return []
    
    async def _get_price_data(self, symbol: str, period_days: int) -> List[Dict[str, Any]]:
        """Récupérer données prix depuis base"""
        
        start_date = datetime.now() - timedelta(days=period_days)
        
        # Utiliser données existantes du système de collecte
        query = """
        SELECT date(timestamp) as date, 
               AVG(open) as price_open,
               AVG(close) as price_close,
               AVG(volume) as volume
        FROM market_data 
        WHERE symbol = ? AND timestamp >= ?
        GROUP BY date(timestamp)
        ORDER BY date
        """
        
        try:
            results = await self.db_manager.fetch_all(query, (symbol, start_date))
            
            price_data = []
            for result in results:
                price_open = result[1] or 0
                price_close = result[2] or price_open
                price_change_pct = 0
                
                if price_open > 0:
                    price_change_pct = ((price_close - price_open) / price_open) * 100
                
                price_data.append({
                    'date': result[0],
                    'price_open': price_open,
                    'price_close': price_close,
                    'price_change_pct': price_change_pct,
                    'volume': result[3] or 0
                })
            
            return price_data
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch price data for {symbol}: {e}")
            # Générer données simulées pour développement
            return self._generate_mock_price_data(symbol, period_days)
    
    def _generate_mock_price_data(self, symbol: str, period_days: int) -> List[Dict[str, Any]]:
        """Générer données prix simulées pour développement"""
        import random
        
        price_data = []
        base_price = 50000 if symbol == 'BTC' else 3000 if symbol == 'ETH' else 100
        current_price = base_price
        
        for i in range(period_days):
            date = (datetime.now() - timedelta(days=period_days-i)).date()
            
            # Mouvement prix aléatoire avec tendance
            change_pct = random.uniform(-5, 5)
            new_price = current_price * (1 + change_pct/100)
            
            price_data.append({
                'date': date,
                'price_open': current_price,
                'price_close': new_price,
                'price_change_pct': change_pct,
                'volume': random.uniform(1000, 10000)
            })
            
            current_price = new_price
        
        logger.info(f"🔧 Generated {len(price_data)} mock price points for {symbol}")
        return price_data
    
    def _align_data(self, sentiment_data: List[Dict], price_data: List[Dict]) -> List[Dict[str, Any]]:
        """Aligner données sentiment et prix par date"""
        
        # Conversion en DataFrames pour facilité
        df_sentiment = pd.DataFrame(sentiment_data)
        df_price = pd.DataFrame(price_data)
        
        if df_sentiment.empty or df_price.empty:
            return []
        
        # Merger sur date
        df_aligned = pd.merge(df_sentiment, df_price, on='date', how='inner')
        
        # Conversion retour en liste de dicts
        return df_aligned.to_dict('records')
    
    async def _compute_correlation(self, symbol: str, period_days: int, aligned_data: List[Dict]) -> CorrelationResult:
        """Calculer corrélation avec tests statistiques"""
        
        # Extraction séries temporelles
        sentiment_scores = [d['sentiment_score'] for d in aligned_data]
        price_changes = [d['price_change_pct'] for d in aligned_data]
        
        # Calcul corrélation Pearson
        correlation = np.corrcoef(sentiment_scores, price_changes)[0, 1]
        
        # Test significativité (approximation)
        n = len(aligned_data)
        if n > 2:
            t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2))
            try:
                from scipy import stats
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
            except ImportError:
                # Fallback si scipy non disponible
                p_value = 0.05 if abs(t_stat) > 2 else 0.1
        else:
            p_value = 1.0
        
        # Métriques agrégées
        sentiment_avg = np.mean(sentiment_scores)
        price_change_avg = np.mean(price_changes)
        
        # Significativité
        is_significant = p_value < self.significance_level and abs(correlation) >= self.min_correlation_threshold
        
        return CorrelationResult(
            symbol=symbol,
            period_days=period_days,
            correlation=float(correlation) if not np.isnan(correlation) else 0.0,
            p_value=float(p_value) if not np.isnan(p_value) else 1.0,
            sample_size=n,
            sentiment_avg=float(sentiment_avg),
            price_change=float(price_change_avg),
            calculated_at=datetime.now(),
            is_significant=is_significant
        )
    
    async def _save_correlation_result(self, result: CorrelationResult):
        """Sauvegarder résultat corrélation en base"""
        
        query = """
        INSERT OR REPLACE INTO sentiment_correlations 
        (symbol, period_days, correlation, p_value, sample_size, 
         sentiment_avg, price_change, calculated_at, is_significant)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        try:
            await self.db_manager.execute_query(query, (
                result.symbol,
                result.period_days,
                result.correlation,
                result.p_value,
                result.sample_size,
                result.sentiment_avg,
                result.price_change,
                result.calculated_at,
                result.is_significant
            ))
        except Exception as e:
            logger.error(f"❌ Failed to save correlation result: {e}")
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Vérifier validité cache"""
        if cache_key not in self.correlation_cache:
            return False
        
        cached_time = self.correlation_cache[cache_key]['timestamp']
        return (datetime.now() - cached_time).total_seconds() < self.cache_ttl
    
    async def get_multi_symbol_correlations(self, symbols: List[str], period_days: int = 30) -> Dict[str, CorrelationResult]:
        """Calculer corrélations pour plusieurs symboles"""
        
        tasks = [
            self.calculate_correlation(symbol, period_days) 
            for symbol in symbols
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        correlation_results = {}
        for i, result in enumerate(results):
            if isinstance(result, CorrelationResult):
                correlation_results[symbols[i]] = result
            else:
                logger.error(f"❌ Correlation failed for {symbols[i]}: {result}")
                # Résultat par défaut
                correlation_results[symbols[i]] = CorrelationResult(
                    symbol=symbols[i],
                    period_days=period_days,
                    correlation=0.0,
                    p_value=1.0,
                    sample_size=0,
                    sentiment_avg=0.0,
                    price_change=0.0,
                    calculated_at=datetime.now(),
                    is_significant=False
                )
        
        return correlation_results
    
    async def get_correlation_trend(self, symbol: str, days: int = 7) -> List[Dict[str, Any]]:
        """Tendance corrélation sur plusieurs jours"""
        
        query = """
        SELECT DATE(calculated_at) as date, 
               AVG(correlation) as avg_correlation,
               COUNT(*) as count
        FROM sentiment_correlations 
        WHERE symbol = ? AND calculated_at >= ?
        GROUP BY DATE(calculated_at)
        ORDER BY date DESC
        LIMIT ?
        """
        
        start_date = datetime.now() - timedelta(days=days)
        
        try:
            results = await self.db_manager.fetch_all(query, (symbol, start_date, days))
            
            return [
                {
                    'date': result[0],
                    'correlation': result[1],
                    'count': result[2]
                }
                for result in results
            ]
        except Exception as e:
            logger.error(f"❌ Failed to get correlation trend for {symbol}: {e}")
            return []
    
    async def store_sentiment_metrics(self, symbol: str, date: datetime.date, 
                                    sentiment_score: float, sentiment_volume: int,
                                    price_data: Dict[str, float] = None):
        """Stocker métriques sentiment quotidiennes"""
        
        price_data = price_data or {}
        
        query = """
        INSERT OR REPLACE INTO sentiment_metrics 
        (symbol, date, sentiment_score, sentiment_volume, 
         price_open, price_close, price_change_pct, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        try:
            await self.db_manager.execute_query(query, (
                symbol,
                date,
                sentiment_score,
                sentiment_volume,
                price_data.get('price_open'),
                price_data.get('price_close'), 
                price_data.get('price_change_pct'),
                price_data.get('volume')
            ))
        except Exception as e:
            logger.error(f"❌ Failed to store sentiment metrics for {symbol}: {e}")
    
    async def get_correlation_summary(self, period_days: int = 30) -> Dict[str, Any]:
        """Résumé corrélations système"""
        
        query = """
        SELECT symbol, 
               AVG(correlation) as avg_correlation,
               COUNT(*) as analysis_count,
               MAX(calculated_at) as last_analysis,
               SUM(CASE WHEN is_significant = 1 THEN 1 ELSE 0 END) as significant_count
        FROM sentiment_correlations 
        WHERE calculated_at >= ?
        GROUP BY symbol
        ORDER BY avg_correlation DESC
        """
        
        start_date = datetime.now() - timedelta(days=period_days)
        
        try:
            results = await self.db_manager.fetch_all(query, (start_date,))
            
            symbols_data = []
            total_significant = 0
            total_analyses = 0
            
            for result in results:
                symbol_data = {
                    'symbol': result[0],
                    'avg_correlation': result[1],
                    'analysis_count': result[2],
                    'last_analysis': result[3],
                    'significant_count': result[4]
                }
                symbols_data.append(symbol_data)
                total_significant += result[4]
                total_analyses += result[2]
            
            # Métriques globales
            avg_correlation = np.mean([s['avg_correlation'] for s in symbols_data]) if symbols_data else 0.0
            significance_rate = (total_significant / total_analyses * 100) if total_analyses > 0 else 0.0
            
            return {
                'period_days': period_days,
                'symbols_analyzed': len(symbols_data),
                'total_analyses': total_analyses,
                'avg_correlation': round(avg_correlation, 3),
                'significant_correlations': total_significant,
                'significance_rate': round(significance_rate, 1),
                'target_threshold': self.min_correlation_threshold,
                'symbols_data': symbols_data,
                'last_updated': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get correlation summary: {e}")
            return {
                'period_days': period_days,
                'symbols_analyzed': 0,
                'total_analyses': 0,
                'avg_correlation': 0.0,
                'significant_correlations': 0,
                'significance_rate': 0.0,
                'target_threshold': self.min_correlation_threshold,
                'symbols_data': [],
                'last_updated': datetime.now()
            }
    
    async def clear_cache(self):
        """Vider cache corrélations"""
        self.correlation_cache.clear()
        logger.info("🧹 Correlation cache cleared")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Statistiques analyseur corrélation"""
        return {
            'analyzer': 'correlation_analyzer',
            'initialized': self.is_initialized,
            'analysis_count': self.analysis_count,
            'cache_size': len(self.correlation_cache),
            'cache_ttl': self.cache_ttl,
            'min_correlation_threshold': self.min_correlation_threshold,
            'significance_level': self.significance_level,
            'status': 'active' if self.is_initialized else 'inactive'
        }