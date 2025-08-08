"""
🎯 Sentiment Manager - Phase 3.2
Gestionnaire principal du système sentiment
Orchestration news + social + analysis + corrélation
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json

from .news_collector import NewsCollector
from .social_collector import SocialMediaCollector
from .sentiment_analyzer import SentimentAnalyzer
from .correlation_analyzer import CorrelationAnalyzer

from core.logger import get_logger
from core.config import get_settings

logger = get_logger("byjy.ai.sentiment.manager")


class SentimentManager:
    """Gestionnaire principal système sentiment"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.settings = get_settings()
        
        # Composants
        self.news_collector = None
        self.social_collector = None
        self.sentiment_analyzer = None
        self.correlation_analyzer = None
        
        # État du système
        self.is_initialized = False
        self.is_running = False
        
        # Configuration collecte
        self.collection_interval = self.config.get('collection_interval', 300)  # 5 minutes
        self.symbols = self.config.get('symbols', ['BTC', 'ETH', 'ADA', 'SOL'])
        self.news_limit = self.config.get('news_limit', 50)
        self.social_limit = self.config.get('social_limit', 100)
        
        # Tâches async
        self.collection_task = None
        self.correlation_task = None
        
        # Stats
        self.stats = {
            'total_collected': 0,
            'total_analyzed': 0,
            'last_collection': None,
            'last_correlation': None,
            'errors': 0
        }
    
    async def initialize(self) -> bool:
        """Initialisation complète du système sentiment"""
        try:
            logger.info("🚀 Initializing sentiment management system...")
            
            # Initialiser composants
            self.news_collector = NewsCollector(self.config.get('news_config', {}))
            self.social_collector = SocialMediaCollector(self.config.get('social_config', {}))
            self.sentiment_analyzer = SentimentAnalyzer(self.config.get('analyzer_config', {}))
            self.correlation_analyzer = CorrelationAnalyzer(self.config.get('correlation_config', {}))
            
            # Connexion composants
            news_ok = await self.news_collector.connect()
            social_ok = await self.social_collector.connect()
            analyzer_ok = await self.sentiment_analyzer.initialize()
            correlation_ok = await self.correlation_analyzer.initialize()
            
            if not all([news_ok, social_ok, analyzer_ok, correlation_ok]):
                logger.error("❌ Failed to initialize some sentiment components")
                return False
            
            self.is_initialized = True
            logger.info("✅ Sentiment management system initialized successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Sentiment system initialization failed: {e}")
            return False
    
    async def start_collection(self) -> bool:
        """Démarrer collecte automatique sentiment"""
        if not self.is_initialized:
            logger.error("❌ System not initialized")
            return False
        
        if self.is_running:
            logger.warning("⚠️ Collection already running")
            return True
        
        try:
            self.is_running = True
            
            # Démarrer tâches parallèles
            self.collection_task = asyncio.create_task(self._collection_loop())
            self.correlation_task = asyncio.create_task(self._correlation_loop())
            
            logger.info("🔄 Sentiment collection started")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start sentiment collection: {e}")
            self.is_running = False
            return False
    
    async def stop_collection(self):
        """Arrêter collecte sentiment"""
        self.is_running = False
        
        # Annuler tâches
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        
        if self.correlation_task:
            self.correlation_task.cancel()
            try:
                await self.correlation_task
            except asyncio.CancelledError:
                pass
        
        logger.info("🛑 Sentiment collection stopped")
    
    async def _collection_loop(self):
        """Boucle collecte données sentiment"""
        while self.is_running:
            try:
                logger.info("📊 Starting sentiment collection cycle...")
                
                # Collecte parallèle news et social
                news_task = self.news_collector.collect_data(self.symbols, self.news_limit)
                social_task = self.social_collector.collect_data(self.symbols, self.social_limit)
                
                news_data, social_data = await asyncio.gather(news_task, social_task)
                
                # Combiner données
                all_content = []
                
                # Traiter news
                for article in news_data:
                    all_content.append({
                        'source': 'news',
                        'symbol': article['symbol'],
                        'text': f"{article['title']} {article['content']}",
                        'metadata': article
                    })
                
                # Traiter social
                for post in social_data:
                    all_content.append({
                        'source': 'social',
                        'symbol': post['symbol'],
                        'text': post['content'],
                        'metadata': post
                    })
                
                logger.info(f"📰 Collected {len(news_data)} news + {len(social_data)} social posts")
                
                # Analyse sentiment
                if all_content:
                    await self._analyze_and_store_sentiment(all_content)
                
                self.stats['total_collected'] += len(all_content)
                self.stats['last_collection'] = datetime.now()
                
            except Exception as e:
                logger.error(f"❌ Collection cycle error: {e}")
                self.stats['errors'] += 1
            
            # Attendre prochain cycle
            await asyncio.sleep(self.collection_interval)
    
    async def _correlation_loop(self):
        """Boucle calcul corrélations"""
        # Calcul corrélations moins fréquent (toutes les heures)
        correlation_interval = 3600  # 1 heure
        
        while self.is_running:
            try:
                logger.info("📈 Starting correlation analysis cycle...")
                
                # Calculer corrélations pour tous symboles
                correlation_results = await self.correlation_analyzer.get_multi_symbol_correlations(
                    self.symbols, period_days=30
                )
                
                # Log résultats
                for symbol, result in correlation_results.items():
                    correlation_value = result.correlation
                    is_significant = "✅" if result.is_significant else "⚠️"
                    logger.info(f"{is_significant} {symbol}: correlation={correlation_value:.3f}")
                
                self.stats['last_correlation'] = datetime.now()
                
            except Exception as e:
                logger.error(f"❌ Correlation cycle error: {e}")
                self.stats['errors'] += 1
            
            # Attendre prochain cycle
            await asyncio.sleep(correlation_interval)
    
    async def _analyze_and_store_sentiment(self, content_list: List[Dict[str, Any]]):
        """Analyser sentiment et stocker métriques"""
        
        # Grouper par symbole
        symbol_content = {}
        for item in content_list:
            symbol = item['symbol']
            if symbol not in symbol_content:
                symbol_content[symbol] = []
            symbol_content[symbol].append(item['text'])
        
        # Analyser sentiment par symbole
        for symbol, texts in symbol_content.items():
            try:
                # Analyse batch
                results = await self.sentiment_analyzer.analyze_batch(texts)
                
                # Calculer métriques agrégées
                if results:
                    sentiment_scores = [r.score for r in results]
                    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                    volume = len(results)
                    
                    # Stocker métriques quotidiennes
                    await self.correlation_analyzer.store_sentiment_metrics(
                        symbol=symbol,
                        date=datetime.now().date(),
                        sentiment_score=avg_sentiment,
                        sentiment_volume=volume
                    )
                    
                    self.stats['total_analyzed'] += len(results)
                    
                    logger.debug(f"📊 {symbol}: sentiment={avg_sentiment:.3f}, volume={volume}")
                
            except Exception as e:
                logger.error(f"❌ Failed to analyze sentiment for {symbol}: {e}")
    
    async def get_current_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Obtenir sentiment actuel pour un symbole"""
        try:
            # Collecte rapide données récentes
            news_data = await self.news_collector.collect_data([symbol], 10)
            social_data = await self.social_collector.collect_data([symbol], 20)
            
            # Combiner textes
            all_texts = []
            all_texts.extend([f"{item['title']} {item['content']}" for item in news_data])
            all_texts.extend([item['content'] for item in social_data])
            
            if not all_texts:
                return {
                    'symbol': symbol,
                    'sentiment_score': 0.0,
                    'sentiment_label': 'neutral',
                    'confidence': 0.0,
                    'volume': 0,
                    'last_update': datetime.now(),
                    'sources': {'news': 0, 'social': 0}
                }
            
            # Analyse sentiment
            distribution = await self.sentiment_analyzer.get_sentiment_distribution(all_texts)
            
            return {
                'symbol': symbol,
                'sentiment_score': distribution['average_score'],
                'sentiment_label': self._get_dominant_sentiment(distribution['percentages']),
                'confidence': distribution['average_confidence'],
                'volume': distribution['total_analyzed'],
                'distribution': distribution['percentages'],
                'sentiment_index': distribution['sentiment_index'],
                'last_update': datetime.now(),
                'sources': {
                    'news': len(news_data),
                    'social': len(social_data)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get current sentiment for {symbol}: {e}")
            return {
                'symbol': symbol,
                'sentiment_score': 0.0,
                'sentiment_label': 'neutral',
                'confidence': 0.0,
                'volume': 0,
                'error': str(e),
                'last_update': datetime.now(),
                'sources': {'news': 0, 'social': 0}
            }
    
    def _get_dominant_sentiment(self, percentages: Dict[str, float]) -> str:
        """Obtenir sentiment dominant"""
        return max(percentages, key=percentages.get)
    
    async def get_sentiment_correlation(self, symbol: str, period_days: int = 30) -> Dict[str, Any]:
        """Obtenir corrélation sentiment-prix"""
        try:
            result = await self.correlation_analyzer.calculate_correlation(symbol, period_days)
            
            return {
                'symbol': result.symbol,
                'period_days': result.period_days,
                'correlation': result.correlation,
                'p_value': result.p_value,
                'sample_size': result.sample_size,
                'sentiment_avg': result.sentiment_avg,
                'price_change': result.price_change,
                'is_significant': result.is_significant,
                'meets_target': abs(result.correlation) >= 0.6,
                'calculated_at': result.calculated_at
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get correlation for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'meets_target': False
            }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Status complet du système sentiment"""
        
        # Status composants
        components_status = {}
        
        if self.news_collector:
            components_status['news_collector'] = await self.news_collector.health_check()
        
        if self.social_collector:
            components_status['social_collector'] = await self.social_collector.health_check()
        
        if self.sentiment_analyzer:
            components_status['sentiment_analyzer'] = await self.sentiment_analyzer.get_stats()
        
        if self.correlation_analyzer:
            components_status['correlation_analyzer'] = await self.correlation_analyzer.get_stats()
        
        # Résumé corrélations
        correlation_summary = {}
        if self.correlation_analyzer:
            correlation_summary = await self.correlation_analyzer.get_correlation_summary()
        
        return {
            'system': {
                'initialized': self.is_initialized,
                'running': self.is_running,
                'symbols': self.symbols,
                'collection_interval': self.collection_interval
            },
            'components': components_status,
            'stats': self.stats,
            'correlations': correlation_summary,
            'last_updated': datetime.now()
        }
    
    async def shutdown(self):
        """Arrêt propre du système"""
        logger.info("🛑 Shutting down sentiment management system...")
        
        # Arrêter collecte
        await self.stop_collection()
        
        # Déconnecter composants
        if self.news_collector:
            await self.news_collector.disconnect()
        
        if self.social_collector:
            await self.social_collector.disconnect()
        
        self.is_initialized = False
        logger.info("✅ Sentiment system shutdown complete")


# Instance globale
_sentiment_manager = None

def get_sentiment_manager() -> SentimentManager:
    """Obtenir instance globale du gestionnaire sentiment"""
    global _sentiment_manager
    if _sentiment_manager is None:
        _sentiment_manager = SentimentManager()
    return _sentiment_manager