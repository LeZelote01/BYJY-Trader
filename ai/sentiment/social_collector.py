"""
🐦 Social Media Collector - Phase 3.2
Collecteur sentiment réseaux sociaux (Twitter, Reddit)
Simulation pour éviter dépendances API tierces
"""

import asyncio
import random
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json

from .base_collector import BaseCollector
from core.logger import get_logger

logger = get_logger("byjy.ai.sentiment.social")


class SocialMediaCollector(BaseCollector):
    """Collecteur sentiment réseaux sociaux avec données simulées"""
    
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        super().__init__("social_collector", config)
        
        # Configuration sources sociales
        self.social_sources = config.get('social_sources', {
            'twitter': {
                'enabled': True,
                'hashtags': ['#BTC', '#Bitcoin', '#ETH', '#Ethereum', '#Crypto'],
                'accounts': ['@elonmusk', '@VitalikButerin', '@CoinDesk', '@cz_binance'],
                'simulation': True  # Mode simulation
            },
            'reddit': {
                'enabled': True,
                'subreddits': ['cryptocurrency', 'Bitcoin', 'ethereum', 'wallstreetbets'],
                'simulation': True  # Mode simulation
            }
        })
        
        self.collected_count = 0
        
        # Données simulées pour développement
        self.simulation_data = self._generate_simulation_data()
    
    async def connect(self) -> bool:
        """Connexion aux APIs sociales"""
        try:
            # En mode simulation, toujours connecté
            if all(source.get('simulation', False) for source in self.social_sources.values()):
                self.is_connected = True
                logger.info("✅ Social collector connected (simulation mode)")
                return True
            
            # TODO: Implémentation réelle APIs
            # - Twitter API v2
            # - Reddit API
            # - Discord webhooks
            
            logger.warning("⚠️ Real social APIs not implemented yet - using simulation")
            self.is_connected = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Social collector connection failed: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Déconnexion"""
        self.is_connected = False
        logger.info("🔌 Social collector disconnected")
    
    async def collect_data(self, symbols: List[str], limit: int = 100) -> List[Dict[str, Any]]:
        """
        Collecte données sociales pour les symboles donnés
        
        Args:
            symbols: Liste symboles (BTC, ETH, etc.)
            limit: Nombre max de posts
            
        Returns:
            Liste de posts sociaux avec sentiment
        """
        if not self.is_connected:
            logger.error("❌ Social collector not connected")
            return []
        
        await self._rate_limit_check()
        
        all_posts = []
        
        # Collecte depuis chaque source active
        for source_name, source_config in self.social_sources.items():
            if not source_config['enabled']:
                continue
            
            try:
                if source_config.get('simulation', True):
                    posts = await self._collect_simulation_data(
                        source_name, source_config, symbols, limit // len(self.social_sources)
                    )
                    all_posts.extend(posts)
                else:
                    # TODO: Implémentation APIs réelles
                    posts = await self._collect_real_data(
                        source_name, source_config, symbols, limit // len(self.social_sources)
                    )
                    all_posts.extend(posts)
                    
            except Exception as e:
                logger.error(f"❌ Failed to collect from {source_name}: {e}")
                continue
        
        # Trier par date (plus récent d'abord)
        all_posts.sort(key=lambda x: x['published_at'], reverse=True)
        
        # Limiter au nombre demandé
        result = all_posts[:limit]
        
        self.collected_count += len(result)
        logger.info(f"🐦 Collected {len(result)} social posts from {len([s for s in self.social_sources.values() if s['enabled']])} sources")
        
        return result
    
    async def _collect_simulation_data(self, source_name: str, config: Dict[str, Any],
                                     symbols: List[str], limit: int) -> List[Dict[str, Any]]:
        """Collecte données simulées pour développement"""
        posts = []
        
        # Générer posts simulés
        for i in range(limit):
            symbol = random.choice(symbols)
            
            # Templates de posts par source
            if source_name == 'twitter':
                post_templates = [
                    f"🚀 {symbol} looking bullish! Great momentum #crypto",
                    f"📈 {symbol} breaking resistance levels! #trading",
                    f"💎 {symbol} holders staying strong! #HODL",
                    f"⚠️ {symbol} showing some weakness, watch support",
                    f"🔥 {symbol} volume increasing significantly!",
                    f"😱 {symbol} dumping hard, market fear rising",
                    f"🎯 {symbol} hitting key targets, bullish signal!",
                    f"📊 {symbol} technical analysis suggests uptrend"
                ]
            else:  # reddit
                post_templates = [
                    f"Analysis: {symbol} fundamentals still strong despite recent dip",
                    f"DD: Why {symbol} is undervalued at current prices",
                    f"Discussion: {symbol} long-term prospects looking good",
                    f"Warning: {symbol} showing concerning patterns",
                    f"News: Major development for {symbol} ecosystem",
                    f"Opinion: {symbol} overvalued in current market",
                    f"Chart: {symbol} technical breakout imminent",
                    f"Update: {symbol} community growing rapidly"
                ]
            
            content = random.choice(post_templates)
            
            # Sentiment simulé basé sur contenu
            sentiment_score = self._simulate_sentiment(content)
            
            post = {
                'id': f"{source_name}_{i}_{random.randint(1000, 9999)}",
                'source': source_name,
                'symbol': symbol,
                'title': content[:50] + '...' if len(content) > 50 else content,
                'content': content,
                'url': f"https://{source_name}.com/post/{random.randint(10000, 99999)}",
                'author': f"user_{random.randint(1, 1000)}",
                'published_at': datetime.now() - timedelta(
                    minutes=random.randint(1, 1440)  # 1 min à 24h
                ),
                'collected_at': datetime.now(),
                'raw_data': {
                    'likes': random.randint(0, 500),
                    'shares': random.randint(0, 100),
                    'comments': random.randint(0, 50),
                    'sentiment_score': sentiment_score,
                    'engagement': random.uniform(0.1, 10.0)
                }
            }
            
            if self._validate_data(post):
                posts.append(post)
        
        return posts
    
    async def _collect_real_data(self, source_name: str, config: Dict[str, Any],
                                symbols: List[str], limit: int) -> List[Dict[str, Any]]:
        """Collecte données réelles des APIs"""
        # TODO: Implémentation APIs réelles
        # 
        # Twitter API v2:
        # - Recherche tweets par hashtags/mentions
        # - Authentification Bearer Token
        # - Gestion rate limits (300 req/15min)
        #
        # Reddit API:  
        # - Recherche posts par subreddits
        # - Authentification OAuth2
        # - Gestion rate limits (60 req/min)
        
        logger.info(f"🚧 Real API collection for {source_name} not implemented yet")
        return []
    
    def _simulate_sentiment(self, content: str) -> float:
        """Simuler score sentiment basé sur mots-clés"""
        content_lower = content.lower()
        
        # Mots positifs/négatifs
        positive_words = ['bullish', 'moon', 'pump', 'up', 'buy', 'bull', 'good', 'strong', 'high']
        negative_words = ['bearish', 'dump', 'down', 'sell', 'bear', 'bad', 'weak', 'low', 'crash']
        
        positive_count = sum(1 for word in positive_words if word in content_lower)
        negative_count = sum(1 for word in negative_words if word in content_lower)
        
        # Score entre -1 (très négatif) et +1 (très positif)
        if positive_count + negative_count == 0:
            return random.uniform(-0.2, 0.2)  # Neutre avec variation
        
        score = (positive_count - negative_count) / (positive_count + negative_count)
        return max(-1.0, min(1.0, score + random.uniform(-0.1, 0.1)))
    
    def _generate_simulation_data(self) -> Dict[str, Any]:
        """Générer données de base pour simulation"""
        return {
            'last_generated': datetime.now(),
            'total_users': random.randint(10000, 50000),
            'active_hashtags': ['#BTC', '#ETH', '#Crypto', '#Trading', '#HODL'],
            'trending_topics': ['Bitcoin', 'Ethereum', 'DeFi', 'NFTs']
        }
    
    async def get_social_metrics(self, symbol: str) -> Dict[str, Any]:
        """Métriques sociales pour un symbole"""
        # Simulation métriques
        base_engagement = random.uniform(100, 1000)
        
        return {
            'symbol': symbol,
            'mentions_24h': int(base_engagement),
            'sentiment_avg': random.uniform(-0.5, 0.8),
            'engagement_score': base_engagement,
            'trending_rank': random.randint(1, 50),
            'volume_change': random.uniform(-50, 200),
            'sources': list(self.social_sources.keys()),
            'last_update': datetime.now()
        }
    
    async def get_daily_stats(self) -> Dict[str, Any]:
        """Statistiques quotidiennes de collecte"""
        return {
            'collector': self.name,
            'total_collected': self.collected_count,
            'active_sources': len([s for s in self.social_sources.values() if s['enabled']]),
            'sources': list(self.social_sources.keys()),
            'simulation_mode': all(s.get('simulation', False) for s in self.social_sources.values()),
            'status': 'active' if self.is_connected else 'inactive'
        }