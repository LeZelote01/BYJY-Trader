"""
📰 News Collector - Phase 3.2
Collecteur de news financières en temps réel
Sources: CoinDesk, CryptoNews, Yahoo Finance, Reuters
"""

import asyncio
import aiohttp
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import feedparser
from bs4 import BeautifulSoup
import re

from .base_collector import BaseCollector
from core.logger import get_logger

logger = get_logger("byjy.ai.sentiment.news")


class NewsCollector(BaseCollector):
    """Collecteur de news financières multi-sources"""
    
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        super().__init__("news_collector", config)
        
        # Sources de news configurables
        self.news_sources = config.get('news_sources', {
            'coindesk': {
                'url': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
                'type': 'rss',
                'enabled': True,
                'symbols': ['BTC', 'ETH', 'CRYPTO']
            },
            'cryptonews': {
                'url': 'https://cryptonews.com/news/feed/',
                'type': 'rss', 
                'enabled': True,
                'symbols': ['BTC', 'ETH', 'CRYPTO']
            },
            'yahoo_finance': {
                'url': 'https://finance.yahoo.com/rss/headline',
                'type': 'rss',
                'enabled': True,
                'symbols': ['BTC', 'ETH', 'SPY', 'TSLA']
            },
            'reuters_crypto': {
                'url': 'https://www.reuters.com/markets/currencies/crypto-currency/',
                'type': 'scraping',
                'enabled': False,  # Scraping plus complexe
                'symbols': ['BTC', 'ETH', 'CRYPTO']
            }
        })
        
        self.session = None
        self.collected_count = 0
        
    async def connect(self) -> bool:
        """Connexion aux sources de news"""
        try:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={
                    'User-Agent': 'BYJY-Trader News Collector 1.0',
                    'Accept': 'application/rss+xml, application/xml, text/xml'
                }
            )
            
            # Test connexion avec une source
            test_source = next(
                (s for s in self.news_sources.values() if s['enabled']), 
                None
            )
            
            if test_source:
                async with self.session.get(test_source['url']) as response:
                    if response.status == 200:
                        self.is_connected = True
                        logger.info("✅ News collector connected successfully")
                        return True
            
            logger.error("❌ Failed to connect to news sources")
            return False
            
        except Exception as e:
            logger.error(f"❌ News collector connection failed: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Déconnexion"""
        if self.session:
            await self.session.close()
            self.session = None
        self.is_connected = False
        logger.info("🔌 News collector disconnected")
    
    async def collect_data(self, symbols: List[str], limit: int = 100) -> List[Dict[str, Any]]:
        """
        Collecte news pour les symboles donnés
        
        Args:
            symbols: Liste symboles (BTC, ETH, etc.)
            limit: Nombre max d'articles
            
        Returns:
            Liste d'articles avec métadonnées
        """
        if not self.is_connected:
            logger.error("❌ News collector not connected")
            return []
        
        await self._rate_limit_check()
        
        all_articles = []
        
        # Collecte depuis chaque source active
        for source_name, source_config in self.news_sources.items():
            if not source_config['enabled']:
                continue
                
            # Vérifier si source pertinente pour les symboles demandés
            source_symbols = source_config['symbols']
            if not any(symbol in source_symbols for symbol in symbols + ['CRYPTO']):
                continue
            
            try:
                if source_config['type'] == 'rss':
                    articles = await self._collect_from_rss(
                        source_name, source_config, symbols, limit // len(self.news_sources)
                    )
                    all_articles.extend(articles)
                elif source_config['type'] == 'scraping':
                    articles = await self._collect_from_scraping(
                        source_name, source_config, symbols, limit // len(self.news_sources)
                    )
                    all_articles.extend(articles)
                    
            except Exception as e:
                logger.error(f"❌ Failed to collect from {source_name}: {e}")
                continue
        
        # Trier par date de publication (plus récent d'abord)
        all_articles.sort(key=lambda x: x['published_at'], reverse=True)
        
        # Limiter au nombre demandé
        result = all_articles[:limit]
        
        self.collected_count += len(result)
        logger.info(f"📰 Collected {len(result)} news articles from {len([s for s in self.news_sources.values() if s['enabled']])} sources")
        
        return result
    
    async def _collect_from_rss(self, source_name: str, config: Dict[str, Any], 
                               symbols: List[str], limit: int) -> List[Dict[str, Any]]:
        """Collecte depuis feed RSS"""
        articles = []
        
        try:
            async with self.session.get(config['url']) as response:
                if response.status != 200:
                    logger.warning(f"⚠️ RSS feed {source_name} returned {response.status}")
                    return []
                
                content = await response.text()
                
                # Parse RSS feed
                feed = feedparser.parse(content)
                
                for entry in feed.entries[:limit]:
                    # Vérifier si l'article contient les symboles recherchés
                    content_text = entry.get('summary', '') + ' ' + entry.get('title', '')
                    if not self._contains_relevant_keywords(content_text, symbols):
                        continue
                    
                    article = {
                        'id': f"{source_name}_{entry.get('id', entry.get('link', ''))}",
                        'source': source_name,
                        'symbol': self._extract_symbols(content_text, symbols),
                        'title': entry.get('title', ''),
                        'content': entry.get('summary', ''),
                        'url': entry.get('link', ''),
                        'author': entry.get('author', 'Unknown'),
                        'published_at': self._parse_date(entry.get('published', '')),
                        'collected_at': datetime.now(),
                        'raw_data': dict(entry)
                    }
                    
                    if self._validate_data(article):
                        articles.append(article)
                        
        except Exception as e:
            logger.error(f"❌ Error collecting from RSS {source_name}: {e}")
        
        return articles
    
    async def _collect_from_scraping(self, source_name: str, config: Dict[str, Any],
                                   symbols: List[str], limit: int) -> List[Dict[str, Any]]:
        """Collecte par scraping HTML (pour sources sans RSS)"""
        # Implémentation basique pour démonstration
        # En production, chaque site nécessiterait un parser spécifique
        articles = []
        
        try:
            async with self.session.get(config['url']) as response:
                if response.status != 200:
                    return []
                
                content = await response.text()
                soup = BeautifulSoup(content, 'html.parser')
                
                # Exemple générique - à adapter par site
                article_elements = soup.find_all(['article', 'div'], class_=re.compile(r'(article|news|story)'), limit=limit)
                
                for element in article_elements:
                    title_elem = element.find(['h1', 'h2', 'h3', 'h4'])
                    content_elem = element.find(['p', 'div'], class_=re.compile(r'(content|summary|description)'))
                    
                    if not title_elem:
                        continue
                        
                    title = title_elem.get_text().strip()
                    content = content_elem.get_text().strip() if content_elem else title
                    
                    if not self._contains_relevant_keywords(title + ' ' + content, symbols):
                        continue
                    
                    article = {
                        'id': f"{source_name}_{hash(title)}",
                        'source': source_name,
                        'symbol': self._extract_symbols(title + ' ' + content, symbols),
                        'title': title,
                        'content': content,
                        'url': config['url'],
                        'author': 'Unknown',
                        'published_at': datetime.now() - timedelta(hours=1),  # Estimation
                        'collected_at': datetime.now(),
                        'raw_data': {'html': str(element)}
                    }
                    
                    if self._validate_data(article):
                        articles.append(article)
                        
        except Exception as e:
            logger.error(f"❌ Error scraping {source_name}: {e}")
        
        return articles
    
    def _contains_relevant_keywords(self, text: str, symbols: List[str]) -> bool:
        """Vérifier si le texte contient des mots-clés pertinents"""
        text_lower = text.lower()
        
        # Mots-clés par symbole
        symbol_keywords = {
            'BTC': ['bitcoin', 'btc', 'btcusd'],
            'ETH': ['ethereum', 'eth', 'ethusd', 'ether'],
            'ADA': ['cardano', 'ada'],
            'SOL': ['solana', 'sol'],
            'CRYPTO': ['crypto', 'cryptocurrency', 'digital currency', 'blockchain'],
            'SPY': ['s&p 500', 'spy', 'market index'],
            'TSLA': ['tesla', 'tsla', 'elon musk']
        }
        
        for symbol in symbols:
            keywords = symbol_keywords.get(symbol, [symbol.lower()])
            if any(keyword in text_lower for keyword in keywords):
                return True
                
        return False
    
    def _extract_symbols(self, text: str, symbols: List[str]) -> str:
        """Extraire le symbole principal de l'article"""
        text_lower = text.lower()
        
        # Compter mentions par symbole
        symbol_counts = {}
        for symbol in symbols:
            symbol_keywords = {
                'BTC': ['bitcoin', 'btc'],
                'ETH': ['ethereum', 'eth', 'ether'],
                'ADA': ['cardano', 'ada'],
                'SOL': ['solana', 'sol']
            }
            
            keywords = symbol_keywords.get(symbol, [symbol.lower()])
            count = sum(text_lower.count(keyword) for keyword in keywords)
            if count > 0:
                symbol_counts[symbol] = count
        
        # Retourner le symbole le plus mentionné
        if symbol_counts:
            return max(symbol_counts, key=symbol_counts.get)
        
        return symbols[0] if symbols else 'UNKNOWN'
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parser date depuis RSS feed"""
        try:
            # Plusieurs formats possibles
            import dateutil.parser
            return dateutil.parser.parse(date_str)
        except:
            # Fallback: date actuelle
            return datetime.now()
    
    async def get_daily_stats(self) -> Dict[str, Any]:
        """Statistiques quotidiennes de collecte"""
        return {
            'collector': self.name,
            'total_collected': self.collected_count,
            'active_sources': len([s for s in self.news_sources.values() if s['enabled']]),
            'sources': list(self.news_sources.keys()),
            'status': 'active' if self.is_connected else 'inactive'
        }