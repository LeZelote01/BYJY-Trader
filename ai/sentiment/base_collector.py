"""
📰 Base Collector pour sources de données sentiment
Classe abstraite pour standardiser collecte données sentiment
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
from core.logger import get_logger

logger = get_logger("byjy.ai.sentiment.base")


class BaseCollector(ABC):
    """Classe de base abstraite pour collecteurs sentiment"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.is_connected = False
        self.rate_limit = config.get('rate_limit', 100)  # requêtes/heure
        self.last_request_time = None
        
    @abstractmethod
    async def connect(self) -> bool:
        """Connexion au service de données"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Déconnexion du service"""
        pass
    
    @abstractmethod
    async def collect_data(self, symbols: List[str], limit: int = 100) -> List[Dict[str, Any]]:
        """
        Collecte données pour les symboles donnés
        
        Args:
            symbols: Liste des symboles (BTC, ETH, etc.)
            limit: Nombre maximum d'éléments à collecter
            
        Returns:
            Liste de dictionnaires avec structure standardisée:
            {
                'id': str,
                'source': str,
                'symbol': str, 
                'title': str,
                'content': str,
                'url': str,
                'author': str,
                'published_at': datetime,
                'collected_at': datetime,
                'raw_data': dict
            }
        """
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """Vérification santé du collecteur"""
        return {
            'collector': self.name,
            'connected': self.is_connected,
            'last_request': self.last_request_time,
            'rate_limit': self.rate_limit,
            'status': 'healthy' if self.is_connected else 'disconnected'
        }
    
    def _validate_data(self, data: Dict[str, Any]) -> bool:
        """Validation des données collectées"""
        required_fields = ['id', 'source', 'title', 'content', 'published_at']
        return all(field in data for field in required_fields)
    
    async def _rate_limit_check(self) -> None:
        """Gestion du rate limiting"""
        if self.last_request_time:
            time_since_last = (datetime.now() - self.last_request_time).total_seconds()
            min_interval = 3600 / self.rate_limit  # secondes entre requêtes
            
            if time_since_last < min_interval:
                sleep_time = min_interval - time_since_last
                logger.info(f"Rate limit: sleeping {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
        
        self.last_request_time = datetime.now()