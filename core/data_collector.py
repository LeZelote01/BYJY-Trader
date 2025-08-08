"""
🔌 Data Collector Core Interface
Interface unifiée pour la collecte de données multi-sources
"""

import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import pandas as pd

from core.logger import get_logger
from core.config import get_settings
from data.collectors.base_collector import BaseCollector
from data.collectors.binance_collector import BinanceCollector
from data.collectors.yahoo_collector import YahooCollector
from data.collectors.coingecko_collector import CoinGeckoCollector
from data.storage.data_manager import DataManager

logger = get_logger(__name__)
settings = get_settings()

class DataCollector:
    """
    Interface unifiée pour la collecte de données multi-sources
    Compatible avec le système de backtesting
    """
    
    def __init__(self):
        self.data_manager = DataManager()
        self.collectors = {
            'binance': BinanceCollector(),
            'yahoo': YahooCollector(), 
            'coingecko': CoinGeckoCollector()
        }
        
    async def get_historical_data(
        self, 
        symbol: str,
        interval: str = "1d",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        source: str = "yahoo"
    ) -> pd.DataFrame:
        """
        Récupère des données historiques pour le backtesting
        
        Args:
            symbol: Symbole à récupérer (ex: AAPL, BTCUSDT)
            interval: Intervalle temporel (1m, 5m, 1h, 1d, etc.)
            start_time: Date de début
            end_time: Date de fin
            source: Source de données (binance, yahoo, coingecko)
            
        Returns:
            DataFrame avec colonnes OHLCV standardisées
        """
        try:
            if source not in self.collectors:
                raise ValueError(f"Source '{source}' non supportée. Sources disponibles: {list(self.collectors.keys())}")
            
            collector = self.collectors[source]
            
            # Définir les dates par défaut si non spécifiées
            if not end_time:
                end_time = datetime.now()
            if not start_time:
                start_time = end_time - timedelta(days=365)  # 1 an par défaut
            
            # Récupérer les données
            data = await collector.fetch_historical_data(
                symbol=symbol,
                interval=interval,
                start_time=start_time,
                end_time=end_time
            )
            
            if data.empty:
                logger.warning(f"Aucune donnée récupérée pour {symbol} depuis {source}")
                return pd.DataFrame()
            
            # Standardiser les colonnes pour le backtesting
            return self._standardize_columns(data)
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données {symbol}: {e}")
            return pd.DataFrame()
    
    def _standardize_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Standardise les colonnes OHLCV pour compatibilité backtesting
        """
        column_mapping = {
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'open': 'open',
            'high': 'high',
            'low': 'low', 
            'close': 'close',
            'volume': 'volume'
        }
        
        # Renommer les colonnes si nécessaire
        data_copy = data.copy()
        data_copy = data_copy.rename(columns=column_mapping)
        
        # S'assurer que les colonnes requises existent
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in data_copy.columns:
                logger.warning(f"Colonne '{col}' manquante dans les données")
                data_copy[col] = 0.0
        
        # Garder seulement les colonnes OHLCV
        return data_copy[required_columns]
    
    async def get_available_symbols(self, source: str = "yahoo") -> List[str]:
        """
        Récupère la liste des symboles disponibles pour une source
        """
        try:
            if source not in self.collectors:
                return []
            
            collector = self.collectors[source]
            return await collector.get_available_symbols()
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des symboles de {source}: {e}")
            return []
    
    async def validate_symbol(self, symbol: str, source: str = "yahoo") -> bool:
        """
        Valide qu'un symbole existe pour une source donnée
        """
        try:
            available_symbols = await self.get_available_symbols(source)
            return symbol in available_symbols
        except:
            return False
    
    def get_supported_intervals(self, source: str = "yahoo") -> List[str]:
        """
        Récupère les intervalles supportés pour une source
        """
        intervals_map = {
            'yahoo': ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'],
            'binance': ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'],
            'coingecko': ['1d', '7d', '30d', '90d', '365d']
        }
        
        return intervals_map.get(source, ['1d'])
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Vérifie l'état de santé de tous les collecteurs
        """
        health_status = {
            'status': 'healthy',
            'collectors': {},
            'timestamp': datetime.now().isoformat()
        }
        
        for name, collector in self.collectors.items():
            try:
                # Test basique de connectivité
                is_healthy = await collector.health_check() if hasattr(collector, 'health_check') else True
                health_status['collectors'][name] = {
                    'status': 'healthy' if is_healthy else 'unhealthy',
                    'available': True
                }
            except Exception as e:
                health_status['collectors'][name] = {
                    'status': 'unhealthy',
                    'available': False,
                    'error': str(e)
                }
        
        # Déterminer le statut global
        unhealthy_count = sum(1 for c in health_status['collectors'].values() if c['status'] == 'unhealthy')
        if unhealthy_count == len(self.collectors):
            health_status['status'] = 'critical'
        elif unhealthy_count > 0:
            health_status['status'] = 'degraded'
        
        return health_status

# Instance globale
_data_collector_instance = None

def get_data_collector() -> DataCollector:
    """Récupère l'instance singleton du collecteur de données"""
    global _data_collector_instance
    if _data_collector_instance is None:
        _data_collector_instance = DataCollector()
    return _data_collector_instance