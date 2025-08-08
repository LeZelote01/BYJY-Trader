"""
📊 Base Strategy - Classe de Base pour les Stratégies
Classe abstraite pour toutes les stratégies de trading
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
import logging
import pandas as pd

from core.logger import get_logger
from ..engine.execution_handler import TradingSignal, SignalType

logger = get_logger(__name__)


class StrategyStatus(Enum):
    """Status d'une stratégie"""
    INACTIVE = "inactive"
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class StrategyMetrics:
    """Métriques d'une stratégie"""
    total_signals: int = 0
    buy_signals: int = 0
    sell_signals: int = 0
    hold_signals: int = 0
    successful_trades: int = 0
    failed_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    average_trade_duration: float = 0.0  # en heures
    max_drawdown: float = 0.0
    
    def update_win_rate(self):
        """Met à jour le win rate"""
        total_trades = self.successful_trades + self.failed_trades
        self.win_rate = (self.successful_trades / max(1, total_trades)) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_signals": self.total_signals,
            "buy_signals": self.buy_signals,
            "sell_signals": self.sell_signals,
            "hold_signals": self.hold_signals,
            "successful_trades": self.successful_trades,
            "failed_trades": self.failed_trades,
            "total_pnl": self.total_pnl,
            "win_rate": self.win_rate,
            "average_trade_duration": self.average_trade_duration,
            "max_drawdown": self.max_drawdown
        }


class BaseStrategy(ABC):
    """
    Classe de base abstraite pour toutes les stratégies de trading
    Définit l'interface commune et les fonctionnalités de base
    """
    
    def __init__(self, name: str, symbol: str, timeframe: str = "1h", parameters: Optional[Dict] = None):
        # Identifiants
        self.strategy_id: str = str(uuid.uuid4())
        self.name: str = name
        self.symbol: str = symbol.upper()
        self.timeframe: str = timeframe
        
        # Paramètres
        self.parameters: Dict[str, Any] = parameters or {}
        self.default_parameters: Dict[str, Any] = self.get_default_parameters()
        
        # Status et métriques
        self.status: StrategyStatus = StrategyStatus.INACTIVE
        self.metrics: StrategyMetrics = StrategyMetrics()
        
        # Historique
        self.signal_history: List[TradingSignal] = []
        self.data_cache: Optional[pd.DataFrame] = None
        
        # Configuration
        self.created_at: datetime = datetime.now(timezone.utc)
        self.last_signal_at: Optional[datetime] = None
        
        # Fusion des paramètres par défaut avec les paramètres fournis
        self._merge_parameters()
        
        logger.info(f"Stratégie {self.name} créée - ID: {self.strategy_id}, Symbol: {self.symbol}")
    
    @abstractmethod
    def get_default_parameters(self) -> Dict[str, Any]:
        """
        Retourne les paramètres par défaut de la stratégie
        À implémenter dans chaque stratégie
        """
        pass
    
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> TradingSignal:
        """
        Génère un signal de trading basé sur les données
        À implémenter dans chaque stratégie
        
        Args:
            data: DataFrame avec colonnes ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
        Returns:
            TradingSignal: Signal généré
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: TradingSignal, account_balance: float) -> float:
        """
        Calcule la taille de position pour un signal
        À implémenter dans chaque stratégie
        
        Args:
            signal: Signal de trading
            account_balance: Solde du compte
            
        Returns:
            float: Taille de position recommandée
        """
        pass
    
    @abstractmethod
    def should_exit(self, position: Dict[str, Any], current_data: pd.DataFrame) -> bool:
        """
        Détermine si une position doit être fermée
        À implémenter dans chaque stratégie
        
        Args:
            position: Dictionnaire avec les infos de position
            current_data: Données de marché actuelles
            
        Returns:
            bool: True si la position doit être fermée
        """
        pass
    
    def activate(self):
        """Active la stratégie"""
        if self.status == StrategyStatus.INACTIVE:
            self.status = StrategyStatus.ACTIVE
            logger.info(f"Stratégie {self.name} activée")
        else:
            logger.warning(f"Stratégie {self.name} déjà active ou en erreur")
    
    def deactivate(self):
        """Désactive la stratégie"""
        self.status = StrategyStatus.INACTIVE
        logger.info(f"Stratégie {self.name} désactivée")
    
    def pause(self):
        """Met en pause la stratégie"""
        if self.status == StrategyStatus.ACTIVE:
            self.status = StrategyStatus.PAUSED
            logger.info(f"Stratégie {self.name} en pause")
    
    def resume(self):
        """Reprend la stratégie"""
        if self.status == StrategyStatus.PAUSED:
            self.status = StrategyStatus.ACTIVE
            logger.info(f"Stratégie {self.name} reprise")
    
    def process_market_data(self, data: pd.DataFrame) -> Optional[TradingSignal]:
        """
        Traite les données de marché et génère un signal si nécessaire
        
        Args:
            data: DataFrame avec les données de marché
            
        Returns:
            Optional[TradingSignal]: Signal généré ou None
        """
        try:
            if self.status != StrategyStatus.ACTIVE:
                return None
            
            # Validation des données
            if not self._validate_data(data):
                logger.warning(f"Données invalides pour stratégie {self.name}")
                return None
            
            # Cache des données
            self.data_cache = data.copy()
            
            # Génération du signal
            signal = self.generate_signal(data)
            
            if signal:
                # Mise à jour des métriques
                self._update_signal_metrics(signal)
                
                # Ajout à l'historique
                self.signal_history.append(signal)
                self.last_signal_at = datetime.now(timezone.utc)
                
                logger.info(f"Signal généré par {self.name}: {signal.signal_type.value} pour {signal.symbol}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Erreur traitement données par {self.name}: {str(e)}")
            self.status = StrategyStatus.ERROR
            return None
    
    def update_parameters(self, new_parameters: Dict[str, Any]):
        """Met à jour les paramètres de la stratégie"""
        try:
            # Validation des paramètres
            validated_params = self._validate_parameters(new_parameters)
            
            # Mise à jour
            self.parameters.update(validated_params)
            
            logger.info(f"Paramètres mis à jour pour {self.name}: {validated_params}")
            
        except Exception as e:
            logger.error(f"Erreur mise à jour paramètres {self.name}: {str(e)}")
            raise
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Retourne les informations complètes de la stratégie"""
        return {
            "strategy_id": self.strategy_id,
            "name": self.name,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "status": self.status.value,
            "parameters": self.parameters,
            "metrics": self.metrics.to_dict(),
            "created_at": self.created_at.isoformat(),
            "last_signal_at": self.last_signal_at.isoformat() if self.last_signal_at else None,
            "signal_count": len(self.signal_history)
        }
    
    def get_recent_signals(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retourne les signaux récents"""
        try:
            # Trier par timestamp décroissant
            sorted_signals = sorted(
                self.signal_history,
                key=lambda x: x.timestamp,
                reverse=True
            )
            
            # Limiter les résultats
            recent_signals = sorted_signals[:limit]
            
            return [signal.to_dict() for signal in recent_signals]
            
        except Exception as e:
            logger.error(f"Erreur récupération signaux récents {self.name}: {str(e)}")
            return []
    
    # Méthodes utilitaires communes
    def _merge_parameters(self):
        """Fusionne les paramètres par défaut avec ceux fournis"""
        merged_params = self.default_parameters.copy()
        merged_params.update(self.parameters)
        self.parameters = merged_params
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Valide les données de marché"""
        if data is None or data.empty:
            return False
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in data.columns:
                logger.error(f"Colonne manquante dans les données: {col}")
                return False
        
        # Vérification des valeurs nulles
        if data[required_columns].isnull().any().any():
            logger.warning("Valeurs nulles détectées dans les données")
            return False
        
        return True
    
    def _validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valide les paramètres (à surcharger dans les sous-classes)
        """
        # Validation de base - à étendre dans les stratégies spécifiques
        validated = {}
        for key, value in parameters.items():
            if key in self.default_parameters:
                validated[key] = value
            else:
                logger.warning(f"Paramètre inconnu ignoré: {key}")
        
        return validated
    
    def _update_signal_metrics(self, signal: TradingSignal):
        """Met à jour les métriques basées sur le signal"""
        self.metrics.total_signals += 1
        
        if signal.signal_type == SignalType.BUY:
            self.metrics.buy_signals += 1
        elif signal.signal_type == SignalType.SELL:
            self.metrics.sell_signals += 1
        else:
            self.metrics.hold_signals += 1
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les indicateurs techniques communs
        Peut être utilisé par les stratégies dérivées
        """
        try:
            df = data.copy()
            
            # Moving Averages
            df['SMA_10'] = df['close'].rolling(window=10).mean()
            df['SMA_20'] = df['close'].rolling(window=20).mean()
            df['EMA_12'] = df['close'].ewm(span=12).mean()
            df['EMA_26'] = df['close'].ewm(span=26).mean()
            
            # MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['BB_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
            df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur calcul indicateurs: {str(e)}")
            return data
    
    def __str__(self) -> str:
        return f"{self.name}({self.symbol}-{self.timeframe})"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name} [{self.status.value}]>"