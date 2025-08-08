"""
📉 Mean Reversion Strategy - Stratégie de Retour à la Moyenne
Stratégie basée sur RSI et Bollinger Bands
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import pandas as pd
import logging

from core.logger import get_logger
from .base_strategy import BaseStrategy, StrategyMetrics
from ..engine.execution_handler import TradingSignal, SignalType

logger = get_logger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """
    Stratégie de retour à la moyenne basée sur :
    - RSI (Relative Strength Index)
    - Bollinger Bands
    - Z-Score sur les prix
    """
    
    def __init__(self, symbol: str, timeframe: str = "1h", parameters: Optional[Dict] = None):
        super().__init__(
            name="MeanReversion",
            symbol=symbol,
            timeframe=timeframe,
            parameters=parameters
        )
        
        logger.info(f"MeanReversionStrategy initialisée pour {symbol}")
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """Paramètres par défaut de la stratégie mean reversion"""
        return {
            # RSI
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            
            # Bollinger Bands
            "bb_period": 20,
            "bb_std": 2,
            
            # Risk Management
            "stop_loss_percent": 1.5,
            "take_profit_percent": 3.0,
            "max_position_size": 1000.0
        }
    
    def generate_signal(self, data: pd.DataFrame) -> TradingSignal:
        """Génère un signal basé sur mean reversion"""
        try:
            # Calcul des indicateurs
            df = self.calculate_indicators(data)
            
            if len(df) < self.parameters["rsi_period"]:
                return self._create_hold_signal("Données insuffisantes")
            
            current_rsi = df['RSI'].iloc[-1]
            current_price = df['close'].iloc[-1]
            bb_upper = df['BB_upper'].iloc[-1]
            bb_lower = df['BB_lower'].iloc[-1]
            
            # Logic mean reversion
            if current_rsi <= self.parameters["rsi_oversold"] and current_price <= bb_lower:
                signal_type = SignalType.BUY
                confidence = 0.8
            elif current_rsi >= self.parameters["rsi_overbought"] and current_price >= bb_upper:
                signal_type = SignalType.SELL
                confidence = 0.8
            else:
                signal_type = SignalType.HOLD
                confidence = 0.0
            
            # Calcul des prix cibles
            stop_loss, take_profit = self._calculate_targets(signal_type, current_price)
            
            return TradingSignal(
                strategy_id=self.strategy_id,
                symbol=self.symbol,
                signal_type=signal_type,
                confidence=confidence,
                suggested_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
        except Exception as e:
            logger.error(f"Erreur génération signal MeanReversion: {str(e)}")
            return self._create_hold_signal(f"Erreur: {str(e)}")
    
    def calculate_position_size(self, signal: TradingSignal, account_balance: float) -> float:
        """Calcule la taille de position"""
        return min(account_balance * 0.02 * signal.confidence, self.parameters["max_position_size"])
    
    def should_exit(self, position: Dict[str, Any], current_data: pd.DataFrame) -> bool:
        """Détermine si une position doit être fermée"""
        return False  # Simplifié pour l'instant
    
    def _calculate_targets(self, signal_type: SignalType, current_price: float) -> tuple:
        """Calcule stop-loss et take-profit"""
        if signal_type == SignalType.HOLD:
            return None, None
        
        stop_loss_pct = self.parameters["stop_loss_percent"] / 100
        take_profit_pct = self.parameters["take_profit_percent"] / 100
        
        if signal_type == SignalType.BUY:
            stop_loss = current_price * (1 - stop_loss_pct)
            take_profit = current_price * (1 + take_profit_pct)
        else:
            stop_loss = current_price * (1 + stop_loss_pct)
            take_profit = current_price * (1 - take_profit_pct)
        
        return stop_loss, take_profit
    
    def _create_hold_signal(self, reason: str) -> TradingSignal:
        """Crée un signal HOLD"""
        return TradingSignal(
            strategy_id=self.strategy_id,
            symbol=self.symbol,
            signal_type=SignalType.HOLD,
            confidence=0.0,
            metadata={"reason": reason}
        )