"""
📈 Trend Following Strategy - Stratégie de Suivi de Tendance
Stratégie basée sur les moyennes mobiles et le MACD
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import pandas as pd
import logging

from core.logger import get_logger
from .base_strategy import BaseStrategy, StrategyMetrics
from ..engine.execution_handler import TradingSignal, SignalType

logger = get_logger(__name__)


class TrendFollowingStrategy(BaseStrategy):
    """
    Stratégie de suivi de tendance basée sur :
    - Moyennes mobiles (SMA et EMA)
    - MACD (Moving Average Convergence Divergence)
    - ADX (Average Directional Index) - simplifié
    """
    
    def __init__(self, symbol: str, timeframe: str = "1h", parameters: Optional[Dict] = None):
        super().__init__(
            name="TrendFollowing",
            symbol=symbol,
            timeframe=timeframe,
            parameters=parameters
        )
        
        # Variables d'état
        self.current_trend = None  # "bullish", "bearish", "neutral"
        self.trend_strength = 0.0
        self.last_crossover = None
        
        logger.info(f"TrendFollowingStrategy initialisée pour {symbol}")
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """Paramètres par défaut de la stratégie trend following"""
        return {
            # Moyennes mobiles
            "fast_ma_period": 10,
            "slow_ma_period": 20,
            "use_ema": True,
            
            # MACD
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            
            # Filtres de signal
            "min_trend_strength": 0.3,
            "confirmation_periods": 2,
            
            # Risk Management
            "stop_loss_percent": 2.0,
            "take_profit_percent": 4.0,
            "max_position_size": 1000.0,
            
            # Signal thresholds
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "volume_threshold": 1.2  # Multiplicateur volume moyen
        }
    
    def generate_signal(self, data: pd.DataFrame) -> TradingSignal:
        """
        Génère un signal de trading basé sur l'analyse technique
        """
        try:
            # Calcul des indicateurs
            df = self.calculate_indicators(data)
            
            if len(df) < max(self.parameters["slow_ma_period"], self.parameters["macd_slow"]):
                # Pas assez de données historiques
                return self._create_hold_signal("Données insuffisantes")
            
            # Analyse de la tendance
            trend_analysis = self._analyze_trend(df)
            
            # Analyse MACD
            macd_analysis = self._analyze_macd(df)
            
            # Analyse volume
            volume_analysis = self._analyze_volume(df)
            
            # Combinaison des signaux
            signal_type, confidence = self._combine_signals(
                trend_analysis, macd_analysis, volume_analysis
            )
            
            # Calcul des prix cibles
            current_price = df['close'].iloc[-1]
            stop_loss, take_profit = self._calculate_targets(signal_type, current_price)
            
            # Création du signal
            signal = TradingSignal(
                strategy_id=self.strategy_id,
                symbol=self.symbol,
                signal_type=signal_type,
                confidence=confidence,
                suggested_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    "trend_analysis": trend_analysis,
                    "macd_analysis": macd_analysis,
                    "volume_analysis": volume_analysis,
                    "current_trend": self.current_trend,
                    "trend_strength": self.trend_strength
                }
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Erreur génération signal TrendFollowing: {str(e)}")
            return self._create_hold_signal(f"Erreur: {str(e)}")
    
    def _analyze_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyse la tendance basée sur les moyennes mobiles"""
        try:
            # Moyennes mobiles
            fast_ma_col = 'EMA_12' if self.parameters["use_ema"] else 'SMA_10'
            slow_ma_col = 'EMA_26' if self.parameters["use_ema"] else 'SMA_20'
            
            current_fast_ma = df[fast_ma_col].iloc[-1]
            current_slow_ma = df[slow_ma_col].iloc[-1]
            prev_fast_ma = df[fast_ma_col].iloc[-2]
            prev_slow_ma = df[slow_ma_col].iloc[-2]
            
            # Direction de la tendance
            if current_fast_ma > current_slow_ma:
                if prev_fast_ma <= prev_slow_ma:
                    # Crossover bullish
                    self.current_trend = "bullish"
                    self.last_crossover = "golden_cross"
                    trend_signal = "BUY"
                else:
                    trend_signal = "BULLISH"
            elif current_fast_ma < current_slow_ma:
                if prev_fast_ma >= prev_slow_ma:
                    # Crossover bearish
                    self.current_trend = "bearish"
                    self.last_crossover = "death_cross"
                    trend_signal = "SELL"
                else:
                    trend_signal = "BEARISH"
            else:
                trend_signal = "NEUTRAL"
            
            # Force de la tendance
            ma_distance = abs(current_fast_ma - current_slow_ma) / current_slow_ma
            self.trend_strength = min(ma_distance * 10, 1.0)  # Normalisation 0-1
            
            return {
                "signal": trend_signal,
                "strength": self.trend_strength,
                "current_trend": self.current_trend,
                "fast_ma": current_fast_ma,
                "slow_ma": current_slow_ma,
                "crossover": self.last_crossover
            }
            
        except Exception as e:
            logger.error(f"Erreur analyse tendance: {str(e)}")
            return {"signal": "NEUTRAL", "strength": 0.0, "error": str(e)}
    
    def _analyze_macd(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyse MACD"""
        try:
            current_macd = df['MACD'].iloc[-1]
            current_signal = df['MACD_signal'].iloc[-1]
            current_histogram = df['MACD_histogram'].iloc[-1]
            
            prev_macd = df['MACD'].iloc[-2]
            prev_signal = df['MACD_signal'].iloc[-2]
            prev_histogram = df['MACD_histogram'].iloc[-2]
            
            # Signal MACD
            if current_macd > current_signal:
                if prev_macd <= prev_signal:
                    # Crossover bullish MACD
                    macd_signal = "BUY"
                    crossover = "bullish"
                else:
                    macd_signal = "BULLISH"
                    crossover = None
            elif current_macd < current_signal:
                if prev_macd >= prev_signal:
                    # Crossover bearish MACD
                    macd_signal = "SELL"
                    crossover = "bearish"
                else:
                    macd_signal = "BEARISH"
                    crossover = None
            else:
                macd_signal = "NEUTRAL"
                crossover = None
            
            # Momentum (histogramme)
            momentum = "INCREASING" if current_histogram > prev_histogram else "DECREASING"
            
            return {
                "signal": macd_signal,
                "crossover": crossover,
                "momentum": momentum,
                "macd": current_macd,
                "signal_line": current_signal,
                "histogram": current_histogram
            }
            
        except Exception as e:
            logger.error(f"Erreur analyse MACD: {str(e)}")
            return {"signal": "NEUTRAL", "error": str(e)}
    
    def _analyze_volume(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyse du volume"""
        try:
            # Volume actuel vs moyenne
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]
            
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Classification du volume
            if volume_ratio >= self.parameters["volume_threshold"]:
                volume_signal = "HIGH"
            elif volume_ratio <= 0.8:
                volume_signal = "LOW"
            else:
                volume_signal = "NORMAL"
            
            return {
                "signal": volume_signal,
                "ratio": volume_ratio,
                "current": current_volume,
                "average": avg_volume
            }
            
        except Exception as e:
            logger.error(f"Erreur analyse volume: {str(e)}")
            return {"signal": "NORMAL", "error": str(e)}
    
    def _combine_signals(self, trend_analysis: Dict, macd_analysis: Dict, volume_analysis: Dict) -> tuple:
        """Combine les différents signaux pour décision finale"""
        try:
            # Initialisation
            buy_strength = 0.0
            sell_strength = 0.0
            
            # Analyse tendance (poids: 40%)
            if trend_analysis["signal"] == "BUY":
                buy_strength += 0.4
            elif trend_analysis["signal"] == "SELL":
                sell_strength += 0.4
            elif trend_analysis["signal"] == "BULLISH":
                buy_strength += 0.2
            elif trend_analysis["signal"] == "BEARISH":
                sell_strength += 0.2
            
            # Bonus pour la force de tendance
            trend_bonus = trend_analysis.get("strength", 0) * 0.2
            if buy_strength > sell_strength:
                buy_strength += trend_bonus
            elif sell_strength > buy_strength:
                sell_strength += trend_bonus
            
            # Analyse MACD (poids: 30%)
            if macd_analysis["signal"] == "BUY":
                buy_strength += 0.3
            elif macd_analysis["signal"] == "SELL":
                sell_strength += 0.3
            elif macd_analysis["signal"] == "BULLISH":
                buy_strength += 0.15
            elif macd_analysis["signal"] == "BEARISH":
                sell_strength += 0.15
            
            # Bonus pour crossover MACD
            if macd_analysis.get("crossover") == "bullish":
                buy_strength += 0.1
            elif macd_analysis.get("crossover") == "bearish":
                sell_strength += 0.1
            
            # Analyse volume (poids: 20%)
            volume_boost = 0.2 if volume_analysis["signal"] == "HIGH" else 0.1
            if buy_strength > sell_strength:
                buy_strength += volume_boost
            elif sell_strength > buy_strength:
                sell_strength += volume_boost
            
            # Malus pour volume faible
            if volume_analysis["signal"] == "LOW":
                buy_strength *= 0.7
                sell_strength *= 0.7
            
            # Détermination du signal final
            confidence = max(buy_strength, sell_strength)
            min_confidence = self.parameters.get("min_trend_strength", 0.3)
            
            if buy_strength > sell_strength and confidence >= min_confidence:
                return SignalType.BUY, min(confidence, 1.0)
            elif sell_strength > buy_strength and confidence >= min_confidence:
                return SignalType.SELL, min(confidence, 1.0)
            else:
                return SignalType.HOLD, confidence
            
        except Exception as e:
            logger.error(f"Erreur combinaison signaux: {str(e)}")
            return SignalType.HOLD, 0.0
    
    def _calculate_targets(self, signal_type: SignalType, current_price: float) -> tuple:
        """Calcule stop-loss et take-profit"""
        try:
            if signal_type == SignalType.HOLD:
                return None, None
            
            stop_loss_pct = self.parameters["stop_loss_percent"] / 100
            take_profit_pct = self.parameters["take_profit_percent"] / 100
            
            if signal_type == SignalType.BUY:
                stop_loss = current_price * (1 - stop_loss_pct)
                take_profit = current_price * (1 + take_profit_pct)
            elif signal_type == SignalType.SELL:
                stop_loss = current_price * (1 + stop_loss_pct)
                take_profit = current_price * (1 - take_profit_pct)
            else:
                return None, None
            
            return stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"Erreur calcul targets: {str(e)}")
            return None, None
    
    def calculate_position_size(self, signal: TradingSignal, account_balance: float) -> float:
        """
        Calcule la taille de position basée sur le risk management
        """
        try:
            # Taille de base basée sur la confiance
            base_percentage = 0.02  # 2% du compte de base
            confidence_multiplier = signal.confidence
            
            # Ajustement basé sur la force de tendance
            trend_multiplier = self.trend_strength
            
            # Taille finale
            position_percentage = base_percentage * confidence_multiplier * (1 + trend_multiplier)
            position_size = account_balance * position_percentage
            
            # Limitation selon les paramètres
            max_size = self.parameters["max_position_size"]
            
            return min(position_size, max_size)
            
        except Exception as e:
            logger.error(f"Erreur calcul position size: {str(e)}")
            return self.parameters["max_position_size"] * 0.1  # Position conservative
    
    def should_exit(self, position: Dict[str, Any], current_data: pd.DataFrame) -> bool:
        """
        Détermine si une position doit être fermée
        """
        try:
            # Vérification stop-loss et take-profit (géré par PositionManager)
            if position.get("should_stop_loss") or position.get("should_take_profit"):
                return True
            
            # Calcul des indicateurs actuels
            df = self.calculate_indicators(current_data)
            if len(df) < 2:
                return False
            
            # Vérification changement de tendance
            trend_analysis = self._analyze_trend(df)
            position_side = position["side"]
            
            # Sortie si changement de tendance contraire
            if position_side == "long":
                if trend_analysis["signal"] == "SELL" or trend_analysis["current_trend"] == "bearish":
                    logger.info(f"Signal de sortie LONG détecté: {trend_analysis['signal']}")
                    return True
            elif position_side == "short":
                if trend_analysis["signal"] == "BUY" or trend_analysis["current_trend"] == "bullish":
                    logger.info(f"Signal de sortie SHORT détecté: {trend_analysis['signal']}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Erreur should_exit: {str(e)}")
            return False
    
    def _create_hold_signal(self, reason: str) -> TradingSignal:
        """Crée un signal HOLD avec raison"""
        return TradingSignal(
            strategy_id=self.strategy_id,
            symbol=self.symbol,
            signal_type=SignalType.HOLD,
            confidence=0.0,
            metadata={"reason": reason}
        )
    
    def get_strategy_description(self) -> str:
        """Description de la stratégie"""
        return f"""
        Stratégie Trend Following pour {self.symbol}:
        - Moyennes mobiles: {self.parameters['fast_ma_period']}/{self.parameters['slow_ma_period']}
        - MACD: {self.parameters['macd_fast']}/{self.parameters['macd_slow']}/{self.parameters['macd_signal']}
        - Stop Loss: {self.parameters['stop_loss_percent']}%
        - Take Profit: {self.parameters['take_profit_percent']}%
        - Force tendance min: {self.parameters['min_trend_strength']}
        """