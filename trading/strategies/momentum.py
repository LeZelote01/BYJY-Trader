"""
📈 Momentum Strategy - Stratégie de Momentum
Stratégie basée sur les indicateurs de momentum et volume
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import pandas as pd
import logging

from core.logger import get_logger
from .base_strategy import BaseStrategy, StrategyMetrics
from ..engine.execution_handler import TradingSignal, SignalType

logger = get_logger(__name__)


class MomentumStrategy(BaseStrategy):
    """
    Stratégie de momentum basée sur :
    - Rate of Change (ROC)
    - Money Flow Index (MFI)  
    - Volume Price Trend (VPT)
    - Williams %R
    """
    
    def __init__(self, symbol: str, timeframe: str = "1h", parameters: Optional[Dict] = None):
        super().__init__(
            name="Momentum",
            symbol=symbol,
            timeframe=timeframe,
            parameters=parameters
        )
        
        # Variables d'état
        self.momentum_strength = 0.0
        self.volume_trend = "neutral"
        self.last_momentum_signal = None
        
        logger.info(f"MomentumStrategy initialisée pour {symbol}")
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """Paramètres par défaut de la stratégie momentum"""
        return {
            # Rate of Change
            "roc_period": 12,
            "roc_threshold": 2.0,  # % seuil de momentum
            
            # Money Flow Index
            "mfi_period": 14,
            "mfi_oversold": 20,
            "mfi_overbought": 80,
            
            # Williams %R
            "williams_period": 14,
            "williams_oversold": -80,
            "williams_overbought": -20,
            
            # Volume
            "volume_sma_period": 20,
            "volume_threshold": 1.5,  # Multiplicateur volume moyen
            
            # Risk Management
            "stop_loss_percent": 2.5,
            "take_profit_percent": 5.0,
            "max_position_size": 1000.0,
            
            # Signal thresholds
            "min_momentum_strength": 0.4,
            "confirmation_periods": 1
        }
    
    def generate_signal(self, data: pd.DataFrame) -> TradingSignal:
        """
        Génère un signal de trading basé sur l'analyse de momentum
        """
        try:
            # Calcul des indicateurs
            df = self._calculate_momentum_indicators(data)
            
            if len(df) < max(self.parameters["roc_period"], self.parameters["mfi_period"]):
                return self._create_hold_signal("Données insuffisantes")
            
            # Analyse Rate of Change
            roc_analysis = self._analyze_roc(df)
            
            # Analyse Money Flow Index  
            mfi_analysis = self._analyze_mfi(df)
            
            # Analyse Williams %R
            williams_analysis = self._analyze_williams(df)
            
            # Analyse Volume
            volume_analysis = self._analyze_volume_momentum(df)
            
            # Combinaison des signaux
            signal_type, confidence = self._combine_momentum_signals(
                roc_analysis, mfi_analysis, williams_analysis, volume_analysis
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
                    "roc_analysis": roc_analysis,
                    "mfi_analysis": mfi_analysis,
                    "williams_analysis": williams_analysis,
                    "volume_analysis": volume_analysis,
                    "momentum_strength": self.momentum_strength,
                    "volume_trend": self.volume_trend
                }
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Erreur génération signal Momentum: {str(e)}")
            return self._create_hold_signal(f"Erreur: {str(e)}")
    
    def _calculate_momentum_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcule les indicateurs de momentum"""
        try:
            df = data.copy()
            
            # Rate of Change (ROC)
            roc_period = self.parameters["roc_period"]
            df['ROC'] = ((df['close'] - df['close'].shift(roc_period)) / df['close'].shift(roc_period)) * 100
            
            # Money Flow Index (MFI)
            df = self._calculate_mfi(df)
            
            # Williams %R
            williams_period = self.parameters["williams_period"]
            highest_high = df['high'].rolling(window=williams_period).max()
            lowest_low = df['low'].rolling(window=williams_period).min()
            df['Williams_R'] = -100 * (highest_high - df['close']) / (highest_high - lowest_low)
            
            # Volume SMA
            volume_period = self.parameters["volume_sma_period"]
            df['Volume_SMA'] = df['volume'].rolling(window=volume_period).mean()
            
            # Volume Price Trend (VPT)
            df['VPT'] = (df['volume'] * ((df['close'] - df['close'].shift(1)) / df['close'].shift(1))).cumsum()
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur calcul indicateurs momentum: {str(e)}")
            return data
    
    def _calculate_mfi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule le Money Flow Index"""
        try:
            mfi_period = self.parameters["mfi_period"]
            
            # Typical Price
            df['TP'] = (df['high'] + df['low'] + df['close']) / 3
            
            # Raw Money Flow
            df['RMF'] = df['TP'] * df['volume']
            
            # Positive et Negative Money Flow
            df['PMF'] = df['RMF'].where(df['TP'] > df['TP'].shift(1), 0)
            df['NMF'] = df['RMF'].where(df['TP'] < df['TP'].shift(1), 0)
            
            # Money Flow Ratio
            pmf_sum = df['PMF'].rolling(window=mfi_period).sum()
            nmf_sum = df['NMF'].rolling(window=mfi_period).sum()
            
            df['MFR'] = pmf_sum / nmf_sum.replace(0, 1)
            
            # Money Flow Index
            df['MFI'] = 100 - (100 / (1 + df['MFR']))
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur calcul MFI: {str(e)}")
            return df
    
    def _analyze_roc(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyse Rate of Change"""
        try:
            current_roc = df['ROC'].iloc[-1]
            prev_roc = df['ROC'].iloc[-2]
            threshold = self.parameters["roc_threshold"]
            
            # Direction du momentum
            if current_roc > threshold and current_roc > prev_roc:
                roc_signal = "STRONG_BUY"
                strength = min(abs(current_roc) / 10, 1.0)
            elif current_roc > 0 and current_roc > prev_roc:
                roc_signal = "BUY"
                strength = min(abs(current_roc) / 20, 0.8)
            elif current_roc < -threshold and current_roc < prev_roc:
                roc_signal = "STRONG_SELL"
                strength = min(abs(current_roc) / 10, 1.0)
            elif current_roc < 0 and current_roc < prev_roc:
                roc_signal = "SELL"
                strength = min(abs(current_roc) / 20, 0.8)
            else:
                roc_signal = "NEUTRAL"
                strength = 0.0
            
            return {
                "signal": roc_signal,
                "strength": strength,
                "current_roc": current_roc,
                "prev_roc": prev_roc,
                "trend": "accelerating" if current_roc > prev_roc else "decelerating"
            }
            
        except Exception as e:
            logger.error(f"Erreur analyse ROC: {str(e)}")
            return {"signal": "NEUTRAL", "strength": 0.0, "error": str(e)}
    
    def _analyze_mfi(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyse Money Flow Index"""
        try:
            current_mfi = df['MFI'].iloc[-1]
            prev_mfi = df['MFI'].iloc[-2]
            
            oversold = self.parameters["mfi_oversold"]
            overbought = self.parameters["mfi_overbought"]
            
            # Signal MFI
            if current_mfi < oversold:
                if prev_mfi >= oversold:
                    mfi_signal = "BUY"  # Entrée en zone oversold
                else:
                    mfi_signal = "OVERSOLD"
            elif current_mfi > overbought:
                if prev_mfi <= overbought:
                    mfi_signal = "SELL"  # Entrée en zone overbought
                else:
                    mfi_signal = "OVERBOUGHT"
            else:
                mfi_signal = "NEUTRAL"
            
            # Force du signal
            if mfi_signal in ["BUY", "SELL"]:
                strength = 0.8
            elif mfi_signal in ["OVERSOLD", "OVERBOUGHT"]:
                strength = 0.6
            else:
                strength = 0.3
            
            return {
                "signal": mfi_signal,
                "strength": strength,
                "current_mfi": current_mfi,
                "prev_mfi": prev_mfi,
                "divergence": current_mfi - prev_mfi
            }
            
        except Exception as e:
            logger.error(f"Erreur analyse MFI: {str(e)}")
            return {"signal": "NEUTRAL", "strength": 0.0, "error": str(e)}
    
    def _analyze_williams(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyse Williams %R"""
        try:
            current_williams = df['Williams_R'].iloc[-1]
            prev_williams = df['Williams_R'].iloc[-2]
            
            oversold = self.parameters["williams_oversold"]
            overbought = self.parameters["williams_overbought"]
            
            # Signal Williams %R
            if current_williams < oversold:
                if prev_williams >= oversold:
                    williams_signal = "BUY"
                else:
                    williams_signal = "OVERSOLD"
            elif current_williams > overbought:
                if prev_williams <= overbought:
                    williams_signal = "SELL"
                else:
                    williams_signal = "OVERBOUGHT"
            else:
                williams_signal = "NEUTRAL"
            
            return {
                "signal": williams_signal,
                "current_value": current_williams,
                "prev_value": prev_williams,
                "momentum": "increasing" if current_williams > prev_williams else "decreasing"
            }
            
        except Exception as e:
            logger.error(f"Erreur analyse Williams: {str(e)}")
            return {"signal": "NEUTRAL", "error": str(e)}
    
    def _analyze_volume_momentum(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyse du momentum du volume"""
        try:
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['Volume_SMA'].iloc[-1]
            current_vpt = df['VPT'].iloc[-1]
            prev_vpt = df['VPT'].iloc[-2]
            
            # Ratio volume
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Trend VPT
            vpt_trend = "bullish" if current_vpt > prev_vpt else "bearish"
            
            # Signal volume
            threshold = self.parameters["volume_threshold"]
            if volume_ratio >= threshold and vpt_trend == "bullish":
                volume_signal = "STRONG_BUY"
                strength = min(volume_ratio / 2, 1.0)
            elif volume_ratio >= threshold and vpt_trend == "bearish":
                volume_signal = "STRONG_SELL"
                strength = min(volume_ratio / 2, 1.0)
            elif volume_ratio >= 1.2:
                volume_signal = "CONFIRMATION"
                strength = 0.6
            else:
                volume_signal = "WEAK"
                strength = 0.2
            
            self.volume_trend = vpt_trend
            
            return {
                "signal": volume_signal,
                "strength": strength,
                "volume_ratio": volume_ratio,
                "vpt_trend": vpt_trend,
                "current_volume": current_volume,
                "avg_volume": avg_volume
            }
            
        except Exception as e:
            logger.error(f"Erreur analyse volume momentum: {str(e)}")
            return {"signal": "WEAK", "strength": 0.0, "error": str(e)}
    
    def _combine_momentum_signals(self, roc_analysis: Dict, mfi_analysis: Dict, 
                                williams_analysis: Dict, volume_analysis: Dict) -> tuple:
        """Combine les signaux de momentum pour décision finale"""
        try:
            buy_strength = 0.0
            sell_strength = 0.0
            
            # ROC Analysis (poids: 35%)
            roc_signal = roc_analysis.get("signal", "NEUTRAL")
            roc_strength = roc_analysis.get("strength", 0.0)
            
            if roc_signal == "STRONG_BUY":
                buy_strength += 0.35 * roc_strength
            elif roc_signal == "BUY":
                buy_strength += 0.25 * roc_strength
            elif roc_signal == "STRONG_SELL":
                sell_strength += 0.35 * roc_strength
            elif roc_signal == "SELL":
                sell_strength += 0.25 * roc_strength
            
            # MFI Analysis (poids: 25%)
            mfi_signal = mfi_analysis.get("signal", "NEUTRAL")
            mfi_strength = mfi_analysis.get("strength", 0.0)
            
            if mfi_signal in ["BUY", "OVERSOLD"]:
                buy_strength += 0.25 * mfi_strength
            elif mfi_signal in ["SELL", "OVERBOUGHT"]:
                sell_strength += 0.25 * mfi_strength
            
            # Williams %R Analysis (poids: 20%)
            williams_signal = williams_analysis.get("signal", "NEUTRAL")
            
            if williams_signal in ["BUY", "OVERSOLD"]:
                buy_strength += 0.20
            elif williams_signal in ["SELL", "OVERBOUGHT"]:
                sell_strength += 0.20
            
            # Volume Analysis (poids: 20%)
            volume_signal = volume_analysis.get("signal", "WEAK")
            volume_strength = volume_analysis.get("strength", 0.0)
            
            if volume_signal == "STRONG_BUY":
                buy_strength += 0.20 * volume_strength
            elif volume_signal == "STRONG_SELL":
                sell_strength += 0.20 * volume_strength
            elif volume_signal == "CONFIRMATION":
                # Boost le signal dominant
                if buy_strength > sell_strength:
                    buy_strength *= 1.2
                elif sell_strength > buy_strength:
                    sell_strength *= 1.2
            
            # Calcul force momentum globale
            self.momentum_strength = max(buy_strength, sell_strength)
            
            # Détermination signal final
            confidence = self.momentum_strength
            min_strength = self.parameters.get("min_momentum_strength", 0.4)
            
            if buy_strength > sell_strength and confidence >= min_strength:
                return SignalType.BUY, min(confidence, 1.0)
            elif sell_strength > buy_strength and confidence >= min_strength:
                return SignalType.SELL, min(confidence, 1.0)
            else:
                return SignalType.HOLD, confidence
            
        except Exception as e:
            logger.error(f"Erreur combinaison signaux momentum: {str(e)}")
            return SignalType.HOLD, 0.0
    
    def _calculate_targets(self, signal_type: SignalType, current_price: float) -> tuple:
        """Calcule stop-loss et take-profit"""
        try:
            if signal_type == SignalType.HOLD:
                return None, None
            
            stop_loss_pct = self.parameters["stop_loss_percent"] / 100
            take_profit_pct = self.parameters["take_profit_percent"] / 100
            
            # Ajustement basé sur la force du momentum
            momentum_multiplier = 1 + (self.momentum_strength * 0.5)
            
            if signal_type == SignalType.BUY:
                stop_loss = current_price * (1 - stop_loss_pct)
                take_profit = current_price * (1 + take_profit_pct * momentum_multiplier)
            elif signal_type == SignalType.SELL:
                stop_loss = current_price * (1 + stop_loss_pct)
                take_profit = current_price * (1 - take_profit_pct * momentum_multiplier)
            else:
                return None, None
            
            return stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"Erreur calcul targets momentum: {str(e)}")
            return None, None
    
    def calculate_position_size(self, signal: TradingSignal, account_balance: float) -> float:
        """Calcule la taille de position basée sur le momentum"""
        try:
            # Taille de base
            base_percentage = 0.03  # 3% de base (plus agressif pour momentum)
            
            # Ajustement basé sur la confiance et force momentum
            momentum_multiplier = self.momentum_strength
            confidence_multiplier = signal.confidence
            
            # Taille finale
            position_percentage = base_percentage * confidence_multiplier * momentum_multiplier
            position_size = account_balance * position_percentage
            
            # Limitation
            max_size = self.parameters["max_position_size"]
            return min(position_size, max_size)
            
        except Exception as e:
            logger.error(f"Erreur calcul position size momentum: {str(e)}")
            return self.parameters["max_position_size"] * 0.1
    
    def should_exit(self, position: Dict[str, Any], current_data: pd.DataFrame) -> bool:
        """Détermine si une position doit être fermée selon momentum"""
        try:
            # Stop-loss et take-profit standards
            if position.get("should_stop_loss") or position.get("should_take_profit"):
                return True
            
            # Calcul indicateurs actuels
            df = self._calculate_momentum_indicators(current_data)
            if len(df) < 2:
                return False
            
            # Analyse momentum actuel
            roc_analysis = self._analyze_roc(df)
            position_side = position["side"]
            
            # Sortie si perte de momentum
            if position_side == "long":
                if roc_analysis["signal"] in ["STRONG_SELL", "SELL"]:
                    logger.info(f"Signal sortie LONG momentum: {roc_analysis['signal']}")
                    return True
            elif position_side == "short":
                if roc_analysis["signal"] in ["STRONG_BUY", "BUY"]:
                    logger.info(f"Signal sortie SHORT momentum: {roc_analysis['signal']}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Erreur should_exit momentum: {str(e)}")
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
        Stratégie Momentum pour {self.symbol}:
        - ROC Period: {self.parameters['roc_period']}
        - MFI Period: {self.parameters['mfi_period']}  
        - Williams %R Period: {self.parameters['williams_period']}
        - Volume Threshold: {self.parameters['volume_threshold']}x
        - Stop Loss: {self.parameters['stop_loss_percent']}%
        - Take Profit: {self.parameters['take_profit_percent']}%
        - Min Momentum Strength: {self.parameters['min_momentum_strength']}
        """