"""
⚖️ Arbitrage Strategy - Stratégie d'Arbitrage
Stratégie basée sur les différences de prix entre exchanges ou paires
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import logging

from core.logger import get_logger
from .base_strategy import BaseStrategy, StrategyMetrics
from ..engine.execution_handler import TradingSignal, SignalType

logger = get_logger(__name__)


class ArbitrageStrategy(BaseStrategy):
    """
    Stratégie d'arbitrage basée sur :
    - Spread analysis entre paires corrélées
    - Triangular arbitrage opportunities
    - Statistical arbitrage basé sur cointegration
    - Time arbitrage (latence)
    """
    
    def __init__(self, symbol: str, timeframe: str = "1h", parameters: Optional[Dict] = None):
        super().__init__(
            name="Arbitrage",
            symbol=symbol,
            timeframe=timeframe,
            parameters=parameters
        )
        
        # Variables d'état
        self.spread_history: List[float] = []
        self.mean_reversion_level = 0.0
        self.spread_std = 0.0
        self.cointegration_score = 0.0
        
        # Paires pour arbitrage
        self.reference_pairs = self._initialize_reference_pairs()
        
        logger.info(f"ArbitrageStrategy initialisée pour {symbol}")
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """Paramètres par défaut de la stratégie arbitrage"""
        return {
            # Spread Analysis
            "spread_lookback": 50,
            "spread_threshold": 2.0,  # Écart-types
            "mean_reversion_period": 20,
            
            # Statistical Arbitrage
            "cointegration_period": 100,
            "min_cointegration_score": 0.7,
            "zscore_entry": 2.0,
            "zscore_exit": 0.5,
            
            # Triangular Arbitrage
            "min_triangular_profit": 0.1,  # % minimum
            "transaction_cost": 0.1,  # % coûts transaction
            
            # Risk Management
            "stop_loss_percent": 1.0,  # Plus serré pour arbitrage
            "take_profit_percent": 2.0,
            "max_position_size": 2000.0,  # Plus élevé car moins risqué
            "max_spread_age": 5,  # Secondes max pour spread
            
            # Pairs Trading
            "correlation_threshold": 0.8,
            "correlation_period": 30,
            
            # Execution
            "max_latency": 100,  # millisecondes
            "slippage_tolerance": 0.05  # %
        }
    
    def _initialize_reference_pairs(self) -> List[str]:
        """Initialise les paires de référence pour arbitrage"""
        try:
            base_asset = self.symbol[:3]  # Ex: BTC pour BTCUSDT
            
            # Paires communes pour arbitrage
            common_pairs = {
                "BTC": ["BTCUSDT", "BTCBUSD", "BTCETH"],
                "ETH": ["ETHUSDT", "ETHBUSD", "ETHBTC"],
                "ADA": ["ADAUSDT", "ADABUSD", "ADABTC"],
                "SOL": ["SOLUSDT", "SOLBUSD", "SOLBTC"]
            }
            
            return common_pairs.get(base_asset, [self.symbol])
            
        except Exception as e:
            logger.error(f"Erreur initialisation paires référence: {str(e)}")
            return [self.symbol]
    
    def generate_signal(self, data: pd.DataFrame) -> TradingSignal:
        """
        Génère un signal de trading basé sur l'analyse d'arbitrage
        """
        try:
            # Calcul des indicateurs d'arbitrage
            df = self._calculate_arbitrage_indicators(data)
            
            if len(df) < self.parameters["spread_lookback"]:
                return self._create_hold_signal("Données insuffisantes pour arbitrage")
            
            # Analyse spread statistique
            spread_analysis = self._analyze_statistical_spread(df)
            
            # Analyse triangular arbitrage
            triangular_analysis = self._analyze_triangular_arbitrage(df)
            
            # Analyse pairs trading
            pairs_analysis = self._analyze_pairs_trading(df)
            
            # Combinaison des signaux d'arbitrage
            signal_type, confidence = self._combine_arbitrage_signals(
                spread_analysis, triangular_analysis, pairs_analysis
            )
            
            # Calcul des prix cibles (plus conservateurs)
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
                    "spread_analysis": spread_analysis,
                    "triangular_analysis": triangular_analysis,
                    "pairs_analysis": pairs_analysis,
                    "arbitrage_type": self._determine_arbitrage_type(spread_analysis, triangular_analysis),
                    "expected_profit": self._calculate_expected_profit(signal_type, confidence),
                    "risk_score": self._calculate_risk_score(df)
                }
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Erreur génération signal Arbitrage: {str(e)}")
            return self._create_hold_signal(f"Erreur: {str(e)}")
    
    def _calculate_arbitrage_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcule les indicateurs d'arbitrage"""
        try:
            df = data.copy()
            
            # Prix de référence (moyenne mobile)
            ref_period = self.parameters["mean_reversion_period"]
            df['Price_MA'] = df['close'].rolling(window=ref_period).mean()
            
            # Spread par rapport à la moyenne
            df['Spread'] = df['close'] - df['Price_MA']
            df['Spread_Pct'] = (df['Spread'] / df['Price_MA']) * 100
            
            # Z-Score du spread
            spread_std = df['Spread'].rolling(window=self.parameters["spread_lookback"]).std()
            df['Spread_ZScore'] = df['Spread'] / spread_std
            
            # Volatilité
            df['Volatility'] = df['close'].pct_change().rolling(window=20).std() * 100
            
            # Volume relatif
            df['Volume_Ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
            
            # Bid-Ask Spread simulé (basé sur volatilité)
            df['Bid_Ask_Spread'] = df['Volatility'] * 0.1  # Approximation
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur calcul indicateurs arbitrage: {str(e)}")
            return data
    
    def _analyze_statistical_spread(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyse spread statistique pour mean reversion"""
        try:
            current_zscore = df['Spread_ZScore'].iloc[-1]
            current_spread_pct = df['Spread_Pct'].iloc[-1]
            
            entry_threshold = self.parameters["zscore_entry"]
            exit_threshold = self.parameters["zscore_exit"]
            
            # Signal basé sur Z-Score
            if current_zscore > entry_threshold:
                # Prix trop haut, vendre (mean reversion)
                spread_signal = "SELL"
                strength = min(abs(current_zscore) / 3, 1.0)
            elif current_zscore < -entry_threshold:
                # Prix trop bas, acheter (mean reversion)
                spread_signal = "BUY" 
                strength = min(abs(current_zscore) / 3, 1.0)
            elif abs(current_zscore) < exit_threshold:
                # Proche de la moyenne, sortie
                spread_signal = "EXIT"
                strength = 0.3
            else:
                spread_signal = "HOLD"
                strength = 0.0
            
            # Mise à jour historique spread
            self.spread_history.append(current_spread_pct)
            if len(self.spread_history) > self.parameters["spread_lookback"]:
                self.spread_history.pop(0)
            
            return {
                "signal": spread_signal,
                "strength": strength,
                "zscore": current_zscore,
                "spread_pct": current_spread_pct,
                "mean_reversion_confidence": min(abs(current_zscore) / entry_threshold, 1.0)
            }
            
        except Exception as e:
            logger.error(f"Erreur analyse spread statistique: {str(e)}")
            return {"signal": "HOLD", "strength": 0.0, "error": str(e)}
    
    def _analyze_triangular_arbitrage(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyse opportunités arbitrage triangulaire"""
        try:
            # Simulation arbitrage triangulaire simplifié
            # En réalité nécessiterait prix multiple exchanges
            
            current_price = df['close'].iloc[-1]
            volatility = df['Volatility'].iloc[-1]
            bid_ask_spread = df['Bid_Ask_Spread'].iloc[-1]
            
            # Calcul opportunité théorique basée sur volatilité
            transaction_cost = self.parameters["transaction_cost"]
            min_profit = self.parameters["min_triangular_profit"]
            
            # Opportunité = (volatilité - coûts) / prix
            potential_profit = (volatility - bid_ask_spread - transaction_cost)
            profit_ratio = potential_profit / 100  # Normalisation
            
            if profit_ratio > min_profit / 100:
                triangular_signal = "OPPORTUNITY"
                strength = min(profit_ratio * 10, 1.0)
            else:
                triangular_signal = "NO_OPPORTUNITY"
                strength = 0.0
            
            return {
                "signal": triangular_signal,
                "strength": strength,
                "potential_profit": potential_profit,
                "profit_ratio": profit_ratio,
                "transaction_cost": transaction_cost
            }
            
        except Exception as e:
            logger.error(f"Erreur analyse triangular arbitrage: {str(e)}")
            return {"signal": "NO_OPPORTUNITY", "strength": 0.0, "error": str(e)}
    
    def _analyze_pairs_trading(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyse pairs trading (corrélation entre paires)"""
        try:
            # Simulation pairs trading - en réalité nécessiterait données de paires corrélées
            
            current_price = df['close'].iloc[-1]
            price_ma = df['Price_MA'].iloc[-1]
            volume_ratio = df['Volume_Ratio'].iloc[-1]
            
            # Calcul correlation théorique basé sur stabilité prix/volume
            price_stability = abs(current_price - price_ma) / price_ma
            volume_stability = abs(volume_ratio - 1.0)
            
            # Score de corrélation simulé
            correlation_score = max(0, 1 - price_stability - volume_stability)
            
            correlation_threshold = self.parameters["correlation_threshold"]
            
            if correlation_score > correlation_threshold:
                pairs_signal = "CORRELATED"
                strength = correlation_score
            else:
                pairs_signal = "UNCORRELATED"
                strength = 0.0
            
            return {
                "signal": pairs_signal,
                "strength": strength,
                "correlation_score": correlation_score,
                "price_stability": price_stability,
                "volume_stability": volume_stability
            }
            
        except Exception as e:
            logger.error(f"Erreur analyse pairs trading: {str(e)}")
            return {"signal": "UNCORRELATED", "strength": 0.0, "error": str(e)}
    
    def _combine_arbitrage_signals(self, spread_analysis: Dict, triangular_analysis: Dict, 
                                 pairs_analysis: Dict) -> tuple:
        """Combine les signaux d'arbitrage pour décision finale"""
        try:
            buy_strength = 0.0
            sell_strength = 0.0
            
            # Statistical Arbitrage (poids: 50%)
            spread_signal = spread_analysis.get("signal", "HOLD")
            spread_strength = spread_analysis.get("strength", 0.0)
            
            if spread_signal == "BUY":
                buy_strength += 0.5 * spread_strength
            elif spread_signal == "SELL":
                sell_strength += 0.5 * spread_strength
            elif spread_signal == "EXIT":
                # Signal de sortie - priorité à la position contraire
                buy_strength += 0.2 * spread_strength
                sell_strength += 0.2 * spread_strength
            
            # Triangular Arbitrage (poids: 30%)
            triangular_signal = triangular_analysis.get("signal", "NO_OPPORTUNITY")
            triangular_strength = triangular_analysis.get("strength", 0.0)
            
            if triangular_signal == "OPPORTUNITY":
                # Boost le signal statistique si opportunité triangulaire
                if buy_strength > sell_strength:
                    buy_strength += 0.3 * triangular_strength
                else:
                    sell_strength += 0.3 * triangular_strength
            
            # Pairs Trading (poids: 20%)
            pairs_signal = pairs_analysis.get("signal", "UNCORRELATED")
            pairs_strength = pairs_analysis.get("strength", 0.0)
            
            if pairs_signal == "CORRELATED":
                # Renforce la confiance si corrélation détectée
                buy_strength *= (1 + 0.2 * pairs_strength)
                sell_strength *= (1 + 0.2 * pairs_strength)
            
            # Détermination signal final
            confidence = max(buy_strength, sell_strength)
            min_confidence = 0.3  # Seuil plus bas pour arbitrage
            
            if buy_strength > sell_strength and confidence >= min_confidence:
                return SignalType.BUY, min(confidence, 1.0)
            elif sell_strength > buy_strength and confidence >= min_confidence:
                return SignalType.SELL, min(confidence, 1.0)
            else:
                return SignalType.HOLD, confidence
            
        except Exception as e:
            logger.error(f"Erreur combinaison signaux arbitrage: {str(e)}")
            return SignalType.HOLD, 0.0
    
    def _determine_arbitrage_type(self, spread_analysis: Dict, triangular_analysis: Dict) -> str:
        """Détermine le type d'arbitrage principal"""
        spread_strength = spread_analysis.get("strength", 0.0)
        triangular_strength = triangular_analysis.get("strength", 0.0)
        
        if triangular_strength > spread_strength:
            return "triangular"
        elif spread_strength > 0:
            return "statistical"
        else:
            return "none"
    
    def _calculate_expected_profit(self, signal_type: SignalType, confidence: float) -> float:
        """Calcule le profit attendu"""
        if signal_type == SignalType.HOLD:
            return 0.0
        
        base_profit = self.parameters["take_profit_percent"] / 100
        return base_profit * confidence
    
    def _calculate_risk_score(self, df: pd.DataFrame) -> float:
        """Calcule le score de risque pour arbitrage"""
        try:
            volatility = df['Volatility'].iloc[-1]
            volume_ratio = df['Volume_Ratio'].iloc[-1]
            
            # Risque faible pour arbitrage = volatilité faible + volume élevé
            vol_risk = min(volatility / 5, 1.0)  # Normalisation
            volume_risk = max(0, 1 - volume_ratio) if volume_ratio < 1 else 0
            
            return (vol_risk + volume_risk) / 2
            
        except Exception as e:
            logger.error(f"Erreur calcul risk score: {str(e)}")
            return 0.5
    
    def _calculate_targets(self, signal_type: SignalType, current_price: float) -> tuple:
        """Calcule stop-loss et take-profit pour arbitrage"""
        try:
            if signal_type == SignalType.HOLD:
                return None, None
            
            # Targets plus serrés pour arbitrage
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
            logger.error(f"Erreur calcul targets arbitrage: {str(e)}")
            return None, None
    
    def calculate_position_size(self, signal: TradingSignal, account_balance: float) -> float:
        """Calcule la taille de position pour arbitrage"""
        try:
            # Arbitrage = moins risqué, position plus importante
            base_percentage = 0.05  # 5% de base
            
            # Ajustement basé sur confiance et type arbitrage
            confidence_multiplier = signal.confidence
            
            # Bonus si arbitrage triangulaire détecté
            arbitrage_type = signal.metadata.get("arbitrage_type", "none")
            type_multiplier = 1.5 if arbitrage_type == "triangular" else 1.0
            
            # Réduction si risque élevé
            risk_score = signal.metadata.get("risk_score", 0.5)
            risk_multiplier = max(0.5, 1 - risk_score)
            
            # Taille finale
            position_percentage = base_percentage * confidence_multiplier * type_multiplier * risk_multiplier
            position_size = account_balance * position_percentage
            
            # Limitation
            max_size = self.parameters["max_position_size"]
            return min(position_size, max_size)
            
        except Exception as e:
            logger.error(f"Erreur calcul position size arbitrage: {str(e)}")
            return self.parameters["max_position_size"] * 0.1
    
    def should_exit(self, position: Dict[str, Any], current_data: pd.DataFrame) -> bool:
        """Détermine si une position d'arbitrage doit être fermée"""
        try:
            # Stops standards
            if position.get("should_stop_loss") or position.get("should_take_profit"):
                return True
            
            # Vérification expiration arbitrage
            entry_time = position.get("entry_time")
            if entry_time:
                time_elapsed = (datetime.now(timezone.utc) - entry_time).total_seconds()
                max_age = self.parameters.get("max_spread_age", 5) * 60  # Conversion minutes
                if time_elapsed > max_age:
                    logger.info("Position arbitrage expirée par âge")
                    return True
            
            # Calcul indicateurs actuels
            df = self._calculate_arbitrage_indicators(current_data)
            if len(df) < 2:
                return False
            
            # Vérification disparition opportunité
            spread_analysis = self._analyze_statistical_spread(df)
            
            # Sortie si signal EXIT ou convergence vers moyenne
            if spread_analysis.get("signal") == "EXIT":
                logger.info("Signal EXIT arbitrage détecté")
                return True
            
            # Sortie si Z-score proche de 0 (retour moyenne)
            zscore = abs(spread_analysis.get("zscore", 0))
            if zscore < self.parameters.get("zscore_exit", 0.5):
                logger.info("Arbitrage convergé vers moyenne")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Erreur should_exit arbitrage: {str(e)}")
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
        Stratégie Arbitrage pour {self.symbol}:
        - Spread Lookback: {self.parameters['spread_lookback']} periods
        - Z-Score Entry: {self.parameters['zscore_entry']}
        - Z-Score Exit: {self.parameters['zscore_exit']}
        - Min Triangular Profit: {self.parameters['min_triangular_profit']}%
        - Transaction Cost: {self.parameters['transaction_cost']}%
        - Stop Loss: {self.parameters['stop_loss_percent']}%
        - Take Profit: {self.parameters['take_profit_percent']}%
        - Correlation Threshold: {self.parameters['correlation_threshold']}
        """