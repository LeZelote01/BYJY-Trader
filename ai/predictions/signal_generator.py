# Trading Signal Generator for BYJY-Trader
# Phase 2.2 - AI-powered trading signals

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

from .predictor import AIPredictor
from core.logger import get_logger

logger = get_logger(__name__)

class SignalType(Enum):
    """Trading signal types."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

class RiskLevel(Enum):
    """Risk assessment levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class SignalGenerator:
    """
    AI-powered trading signal generator.
    Combines predictions with technical analysis to generate actionable signals.
    """
    
    def __init__(self):
        """Initialize Signal Generator."""
        self.predictor = AIPredictor()
        
        # Signal thresholds
        self.price_change_thresholds = {
            'strong_buy': 0.05,    # 5% predicted increase
            'buy': 0.02,           # 2% predicted increase
            'sell': -0.02,         # 2% predicted decrease
            'strong_sell': -0.05   # 5% predicted decrease
        }
        
        # Risk assessment parameters
        self.risk_thresholds = {
            'volatility_high': 0.08,   # 8% volatility = high risk
            'volatility_medium': 0.04  # 4% volatility = medium risk
        }
        
        logger.info("Signal Generator initialized")
    
    async def initialize(self):
        """Initialize signal generator."""
        await self.predictor.initialize()
        logger.info("Signal Generator ready")
    
    async def generate_signal(self,
                            symbol: str,
                            horizons: List[str] = None,
                            model_name: str = 'lstm') -> Dict[str, Any]:
        """
        Generate trading signal for a symbol.
        
        Args:
            symbol: Trading symbol
            horizons: List of time horizons to consider
            model_name: AI model to use
            
        Returns:
            Dict: Trading signal with recommendations
        """
        try:
            if horizons is None:
                horizons = ['1h', '4h', '1d']
            
            # Get predictions for multiple horizons
            predictions = await self.predictor.predict_multiple_horizons(
                symbol, horizons, model_name
            )
            
            if 'error' in predictions:
                return {
                    'symbol': symbol,
                    'error': predictions['error'],
                    'timestamp': datetime.now().isoformat()
                }
            
            # Analyze predictions to generate signal
            signal_analysis = self._analyze_predictions(predictions)
            
            # Generate final signal
            signal = self._generate_final_signal(signal_analysis)
            
            # Add risk assessment
            risk_assessment = await self._assess_risk(symbol, predictions)
            
            # Combine into final signal
            final_signal = {
                'symbol': symbol,
                'signal': signal['type'].value,
                'signal_strength': signal['strength'],
                'confidence': signal['confidence'],
                'risk_level': risk_assessment['level'].value,
                'risk_score': risk_assessment['score'],
                'recommendations': self._generate_recommendations(signal, risk_assessment),
                'horizon_analysis': signal_analysis,
                'model_used': model_name,
                'timestamp': datetime.now().isoformat(),
                'valid_until': (datetime.now() + timedelta(minutes=30)).isoformat()
            }
            
            logger.info(f"Signal generated for {symbol}: {final_signal['signal']} ({final_signal['confidence']:.2f})")
            return final_signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def generate_batch_signals(self,
                                   symbols: List[str],
                                   horizons: List[str] = None,
                                   model_name: str = 'lstm') -> Dict[str, Any]:
        """
        Generate signals for multiple symbols.
        
        Args:
            symbols: List of symbols
            horizons: Time horizons
            model_name: AI model to use
            
        Returns:
            Dict: Batch signal results
        """
        results = {
            'model_used': model_name,
            'horizons': horizons or ['1h', '4h', '1d'],
            'timestamp': datetime.now().isoformat(),
            'signals': {},
            'summary': {
                'strong_buy': 0,
                'buy': 0,
                'hold': 0,
                'sell': 0,
                'strong_sell': 0,
                'errors': 0
            }
        }
        
        for symbol in symbols:
            try:
                signal = await self.generate_signal(symbol, horizons, model_name)
                results['signals'][symbol] = signal
                
                if 'error' in signal:
                    results['summary']['errors'] += 1
                else:
                    signal_type = signal['signal'].lower().replace(' ', '_')
                    if signal_type in results['summary']:
                        results['summary'][signal_type] += 1
                        
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
                results['signals'][symbol] = {
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                results['summary']['errors'] += 1
        
        return results
    
    def _analyze_predictions(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze predictions across horizons.
        
        Args:
            predictions: Predictions from multiple horizons
            
        Returns:
            Dict: Analysis results
        """
        analysis = {
            'horizon_signals': {},
            'consensus': 'HOLD',
            'consistency': 0.0,
            'average_change': 0.0
        }
        
        valid_predictions = []
        signal_votes = []
        
        for horizon, pred in predictions['predictions'].items():
            if 'error' in pred:
                analysis['horizon_signals'][horizon] = {
                    'signal': 'UNKNOWN',
                    'change_percent': 0,
                    'error': pred['error']
                }
                continue
            
            change_percent = pred.get('price_change_percent', 0)
            valid_predictions.append(change_percent)
            
            # Determine signal for this horizon
            if change_percent >= self.price_change_thresholds['strong_buy'] * 100:
                signal = SignalType.STRONG_BUY
            elif change_percent >= self.price_change_thresholds['buy'] * 100:
                signal = SignalType.BUY
            elif change_percent <= self.price_change_thresholds['strong_sell'] * 100:
                signal = SignalType.STRONG_SELL
            elif change_percent <= self.price_change_thresholds['sell'] * 100:
                signal = SignalType.SELL
            else:
                signal = SignalType.HOLD
            
            signal_votes.append(signal)
            analysis['horizon_signals'][horizon] = {
                'signal': signal.value,
                'change_percent': change_percent,
                'confidence': pred.get('prediction_quality', 'MEDIUM'),
                'current_price': pred.get('current_price', 0),
                'predicted_price': pred.get('predicted_price', 0)
            }
        
        if valid_predictions:
            analysis['average_change'] = np.mean(valid_predictions)
            
            # Calculate consistency (how much signals agree)
            if signal_votes:
                signal_counts = {}
                for signal in signal_votes:
                    signal_counts[signal] = signal_counts.get(signal, 0) + 1
                
                max_votes = max(signal_counts.values())
                analysis['consistency'] = max_votes / len(signal_votes)
                
                # Consensus is the most frequent signal
                consensus_signal = max(signal_counts, key=signal_counts.get)
                analysis['consensus'] = consensus_signal.value
        
        return analysis
    
    def _generate_final_signal(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate final trading signal from analysis.
        
        Args:
            analysis: Signal analysis results
            
        Returns:
            Dict: Final signal information
        """
        consensus = analysis['consensus']
        consistency = analysis['consistency']
        average_change = analysis['average_change']
        
        # Map consensus to SignalType
        signal_type = SignalType(consensus) if consensus != 'UNKNOWN' else SignalType.HOLD
        
        # Calculate signal strength (0-100)
        base_strength = abs(average_change) * 10  # Convert % to 0-100 scale
        consistency_bonus = consistency * 20  # Consistency adds up to 20 points
        signal_strength = min(100, base_strength + consistency_bonus)
        
        # Calculate confidence (0-1)
        confidence = consistency * 0.7 + min(1.0, abs(average_change) / 5) * 0.3
        
        return {
            'type': signal_type,
            'strength': float(signal_strength),
            'confidence': float(confidence),
            'average_change_percent': float(average_change),
            'consistency': float(consistency)
        }
    
    async def _assess_risk(self, symbol: str, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess risk level for the trading signal.
        
        Args:
            symbol: Trading symbol
            predictions: Prediction results
            
        Returns:
            Dict: Risk assessment
        """
        try:
            # Get recent historical data for volatility calculation
            historical_data = await self.predictor._get_prediction_data(symbol)
            
            if historical_data.empty:
                return {
                    'level': RiskLevel.HIGH,
                    'score': 0.8,
                    'factors': ['insufficient_data']
                }
            
            # Calculate recent volatility
            recent_returns = historical_data['close'].tail(30).pct_change()
            volatility = recent_returns.std()
            
            # Assess prediction consistency
            consistency = 0.5  # Default
            for horizon, pred in predictions['predictions'].items():
                if 'error' not in pred and pred.get('confidence_level'):
                    consistency = max(consistency, pred.get('confidence_level', 0.5))
            
            # Calculate risk score (0-1, higher = more risky)
            volatility_risk = min(1.0, volatility / self.risk_thresholds['volatility_high'])
            prediction_risk = 1.0 - consistency
            
            risk_score = (volatility_risk * 0.6 + prediction_risk * 0.4)
            
            # Determine risk level
            if risk_score >= 0.7:
                risk_level = RiskLevel.HIGH
            elif risk_score >= 0.4:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW
            
            risk_factors = []
            if volatility > self.risk_thresholds['volatility_high']:
                risk_factors.append('high_volatility')
            if consistency < 0.6:
                risk_factors.append('low_prediction_confidence')
            
            return {
                'level': risk_level,
                'score': float(risk_score),
                'volatility': float(volatility),
                'consistency': float(consistency),
                'factors': risk_factors
            }
            
        except Exception as e:
            logger.error(f"Error assessing risk for {symbol}: {e}")
            return {
                'level': RiskLevel.HIGH,
                'score': 0.8,
                'factors': ['assessment_error']
            }
    
    def _generate_recommendations(self, 
                                signal: Dict[str, Any], 
                                risk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading recommendations based on signal and risk.
        
        Args:
            signal: Signal information
            risk: Risk assessment
            
        Returns:
            Dict: Trading recommendations
        """
        recommendations = {
            'action': signal['type'].value,
            'position_size': 'standard',
            'stop_loss_percent': 5.0,
            'take_profit_percent': 10.0,
            'time_horizon': 'short_term',
            'notes': []
        }
        
        # Adjust position size based on risk
        if risk['level'] == RiskLevel.HIGH:
            recommendations['position_size'] = 'small'
            recommendations['stop_loss_percent'] = 3.0
            recommendations['notes'].append('Reduce position size due to high risk')
        elif risk['level'] == RiskLevel.LOW:
            recommendations['position_size'] = 'large'
            recommendations['stop_loss_percent'] = 7.0
            recommendations['notes'].append('Consider larger position due to low risk')
        
        # Adjust based on signal strength
        if signal['strength'] > 80:
            recommendations['take_profit_percent'] = 15.0
            recommendations['notes'].append('Strong signal - consider higher profit target')
        elif signal['strength'] < 40:
            recommendations['take_profit_percent'] = 5.0
            recommendations['notes'].append('Weak signal - conservative profit target')
        
        # Adjust based on confidence
        if signal['confidence'] < 0.5:
            recommendations['notes'].append('Low confidence - consider waiting for better setup')
        
        return recommendations