"""
📊 Market Regime Detector Module
Automatic detection of market regimes
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """
    Market regime detection using machine learning
    Phase 3.4 - Adaptive Strategies Component
    """
    
    def __init__(self, 
                 n_regimes: int = 3,
                 window_size: int = 50,
                 min_regime_duration: int = 5):
        """
        Initialize Market Regime Detector
        
        Args:
            n_regimes: Number of market regimes to detect
            window_size: Window size for regime analysis
            min_regime_duration: Minimum duration for regime stability
        """
        self.n_regimes = n_regimes
        self.window_size = window_size
        self.min_regime_duration = min_regime_duration
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_regimes, random_state=42)
        self.regime_history = []
        
        logger.info(f"MarketRegimeDetector initialized with {n_regimes} regimes")
    
    def detect_current_regime(self,
                             price_data: List[float],
                             volume_data: List[float] = None,
                             indicators: Dict[str, List[float]] = None) -> Dict[str, Any]:
        """
        Detect current market regime
        
        Args:
            price_data: Historical price data
            volume_data: Historical volume data
            indicators: Technical indicators
            
        Returns:
            Current regime information
        """
        regime_info = {
            'regime_id': 0,
            'regime_name': 'unknown',
            'confidence': 0.0,
            'characteristics': {},
            'stability': 0.0
        }
        
        if len(price_data) < self.window_size:
            logger.warning(f"Insufficient data for regime detection: {len(price_data)} < {self.window_size}")
            return regime_info
        
        try:
            # Extract features for regime detection
            features = self._extract_regime_features(price_data, volume_data, indicators)
            
            if not features:
                return regime_info
            
            # Normalize features
            features_array = np.array([features]).reshape(1, -1)
            features_scaled = self.scaler.fit_transform(features_array)
            
            # Predict regime
            regime_id = self.kmeans.fit_predict(features_scaled)[0]
            
            # Calculate confidence (distance to cluster centers)
            distances = self.kmeans.transform(features_scaled)[0]
            confidence = 1.0 - (distances[regime_id] / np.sum(distances))
            
            # Determine regime characteristics
            regime_characteristics = self._analyze_regime_characteristics(features, regime_id)
            
            # Calculate regime stability
            stability = self._calculate_regime_stability(regime_id)
            
            regime_info.update({
                'regime_id': int(regime_id),
                'regime_name': self._get_regime_name(regime_id, regime_characteristics),
                'confidence': float(confidence),
                'characteristics': regime_characteristics,
                'stability': stability
            })
            
            # Update regime history
            self.regime_history.append({
                'regime_id': regime_id,
                'timestamp': len(self.regime_history),
                'features': features
            })
            
            logger.info(f"Detected regime: {regime_info['regime_name']} (confidence: {confidence:.3f})")
            
        except Exception as e:
            logger.error(f"Regime detection failed: {e}")
            regime_info['error'] = str(e)
        
        return regime_info
    
    def get_regime_transitions(self) -> List[Dict[str, Any]]:
        """Get regime transition points"""
        transitions = []
        
        if len(self.regime_history) < 2:
            return transitions
        
        current_regime = self.regime_history[0]['regime_id']
        transition_start = 0
        
        for i, regime_data in enumerate(self.regime_history[1:], 1):
            if regime_data['regime_id'] != current_regime:
                # Regime transition detected
                transitions.append({
                    'from_regime': current_regime,
                    'to_regime': regime_data['regime_id'],
                    'transition_point': i,
                    'duration': i - transition_start
                })
                
                current_regime = regime_data['regime_id']
                transition_start = i
        
        return transitions
    
    def predict_regime_duration(self, current_regime_id: int) -> Dict[str, Any]:
        """Predict expected duration of current regime"""
        prediction = {
            'expected_duration': 0,
            'confidence': 0.0,
            'historical_average': 0.0
        }
        
        if not self.regime_history:
            return prediction
        
        # Calculate historical durations for this regime
        regime_durations = []
        current_regime = None
        regime_start = 0
        
        for i, regime_data in enumerate(self.regime_history):
            regime_id = regime_data['regime_id']
            
            if current_regime != regime_id:
                if current_regime == current_regime_id and i > regime_start:
                    regime_durations.append(i - regime_start)
                
                current_regime = regime_id
                regime_start = i
        
        if regime_durations:
            avg_duration = np.mean(regime_durations)
            std_duration = np.std(regime_durations)
            
            prediction.update({
                'expected_duration': int(avg_duration),
                'confidence': 1.0 / (1.0 + std_duration / max(avg_duration, 1.0)),
                'historical_average': float(avg_duration)
            })
        
        return prediction
    
    def _extract_regime_features(self,
                                price_data: List[float],
                                volume_data: Optional[List[float]],
                                indicators: Optional[Dict[str, List[float]]]) -> List[float]:
        """Extract features for regime detection"""
        if len(price_data) < self.window_size:
            return []
        
        # Use latest window
        recent_prices = price_data[-self.window_size:]
        
        features = []
        
        # Price-based features
        returns = [recent_prices[i] - recent_prices[i-1] for i in range(1, len(recent_prices))]
        
        features.extend([
            np.mean(returns),  # Mean return
            np.std(returns),   # Volatility
            np.mean([abs(r) for r in returns]),  # Mean absolute return
            min(returns),      # Min return
            max(returns),      # Max return
        ])
        
        # Volume features (if available)
        if volume_data and len(volume_data) >= self.window_size:
            recent_volume = volume_data[-self.window_size:]
            features.extend([
                np.mean(recent_volume),
                np.std(recent_volume)
            ])
        else:
            features.extend([0.0, 0.0])  # Placeholder
        
        # Technical indicator features (if available)
        if indicators:
            for indicator_name, values in indicators.items():
                if len(values) >= self.window_size:
                    recent_values = values[-self.window_size:]
                    features.extend([
                        np.mean(recent_values),
                        np.std(recent_values)
                    ])
        
        return features
    
    def _analyze_regime_characteristics(self, features: List[float], regime_id: int) -> Dict[str, Any]:
        """Analyze characteristics of detected regime"""
        if len(features) < 5:
            return {}
        
        characteristics = {
            'volatility_level': 'medium',
            'trend_direction': 'sideways',
            'volume_level': 'normal'
        }
        
        # Analyze volatility
        volatility = features[1] if len(features) > 1 else 0.0
        if volatility > 0.02:
            characteristics['volatility_level'] = 'high'
        elif volatility < 0.005:
            characteristics['volatility_level'] = 'low'
        
        # Analyze trend
        mean_return = features[0] if len(features) > 0 else 0.0
        if mean_return > 0.001:
            characteristics['trend_direction'] = 'bullish'
        elif mean_return < -0.001:
            characteristics['trend_direction'] = 'bearish'
        
        # Analyze volume (if available)
        if len(features) > 6:
            volume_mean = features[5]
            volume_std = features[6]
            
            if volume_std > volume_mean * 0.5:
                characteristics['volume_level'] = 'volatile'
            elif volume_mean > 1000000:  # Arbitrary threshold
                characteristics['volume_level'] = 'high'
        
        return characteristics
    
    def _get_regime_name(self, regime_id: int, characteristics: Dict[str, Any]) -> str:
        """Get descriptive name for regime"""
        volatility = characteristics.get('volatility_level', 'medium')
        trend = characteristics.get('trend_direction', 'sideways')
        
        regime_names = {
            ('low', 'bullish'): 'steady_bull',
            ('low', 'bearish'): 'steady_bear', 
            ('low', 'sideways'): 'quiet_market',
            ('medium', 'bullish'): 'normal_bull',
            ('medium', 'bearish'): 'normal_bear',
            ('medium', 'sideways'): 'choppy_market',
            ('high', 'bullish'): 'volatile_bull',
            ('high', 'bearish'): 'volatile_bear',
            ('high', 'sideways'): 'high_volatility'
        }
        
        return regime_names.get((volatility, trend), f'regime_{regime_id}')
    
    def _calculate_regime_stability(self, current_regime_id: int) -> float:
        """Calculate stability of current regime"""
        if len(self.regime_history) < self.min_regime_duration:
            return 0.0
        
        recent_regimes = [r['regime_id'] for r in self.regime_history[-self.min_regime_duration:]]
        stability = recent_regimes.count(current_regime_id) / len(recent_regimes)
        
        return stability
    
    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get statistics about regime detection"""
        if not self.regime_history:
            return {}
        
        regime_counts = {}
        for regime_data in self.regime_history:
            regime_id = regime_data['regime_id']
            regime_counts[regime_id] = regime_counts.get(regime_id, 0) + 1
        
        transitions = self.get_regime_transitions()
        
        return {
            'total_observations': len(self.regime_history),
            'regime_distribution': regime_counts,
            'number_of_transitions': len(transitions),
            'average_regime_duration': np.mean([t['duration'] for t in transitions]) if transitions else 0,
            'current_regime': self.regime_history[-1]['regime_id'] if self.regime_history else None
        }