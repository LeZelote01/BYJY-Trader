"""
🔍 Pattern Recognizer Module
Recognition of learning patterns and market regime patterns
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class PatternRecognizer:
    """
    Pattern recognition for learning curves and market regimes
    Phase 3.4 - Meta-Learning Component
    """
    
    def __init__(self, 
                 window_size: int = 50,
                 similarity_threshold: float = 0.8):
        """
        Initialize Pattern Recognizer
        
        Args:
            window_size: Size of the pattern window
            similarity_threshold: Threshold for pattern similarity
        """
        self.window_size = window_size
        self.similarity_threshold = similarity_threshold
        self.learning_patterns = {}
        self.market_patterns = {}
        
        logger.info(f"PatternRecognizer initialized with window_size={window_size}")
    
    def recognize_learning_patterns(self, 
                                  learning_curves: List[List[float]],
                                  model_types: List[str]) -> Dict[str, Any]:
        """
        Recognize patterns in learning curves
        
        Args:
            learning_curves: List of learning curves for different models
            model_types: Types of models corresponding to curves
            
        Returns:
            Dictionary containing recognized patterns
        """
        patterns = {
            'convergence_patterns': [],
            'optimization_patterns': [],
            'performance_patterns': []
        }
        
        for i, curve in enumerate(learning_curves):
            if len(curve) < self.window_size:
                continue
                
            model_type = model_types[i] if i < len(model_types) else 'unknown'
            
            # Analyze convergence pattern
            convergence = self._analyze_convergence_pattern(curve)
            patterns['convergence_patterns'].append({
                'model_type': model_type,
                'convergence_rate': convergence['rate'],
                'stability': convergence['stability'],
                'final_performance': curve[-1] if curve else 0.0
            })
        
        logger.info(f"Recognized {len(patterns['convergence_patterns'])} learning patterns")
        return patterns
    
    def recognize_market_patterns(self, 
                                price_data: List[float],
                                volume_data: List[float],
                                indicators: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Recognize market regime patterns
        
        Args:
            price_data: Historical price data
            volume_data: Historical volume data  
            indicators: Technical indicators
            
        Returns:
            Dictionary containing market patterns
        """
        patterns = {
            'regime_changes': [],
            'volatility_patterns': [],
            'trend_patterns': []
        }
        
        if len(price_data) < self.window_size:
            return patterns
            
        # Detect regime changes
        regime_changes = self._detect_regime_changes(price_data)
        patterns['regime_changes'] = regime_changes
        
        # Analyze volatility patterns
        volatility_patterns = self._analyze_volatility_patterns(price_data)
        patterns['volatility_patterns'] = volatility_patterns
        
        logger.info(f"Recognized market patterns with {len(regime_changes)} regime changes")
        return patterns
    
    def find_similar_patterns(self,
                            current_pattern: List[float],
                            historical_patterns: List[List[float]]) -> List[Tuple[int, float]]:
        """
        Find similar patterns in historical data
        
        Args:
            current_pattern: Current pattern to match
            historical_patterns: Historical patterns to search
            
        Returns:
            List of (index, similarity_score) tuples
        """
        similarities = []
        
        for i, hist_pattern in enumerate(historical_patterns):
            similarity = self._calculate_pattern_similarity(current_pattern, hist_pattern)
            if similarity >= self.similarity_threshold:
                similarities.append((i, similarity))
        
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        logger.debug(f"Found {len(similarities)} similar patterns")
        return similarities
    
    def _analyze_convergence_pattern(self, learning_curve: List[float]) -> Dict[str, float]:
        """Analyze convergence characteristics of learning curve"""
        if len(learning_curve) < 2:
            return {'rate': 0.0, 'stability': 0.0}
            
        # Calculate convergence rate (improvement over time)
        improvements = [learning_curve[i+1] - learning_curve[i] for i in range(len(learning_curve)-1)]
        avg_improvement = np.mean(improvements) if improvements else 0.0
        
        # Calculate stability (variance in later stages)
        later_half = learning_curve[len(learning_curve)//2:]
        stability = 1.0 / (1.0 + np.var(later_half)) if len(later_half) > 1 else 0.0
        
        return {
            'rate': float(avg_improvement),
            'stability': float(stability)
        }
    
    def _detect_regime_changes(self, price_data: List[float]) -> List[Dict[str, Any]]:
        """Detect regime changes in price data"""
        regime_changes = []
        
        if len(price_data) < self.window_size:
            return regime_changes
            
        # Simple regime change detection based on volatility shifts
        for i in range(self.window_size, len(price_data) - self.window_size):
            window_before = price_data[i-self.window_size:i]
            window_after = price_data[i:i+self.window_size]
            
            vol_before = np.std(window_before)
            vol_after = np.std(window_after)
            
            # Detect significant volatility shift
            if vol_after > 1.5 * vol_before or vol_after < 0.5 * vol_before:
                regime_changes.append({
                    'index': i,
                    'type': 'volatility_shift',
                    'vol_before': float(vol_before),
                    'vol_after': float(vol_after),
                    'intensity': float(abs(vol_after - vol_before) / vol_before)
                })
        
        return regime_changes
    
    def _analyze_volatility_patterns(self, price_data: List[float]) -> List[Dict[str, Any]]:
        """Analyze volatility patterns in price data"""
        patterns = []
        
        if len(price_data) < self.window_size:
            return patterns
            
        # Calculate rolling volatility
        for i in range(self.window_size, len(price_data)):
            window = price_data[i-self.window_size:i]
            volatility = np.std(window)
            
            patterns.append({
                'index': i,
                'volatility': float(volatility),
                'mean_price': float(np.mean(window))
            })
        
        return patterns
    
    def _calculate_pattern_similarity(self, 
                                    pattern1: List[float], 
                                    pattern2: List[float]) -> float:
        """Calculate similarity between two patterns"""
        if len(pattern1) != len(pattern2) or not pattern1:
            return 0.0
            
        # Normalize patterns
        p1_norm = self._normalize_pattern(pattern1)
        p2_norm = self._normalize_pattern(pattern2)
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(p1_norm, p2_norm)[0, 1]
        
        # Return absolute correlation (similarity regardless of direction)
        return abs(correlation) if not np.isnan(correlation) else 0.0
    
    def _normalize_pattern(self, pattern: List[float]) -> List[float]:
        """Normalize pattern to zero mean and unit variance"""
        pattern_array = np.array(pattern)
        mean = np.mean(pattern_array)
        std = np.std(pattern_array)
        
        if std == 0:
            return [0.0] * len(pattern)
            
        return ((pattern_array - mean) / std).tolist()
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get statistics about recognized patterns"""
        return {
            'learning_patterns_count': len(self.learning_patterns),
            'market_patterns_count': len(self.market_patterns),
            'window_size': self.window_size,
            'similarity_threshold': self.similarity_threshold
        }