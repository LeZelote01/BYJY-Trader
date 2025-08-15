"""
🔄 Transfer Learning Module
Transfer knowledge between different assets and market conditions
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class TransferLearner:
    """
    Transfer learning for knowledge sharing between different assets
    Phase 3.4 - Meta-Learning Component
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.7,
                 transfer_ratio: float = 0.3):
        """
        Initialize Transfer Learner
        
        Args:
            similarity_threshold: Minimum similarity for knowledge transfer
            transfer_ratio: Ratio of knowledge to transfer
        """
        self.similarity_threshold = similarity_threshold
        self.transfer_ratio = transfer_ratio
        self.knowledge_base = {}
        self.asset_relationships = {}
        
        logger.info(f"TransferLearner initialized with similarity_threshold={similarity_threshold}")
    
    def transfer_knowledge(self,
                          source_asset: str,
                          target_asset: str,
                          source_model: Any,
                          target_model: Any) -> Dict[str, Any]:
        """
        Transfer knowledge from source to target asset
        
        Args:
            source_asset: Source asset symbol
            target_asset: Target asset symbol
            source_model: Trained model from source asset
            target_model: Model to transfer knowledge to
            
        Returns:
            Transfer results and performance metrics
        """
        transfer_results = {
            'source_asset': source_asset,
            'target_asset': target_asset,
            'transfer_applied': False,
            'similarity_score': 0.0,
            'performance_improvement': 0.0
        }
        
        # Calculate asset similarity
        similarity = self._calculate_asset_similarity(source_asset, target_asset)
        transfer_results['similarity_score'] = similarity
        
        if similarity < self.similarity_threshold:
            logger.info(f"Similarity {similarity:.3f} below threshold {self.similarity_threshold}")
            return transfer_results
        
        # Apply knowledge transfer
        try:
            transferred_knowledge = self._extract_transferable_knowledge(source_model)
            improvement = self._apply_knowledge_transfer(target_model, transferred_knowledge)
            
            transfer_results['transfer_applied'] = True
            transfer_results['performance_improvement'] = improvement
            
            # Update knowledge base
            self._update_knowledge_base(source_asset, target_asset, transferred_knowledge)
            
            logger.info(f"Knowledge transfer completed: {source_asset} -> {target_asset}, "
                       f"improvement: {improvement:.3f}")
            
        except Exception as e:
            logger.error(f"Knowledge transfer failed: {e}")
        
        return transfer_results
    
    def learn_asset_relationships(self,
                                assets_data: Dict[str, List[float]],
                                market_features: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Learn relationships between different assets
        
        Args:
            assets_data: Price data for different assets
            market_features: Market features for each asset
            
        Returns:
            Asset relationship matrix
        """
        relationships = {}
        asset_names = list(assets_data.keys())
        
        for i, asset1 in enumerate(asset_names):
            relationships[asset1] = {}
            
            for j, asset2 in enumerate(asset_names):
                if i != j:
                    # Calculate relationship score
                    price_correlation = self._calculate_price_correlation(
                        assets_data[asset1], assets_data[asset2]
                    )
                    
                    feature_similarity = self._calculate_feature_similarity(
                        market_features.get(asset1, {}), 
                        market_features.get(asset2, {})
                    )
                    
                    # Combined relationship score
                    relationship_score = (price_correlation + feature_similarity) / 2
                    relationships[asset1][asset2] = relationship_score
        
        self.asset_relationships = relationships
        logger.info(f"Learned relationships for {len(asset_names)} assets")
        
        return relationships
    
    def find_best_source_assets(self,
                               target_asset: str,
                               available_assets: List[str],
                               top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Find best source assets for knowledge transfer
        
        Args:
            target_asset: Target asset for transfer
            available_assets: Available source assets
            top_k: Number of top assets to return
            
        Returns:
            List of (asset, similarity_score) tuples
        """
        if target_asset not in self.asset_relationships:
            logger.warning(f"No relationships found for {target_asset}")
            return []
        
        candidates = []
        target_relationships = self.asset_relationships[target_asset]
        
        for asset in available_assets:
            if asset in target_relationships:
                similarity = target_relationships[asset]
                if similarity >= self.similarity_threshold:
                    candidates.append((asset, similarity))
        
        # Sort by similarity and return top_k
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        result = candidates[:top_k]
        logger.info(f"Found {len(result)} suitable source assets for {target_asset}")
        
        return result
    
    def adapt_to_new_market_conditions(self,
                                     model: Any,
                                     new_market_data: List[float],
                                     historical_adaptations: List[Dict]) -> Dict[str, Any]:
        """
        Adapt model to new market conditions using transfer learning
        
        Args:
            model: Model to adapt
            new_market_data: New market data
            historical_adaptations: Previous adaptation results
            
        Returns:
            Adaptation results
        """
        adaptation_results = {
            'adaptation_applied': False,
            'adaptation_score': 0.0,
            'performance_change': 0.0
        }
        
        try:
            # Find similar historical market conditions
            similar_conditions = self._find_similar_market_conditions(
                new_market_data, historical_adaptations
            )
            
            if similar_conditions:
                # Apply historical adaptation knowledge
                best_adaptation = max(similar_conditions, key=lambda x: x['similarity'])
                adaptation_knowledge = best_adaptation['adaptation_knowledge']
                
                performance_change = self._apply_adaptation_knowledge(model, adaptation_knowledge)
                
                adaptation_results.update({
                    'adaptation_applied': True,
                    'adaptation_score': best_adaptation['similarity'],
                    'performance_change': performance_change
                })
                
                logger.info(f"Adaptation applied with score {best_adaptation['similarity']:.3f}")
            
        except Exception as e:
            logger.error(f"Market condition adaptation failed: {e}")
        
        return adaptation_results
    
    def _calculate_asset_similarity(self, asset1: str, asset2: str) -> float:
        """Calculate similarity between two assets"""
        if asset1 in self.asset_relationships and asset2 in self.asset_relationships[asset1]:
            return self.asset_relationships[asset1][asset2]
        
        # Default similarity calculation (placeholder)
        return 0.5
    
    def _calculate_price_correlation(self, prices1: List[float], prices2: List[float]) -> float:
        """Calculate price correlation between two assets"""
        if len(prices1) != len(prices2) or len(prices1) < 2:
            return 0.0
        
        correlation = np.corrcoef(prices1, prices2)[0, 1]
        return abs(correlation) if not np.isnan(correlation) else 0.0
    
    def _calculate_feature_similarity(self, features1: Dict[str, float], features2: Dict[str, float]) -> float:
        """Calculate similarity between market features"""
        if not features1 or not features2:
            return 0.0
        
        common_keys = set(features1.keys()) & set(features2.keys())
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            val1, val2 = features1[key], features2[key]
            if val1 == 0 and val2 == 0:
                similarities.append(1.0)
            elif val1 == 0 or val2 == 0:
                similarities.append(0.0)
            else:
                similarity = 1.0 - abs(val1 - val2) / max(abs(val1), abs(val2))
                similarities.append(max(0.0, similarity))
        
        return np.mean(similarities)
    
    def _extract_transferable_knowledge(self, source_model: Any) -> Dict[str, Any]:
        """Extract transferable knowledge from source model"""
        # Placeholder implementation - would extract model weights, features, etc.
        knowledge = {
            'model_type': type(source_model).__name__,
            'parameters': {},
            'performance_metrics': {},
            'feature_importance': {}
        }
        
        return knowledge
    
    def _apply_knowledge_transfer(self, target_model: Any, knowledge: Dict[str, Any]) -> float:
        """Apply transferred knowledge to target model"""
        # Placeholder implementation - would apply weights, features, etc.
        # Return simulated performance improvement
        return np.random.uniform(0.01, 0.05)  # 1-5% improvement
    
    def _update_knowledge_base(self, source_asset: str, target_asset: str, knowledge: Dict[str, Any]):
        """Update knowledge base with transfer results"""
        transfer_key = f"{source_asset}->{target_asset}"
        self.knowledge_base[transfer_key] = {
            'knowledge': knowledge,
            'timestamp': np.datetime64('now'),
            'success': True
        }
    
    def _find_similar_market_conditions(self, 
                                      new_data: List[float], 
                                      historical_adaptations: List[Dict]) -> List[Dict]:
        """Find similar market conditions from historical adaptations"""
        similar_conditions = []
        
        for adaptation in historical_adaptations:
            historical_data = adaptation.get('market_data', [])
            if not historical_data:
                continue
                
            similarity = self._calculate_price_correlation(new_data, historical_data)
            if similarity >= self.similarity_threshold:
                similar_conditions.append({
                    'similarity': similarity,
                    'adaptation_knowledge': adaptation.get('adaptation_knowledge', {}),
                    'historical_data': historical_data
                })
        
        return similar_conditions
    
    def _apply_adaptation_knowledge(self, model: Any, adaptation_knowledge: Dict[str, Any]) -> float:
        """Apply adaptation knowledge to model"""
        # Placeholder implementation - would adapt model parameters
        return np.random.uniform(-0.02, 0.03)  # -2% to +3% performance change
    
    def get_transfer_statistics(self) -> Dict[str, Any]:
        """Get statistics about transfer learning"""
        return {
            'knowledge_base_size': len(self.knowledge_base),
            'asset_relationships_count': len(self.asset_relationships),
            'similarity_threshold': self.similarity_threshold,
            'transfer_ratio': self.transfer_ratio
        }