"""
🧠 Sentiment Analyzer - Phase 3.2  
Analyseur de sentiment NLP avec modèles pré-entraînés
Classification sentiment: Positive, Negative, Neutral
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import re
import math
from dataclasses import dataclass

from core.logger import get_logger

logger = get_logger("byjy.ai.sentiment.analyzer")


@dataclass
class SentimentResult:
    """Résultat d'analyse sentiment"""
    text: str
    label: str  # 'positive', 'negative', 'neutral'
    score: float  # -1.0 à +1.0
    confidence: float  # 0.0 à 1.0
    keywords: List[str]
    processed_at: datetime


class SentimentAnalyzer:
    """Analyseur sentiment avec NLP avancé"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.model_type = config.get('model_type', 'lexicon')  # 'lexicon', 'transformers', 'ensemble'
        self.is_initialized = False
        
        # Lexiques sentiment (fallback method)
        self.positive_words = set([
            'bullish', 'moon', 'pump', 'up', 'buy', 'bull', 'good', 'strong', 'high',
            'rise', 'gain', 'profit', 'win', 'success', 'growth', 'positive', 'optimistic',
            'confident', 'excellent', 'great', 'amazing', 'fantastic', 'breakthrough',
            'surge', 'rally', 'boom', 'diamond', 'hodl', 'to the moon', 'lambo'
        ])
        
        self.negative_words = set([
            'bearish', 'dump', 'down', 'sell', 'bear', 'bad', 'weak', 'low', 'crash',
            'fall', 'loss', 'lose', 'fail', 'decline', 'drop', 'negative', 'pessimistic',
            'worried', 'terrible', 'awful', 'disaster', 'collapse', 'plummet',
            'correction', 'dip', 'fud', 'panic', 'fear', 'dead', 'scam'
        ])
        
        # Amplificateurs
        self.amplifiers = {
            'very': 1.5, 'extremely': 2.0, 'super': 1.8, 'really': 1.3,
            'totally': 1.6, 'completely': 1.7, 'absolutely': 1.9
        }
        
        # Négations
        self.negations = {
            'not', 'no', 'never', 'none', 'neither', 'nothing', 'nobody',
            'nowhere', "don't", "doesn't", "didn't", "won't", "wouldn't", 
            "can't", "couldn't", "shouldn't", "isn't", "aren't", "wasn't", "weren't"
        }
        
        self.processed_count = 0
        
    async def initialize(self) -> bool:
        """Initialisation de l'analyseur"""
        try:
            if self.model_type == 'transformers':
                # TODO: Charger modèles Transformers (BERT, RoBERTa, FinBERT)
                # from transformers import AutoTokenizer, AutoModelForSequenceClassification
                # self.tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
                # self.model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
                logger.warning("🚧 Transformers model loading not implemented - using lexicon fallback")
                
            elif self.model_type == 'ensemble':
                # TODO: Modèle ensemble combinant plusieurs approches
                logger.warning("🚧 Ensemble model not implemented - using lexicon fallback")
            
            # Mode lexicon toujours disponible
            self.is_initialized = True
            logger.info(f"✅ Sentiment analyzer initialized (method: {self.model_type})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Sentiment analyzer initialization failed: {e}")
            return False
    
    async def analyze_text(self, text: str) -> SentimentResult:
        """
        Analyser sentiment d'un texte
        
        Args:
            text: Texte à analyser
            
        Returns:
            SentimentResult avec label, score et métadonnées
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Préprocessing
            processed_text = self._preprocess_text(text)
            
            # Analyse selon méthode configurée
            if self.model_type == 'transformers' and hasattr(self, 'model'):
                result = await self._analyze_with_transformers(processed_text)
            elif self.model_type == 'ensemble':
                result = await self._analyze_with_ensemble(processed_text)
            else:
                # Méthode lexicon (fallback)
                result = await self._analyze_with_lexicon(processed_text)
            
            self.processed_count += 1
            
            return SentimentResult(
                text=text,
                label=result['label'],
                score=result['score'],
                confidence=result['confidence'],
                keywords=result['keywords'],
                processed_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"❌ Sentiment analysis failed for text: {text[:50]}... Error: {e}")
            return SentimentResult(
                text=text,
                label='neutral',
                score=0.0,
                confidence=0.0,
                keywords=[],
                processed_at=datetime.now()
            )
    
    async def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """
        Analyser sentiment en batch
        
        Args:
            texts: Liste de textes à analyser
            
        Returns:
            Liste de SentimentResult
        """
        tasks = [self.analyze_text(text) for text in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filtrer les exceptions
        valid_results = []
        for result in results:
            if isinstance(result, SentimentResult):
                valid_results.append(result)
            else:
                logger.error(f"❌ Batch analysis error: {result}")
                # Ajouter résultat neutre par défaut
                valid_results.append(SentimentResult(
                    text="", label='neutral', score=0.0, confidence=0.0,
                    keywords=[], processed_at=datetime.now()
                ))
        
        return valid_results
    
    def _preprocess_text(self, text: str) -> str:
        """Préprocessing du texte"""
        if not text:
            return ""
        
        # Normalisation
        text = text.lower().strip()
        
        # Suppression URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Suppression mentions (@user)
        text = re.sub(r'@[A-Za-z0-9_]+', '', text)
        
        # Suppression hashtags (garder contenu)
        text = re.sub(r'#([A-Za-z0-9_]+)', r'\1', text)
        
        # Suppression caractères spéciaux excessifs
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Espaces multiples
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    async def _analyze_with_lexicon(self, text: str) -> Dict[str, Any]:
        """Analyse sentiment basée sur lexique"""
        if not text:
            return {'label': 'neutral', 'score': 0.0, 'confidence': 0.0, 'keywords': []}
        
        words = text.split()
        sentiment_score = 0.0
        found_keywords = []
        
        i = 0
        while i < len(words):
            word = words[i]
            
            # Vérifier négations
            is_negated = False
            if i > 0 and words[i-1] in self.negations:
                is_negated = True
            
            # Vérifier amplificateurs
            amplifier = 1.0
            if i > 0 and words[i-1] in self.amplifiers:
                amplifier = self.amplifiers[words[i-1]]
            
            # Score sentiment du mot
            word_sentiment = 0.0
            if word in self.positive_words:
                word_sentiment = 1.0
                found_keywords.append(f"+{word}")
            elif word in self.negative_words:
                word_sentiment = -1.0
                found_keywords.append(f"-{word}")
            
            # Application modificateurs
            if word_sentiment != 0.0:
                if is_negated:
                    word_sentiment *= -1
                    found_keywords[-1] = f"!{found_keywords[-1]}"
                
                word_sentiment *= amplifier
                sentiment_score += word_sentiment
            
            i += 1
        
        # Normalisation score
        if len(words) > 0:
            sentiment_score = max(-1.0, min(1.0, sentiment_score / len(words) * 10))
        
        # Détermination label
        if sentiment_score > 0.1:
            label = 'positive'
        elif sentiment_score < -0.1:
            label = 'negative'  
        else:
            label = 'neutral'
        
        # Confidence basée sur l'amplitude du score
        confidence = min(1.0, abs(sentiment_score) * 2)
        
        return {
            'label': label,
            'score': sentiment_score,
            'confidence': confidence,
            'keywords': found_keywords
        }
    
    async def _analyze_with_transformers(self, text: str) -> Dict[str, Any]:
        """Analyse sentiment avec modèles Transformers"""
        # TODO: Implémentation avec transformers library
        # 
        # inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        # outputs = self.model(**inputs)
        # predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        # 
        # Mapping des labels selon le modèle utilisé
        
        logger.info("🚧 Transformers analysis not implemented - using lexicon fallback")
        return await self._analyze_with_lexicon(text)
    
    async def _analyze_with_ensemble(self, text: str) -> Dict[str, Any]:
        """Analyse sentiment avec modèle ensemble"""
        # TODO: Combinaison plusieurs méthodes avec pondération
        #
        # lexicon_result = await self._analyze_with_lexicon(text)
        # transformer_result = await self._analyze_with_transformers(text)
        # 
        # Fusion pondérée des résultats
        
        logger.info("🚧 Ensemble analysis not implemented - using lexicon fallback") 
        return await self._analyze_with_lexicon(text)
    
    async def get_sentiment_distribution(self, texts: List[str]) -> Dict[str, Any]:
        """Distribution sentiment sur ensemble de textes"""
        results = await self.analyze_batch(texts)
        
        # Compter par label
        label_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        total_score = 0.0
        total_confidence = 0.0
        
        for result in results:
            label_counts[result.label] += 1
            total_score += result.score
            total_confidence += result.confidence
        
        total = len(results)
        if total == 0:
            return {
                'distribution': label_counts,
                'percentages': {'positive': 0, 'negative': 0, 'neutral': 0},
                'average_score': 0.0,
                'average_confidence': 0.0,
                'total_analyzed': 0
            }
        
        return {
            'distribution': label_counts,
            'percentages': {
                'positive': round(label_counts['positive'] / total * 100, 2),
                'negative': round(label_counts['negative'] / total * 100, 2),
                'neutral': round(label_counts['neutral'] / total * 100, 2)
            },
            'average_score': round(total_score / total, 3),
            'average_confidence': round(total_confidence / total, 3),
            'total_analyzed': total,
            'sentiment_index': self._calculate_sentiment_index(label_counts, total)
        }
    
    def _calculate_sentiment_index(self, label_counts: Dict[str, int], total: int) -> float:
        """Calcul index sentiment global (-100 à +100)"""
        if total == 0:
            return 0.0
        
        positive_ratio = label_counts['positive'] / total
        negative_ratio = label_counts['negative'] / total
        
        # Index entre -100 (100% négatif) et +100 (100% positif)
        sentiment_index = (positive_ratio - negative_ratio) * 100
        
        return round(sentiment_index, 2)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Statistiques de l'analyseur"""
        return {
            'analyzer': 'sentiment_analyzer',
            'model_type': self.model_type,
            'initialized': self.is_initialized,
            'processed_count': self.processed_count,
            'vocabulary_size': len(self.positive_words) + len(self.negative_words),
            'positive_words': len(self.positive_words),
            'negative_words': len(self.negative_words),
            'status': 'active' if self.is_initialized else 'inactive'
        }