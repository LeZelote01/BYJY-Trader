"""
🧠 Meta-Learning Module
Pattern recognition and adaptive learning systems
"""

from .meta_learner import MetaLearner
from .adaptation_engine import AdaptationEngine
from .pattern_recognizer import PatternRecognizer
from .transfer_learning import TransferLearner
from .few_shot_learner import FewShotLearner

__all__ = [
    'MetaLearner',
    'AdaptationEngine',
    'PatternRecognizer',
    'TransferLearner',
    'FewShotLearner'
]