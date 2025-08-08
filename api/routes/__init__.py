"""
🌐 API Routes - BYJY-Trader
Import centralisé de toutes les routes API
"""

from . import (
    health,
    system,
    trading,
    auth,
    data_collection,
    ai_predictions,
    connectors,
    backtesting,
    strategies,
    risk_management,
    ensemble_predictions,
    sentiment_analysis  # 🆕 Phase 3.2
)

__all__ = [
    'health',
    'system', 
    'trading',
    'auth',
    'data_collection',
    'ai_predictions',
    'connectors',
    'backtesting',
    'strategies',
    'risk_management',
    'ensemble_predictions',
    'sentiment_analysis'  # 🆕 Phase 3.2
]