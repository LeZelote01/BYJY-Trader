"""
🗄️ Database Models
Modèles de base de données SQLAlchemy pour BYJY-Trader
"""

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Base pour tous les modèles
Base = declarative_base()

# Import de tous les modèles pour registration automatique
from .trading import TradingPair, Order, Trade, Position
from .strategy import Strategy, StrategyExecution
from .system import SystemLog, Configuration
# User models from user_db to avoid conflicts

__all__ = [
    'Base',
    'TradingPair', 'Order', 'Trade', 'Position',
    'Strategy', 'StrategyExecution',
    'SystemLog', 'Configuration'
]