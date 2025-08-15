"""
🧠 Strategy Models
Modèles pour les stratégies de trading
"""

import uuid
from decimal import Decimal
from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import Column, String, Numeric, DateTime, Enum as SqlEnum, Boolean, Text, Integer, ForeignKey
from sqlalchemy.orm import relationship
from . import Base


class StrategyStatus(str, Enum):
    """Status des stratégies"""
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


class StrategyType(str, Enum):
    """Types de stratégies"""
    GRID = "GRID"
    DCA = "DCA"
    ARBITRAGE = "ARBITRAGE"
    MOMENTUM = "MOMENTUM"
    MEAN_REVERSION = "MEAN_REVERSION"
    AI_POWERED = "AI_POWERED"


class Strategy(Base):
    """Modèle pour les stratégies de trading"""
    __tablename__ = "strategies"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    strategy_type = Column(SqlEnum(StrategyType), nullable=False)
    
    # Configuration de la stratégie (JSON)
    config = Column(Text, nullable=False)  # JSON config
    
    # Status et contrôle
    status = Column(SqlEnum(StrategyStatus), default=StrategyStatus.STOPPED, nullable=False)
    is_active = Column(Boolean, default=False, nullable=False)
    
    # Métrics de performance
    total_trades = Column(Integer, default=0, nullable=False)
    winning_trades = Column(Integer, default=0, nullable=False)
    losing_trades = Column(Integer, default=0, nullable=False)
    total_pnl = Column(Numeric(20, 8), default=0, nullable=False)
    max_drawdown = Column(Numeric(10, 4), default=0, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    stopped_at = Column(DateTime, nullable=True)
    
    # Relations
    orders = relationship("Order", back_populates="strategy")
    positions = relationship("Position", back_populates="strategy")
    executions = relationship("StrategyExecution", back_populates="strategy")


class StrategyExecution(Base):
    """Modèle pour l'historique d'exécution des stratégies"""
    __tablename__ = "strategy_executions"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    strategy_id = Column(String(36), ForeignKey("strategies.id"), nullable=False, index=True)
    
    # Détails de l'exécution
    action = Column(String(50), nullable=False)  # "BUY", "SELL", "ADJUST", etc.
    result = Column(String(20), nullable=False)  # "SUCCESS", "FAILURE", "PARTIAL"
    
    # Données de l'exécution
    parameters = Column(Text, nullable=True)  # JSON des paramètres
    result_data = Column(Text, nullable=True)  # JSON du résultat
    error_message = Column(Text, nullable=True)
    
    # Performance
    execution_time_ms = Column(Integer, nullable=True)
    
    # Timestamp
    executed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relations
    strategy = relationship("Strategy", back_populates="executions")