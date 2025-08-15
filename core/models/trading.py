"""
📈 Trading Models
Modèles pour les données de trading
"""

import uuid
from decimal import Decimal
from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import Column, String, Numeric, DateTime, Enum as SqlEnum, Boolean, Text, ForeignKey
from sqlalchemy.orm import relationship
from . import Base


class OrderStatus(str, Enum):
    """Status des ordres"""
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class OrderType(str, Enum):
    """Types d'ordres"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSide(str, Enum):
    """Côté de l'ordre"""
    BUY = "BUY"
    SELL = "SELL"


class TradingPair(Base):
    """Modèle pour les paires de trading"""
    __tablename__ = "trading_pairs"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    symbol = Column(String(20), unique=True, nullable=False, index=True)
    base_asset = Column(String(10), nullable=False)
    quote_asset = Column(String(10), nullable=False)
    min_quantity = Column(Numeric(20, 8), nullable=False, default=0.00001)
    max_quantity = Column(Numeric(20, 8), nullable=True)
    step_size = Column(Numeric(20, 8), nullable=False, default=0.00001)
    min_price = Column(Numeric(20, 8), nullable=False, default=0.00000001)
    max_price = Column(Numeric(20, 8), nullable=True)
    tick_size = Column(Numeric(20, 8), nullable=False, default=0.00000001)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relations
    orders = relationship("Order", back_populates="trading_pair")
    trades = relationship("Trade", back_populates="trading_pair")
    positions = relationship("Position", back_populates="trading_pair")


class Order(Base):
    """Modèle pour les ordres"""
    __tablename__ = "orders"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    exchange_id = Column(String(100), nullable=True, index=True)
    trading_pair_id = Column(String(36), ForeignKey("trading_pairs.id"), nullable=False)
    strategy_id = Column(String(36), ForeignKey("strategies.id"), nullable=True)
    
    # Détails de l'ordre
    order_type = Column(SqlEnum(OrderType), nullable=False)
    side = Column(SqlEnum(OrderSide), nullable=False)
    quantity = Column(Numeric(20, 8), nullable=False)
    price = Column(Numeric(20, 8), nullable=True)
    stop_price = Column(Numeric(20, 8), nullable=True)
    
    # Status et exécution
    status = Column(SqlEnum(OrderStatus), default=OrderStatus.PENDING, nullable=False)
    filled_quantity = Column(Numeric(20, 8), default=0, nullable=False)
    average_price = Column(Numeric(20, 8), nullable=True)
    commission = Column(Numeric(20, 8), default=0, nullable=False)
    commission_asset = Column(String(10), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    executed_at = Column(DateTime, nullable=True)
    
    # Métadonnées
    extra_data = Column(Text, nullable=True)  # JSON pour données supplémentaires
    
    # Relations
    trading_pair = relationship("TradingPair", back_populates="orders")
    strategy = relationship("Strategy", back_populates="orders")
    trades = relationship("Trade", back_populates="order")


class Trade(Base):
    """Modèle pour les trades exécutés"""
    __tablename__ = "trades"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    exchange_id = Column(String(100), nullable=True, index=True)
    order_id = Column(String(36), ForeignKey("orders.id"), nullable=False)
    trading_pair_id = Column(String(36), ForeignKey("trading_pairs.id"), nullable=False)
    
    # Détails du trade
    side = Column(SqlEnum(OrderSide), nullable=False)
    quantity = Column(Numeric(20, 8), nullable=False)
    price = Column(Numeric(20, 8), nullable=False)
    quote_quantity = Column(Numeric(20, 8), nullable=False)
    
    # Commission
    commission = Column(Numeric(20, 8), default=0, nullable=False)
    commission_asset = Column(String(10), nullable=True)
    
    # Timestamps
    executed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relations
    order = relationship("Order", back_populates="trades")
    trading_pair = relationship("TradingPair", back_populates="trades")


class Position(Base):
    """Modèle pour les positions"""
    __tablename__ = "positions"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    trading_pair_id = Column(String(36), ForeignKey("trading_pairs.id"), nullable=False)
    strategy_id = Column(String(36), ForeignKey("strategies.id"), nullable=True)
    
    # Détails de la position
    side = Column(SqlEnum(OrderSide), nullable=False)
    quantity = Column(Numeric(20, 8), nullable=False)
    entry_price = Column(Numeric(20, 8), nullable=False)
    current_price = Column(Numeric(20, 8), nullable=True)
    
    # PnL
    unrealized_pnl = Column(Numeric(20, 8), default=0, nullable=False)
    realized_pnl = Column(Numeric(20, 8), default=0, nullable=False)
    
    # Status
    is_open = Column(Boolean, default=True, nullable=False)
    
    # Timestamps
    opened_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    closed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relations
    trading_pair = relationship("TradingPair", back_populates="positions")
    strategy = relationship("Strategy", back_populates="positions")