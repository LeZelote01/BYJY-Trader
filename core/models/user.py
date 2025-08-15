"""
👤 User Models
Modèles pour la gestion des utilisateurs et API keys
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import Column, String, DateTime, Boolean, Text, Enum as SqlEnum
from sqlalchemy.orm import relationship
from . import Base


class ApiKeyStatus(str, Enum):
    """Status des clés API"""
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    EXPIRED = "EXPIRED"
    REVOKED = "REVOKED"


class User(Base):
    """Modèle utilisateur"""
    __tablename__ = "users"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=True, index=True)
    password_hash = Column(String(255), nullable=False)
    
    # Profile
    first_name = Column(String(50), nullable=True)
    last_name = Column(String(50), nullable=True)
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    last_login = Column(DateTime, nullable=True)
    
    # Relations
    api_keys = relationship("ApiKey", back_populates="user")


class ApiKey(Base):
    """Modèle pour les clés API des exchanges"""
    __tablename__ = "api_keys"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), nullable=False, index=True)  # Pas de FK pour flexibilité
    
    # Exchange info
    exchange_name = Column(String(50), nullable=False, index=True)
    key_name = Column(String(100), nullable=False)
    
    # Clés (chiffrées)
    api_key = Column(Text, nullable=False)
    api_secret = Column(Text, nullable=False)
    passphrase = Column(Text, nullable=True)  # Pour certains exchanges
    
    # Permissions
    permissions = Column(Text, nullable=True)  # JSON des permissions
    
    # Status et sécurité
    status = Column(SqlEnum(ApiKeyStatus), default=ApiKeyStatus.ACTIVE, nullable=False)
    is_testnet = Column(Boolean, default=True, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    last_used = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    
    # Relations
    user = relationship("User", back_populates="api_keys")