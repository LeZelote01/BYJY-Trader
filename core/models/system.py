"""
⚙️ System Models
Modèles pour les logs système et configurations
"""

import uuid
from datetime import datetime
from enum import Enum

from sqlalchemy import Column, String, DateTime, Text, Enum as SqlEnum, Integer
from . import Base


class LogLevel(str, Enum):
    """Niveaux de logs"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class SystemLog(Base):
    """Modèle pour les logs système"""
    __tablename__ = "system_logs"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Log details
    level = Column(SqlEnum(LogLevel), nullable=False, index=True)
    module = Column(String(100), nullable=False, index=True)
    message = Column(Text, nullable=False)
    
    # Context
    function_name = Column(String(100), nullable=True)
    line_number = Column(Integer, nullable=True)
    
    # Additional data
    extra_data = Column(Text, nullable=True)  # JSON
    stack_trace = Column(Text, nullable=True)
    
    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)


class Configuration(Base):
    """Modèle pour les configurations système"""
    __tablename__ = "configurations"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Configuration key-value
    key = Column(String(200), nullable=False, unique=True, index=True)
    value = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    
    # Metadata
    config_type = Column(String(50), nullable=False, default="user")
    is_encrypted = Column(String(10), nullable=False, default="false")  # "true"/"false" string
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)