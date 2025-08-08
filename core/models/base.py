"""
🏗️ Base Model
Modèle de base pour tous les modèles SQLAlchemy
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, String
from sqlalchemy.ext.declarative import declarative_base, declared_attr

Base = declarative_base()

class BaseModel:
    """Modèle de base avec champs communs"""
    
    @declared_attr
    def __tablename__(cls):
        """Nom de table automatique basé sur le nom de classe"""
        return cls.__name__.lower()
    
    # Clé primaire UUID
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), unique=True)
    
    # Timestamps automatiques
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), 
                       onupdate=lambda: datetime.now(timezone.utc), nullable=False)
    
    def to_dict(self) -> dict:
        """Convertir le modèle en dictionnaire"""
        result = {}
        for column in self.__table__.columns:
            value = getattr(self, column.name)
            # Convertir UUID et datetime en string pour JSON
            if isinstance(value, uuid.UUID):
                value = str(value)
            elif isinstance(value, datetime):
                value = value.isoformat()
            result[column.name] = value
        return result
    
    def __repr__(self):
        """Représentation string du modèle"""
        return f"<{self.__class__.__name__}(id={self.id})>"