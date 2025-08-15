"""
👥 User Database Models
Modèles de base de données pour les utilisateurs
"""

import uuid
from datetime import datetime, timezone
from typing import Optional, List
from sqlalchemy import Column, String, Boolean, DateTime, Integer, Text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.exc import IntegrityError

from core.models import Base
from core.database import get_database_manager
from core.logger import get_logger
from core.auth_models import UserInDB, UserCreate, User
from core.auth import get_auth_manager

logger = get_logger("byjy.user_db")


class UserModel(Base):
    """Modèle SQLAlchemy pour les utilisateurs"""
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    full_name = Column(String(255), nullable=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    last_login = Column(DateTime(timezone=True), nullable=True)
    failed_login_attempts = Column(Integer, default=0, nullable=False)
    locked_until = Column(DateTime(timezone=True), nullable=True)
    
    def __repr__(self):
        return f"<User(id={self.id}, username={self.username}, email={self.email})>"
    
    def to_user_in_db(self) -> UserInDB:
        """Convertit en UserInDB"""
        return UserInDB(
            id=self.id,
            username=self.username,
            email=self.email,
            full_name=self.full_name,
            hashed_password=self.hashed_password,
            is_active=self.is_active,
            is_admin=self.is_admin,
            created_at=self.created_at,
            last_login=self.last_login,
            failed_login_attempts=self.failed_login_attempts,
            locked_until=self.locked_until
        )
    
    def to_user(self) -> User:
        """Convertit en User (sans mot de passe)"""
        return User(
            id=self.id,
            username=self.username,
            email=self.email,
            full_name=self.full_name,
            is_active=self.is_active,
            is_admin=self.is_admin,
            created_at=self.created_at,
            last_login=self.last_login
        )


class UserDatabase:
    """Gestionnaire de base de données pour les utilisateurs"""
    
    def __init__(self):
        self.auth_manager = get_auth_manager()
    
    async def create_user(self, user_data: UserCreate) -> User:
        """Crée un nouvel utilisateur"""
        try:
            # Hasher le mot de passe
            hashed_password = self.auth_manager.hash_password(user_data.password)
            
            # Créer le modèle utilisateur
            user_model = UserModel(
                username=user_data.username,
                email=user_data.email,
                full_name=user_data.full_name,
                hashed_password=hashed_password,
                is_active=user_data.is_active
            )
            
            # Sauvegarder en base
            db_manager = get_database_manager()
            async with db_manager.get_async_session() as session:
                session.add(user_model)
                await session.commit()
                await session.refresh(user_model)
            
            logger.info(f"User created successfully: {user_model.username}")
            return user_model.to_user()
            
        except IntegrityError as e:
            logger.error(f"User creation failed - integrity error: {e}")
            if "username" in str(e):
                raise ValueError("Username already exists")
            elif "email" in str(e):
                raise ValueError("Email already exists")
            else:
                raise ValueError("User creation failed - duplicate data")
        except Exception as e:
            logger.error(f"User creation failed: {e}")
            raise ValueError(f"User creation failed: {str(e)}")
    
    async def get_user_by_username(self, username: str) -> Optional[UserInDB]:
        """Récupère un utilisateur par nom d'utilisateur"""
        try:
            db_manager = get_database_manager()
            async with db_manager.get_async_session() as session:
                result = await session.execute(
                    select(UserModel).where(UserModel.username == username)
                )
                user_model = result.scalar_one_or_none()
                
                if user_model:
                    return user_model.to_user_in_db()
                return None
                
        except Exception as e:
            logger.error(f"Failed to get user by username {username}: {e}")
            return None
    
    async def get_user_by_email(self, email: str) -> Optional[UserInDB]:
        """Récupère un utilisateur par email"""
        try:
            db_manager = get_database_manager()
            async with db_manager.get_async_session() as session:
                result = await session.execute(
                    select(UserModel).where(UserModel.email == email)
                )
                user_model = result.scalar_one_or_none()
                
                if user_model:
                    return user_model.to_user_in_db()
                return None
                
        except Exception as e:
            logger.error(f"Failed to get user by email {email}: {e}")
            return None
    
    async def get_user_by_id(self, user_id: str) -> Optional[UserInDB]:
        """Récupère un utilisateur par ID"""
        try:
            db_manager = get_database_manager()
            async with db_manager.get_async_session() as session:
                result = await session.execute(
                    select(UserModel).where(UserModel.id == user_id)
                )
                user_model = result.scalar_one_or_none()
                
                if user_model:
                    return user_model.to_user_in_db()
                return None
                
        except Exception as e:
            logger.error(f"Failed to get user by ID {user_id}: {e}")
            return None
    
    async def update_user_login(self, user_id: str, login_time: Optional[datetime] = None):
        """Met à jour la dernière connexion d'un utilisateur"""
        try:
            if login_time is None:
                login_time = datetime.now(timezone.utc)
            
            db_manager = get_database_manager()
            async with db_manager.get_async_session() as session:
                result = await session.execute(
                    select(UserModel).where(UserModel.id == user_id)
                )
                user_model = result.scalar_one_or_none()
                
                if user_model:
                    user_model.last_login = login_time
                    await session.commit()
                    logger.info(f"Updated last login for user: {user_model.username}")
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"Failed to update user login for {user_id}: {e}")
            return False
    
    async def update_failed_attempts(self, user_id: str, attempts: int, locked_until: Optional[datetime] = None):
        """Met à jour les tentatives échouées et le verrouillage"""
        try:
            db_manager = get_database_manager()
            async with db_manager.get_async_session() as session:
                result = await session.execute(
                    select(UserModel).where(UserModel.id == user_id)
                )
                user_model = result.scalar_one_or_none()
                
                if user_model:
                    user_model.failed_login_attempts = attempts
                    user_model.locked_until = locked_until
                    await session.commit()
                    logger.info(f"Updated failed attempts for user {user_model.username}: {attempts}")
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"Failed to update failed attempts for {user_id}: {e}")
            return False
    
    async def list_users(self, skip: int = 0, limit: int = 100, active_only: bool = True) -> List[User]:
        """Liste les utilisateurs"""
        try:
            db_manager = get_database_manager()
            async with db_manager.get_async_session() as session:
                query = select(UserModel).offset(skip).limit(limit)
                
                if active_only:
                    query = query.where(UserModel.is_active == True)
                
                result = await session.execute(query)
                user_models = result.scalars().all()
                
                return [user_model.to_user() for user_model in user_models]
                
        except Exception as e:
            logger.error(f"Failed to list users: {e}")
            return []
    
    async def user_exists(self, username: str = None, email: str = None) -> bool:
        """Vérifie si un utilisateur existe"""
        try:
            db_manager = get_database_manager()
            async with db_manager.get_async_session() as session:
                if username:
                    result = await session.execute(
                        select(UserModel).where(UserModel.username == username)
                    )
                elif email:
                    result = await session.execute(
                        select(UserModel).where(UserModel.email == email)
                    )
                else:
                    return False
                
                user_model = result.scalar_one_or_none()
                return user_model is not None
                
        except Exception as e:
            logger.error(f"Failed to check user existence: {e}")
            return False
    
    async def create_admin_user(self) -> Optional[User]:
        """Crée un utilisateur admin par défaut si aucun n'existe"""
        try:
            # Vérifier s'il y a déjà un admin
            db_manager = get_database_manager()
            async with db_manager.get_async_session() as session:
                result = await session.execute(
                    select(UserModel).where(UserModel.is_admin == True)
                )
                admin_exists = result.scalar_one_or_none()
                
                if admin_exists:
                    logger.info("Admin user already exists")
                    return None
                
                # Créer l'admin par défaut
                admin_password = "Admin123!"  # À changer immédiatement
                hashed_password = self.auth_manager.hash_password(admin_password)
                
                admin_user = UserModel(
                    username="admin",
                    email="admin@byjy-trader.local",
                    full_name="System Administrator",
                    hashed_password=hashed_password,
                    is_active=True,
                    is_admin=True
                )
                
                session.add(admin_user)
                await session.commit()
                await session.refresh(admin_user)
                
                logger.warning(f"Default admin user created - USERNAME: admin, PASSWORD: {admin_password}")
                logger.warning("CHANGE THE DEFAULT PASSWORD IMMEDIATELY!")
                
                return admin_user.to_user()
                
        except Exception as e:
            logger.error(f"Failed to create admin user: {e}")
            return None


# Instance globale
user_db = UserDatabase()


def get_user_database() -> UserDatabase:
    """Récupère l'instance de la base de données utilisateurs"""
    return user_db