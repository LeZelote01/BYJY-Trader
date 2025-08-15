"""
🔐 Authentication System
Système d'authentification sécurisé pour BYJY-Trader
"""

import os
import jwt
import bcrypt
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from fastapi import HTTPException, status
from passlib.context import CryptContext
import secrets
import uuid

from core.config import get_settings
from core.logger import get_logger
from core.auth_models import UserInDB, TokenData, SessionInfo

logger = get_logger("byjy.auth")
settings = get_settings()

# Configuration cryptographique
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Constantes sécurité
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_DURATION = timedelta(minutes=30)
SESSION_TIMEOUT = timedelta(hours=24)
TOKEN_EXPIRE_MINUTES = 30


class AuthenticationError(Exception):
    """Exception personnalisée pour l'authentification"""
    pass


class AuthManager:
    """Gestionnaire d'authentification sécurisé"""
    
    def __init__(self):
        self.secret_key = settings.secret_key
        self.algorithm = settings.jwt_algorithm
        self.access_token_expire_minutes = settings.jwt_access_token_expire_minutes
        self._sessions: Dict[str, SessionInfo] = {}
        
        # Générer une clé secrète sécurisée si nécessaire
        if self.secret_key == "dev-secret-key-change-in-production":
            if settings.is_production():
                raise ValueError("Production secret key must be changed!")
            logger.warning("Using development secret key - CHANGE IN PRODUCTION!")
    
    def hash_password(self, password: str) -> str:
        """Hash un mot de passe avec bcrypt"""
        try:
            # Générer salt et hasher
            salt = bcrypt.gensalt()
            hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
            return hashed.decode('utf-8')
        except Exception as e:
            logger.error(f"Password hashing failed: {e}")
            raise AuthenticationError("Password hashing failed")
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Vérifie un mot de passe contre son hash"""
        try:
            return bcrypt.checkpw(
                plain_password.encode('utf-8'), 
                hashed_password.encode('utf-8')
            )
        except Exception as e:
            logger.error(f"Password verification failed: {e}")
            return False
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Crée un token JWT d'accès"""
        try:
            to_encode = data.copy()
            
            # Définir l'expiration
            if expires_delta:
                expire = datetime.now(timezone.utc) + expires_delta
            else:
                expire = datetime.now(timezone.utc) + timedelta(minutes=self.access_token_expire_minutes)
            
            to_encode.update({
                "exp": expire,
                "iat": datetime.now(timezone.utc),
                "type": "access"
            })
            
            # Encoder le token
            encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
            
            logger.info(f"Access token created for user: {data.get('sub', 'unknown')}")
            return encoded_jwt
            
        except Exception as e:
            logger.error(f"Token creation failed: {e}")
            raise AuthenticationError("Token creation failed")
    
    def verify_token(self, token: str) -> TokenData:
        """Vérifie et décode un token JWT"""
        try:
            # Décoder le token
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Extraire les données
            username: str = payload.get("sub")
            user_id: str = payload.get("user_id")
            exp_timestamp: int = payload.get("exp")
            
            if username is None or user_id is None:
                raise AuthenticationError("Invalid token payload")
            
            # Vérifier l'expiration
            if exp_timestamp:
                exp_datetime = datetime.fromtimestamp(exp_timestamp, tz=timezone.utc)
                if datetime.now(timezone.utc) > exp_datetime:
                    raise AuthenticationError("Token expired")
            
            return TokenData(
                username=username,
                user_id=user_id,
                exp=datetime.fromtimestamp(exp_timestamp, tz=timezone.utc) if exp_timestamp else None
            )
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            raise AuthenticationError("Token expired")
        except jwt.JWTError as e:
            logger.error(f"Token verification failed: {e}")
            raise AuthenticationError("Invalid token")
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            raise AuthenticationError("Token verification failed")
    
    def create_session(self, user: UserInDB, ip_address: Optional[str] = None, 
                      user_agent: Optional[str] = None) -> str:
        """Crée une nouvelle session utilisateur"""
        try:
            session_id = str(uuid.uuid4())
            session = SessionInfo(
                session_id=session_id,
                user_id=user.id,
                username=user.username,
                created_at=datetime.now(timezone.utc),
                last_activity=datetime.now(timezone.utc),
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            self._sessions[session_id] = session
            logger.info(f"Session created for user {user.username}: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Session creation failed: {e}")
            raise AuthenticationError("Session creation failed")
    
    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """Récupère une session"""
        session = self._sessions.get(session_id)
        if session and self.is_session_valid(session):
            # Mettre à jour la dernière activité
            session.last_activity = datetime.now(timezone.utc)
            return session
        return None
    
    def is_session_valid(self, session: SessionInfo) -> bool:
        """Vérifie si une session est valide"""
        if not session.is_active:
            return False
        
        # Vérifier le timeout
        if datetime.now(timezone.utc) - session.last_activity > SESSION_TIMEOUT:
            self.invalidate_session(session.session_id)
            return False
        
        return True
    
    def invalidate_session(self, session_id: str) -> bool:
        """Invalide une session"""
        if session_id in self._sessions:
            self._sessions[session_id].is_active = False
            logger.info(f"Session invalidated: {session_id}")
            return True
        return False
    
    def cleanup_expired_sessions(self):
        """Nettoie les sessions expirées"""
        expired_sessions = []
        current_time = datetime.now(timezone.utc)
        
        for session_id, session in self._sessions.items():
            if current_time - session.last_activity > SESSION_TIMEOUT:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self._sessions[session_id]
            logger.info(f"Expired session cleaned up: {session_id}")
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def get_user_sessions(self, user_id: str) -> list[SessionInfo]:
        """Récupère toutes les sessions actives d'un utilisateur"""
        user_sessions = []
        for session in self._sessions.values():
            if session.user_id == user_id and self.is_session_valid(session):
                user_sessions.append(session)
        return user_sessions
    
    def invalidate_all_user_sessions(self, user_id: str) -> int:
        """Invalide toutes les sessions d'un utilisateur"""
        count = 0
        for session in self._sessions.values():
            if session.user_id == user_id and session.is_active:
                session.is_active = False
                count += 1
        
        logger.info(f"Invalidated {count} sessions for user: {user_id}")
        return count
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Génère un token sécurisé aléatoire"""
        return secrets.token_urlsafe(length)
    
    def is_user_locked(self, user: UserInDB) -> bool:
        """Vérifie si un utilisateur est verrouillé"""
        if user.locked_until and datetime.now(timezone.utc) < user.locked_until:
            return True
        return False
    
    def should_lock_user(self, user: UserInDB) -> bool:
        """Détermine si un utilisateur doit être verrouillé"""
        return user.failed_login_attempts >= MAX_LOGIN_ATTEMPTS
    
    def lock_user(self, user: UserInDB) -> datetime:
        """Verrouille un utilisateur"""
        lockout_until = datetime.now(timezone.utc) + LOCKOUT_DURATION
        user.locked_until = lockout_until
        logger.warning(f"User locked due to failed login attempts: {user.username}")
        return lockout_until
    
    def reset_failed_attempts(self, user: UserInDB):
        """Remet à zéro les tentatives échouées"""
        user.failed_login_attempts = 0
        user.locked_until = None
    
    def increment_failed_attempts(self, user: UserInDB):
        """Incrémente les tentatives échouées"""
        user.failed_login_attempts += 1
        logger.warning(f"Failed login attempt for user {user.username}. Count: {user.failed_login_attempts}")


# Instance globale
auth_manager = AuthManager()


def get_auth_manager() -> AuthManager:
    """Récupère l'instance du gestionnaire d'authentification"""
    return auth_manager