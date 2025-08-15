"""
🔐 Authentication Models
Modèles Pydantic pour l'authentification
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, validator
import re


class UserBase(BaseModel):
    """Base model pour utilisateur"""
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    is_active: bool = True
    
    @validator('username')
    def validate_username(cls, v):
        if len(v) < 3:
            raise ValueError('Username must be at least 3 characters long')
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Username can only contain letters, numbers, hyphens and underscores')
        return v


class UserCreate(UserBase):
    """Model pour création d'utilisateur"""
    password: str
    confirm_password: str
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('Password must contain at least one special character')
        return v
    
    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match')
        return v


class UserUpdate(BaseModel):
    """Model pour mise à jour d'utilisateur"""
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    is_active: Optional[bool] = None


class UserInDB(UserBase):
    """Model pour utilisateur en base"""
    id: str
    hashed_password: str
    created_at: datetime
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    is_admin: bool = False


class User(UserBase):
    """Model pour utilisateur (réponse API)"""
    id: str
    created_at: datetime
    last_login: Optional[datetime] = None
    is_admin: bool = False


class LoginRequest(BaseModel):
    """Model pour requête de connexion"""
    username: str
    password: str
    remember_me: bool = False


class LoginResponse(BaseModel):
    """Model pour réponse de connexion"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: User


class TokenData(BaseModel):
    """Model pour données du token"""
    username: Optional[str] = None
    user_id: Optional[str] = None
    exp: Optional[datetime] = None


class PasswordChange(BaseModel):
    """Model pour changement de mot de passe"""
    current_password: str
    new_password: str
    confirm_password: str
    
    @validator('new_password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('Password must contain at least one special character')
        return v
    
    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('Passwords do not match')
        return v


class PasswordReset(BaseModel):
    """Model pour réinitialisation de mot de passe"""
    email: EmailStr


class SessionInfo(BaseModel):
    """Model pour informations de session"""
    session_id: str
    user_id: str
    username: str
    created_at: datetime
    last_activity: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_active: bool = True