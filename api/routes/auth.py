"""
🔐 Authentication Routes
Endpoints pour l'authentification et autorisation
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import jwt

from core.config import get_settings
from core.logger import get_logger

logger = get_logger("byjy.api.auth")
router = APIRouter()
security = HTTPBearer()

class LoginRequest(BaseModel):
    """Requête de connexion"""
    username: str
    password: str

class LoginResponse(BaseModel):
    """Réponse de connexion"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int

class UserInfo(BaseModel):
    """Informations utilisateur"""
    username: str
    role: str
    created_at: datetime

@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """
    Connexion utilisateur - Version simplifiée pour développement
    """
    try:
        settings = get_settings()
        
        # Pour l'instant, utilisateur par défaut pour développement
        if request.username == "admin" and request.password == "admin":
            # Créer un token JWT
            payload = {
                "sub": request.username,
                "role": "admin",
                "exp": datetime.utcnow() + timedelta(minutes=settings.jwt_access_token_expire_minutes)
            }
            
            token = jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)
            
            return LoginResponse(
                access_token=token,
                expires_in=settings.jwt_access_token_expire_minutes * 60
            )
        else:
            # Retourner directement HTTP 401 pour les credentials invalides
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
    
    except HTTPException:
        # Re-lever les HTTPException déjà correctes
        raise
    except jwt.PyJWTError as e:
        logger.error(f"JWT error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token generation failed"
        )
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@router.get("/me", response_model=UserInfo)
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Récupère les informations de l'utilisateur connecté
    """
    try:
        settings = get_settings()
        token = credentials.credentials
        
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])
        username = payload.get("sub")
        role = payload.get("role", "user")
        
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        return UserInfo(
            username=username,
            role=role,
            created_at=datetime.now(timezone.utc)
        )
    
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired"
        )
    except jwt.PyJWTError as e:
        logger.error(f"JWT validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    except Exception as e:
        logger.error(f"User info retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user information"
        )

@router.post("/logout")
async def logout():
    """
    Déconnexion utilisateur
    """
    # Pour une déconnexion côté serveur, on devrait invalider le token
    # Pour l'instant, retourner simplement un message de succès
    return {"message": "Logged out successfully"}

@router.post("/refresh")
async def refresh_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Rafraîchit le token d'accès
    """
    try:
        settings = get_settings()
        token = credentials.credentials
        
        # Vérifier le token existant
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])
        username = payload.get("sub")
        role = payload.get("role", "user")
        
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Créer un nouveau token
        new_payload = {
            "sub": username,
            "role": role,
            "exp": datetime.utcnow() + timedelta(minutes=settings.jwt_access_token_expire_minutes)
        }
        
        new_token = jwt.encode(new_payload, settings.secret_key, algorithm=settings.jwt_algorithm)
        
        return LoginResponse(
            access_token=new_token,
            expires_in=settings.jwt_access_token_expire_minutes * 60
        )
    
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired"
        )
    except jwt.PyJWTError as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )