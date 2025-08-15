"""
🏥 Health Check Routes
Endpoints pour vérifier l'état du système
"""

from datetime import datetime, timezone
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from core.config import get_settings
from core.database import get_database_manager
from core.logger import get_logger

logger = get_logger("byjy.api.health")
router = APIRouter()

class HealthResponse(BaseModel):
    """Modèle de réponse pour le health check"""
    status: str
    timestamp: str
    version: str
    environment: str
    uptime_seconds: float
    components: Dict[str, Any]

# Variable pour tracking uptime
_start_time = datetime.now(timezone.utc)

@router.get("/", response_model=HealthResponse)
async def health_check():
    """
    Health check complet du système
    Vérifie tous les composants critiques
    """
    settings = get_settings()
    current_time = datetime.now(timezone.utc)
    uptime = (current_time - _start_time).total_seconds()
    
    # Vérification des composants
    components = {}
    overall_status = "healthy"
    
    try:
        # Database health check
        db_manager = get_database_manager()
        db_health = await db_manager.health_check()
        components["database"] = db_health
        
        if db_health["status"] != "healthy":
            overall_status = "degraded"
    
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        components["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        overall_status = "unhealthy"
    
    # Configuration health
    components["configuration"] = {
        "status": "healthy",
        "environment": settings.environment,
        "debug": settings.debug
    }
    
    # Logging health
    components["logging"] = {
        "status": "healthy",
        "log_level": settings.log_level,
        "logs_dir": str(settings.logs_dir)
    }
    
    return HealthResponse(
        status=overall_status,
        timestamp=current_time.isoformat(),
        version="0.1.0",
        environment=settings.environment,
        uptime_seconds=uptime,
        components=components
    )

@router.get("/ping")
async def ping():
    """Simple ping endpoint"""
    return {"ping": "pong", "timestamp": datetime.now(timezone.utc).isoformat()}

@router.get("/ready")
async def readiness():
    """Readiness probe pour Kubernetes"""
    try:
        db_manager = get_database_manager()
        db_health = await db_manager.health_check()
        
        if db_health["status"] == "healthy":
            return {"status": "ready"}
        else:
            raise HTTPException(status_code=503, detail="Database not ready")
    
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")

@router.get("/live")
async def liveness():
    """Liveness probe pour Kubernetes"""
    return {"status": "alive", "timestamp": datetime.now(timezone.utc).isoformat()}