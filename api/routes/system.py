"""
⚙️ System Routes
Endpoints pour la gestion système et configuration
"""

import os
import psutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from core.config import get_settings, Settings
from core.database import get_database_manager
from core.logger import get_logger

logger = get_logger("byjy.api.system")
router = APIRouter()

class SystemInfo(BaseModel):
    """Informations système"""
    cpu_percent: float
    memory_total: int
    memory_used: int
    memory_percent: float
    disk_total: int
    disk_used: int
    disk_percent: float
    python_version: str
    platform: str
    uptime: str

class ConfigInfo(BaseModel):
    """Informations de configuration (sans secrets)"""
    environment: str
    debug: bool
    log_level: str
    api_host: str
    api_port: int
    database_path: str
    max_workers: int

@router.get("/info", response_model=SystemInfo)
async def get_system_info():
    """
    Récupère les informations système actuelles
    """
    try:
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory
        memory = psutil.virtual_memory()
        
        # Disk
        disk = psutil.disk_usage('/')
        
        # System info
        import sys
        import platform
        
        # Uptime (approximatif)
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        uptime = str(datetime.now() - boot_time)
        
        return SystemInfo(
            cpu_percent=cpu_percent,
            memory_total=memory.total,
            memory_used=memory.used,
            memory_percent=memory.percent,
            disk_total=disk.total,
            disk_used=disk.used,
            disk_percent=disk.percent,
            python_version=sys.version,
            platform=platform.platform(),
            uptime=uptime
        )
        
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system information")

@router.get("/config", response_model=ConfigInfo)
async def get_config_info():
    """
    Récupère les informations de configuration (sans secrets)
    """
    settings = get_settings()
    
    return ConfigInfo(
        environment=settings.environment,
        debug=settings.debug,
        log_level=settings.log_level,
        api_host=settings.api_host,
        api_port=settings.api_port,
        database_path=str(settings.get_database_path()),
        max_workers=settings.max_workers
    )

@router.post("/database/backup")
async def create_database_backup():
    """
    Crée une sauvegarde de la base de données
    """
    try:
        db_manager = get_database_manager()
        success = db_manager.backup_database()
        
        if success:
            return {"message": "Database backup created successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to create database backup")
    
    except Exception as e:
        logger.error(f"Backup creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/logs")
async def get_recent_logs(lines: int = 50):
    """
    Récupère les logs récents
    """
    try:
        settings = get_settings()
        log_file = settings.logs_dir / "byjy_trader.log"
        
        if not log_file.exists():
            raise HTTPException(status_code=404, detail="Log file not found")
        
        # Lire les dernières lignes
        with open(log_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
        
        return {
            "total_lines": len(all_lines),
            "returned_lines": len(recent_lines),
            "logs": [line.strip() for line in recent_lines]
        }
    
    except Exception as e:
        logger.error(f"Failed to read logs: {e}")
        raise HTTPException(status_code=500, detail="Failed to read log file")

@router.get("/directories")
async def get_directory_info():
    """
    Informations sur les répertoires du projet
    """
    settings = get_settings()
    
    directories = {
        "root": str(settings.root_dir),
        "config": str(settings.config_dir),
        "data": str(settings.data_dir),
        "logs": str(settings.logs_dir),
        "database": str(settings.database_dir),
        "backups": str(settings.backups_dir),
        "models": str(settings.root_dir / settings.models_path)
    }
    
    # Vérifier l'existence et la taille
    directory_info = {}
    for name, path in directories.items():
        path_obj = Path(path)
        directory_info[name] = {
            "path": path,
            "exists": path_obj.exists(),
            "is_directory": path_obj.is_dir() if path_obj.exists() else False,
            "size_mb": round(sum(f.stat().st_size for f in path_obj.rglob('*') if f.is_file()) / 1024 / 1024, 2) if path_obj.exists() and path_obj.is_dir() else 0
        }
    
    return directory_info