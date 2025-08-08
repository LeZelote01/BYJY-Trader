"""
⚙️ Configuration Management
Gestion centralisée de toute la configuration BYJY-Trader
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from functools import lru_cache

from .path_utils import (
    get_project_root, 
    get_config_dir, 
    get_data_dir, 
    get_logs_dir, 
    get_database_dir, 
    get_backups_dir,
    get_models_dir,
    ensure_directory_exists,
    is_portable_installation
)


class Settings(BaseSettings):
    """Configuration principale BYJY-Trader"""
    
    # Environnement
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Base de projet
    project_name: str = "BYJY-Trader"
    version: str = "0.1.0"
    description: str = "Bot de Trading Personnel Avancé avec IA"
    
    # Chemins - Utilisation du système de détection automatique robuste
    root_dir: Path = Field(default_factory=get_project_root)
    config_dir: Path = Field(default_factory=get_config_dir)
    data_dir: Path = Field(default_factory=get_data_dir)
    logs_dir: Path = Field(default_factory=get_logs_dir)
    database_dir: Path = Field(default_factory=get_database_dir)
    backups_dir: Path = Field(default_factory=get_backups_dir)
    
    # Base de données
    database_url: str = Field(default="sqlite:///database/byjy_trader.db", env="DATABASE_URL")
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_reload: bool = Field(default=True, env="API_RELOAD")
    
    # Sécurité
    secret_key: str = Field(default="dev-secret-key-change-in-production", env="SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_access_token_expire_minutes: int = Field(default=30, env="JWT_ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Performance
    max_workers: int = Field(default=4, env="MAX_WORKERS")
    connection_pool_size: int = Field(default=20, env="CONNECTION_POOL_SIZE")
    request_timeout: int = Field(default=30, env="REQUEST_TIMEOUT")
    
    # Trading Configuration
    binance_api_key: Optional[str] = Field(default=None, env="BINANCE_API_KEY")
    binance_secret_key: Optional[str] = Field(default=None, env="BINANCE_SECRET_KEY")
    binance_testnet: bool = Field(default=True, env="BINANCE_TESTNET")
    
    # Risk Management
    max_position_size: float = Field(default=1000.0, env="MAX_POSITION_SIZE")
    max_daily_loss: float = Field(default=500.0, env="MAX_DAILY_LOSS")
    stop_loss_percentage: float = Field(default=2.0, env="STOP_LOSS_PERCENTAGE")
    take_profit_percentage: float = Field(default=3.0, env="TAKE_PROFIT_PERCENTAGE")
    
    # IA Configuration - Utilisation du système de détection automatique
    models_path: str = Field(default="ai/models", env="MODELS_PATH")
    enable_ai_predictions: bool = Field(default=True, env="ENABLE_AI_PREDICTIONS")
    model_retrain_interval: int = Field(default=24, env="MODEL_RETRAIN_INTERVAL")
    
    # Portabilité - Détection automatique
    is_portable: bool = Field(default_factory=is_portable_installation)
    
    # Backup
    backup_enabled: bool = Field(default=True, env="BACKUP_ENABLED")
    backup_interval: int = Field(default=3600, env="BACKUP_INTERVAL")
    backup_retention_days: int = Field(default=30, env="BACKUP_RETENTION_DAYS")
    
    @validator("root_dir", "config_dir", "data_dir", "logs_dir", "database_dir", "backups_dir")
    def ensure_path_absolute(cls, v):
        """S'assure que les chemins sont absolus"""
        if isinstance(v, str):
            v = Path(v)
        return v.resolve()
    
    @validator("log_level")
    def validate_log_level(cls, v):
        """Valide le niveau de log"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of: {valid_levels}")
        return v.upper()
    
    def create_directories(self) -> None:
        """Crée tous les répertoires nécessaires avec le système robuste"""
        directories = [
            self.config_dir,
            self.data_dir,
            self.logs_dir,
            self.database_dir,
            self.backups_dir,
            self.root_dir / self.models_path,
            self.data_dir / "cache",
        ]
        
        for directory in directories:
            ensure_directory_exists(directory, relative_to_project=False)
    
    def load_config_file(self, config_file: Optional[Path] = None) -> Dict[str, Any]:
        """Charge un fichier de configuration JSON"""
        if config_file is None:
            config_file = self.config_dir / "config.json"
        
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load config file {config_file}: {e}")
                return {}
        return {}
    
    def save_config_file(self, config_data: Dict[str, Any], config_file: Optional[Path] = None) -> bool:
        """Sauvegarde la configuration dans un fichier JSON"""
        if config_file is None:
            config_file = self.config_dir / "config.json"
        
        try:
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            return True
        except IOError as e:
            print(f"Error: Could not save config file {config_file}: {e}")
            return False
    
    def get_database_path(self) -> Path:
        """Retourne le chemin absolu de la base de données"""
        if self.database_url.startswith("sqlite:///"):
            db_path = self.database_url.replace("sqlite:///", "")
            if not Path(db_path).is_absolute():
                return self.root_dir / db_path
            return Path(db_path)
        return self.database_dir / "byjy_trader.db"
    
    def is_production(self) -> bool:
        """Vérifie si on est en production"""
        return self.environment.lower() == "production"
    
    def is_development(self) -> bool:
        """Vérifie si on est en développement"""
        return self.environment.lower() == "development"
    
    def is_testing(self) -> bool:
        """Vérifie si on est en mode test"""
        return self.environment.lower() == "testing"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Retourne l'instance de configuration (singleton)
    Utilise lru_cache pour éviter de recharger à chaque appel
    """
    settings = Settings()
    settings.create_directories()
    return settings


# Instance globale pour import facile
settings = get_settings()