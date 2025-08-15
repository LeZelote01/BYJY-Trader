"""
⚙️ Dynamic Configuration Manager
Gestionnaire de configuration dynamique avec hot-reload et versioning
"""

import asyncio
import json
import hashlib
import threading
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Callable, Set
from dataclasses import dataclass
from threading import Lock
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .config import Settings, get_settings
from .logger import get_logger

logger = get_logger("byjy.config_manager")


@dataclass
class ConfigVersion:
    """Représente une version de configuration"""
    version: str
    timestamp: datetime
    config_data: Dict[str, Any]
    hash: str
    description: Optional[str] = None


class ConfigChangeHandler(FileSystemEventHandler):
    """Handler pour surveiller les changements de fichiers de configuration"""
    
    def __init__(self, config_manager: 'DynamicConfigManager'):
        self.config_manager = config_manager
        self.debounce_time = 1.0  # Secondes
        self.last_event_time = 0
        
    def on_modified(self, event):
        """Appelé quand un fichier de configuration est modifié"""
        if event.is_directory:
            return
            
        # Filtrer les fichiers de configuration
        if not (event.src_path.endswith('.json') or event.src_path.endswith('.env')):
            return
            
        # Debouncing pour éviter les multiples événements
        current_time = time.time()
        if current_time - self.last_event_time < self.debounce_time:
            return
        self.last_event_time = current_time
        
        logger.info(f"Configuration file changed: {event.src_path}")
        
        # Déclencher le reload asynchrone (en thread-safe)
        try:
            # Créer une tâche seulement si on est dans un event loop actif
            import threading
            if hasattr(asyncio, '_get_running_loop'):
                loop = asyncio.get_running_loop()
                loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(self.config_manager.reload_config())
                )
            else:
                # Fallback pour versions plus anciennes
                self.config_manager._schedule_reload = True
        except RuntimeError:
            # Pas d'event loop actif, on marque juste pour reload
            self.config_manager._schedule_reload = True


class DynamicConfigManager:
    """Gestionnaire de configuration dynamique avec hot-reload et versioning"""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._config_data: Dict[str, Any] = {}
        self._config_versions: List[ConfigVersion] = []
        self._observers: Set[Callable[[Dict[str, Any]], None]] = set()
        self._lock = Lock()
        self._file_observer: Optional[Observer] = None
        self._initialized = False
        self._schedule_reload = False
        
        # Fichier de configuration principal
        self.config_file = self.settings.config_dir / "dynamic_config.json"
        self.versions_file = self.settings.config_dir / "config_versions.json"
        
        # Configuration par défaut
        self._default_config = {
            "trading": {
                "max_position_size": 1000.0,
                "max_daily_loss": 500.0,
                "stop_loss_percentage": 2.0,
                "take_profit_percentage": 3.0,
                "enabled": True
            },
            "risk_management": {
                "max_drawdown": 10.0,
                "max_concurrent_orders": 5,
                "position_sizing_method": "fixed",
                "risk_per_trade": 1.0
            },
            "api": {
                "rate_limit_requests_per_minute": 100,
                "timeout_seconds": 30,
                "retry_attempts": 3,
                "retry_delay": 1.0
            },
            "logging": {
                "level": "INFO",
                "file_rotation_mb": 10,
                "max_backup_files": 5,
                "console_output": True
            },
            "system": {
                "auto_backup_interval_hours": 24,
                "cleanup_old_logs_days": 30,
                "performance_monitoring": True
            }
        }
    
    async def initialize(self) -> bool:
        """Initialise le gestionnaire de configuration"""
        try:
            with self._lock:
                if self._initialized:
                    return True
                
                # Créer les répertoires nécessaires
                self.settings.config_dir.mkdir(parents=True, exist_ok=True)
                
                # Charger la configuration existante ou créer la configuration par défaut
                await self._load_config()
                
                # Charger l'historique des versions
                await self._load_versions()
                
                # Démarrer la surveillance des fichiers (désactivée temporairement)
                # await self._start_file_watching()
                
                self._initialized = True
                
            logger.info("Dynamic Configuration Manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Dynamic Configuration Manager: {e}")
            return False
    
    async def _load_config(self) -> None:
        """Charge la configuration depuis le fichier"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self._config_data = json.load(f)
                logger.info(f"Configuration loaded from {self.config_file}")
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load config: {e}. Using default configuration.")
                self._config_data = self._default_config.copy()
        else:
            # Créer la configuration par défaut
            self._config_data = self._default_config.copy()
            await self._save_config()
            logger.info("Created default configuration file")
    
    async def _load_versions(self) -> None:
        """Charge l'historique des versions"""
        if self.versions_file.exists():
            try:
                with open(self.versions_file, 'r', encoding='utf-8') as f:
                    versions_data = json.load(f)
                    
                self._config_versions = []
                for version_data in versions_data:
                    version = ConfigVersion(
                        version=version_data['version'],
                        timestamp=datetime.fromisoformat(version_data['timestamp']),
                        config_data=version_data['config_data'],
                        hash=version_data['hash'],
                        description=version_data.get('description')
                    )
                    self._config_versions.append(version)
                    
                logger.info(f"Loaded {len(self._config_versions)} configuration versions")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not load version history: {e}")
                self._config_versions = []
    
    async def _save_config(self) -> bool:
        """Sauvegarde la configuration actuelle"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self._config_data, f, indent=2, ensure_ascii=False)
            return True
        except IOError as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    async def _save_versions(self) -> bool:
        """Sauvegarde l'historique des versions"""
        try:
            versions_data = []
            for version in self._config_versions:
                versions_data.append({
                    'version': version.version,
                    'timestamp': version.timestamp.isoformat(),
                    'config_data': version.config_data,
                    'hash': version.hash,
                    'description': version.description
                })
            
            with open(self.versions_file, 'w', encoding='utf-8') as f:
                json.dump(versions_data, f, indent=2, ensure_ascii=False)
            return True
        except IOError as e:
            logger.error(f"Failed to save version history: {e}")
            return False
    
    def _calculate_config_hash(self, config_data: Dict[str, Any]) -> str:
        """Calcule le hash de la configuration"""
        config_str = json.dumps(config_data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(config_str.encode('utf-8')).hexdigest()[:16]
    
    async def _create_version(self, description: Optional[str] = None) -> str:
        """Crée une nouvelle version de la configuration"""
        config_hash = self._calculate_config_hash(self._config_data)
        
        # Vérifier si cette version existe déjà
        for version in self._config_versions:
            if version.hash == config_hash:
                return version.version
        
        # Créer nouvelle version
        version_number = f"v{len(self._config_versions) + 1:04d}"
        new_version = ConfigVersion(
            version=version_number,
            timestamp=datetime.now(timezone.utc),
            config_data=self._config_data.copy(),
            hash=config_hash,
            description=description
        )
        
        self._config_versions.append(new_version)
        
        # Garder seulement les 50 dernières versions
        if len(self._config_versions) > 50:
            self._config_versions = self._config_versions[-50:]
        
        await self._save_versions()
        return version_number
    
    async def _start_file_watching(self) -> None:
        """Démarre la surveillance des fichiers de configuration"""
        try:
            if self._file_observer is not None:
                return
                
            event_handler = ConfigChangeHandler(self)
            self._file_observer = Observer()
            self._file_observer.schedule(
                event_handler, 
                str(self.settings.config_dir), 
                recursive=False
            )
            self._file_observer.start()
            
            logger.info("File watching started for configuration hot-reload")
        except Exception as e:
            logger.error(f"Failed to start file watching: {e}")
    
    async def stop_file_watching(self) -> None:
        """Arrête la surveillance des fichiers"""
        if self._file_observer is not None:
            self._file_observer.stop()
            self._file_observer.join()
            self._file_observer = None
            logger.info("File watching stopped")
    
    def add_observer(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Ajoute un observateur pour les changements de configuration"""
        with self._lock:
            self._observers.add(callback)
    
    def remove_observer(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Supprime un observateur"""
        with self._lock:
            self._observers.discard(callback)
    
    async def _notify_observers(self) -> None:
        """Notifie tous les observateurs des changements"""
        with self._lock:
            observers = self._observers.copy()
        
        for observer in observers:
            try:
                observer(self._config_data.copy())
            except Exception as e:
                logger.error(f"Error notifying config observer: {e}")
    
    def get_config(self, key_path: Optional[str] = None) -> Any:
        """Récupère une valeur de configuration"""
        with self._lock:
            if key_path is None:
                return self._config_data.copy()
            
            keys = key_path.split('.')
            value = self._config_data
            
            try:
                for key in keys:
                    value = value[key]
                return value
            except (KeyError, TypeError):
                logger.warning(f"Configuration key not found: {key_path}")
                return None
    
    async def set_config(self, key_path: str, value: Any, description: Optional[str] = None, 
                        create_version: bool = True) -> bool:
        """Met à jour une valeur de configuration"""
        try:
            # Validation avant modification
            if not await self._validate_config_change(key_path, value):
                logger.error(f"Configuration validation failed for {key_path}={value}")
                return False
            
            # Créer une sauvegarde avant modification (si demandé)
            backup_version = None
            if create_version:
                backup_version = await self._create_version(f"Backup before changing {key_path}")
            
            # Appliquer le changement dans un contexte thread-safe
            with self._lock:
                keys = key_path.split('.')
                config = self._config_data
                
                # Naviguer jusqu'au parent
                for key in keys[:-1]:
                    if key not in config:
                        config[key] = {}
                    config = config[key]
                
                # Modifier la valeur
                old_value = config.get(keys[-1])
                config[keys[-1]] = value
            
            # Sauvegarder (hors du lock)
            success = await self._save_config()
            
            if success:
                # Créer une version avec la modification (si demandé)
                version = None
                if create_version:
                    version = await self._create_version(
                        description or f"Changed {key_path}: {old_value} -> {value}"
                    )
                    logger.info(f"Configuration updated: {key_path}={value} (version {version})")
                
                # Notifier les observateurs
                await self._notify_observers()
                return True
            else:
                # Rollback en cas d'échec de sauvegarde
                with self._lock:
                    if old_value is not None:
                        config[keys[-1]] = old_value
                    else:
                        config.pop(keys[-1], None)
                return False
                    
                    
        except Exception as e:
            logger.error(f"Failed to set configuration {key_path}={value}: {e}")
            return False
    
    async def _validate_config_change(self, key_path: str, value: Any) -> bool:
        """Valide un changement de configuration"""
        try:
            # Règles de validation spécifiques
            validation_rules = {
                "trading.max_position_size": lambda v: isinstance(v, (int, float)) and v > 0,
                "trading.max_daily_loss": lambda v: isinstance(v, (int, float)) and v > 0,
                "trading.stop_loss_percentage": lambda v: isinstance(v, (int, float)) and 0 < v <= 50,
                "trading.take_profit_percentage": lambda v: isinstance(v, (int, float)) and 0 < v <= 100,
                "risk_management.max_drawdown": lambda v: isinstance(v, (int, float)) and 0 < v <= 50,
                "risk_management.max_concurrent_orders": lambda v: isinstance(v, int) and v > 0,
                "api.rate_limit_requests_per_minute": lambda v: isinstance(v, int) and v > 0,
                "api.timeout_seconds": lambda v: isinstance(v, (int, float)) and v > 0,
                "logging.level": lambda v: v in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            }
            
            # Appliquer la règle de validation si elle existe
            if key_path in validation_rules:
                if not validation_rules[key_path](value):
                    logger.error(f"Validation failed for {key_path}={value}")
                    return False
            
            # Validation générale des types
            if isinstance(value, str) and len(value) > 1000:
                logger.error(f"String value too long for {key_path}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error during validation of {key_path}={value}: {e}")
            return False
    
    async def reload_config(self) -> bool:
        """Recharge la configuration depuis le fichier (hot-reload)"""
        try:
            logger.info("Hot-reloading configuration...")
            
            with self._lock:
                old_config = self._config_data.copy()
                
                # Recharger depuis le fichier
                await self._load_config()
                
                # Vérifier si la configuration a changé
                if old_config != self._config_data:
                    # Créer une version
                    version = await self._create_version("Hot-reload from file")
                    logger.info(f"Configuration hot-reloaded successfully (version {version})")
                    
                    # Notifier les observateurs
                    await self._notify_observers()
                    return True
                else:
                    logger.info("Configuration unchanged after hot-reload")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to hot-reload configuration: {e}")
            return False
    
    def get_versions(self, limit: int = 10) -> List[ConfigVersion]:
        """Retourne l'historique des versions (les plus récentes en premier)"""
        with self._lock:
            return list(reversed(self._config_versions[-limit:]))
    
    async def rollback_to_version(self, version: str) -> bool:
        """Revient à une version spécifique de la configuration"""
        try:
            # Trouver la version (hors du lock)
            target_version = None
            with self._lock:
                for v in self._config_versions:
                    if v.version == version:
                        target_version = v
                        break
                        
            if target_version is None:
                logger.error(f"Version {version} not found")
                return False
            
            # Créer un backup avant rollback
            backup_version = await self._create_version(f"Backup before rollback to {version}")
            
            # Appliquer la configuration de la version cible
            with self._lock:
                self._config_data = target_version.config_data.copy()
            
            # Sauvegarder
            success = await self._save_config()
            
            if success:
                # Créer une nouvelle version pour le rollback
                new_version = await self._create_version(f"Rollback to {version}")
                logger.info(f"Successfully rolled back to version {version} (new version {new_version})")
                
                # Notifier les observateurs
                await self._notify_observers()
                return True
            else:
                logger.error(f"Failed to save configuration during rollback to {version}")
                return False
                    
        except Exception as e:
            logger.error(f"Failed to rollback to version {version}: {e}")
            return False
    
    async def export_config(self, file_path: Path) -> bool:
        """Exporte la configuration actuelle vers un fichier"""
        try:
            with self._lock:
                config_export = {
                    "export_timestamp": datetime.now(timezone.utc).isoformat(),
                    "current_config": self._config_data,
                    "versions": [
                        {
                            "version": v.version,
                            "timestamp": v.timestamp.isoformat(),
                            "hash": v.hash,
                            "description": v.description
                        }
                        for v in self._config_versions[-10:]  # 10 dernières versions
                    ]
                }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_export, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configuration exported to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export configuration: {e}")
            return False
    
    async def get_status(self) -> Dict[str, Any]:
        """Retourne le statut du gestionnaire de configuration"""
        with self._lock:
            return {
                "initialized": self._initialized,
                "config_file": str(self.config_file),
                "config_exists": self.config_file.exists(),
                "versions_count": len(self._config_versions),
                "latest_version": self._config_versions[-1].version if self._config_versions else None,
                "observers_count": len(self._observers),
                "file_watching_active": self._file_observer is not None and self._file_observer.is_alive(),
                "current_config_hash": self._calculate_config_hash(self._config_data)
            }
    
    async def cleanup(self) -> None:
        """Nettoie les ressources"""
        await self.stop_file_watching()
        with self._lock:
            self._observers.clear()
        logger.info("Dynamic Configuration Manager cleaned up")


# Instance globale
_dynamic_config_manager: Optional[DynamicConfigManager] = None


async def get_dynamic_config_manager() -> DynamicConfigManager:
    """Retourne l'instance globale du gestionnaire de configuration dynamique"""
    global _dynamic_config_manager
    if _dynamic_config_manager is None:
        _dynamic_config_manager = DynamicConfigManager()
        await _dynamic_config_manager.initialize()
    return _dynamic_config_manager


# Alias pour faciliter l'import
async def get_config_manager() -> DynamicConfigManager:
    """Alias pour get_dynamic_config_manager"""
    return await get_dynamic_config_manager()