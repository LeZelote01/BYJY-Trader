"""
🔄 Backup Manager
Gestionnaire automatique des sauvegardes de base de données
"""

import asyncio
import schedule
import threading
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional

from .database import get_database_manager
from .config import get_settings
from .logger import get_logger

logger = get_logger("byjy.backup")


class BackupManager:
    """Gestionnaire automatique des sauvegardes"""
    
    def __init__(self):
        self.settings = get_settings()
        self.db_manager = get_database_manager()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
    def create_backup(self, custom_name: str = None) -> bool:
        """
        Crée une sauvegarde manuelle
        
        Args:
            custom_name: Nom personnalisé pour la sauvegarde
            
        Returns:
            bool: True si succès
        """
        try:
            if custom_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"{custom_name}_{timestamp}.db"
                backup_path = self.settings.backups_dir / backup_name
            else:
                backup_path = None
                
            success = self.db_manager.backup_database(backup_path)
            
            if success:
                logger.info(f"Manual backup created successfully")
                return True
            else:
                logger.error("Manual backup failed")
                return False
                
        except Exception as e:
            logger.error(f"Error creating manual backup: {e}")
            return False
    
    def create_scheduled_backup(self) -> bool:
        """Crée une sauvegarde programmée"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"scheduled_backup_{timestamp}.db"
            backup_path = self.settings.backups_dir / backup_name
            
            success = self.db_manager.backup_database(backup_path)
            
            if success:
                logger.info(f"Scheduled backup created: {backup_path}")
                # Nettoyer les anciennes sauvegardes
                self.cleanup_old_backups()
                return True
            else:
                logger.error("Scheduled backup failed")
                return False
                
        except Exception as e:
            logger.error(f"Error creating scheduled backup: {e}")
            return False
    
    def cleanup_old_backups(self) -> int:
        """
        Nettoie les anciennes sauvegardes selon la politique de rétention
        
        Returns:
            int: Nombre de fichiers supprimés
        """
        try:
            backup_dir = self.settings.backups_dir
            if not backup_dir.exists():
                return 0
                
            retention_days = self.settings.backup_retention_days
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            deleted_count = 0
            for backup_file in backup_dir.glob("*.db"):
                # Vérifier l'âge du fichier
                file_time = datetime.fromtimestamp(backup_file.stat().st_mtime)
                
                if file_time < cutoff_date:
                    try:
                        backup_file.unlink()
                        deleted_count += 1
                        logger.info(f"Deleted old backup: {backup_file.name}")
                    except Exception as e:
                        logger.warning(f"Could not delete backup {backup_file.name}: {e}")
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old backup files")
                
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old backups: {e}")
            return 0
    
    def list_backups(self) -> List[dict]:
        """
        Liste toutes les sauvegardes disponibles
        
        Returns:
            List[dict]: Liste des sauvegardes avec métadonnées
        """
        try:
            backup_dir = self.settings.backups_dir
            if not backup_dir.exists():
                return []
                
            backups = []
            for backup_file in backup_dir.glob("*.db"):
                stat = backup_file.stat()
                backups.append({
                    "name": backup_file.name,
                    "path": str(backup_file),
                    "size_bytes": stat.st_size,
                    "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
            
            # Trier par date de création (plus récent en premier)
            backups.sort(key=lambda x: x["created_at"], reverse=True)
            return backups
            
        except Exception as e:
            logger.error(f"Error listing backups: {e}")
            return []
    
    def restore_from_backup(self, backup_name: str) -> bool:
        """
        Restaure depuis une sauvegarde spécifique
        
        Args:
            backup_name: Nom du fichier de sauvegarde
            
        Returns:
            bool: True si succès
        """
        try:
            backup_path = self.settings.backups_dir / backup_name
            
            if not backup_path.exists():
                logger.error(f"Backup file not found: {backup_name}")
                return False
            
            success = self.db_manager.restore_database(backup_path)
            
            if success:
                logger.info(f"Database restored from backup: {backup_name}")
                return True
            else:
                logger.error(f"Failed to restore from backup: {backup_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error restoring from backup {backup_name}: {e}")
            return False
    
    def start_automatic_backups(self) -> bool:
        """Démarre les sauvegardes automatiques"""
        try:
            if not self.settings.backup_enabled:
                logger.info("Automatic backups are disabled in configuration")
                return False
                
            if self._running:
                logger.warning("Automatic backups are already running")
                return True
            
            # Configuration du planning
            interval_hours = self.settings.backup_interval // 3600  # Convertir secondes en heures
            schedule.every(interval_hours).hours.do(self.create_scheduled_backup)
            
            # Créer une sauvegarde initiale
            self.create_scheduled_backup()
            
            # Démarrer le thread de scheduling
            self._running = True
            self._thread = threading.Thread(target=self._scheduler_worker, daemon=True)
            self._thread.start()
            
            logger.info(f"Automatic backups started (interval: {interval_hours}h)")
            return True
            
        except Exception as e:
            logger.error(f"Error starting automatic backups: {e}")
            return False
    
    def stop_automatic_backups(self) -> bool:
        """Arrête les sauvegardes automatiques"""
        try:
            if not self._running:
                logger.info("Automatic backups are not running")
                return True
                
            self._running = False
            schedule.clear()
            
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=5)
            
            logger.info("Automatic backups stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping automatic backups: {e}")
            return False
    
    def _scheduler_worker(self):
        """Worker thread pour les tâches programmées"""
        while self._running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Vérifier toutes les minutes
            except Exception as e:
                logger.error(f"Error in backup scheduler: {e}")
                time.sleep(60)
    
    def get_status(self) -> dict:
        """Retourne le statut du gestionnaire de sauvegarde"""
        try:
            backups = self.list_backups()
            return {
                "automatic_backups_enabled": self.settings.backup_enabled,
                "automatic_backups_running": self._running,
                "backup_interval_hours": self.settings.backup_interval // 3600,
                "retention_days": self.settings.backup_retention_days,
                "backup_directory": str(self.settings.backups_dir),
                "total_backups": len(backups),
                "latest_backup": backups[0] if backups else None,
                "next_scheduled_backup": schedule.next_run() if self._running else None
            }
        except Exception as e:
            logger.error(f"Error getting backup status: {e}")
            return {
                "error": str(e),
                "automatic_backups_running": False
            }


# Instance globale
_backup_manager: Optional[BackupManager] = None


def get_backup_manager() -> BackupManager:
    """Retourne l'instance globale du gestionnaire de sauvegarde"""
    global _backup_manager
    if _backup_manager is None:
        _backup_manager = BackupManager()
    return _backup_manager