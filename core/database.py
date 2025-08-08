"""
🗄️ Database Management
Gestionnaire de base de données avec SQLAlchemy et support SQLite portable
"""

import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager
import sqlite3
from datetime import datetime, timezone

from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import aiosqlite

from .config import get_settings
from .logger import get_logger

# Import des modèles pour création automatique des tables
from .models import Base
from .models.trading import TradingPair, Order, Trade, Position
from .models.strategy import Strategy, StrategyExecution  
from .models.system import SystemLog, Configuration
# Note: UserModel imported dynamically to avoid circular imports
logger = get_logger("byjy.database")


class DatabaseManager:
    """Gestionnaire principal de base de données"""
    
    def __init__(self):
        self.settings = get_settings()
        self.sync_engine = None
        self.async_engine = None
        self.async_session_factory = None
        self.sync_session_factory = None
        self._initialized = False
    
    def _get_database_url(self, async_mode: bool = False) -> str:
        """Génère l'URL de base de données"""
        db_path = self.settings.get_database_path()
        
        # S'assurer que le répertoire existe
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        if async_mode:
            return f"sqlite+aiosqlite:///{db_path}"
        else:
            return f"sqlite:///{db_path}"
    
    def initialize_sync(self) -> None:
        """Initialise la connexion synchrone"""
        if self.sync_engine is not None:
            return
        
        database_url = self._get_database_url(async_mode=False)
        logger.info(f"Initializing sync database connection: {database_url}")
        
        # Configuration pour SQLite portable
        self.sync_engine = create_engine(
            database_url,
            poolclass=StaticPool,
            connect_args={
                "check_same_thread": False,
                "timeout": 30,
                "isolation_level": None,  # autocommit mode
            },
            echo=self.settings.debug,
            future=True
        )
        
        # Factory pour les sessions
        self.sync_session_factory = sessionmaker(
            bind=self.sync_engine,
            autoflush=True,
            autocommit=False
        )
    
    async def initialize_async(self) -> None:
        """Initialise la connexion asynchrone"""
        if self.async_engine is not None:
            return
        
        database_url = self._get_database_url(async_mode=True)
        logger.info(f"Initializing async database connection: {database_url}")
        
        # Configuration pour SQLite async portable
        self.async_engine = create_async_engine(
            database_url,
            poolclass=StaticPool,
            connect_args={
                "check_same_thread": False,
                "timeout": 30,
            },
            echo=self.settings.debug,
            future=True
        )
        
        # Factory pour les sessions async
        self.async_session_factory = async_sessionmaker(
            bind=self.async_engine,
            class_=AsyncSession,
            autoflush=True,
            autocommit=False,
            expire_on_commit=False
        )
    
    def create_tables(self) -> None:
        """Crée toutes les tables"""
        if self.sync_engine is None:
            self.initialize_sync()
        
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=self.sync_engine)
        logger.info("Database tables created successfully")
    
    async def create_tables_async(self) -> bool:
        """Crée toutes les tables de manière asynchrone"""
        try:
            # Import UserModel here to avoid circular imports
            from .user_db import UserModel
            
            async with self.async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            logger.info("Database tables created successfully (async)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create database tables (async): {e}")
            return False
    
    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Context manager pour session asynchrone"""
        if self.async_session_factory is None:
            await self.initialize_async()
        
        async with self.async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error(f"Database session error: {e}")
                raise
            finally:
                await session.close()
    
    def get_sync_session(self):
        """Context manager pour session synchrone"""
        if self.sync_session_factory is None:
            self.initialize_sync()
        
        return self.sync_session_factory()
    
    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Exécute une requête SQL brute (async)"""
        async with self.get_async_session() as session:
            result = await session.execute(text(query), params or {})
            return result
    
    def execute_query_sync(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Exécute une requête SQL brute (sync)"""
        with self.get_sync_session() as session:
            result = session.execute(text(query), params or {})
            session.commit()
            return result
    
    async def health_check(self) -> Dict[str, Any]:
        """Vérifie l'état de la base de données"""
        try:
            async with self.get_async_session() as session:
                # Test simple de connexion
                result = await session.execute(text("SELECT 1 as health_check"))
                row = result.fetchone()
                
                # Informations sur la base
                db_path = self.settings.get_database_path()
                db_size = db_path.stat().st_size if db_path.exists() else 0
                
                return {
                    "status": "healthy",
                    "connection": "ok",
                    "database_path": str(db_path),
                    "database_size_bytes": db_size,
                    "test_query_result": row[0] if row else None,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy", 
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def backup_database(self, backup_path: Optional[Path] = None) -> bool:
        """Crée une sauvegarde de la base de données"""
        try:
            source_path = self.settings.get_database_path()
            if not source_path.exists():
                logger.warning("Database file does not exist, cannot backup")
                return False
            
            if backup_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = self.settings.backups_dir / f"byjy_trader_backup_{timestamp}.db"
            
            # S'assurer que le répertoire de sauvegarde existe
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copie avec SQLite backup API pour cohérence
            with sqlite3.connect(source_path) as source_conn:
                with sqlite3.connect(backup_path) as backup_conn:
                    source_conn.backup(backup_conn)
            
            logger.info(f"Database backup created: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            return False
    
    def restore_database(self, backup_path: Path) -> bool:
        """Restaure la base de données depuis une sauvegarde"""
        try:
            if not backup_path.exists():
                logger.error(f"Backup file does not exist: {backup_path}")
                return False
            
            target_path = self.settings.get_database_path()
            
            # Fermer toutes les connexions
            if self.sync_engine:
                self.sync_engine.dispose()
            if self.async_engine:
                # Note: For sync method, we can't await. This should be handled differently.
                self.async_engine.dispose()
            
            # Copie du backup
            with sqlite3.connect(backup_path) as backup_conn:
                with sqlite3.connect(target_path) as target_conn:
                    backup_conn.backup(target_conn)
            
            # Réinitialiser les engines
            self.sync_engine = None
            self.async_engine = None
            self.sync_session_factory = None
            self.async_session_factory = None
            
            logger.info(f"Database restored from backup: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Database restore failed: {e}")
            return False
    
    async def close(self) -> None:
        """Ferme toutes les connexions"""
        if self.async_engine:
            await self.async_engine.dispose()
            logger.info("Async database engine disposed")
        
        if self.sync_engine:
            self.sync_engine.dispose()
            logger.info("Sync database engine disposed")


# Instance globale
_db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """Retourne l'instance globale du gestionnaire de base de données"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


# Alias pour faciliter l'import
db_manager = get_database_manager()