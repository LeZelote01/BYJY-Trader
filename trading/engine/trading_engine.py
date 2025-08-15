"""
🔧 Trading Engine - Moteur Principal
Orchestration et gestion des stratégies de trading
"""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

from core.logger import get_logger

logger = get_logger(__name__)


class EngineStatus(Enum):
    """Status du trading engine"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class EngineMetrics:
    """Métriques du trading engine"""
    start_time: Optional[datetime] = None
    uptime_seconds: float = 0.0
    active_strategies: int = 0
    total_orders: int = 0
    successful_orders: int = 0
    failed_orders: int = 0
    total_pnl: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "uptime_seconds": self.uptime_seconds,
            "active_strategies": self.active_strategies,
            "total_orders": self.total_orders,
            "successful_orders": self.successful_orders,
            "failed_orders": self.failed_orders,
            "success_rate": (self.successful_orders / max(1, self.total_orders)) * 100,
            "total_pnl": self.total_pnl
        }


class TradingEngine:
    """
    Moteur de trading principal
    Orchestre l'exécution des stratégies et la gestion des ordres
    """
    
    def __init__(self):
        self.engine_id: str = str(uuid.uuid4())
        self.status: EngineStatus = EngineStatus.STOPPED
        self.strategies: Dict[str, Any] = {}
        self.metrics: EngineMetrics = EngineMetrics()
        
        # Gestionnaires
        self.order_manager = None
        self.position_manager = None
        self.execution_handler = None
        
        # Configuration
        self.config = {
            "max_strategies": 10,
            "update_interval": 1.0,  # secondes
            "enable_risk_management": True,
            "paper_trading_mode": True
        }
        
        logger.info(f"Trading Engine initialisé - ID: {self.engine_id}")
    
    async def start(self) -> bool:
        """
        Démarre le trading engine
        """
        try:
            if self.status != EngineStatus.STOPPED:
                logger.warning(f"Engine déjà en cours - Status: {self.status}")
                return False
            
            logger.info("Démarrage du Trading Engine...")
            self.status = EngineStatus.STARTING
            
            # Initialisation des gestionnaires
            await self._initialize_managers()
            
            # Démarrage des stratégies actives
            await self._start_active_strategies()
            
            # Démarrage de la boucle principale
            self._start_main_loop()
            
            self.status = EngineStatus.RUNNING
            self.metrics.start_time = datetime.now(timezone.utc)
            
            logger.info("Trading Engine démarré avec succès")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors du démarrage engine: {str(e)}")
            self.status = EngineStatus.ERROR
            return False
    
    async def stop(self) -> bool:
        """
        Arrête le trading engine
        """
        try:
            if self.status != EngineStatus.RUNNING:
                logger.warning(f"Engine pas en cours d'exécution - Status: {self.status}")
                return False
            
            logger.info("Arrêt du Trading Engine...")
            self.status = EngineStatus.STOPPING
            
            # Arrêt des stratégies
            await self._stop_all_strategies()
            
            # Fermeture des gestionnaires
            await self._shutdown_managers()
            
            self.status = EngineStatus.STOPPED
            
            # Calcul uptime
            if self.metrics.start_time:
                uptime = datetime.now(timezone.utc) - self.metrics.start_time
                self.metrics.uptime_seconds = uptime.total_seconds()
            
            logger.info("Trading Engine arrêté avec succès")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'arrêt engine: {str(e)}")
            self.status = EngineStatus.ERROR
            return False
    
    def add_strategy(self, strategy: Any, strategy_config: Optional[Dict] = None) -> str:
        """
        Ajoute une stratégie au engine
        """
        try:
            if len(self.strategies) >= self.config["max_strategies"]:
                raise ValueError(f"Maximum {self.config['max_strategies']} stratégies autorisées")
            
            strategy_id = str(uuid.uuid4())
            
            strategy_info = {
                "id": strategy_id,
                "strategy": strategy,
                "config": strategy_config or {},
                "status": "inactive",
                "created_at": datetime.now(timezone.utc),
                "metrics": {
                    "total_signals": 0,
                    "active_positions": 0,
                    "pnl": 0.0
                }
            }
            
            self.strategies[strategy_id] = strategy_info
            
            logger.info(f"Stratégie ajoutée - ID: {strategy_id}, Type: {strategy.__class__.__name__}")
            return strategy_id
            
        except Exception as e:
            logger.error(f"Erreur ajout stratégie: {str(e)}")
            raise
    
    def remove_strategy(self, strategy_id: str) -> bool:
        """
        Supprime une stratégie du engine
        """
        try:
            if strategy_id not in self.strategies:
                logger.warning(f"Stratégie inconnue: {strategy_id}")
                return False
            
            strategy_info = self.strategies[strategy_id]
            
            # Désactiver d'abord si active
            if strategy_info["status"] == "active":
                self._deactivate_strategy(strategy_id)
            
            del self.strategies[strategy_id]
            
            logger.info(f"Stratégie supprimée - ID: {strategy_id}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur suppression stratégie: {str(e)}")
            return False
    
    def get_active_strategies(self) -> List[Dict]:
        """
        Retourne la liste des stratégies actives
        """
        active = []
        for strategy_id, info in self.strategies.items():
            if info["status"] == "active":
                active.append({
                    "id": strategy_id,
                    "name": info["strategy"].__class__.__name__,
                    "created_at": info["created_at"].isoformat(),
                    "metrics": info["metrics"]
                })
        return active
    
    def get_engine_status(self) -> Dict[str, Any]:
        """
        Retourne le status complet du engine
        """
        # Mise à jour des métriques temps réel
        if self.metrics.start_time and self.status == EngineStatus.RUNNING:
            uptime = datetime.now(timezone.utc) - self.metrics.start_time
            self.metrics.uptime_seconds = uptime.total_seconds()
        
        self.metrics.active_strategies = len([s for s in self.strategies.values() if s["status"] == "active"])
        
        return {
            "engine_id": self.engine_id,
            "status": self.status.value,
            "config": self.config,
            "strategies_count": len(self.strategies),
            "metrics": self.metrics.to_dict(),
            "active_strategies": self.get_active_strategies()
        }
    
    # Méthodes privées
    async def _initialize_managers(self):
        """Initialise les gestionnaires"""
        from .order_manager import OrderManager
        from .position_manager import PositionManager
        from .execution_handler import ExecutionHandler
        
        self.order_manager = OrderManager()
        self.position_manager = PositionManager()
        self.execution_handler = ExecutionHandler()
        
        logger.info("Gestionnaires initialisés")
    
    async def _start_active_strategies(self):
        """Démarre les stratégies marquées comme actives"""
        active_count = 0
        for strategy_info in self.strategies.values():
            if strategy_info.get("auto_start", False):
                strategy_info["status"] = "active"
                active_count += 1
        
        logger.info(f"{active_count} stratégies démarrées automatiquement")
    
    def _start_main_loop(self):
        """Démarre la boucle principale du engine"""
        # Cette méthode démarrerait la boucle principale en arrière-plan
        # Pour l'instant, juste un placeholder
        logger.info("Boucle principale du engine démarrée")
    
    async def _stop_all_strategies(self):
        """Arrête toutes les stratégies"""
        stopped_count = 0
        for strategy_id, strategy_info in self.strategies.items():
            if strategy_info["status"] == "active":
                strategy_info["status"] = "inactive"
                stopped_count += 1
        
        logger.info(f"{stopped_count} stratégies arrêtées")
    
    async def _shutdown_managers(self):
        """Ferme les gestionnaires"""
        self.order_manager = None
        self.position_manager = None 
        self.execution_handler = None
        logger.info("Gestionnaires fermés")
    
    def _deactivate_strategy(self, strategy_id: str):
        """Désactive une stratégie"""
        if strategy_id in self.strategies:
            self.strategies[strategy_id]["status"] = "inactive"
            logger.info(f"Stratégie désactivée - ID: {strategy_id}")