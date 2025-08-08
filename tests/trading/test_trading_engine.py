"""
🧪 Tests Trading Engine - Phase 2.3
Tests pour le moteur de trading principal
"""

import pytest
import asyncio
from datetime import datetime, timezone

from trading.engine.trading_engine import TradingEngine, EngineStatus
from trading.strategies.trend_following import TrendFollowingStrategy


class TestTradingEngine:
    """Suite de tests pour le TradingEngine"""
    
    @pytest.fixture
    def engine(self):
        """Fixture pour créer un engine de test"""
        return TradingEngine()
    
    @pytest.fixture
    def sample_strategy(self):
        """Fixture pour créer une stratégie de test"""
        return TrendFollowingStrategy(symbol="BTCUSDT", timeframe="1h")
    
    def test_engine_initialization(self, engine):
        """Test l'initialisation du trading engine"""
        assert engine.status == EngineStatus.STOPPED
        assert len(engine.strategies) == 0
        assert engine.engine_id is not None
        assert engine.config["max_strategies"] == 10
        assert engine.config["paper_trading_mode"] is True
    
    @pytest.mark.asyncio
    async def test_engine_start_stop(self, engine):
        """Test le démarrage et arrêt du engine"""
        # Test démarrage
        success = await engine.start()
        assert success is True
        assert engine.status == EngineStatus.RUNNING
        assert engine.metrics.start_time is not None
        
        # Test double démarrage (ne devrait pas fonctionner)
        success = await engine.start()
        assert success is False
        
        # Test arrêt
        success = await engine.stop()
        assert success is True
        assert engine.status == EngineStatus.STOPPED
        assert engine.metrics.uptime_seconds > 0
    
    def test_add_strategy(self, engine, sample_strategy):
        """Test l'ajout d'une stratégie"""
        strategy_id = engine.add_strategy(sample_strategy)
        
        assert strategy_id is not None
        assert len(engine.strategies) == 1
        assert strategy_id in engine.strategies
        
        strategy_info = engine.strategies[strategy_id]
        assert strategy_info["strategy"] == sample_strategy
        assert strategy_info["status"] == "inactive"
        assert strategy_info["created_at"] is not None
    
    def test_add_strategy_max_limit(self, engine):
        """Test la limite maximum de stratégies"""
        # Modifier la limite pour le test
        engine.config["max_strategies"] = 2
        
        # Ajouter 2 stratégies (OK)
        strategy1 = TrendFollowingStrategy("BTCUSDT", "1h")
        strategy2 = TrendFollowingStrategy("ETHUSDT", "1h")
        
        id1 = engine.add_strategy(strategy1)
        id2 = engine.add_strategy(strategy2)
        
        assert id1 is not None
        assert id2 is not None
        
        # Essayer d'ajouter une 3ème (devrait échouer)
        strategy3 = TrendFollowingStrategy("ADAUSDT", "1h")
        
        with pytest.raises(ValueError, match="Maximum 2 stratégies autorisées"):
            engine.add_strategy(strategy3)
    
    def test_remove_strategy(self, engine, sample_strategy):
        """Test la suppression d'une stratégie"""
        # Ajouter une stratégie
        strategy_id = engine.add_strategy(sample_strategy)
        assert len(engine.strategies) == 1
        
        # Supprimer la stratégie
        success = engine.remove_strategy(strategy_id)
        assert success is True
        assert len(engine.strategies) == 0
        
        # Essayer de supprimer une stratégie inexistante
        success = engine.remove_strategy("inexistant")
        assert success is False
    
    def test_get_active_strategies(self, engine, sample_strategy):
        """Test la récupération des stratégies actives"""
        # Aucune stratégie active initialement
        active = engine.get_active_strategies()
        assert len(active) == 0
        
        # Ajouter une stratégie
        strategy_id = engine.add_strategy(sample_strategy)
        
        # Toujours aucune stratégie active (inactive par défaut)
        active = engine.get_active_strategies()
        assert len(active) == 0
        
        # Activer manuellement la stratégie
        engine.strategies[strategy_id]["status"] = "active"
        
        # Maintenant devrait avoir une stratégie active
        active = engine.get_active_strategies()
        assert len(active) == 1
        assert active[0]["id"] == strategy_id
        assert active[0]["name"] == "TrendFollowingStrategy"
    
    def test_get_engine_status(self, engine, sample_strategy):
        """Test la récupération du status du engine"""
        status = engine.get_engine_status()
        
        # Vérifier la structure de base
        assert "engine_id" in status
        assert "status" in status
        assert "config" in status
        assert "strategies_count" in status
        assert "metrics" in status
        assert "active_strategies" in status
        
        # Vérifier les valeurs initiales
        assert status["status"] == "stopped"
        assert status["strategies_count"] == 0
        assert status["metrics"]["active_strategies"] == 0
        
        # Ajouter une stratégie et vérifier
        strategy_id = engine.add_strategy(sample_strategy)
        status = engine.get_engine_status()
        assert status["strategies_count"] == 1
    
    @pytest.mark.asyncio
    async def test_engine_with_strategy_lifecycle(self, engine, sample_strategy):
        """Test du cycle de vie complet engine + stratégie"""
        # Ajouter stratégie
        strategy_id = engine.add_strategy(sample_strategy, {"auto_start": True})
        
        # Démarrer engine
        success = await engine.start()
        assert success is True
        
        # Vérifier que la stratégie a été démarrée automatiquement
        strategy_info = engine.strategies[strategy_id]
        assert strategy_info["status"] == "active"
        
        # Arrêter engine
        success = await engine.stop()
        assert success is True
        
        # Vérifier que la stratégie a été arrêtée
        strategy_info = engine.strategies[strategy_id]
        assert strategy_info["status"] == "inactive"
    
    def test_engine_metrics_update(self, engine):
        """Test la mise à jour des métriques"""
        initial_metrics = engine.metrics.to_dict()
        
        # Vérifier les valeurs initiales
        assert initial_metrics["start_time"] is None
        assert initial_metrics["uptime_seconds"] == 0.0
        assert initial_metrics["active_strategies"] == 0
        
        # Simuler quelques métriques
        engine.metrics.total_orders = 5
        engine.metrics.successful_orders = 4
        engine.metrics.failed_orders = 1
        
        updated_metrics = engine.metrics.to_dict()
        assert updated_metrics["total_orders"] == 5
        assert updated_metrics["success_rate"] == 80.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])