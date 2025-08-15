"""
🧪 Tests complets pour la Configuration Dynamique
Suite de tests pour valider la Fonctionnalité 1.4 selon les critères du Roadmap
"""

import pytest
import pytest_asyncio
import asyncio
import tempfile
import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timezone

from core.dynamic_config import (
    DynamicConfigManager, 
    ConfigVersion,
    get_dynamic_config_manager,
    get_config_manager
)
from core.config import get_settings


class TestDynamicConfigInitialization:
    """Tests pour l'initialisation du gestionnaire de configuration dynamique"""
    
    @pytest_asyncio.fixture
    async def temp_config_manager(self):
        """Gestionnaire de config temporaire pour les tests"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('core.config.get_settings') as mock_settings:
                mock_config = MagicMock()
                mock_config.config_dir = Path(temp_dir) / "config"
                mock_config.debug = False
                mock_settings.return_value = mock_config
                
                config_manager = DynamicConfigManager()
                await config_manager.initialize()
                
                yield config_manager
                
                # Cleanup
                await config_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_initialization_success(self, temp_config_manager):
        """Test initialisation réussie du gestionnaire"""
        config_manager = temp_config_manager
        
        # Vérifier que l'initialisation s'est bien passée
        status = await config_manager.get_status()
        assert status["initialized"] is True
        assert status["config_exists"] is True
        assert status["versions_count"] >= 0
        assert status["file_watching_active"] is True
    
    @pytest.mark.asyncio
    async def test_default_config_creation(self, temp_config_manager):
        """Test création de la configuration par défaut"""
        config_manager = temp_config_manager
        
        # Vérifier que la configuration par défaut est créée
        config = config_manager.get_config()
        
        # Vérifier les sections principales
        assert "trading" in config
        assert "risk_management" in config
        assert "api" in config
        assert "logging" in config
        assert "system" in config
        
        # Vérifier quelques valeurs par défaut
        assert config["trading"]["max_position_size"] == 1000.0
        assert config["trading"]["enabled"] is True
        assert config["logging"]["level"] == "INFO"
    
    @pytest.mark.asyncio
    async def test_config_file_persistence(self, temp_config_manager):
        """Test persistance du fichier de configuration"""
        config_manager = temp_config_manager
        
        # Vérifier que le fichier de configuration existe
        assert config_manager.config_file.exists()
        
        # Vérifier que le contenu peut être lu
        with open(config_manager.config_file, 'r') as f:
            file_config = json.load(f)
        
        current_config = config_manager.get_config()
        assert file_config == current_config


class TestConfigurationAccess:
    """Tests pour l'accès à la configuration"""
    
    @pytest_asyncio.fixture
    async def config_manager(self):
        """Gestionnaire de config pour les tests d'accès"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('core.config.get_settings') as mock_settings:
                mock_config = MagicMock()
                mock_config.config_dir = Path(temp_dir) / "config"
                mock_settings.return_value = mock_config
                
                config_manager = DynamicConfigManager()
                await config_manager.initialize()
                
                yield config_manager
                await config_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_get_config_full(self, config_manager):
        """Test récupération de la configuration complète"""
        config = config_manager.get_config()
        
        assert isinstance(config, dict)
        assert len(config) > 0
        assert "trading" in config
    
    @pytest.mark.asyncio
    async def test_get_config_by_path(self, config_manager):
        """Test récupération de configuration par chemin"""
        # Test accès à une section
        trading_config = config_manager.get_config("trading")
        assert isinstance(trading_config, dict)
        assert "max_position_size" in trading_config
        
        # Test accès à une valeur spécifique
        max_position = config_manager.get_config("trading.max_position_size")
        assert isinstance(max_position, (int, float))
        assert max_position > 0
        
        # Test accès à une clé inexistante
        nonexistent = config_manager.get_config("nonexistent.key")
        assert nonexistent is None
    
    @pytest.mark.asyncio
    async def test_get_config_nested_path(self, config_manager):
        """Test accès à des chemins imbriqués"""
        # Test chemin valide profond
        log_level = config_manager.get_config("logging.level")
        assert log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        
        # Test chemin partiellement invalide
        invalid = config_manager.get_config("trading.nonexistent.deep")
        assert invalid is None


class TestConfigurationModification:
    """Tests pour la modification de configuration"""
    
    @pytest_asyncio.fixture
    async def config_manager(self):
        """Gestionnaire de config pour les tests de modification"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('core.config.get_settings') as mock_settings:
                mock_config = MagicMock()
                mock_config.config_dir = Path(temp_dir) / "config"
                mock_settings.return_value = mock_config
                
                config_manager = DynamicConfigManager()
                await config_manager.initialize()
                
                yield config_manager
                await config_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_set_config_valid_value(self, config_manager):
        """Test modification avec valeur valide"""
        # Modifier une valeur existante
        success = await config_manager.set_config("trading.max_position_size", 2000.0)
        assert success is True
        
        # Vérifier que la valeur a été modifiée
        new_value = config_manager.get_config("trading.max_position_size")
        assert new_value == 2000.0
    
    @pytest.mark.asyncio
    async def test_set_config_new_key(self, config_manager):
        """Test ajout d'une nouvelle clé de configuration"""
        # Ajouter une nouvelle clé
        success = await config_manager.set_config("trading.new_parameter", "test_value")
        assert success is True
        
        # Vérifier que la nouvelle clé existe
        new_value = config_manager.get_config("trading.new_parameter")
        assert new_value == "test_value"
    
    @pytest.mark.asyncio
    async def test_set_config_nested_new_section(self, config_manager):
        """Test création d'une nouvelle section imbriquée"""
        # Créer une nouvelle section
        success = await config_manager.set_config("new_section.subsection.value", 42)
        assert success is True
        
        # Vérifier que la structure a été créée
        value = config_manager.get_config("new_section.subsection.value")
        assert value == 42
        
        section = config_manager.get_config("new_section")
        assert isinstance(section, dict)
        assert "subsection" in section
    
    @pytest.mark.asyncio
    async def test_set_config_validation_success(self, config_manager):
        """Test validation réussie lors de la modification"""
        # Valeurs valides selon les règles de validation
        valid_changes = [
            ("trading.max_position_size", 500.0),
            ("trading.stop_loss_percentage", 1.5),
            ("risk_management.max_concurrent_orders", 10),
            ("api.timeout_seconds", 45.0),
            ("logging.level", "DEBUG")
        ]
        
        for key_path, value in valid_changes:
            success = await config_manager.set_config(key_path, value)
            assert success is True, f"Failed to set {key_path}={value}"
            
            actual_value = config_manager.get_config(key_path)
            assert actual_value == value
    
    @pytest.mark.asyncio
    async def test_set_config_validation_failure(self, config_manager):
        """Test échec de validation lors de la modification"""
        # Valeurs invalides selon les règles de validation
        invalid_changes = [
            ("trading.max_position_size", -100.0),  # Négatif
            ("trading.stop_loss_percentage", 0),     # Zéro
            ("trading.stop_loss_percentage", 60),    # Trop élevé
            ("risk_management.max_concurrent_orders", 0),  # Zéro
            ("api.timeout_seconds", -5),             # Négatif
            ("logging.level", "INVALID_LEVEL")       # Niveau invalide
        ]
        
        for key_path, value in invalid_changes:
            success = await config_manager.set_config(key_path, value)
            assert success is False, f"Should have failed for {key_path}={value}"


class TestConfigurationVersioning:
    """Tests pour le système de versioning"""
    
    @pytest_asyncio.fixture
    async def config_manager(self):
        """Gestionnaire de config pour les tests de versioning"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('core.config.get_settings') as mock_settings:
                mock_config = MagicMock()
                mock_config.config_dir = Path(temp_dir) / "config"
                mock_settings.return_value = mock_config
                
                config_manager = DynamicConfigManager()
                await config_manager.initialize()
                
                yield config_manager
                await config_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_version_creation_on_change(self, config_manager):
        """Test création de version lors des changements"""
        # État initial
        initial_versions = len(config_manager.get_versions())
        
        # Modifier la configuration
        await config_manager.set_config("trading.max_position_size", 1500.0)
        
        # Vérifier qu'une nouvelle version a été créée
        versions = config_manager.get_versions()
        assert len(versions) > initial_versions
        
        # Vérifier la dernière version
        latest_version = versions[0]  # Plus récente en premier
        assert isinstance(latest_version, ConfigVersion)
        assert latest_version.version.startswith("v")
        assert latest_version.timestamp is not None
        assert latest_version.hash is not None
    
    @pytest.mark.asyncio
    async def test_version_with_description(self, config_manager):
        """Test création de version avec description"""
        description = "Test change with custom description"
        
        await config_manager.set_config(
            "trading.enabled", 
            False, 
            description=description
        )
        
        versions = config_manager.get_versions()
        latest_version = versions[0]
        
        assert description in latest_version.description
    
    @pytest.mark.asyncio
    async def test_version_persistence(self, config_manager):
        """Test persistance des versions"""
        # Créer plusieurs versions
        changes = [
            ("trading.max_position_size", 800.0),
            ("risk_management.max_drawdown", 5.0),
            ("api.timeout_seconds", 60.0)
        ]
        
        for key_path, value in changes:
            await config_manager.set_config(key_path, value)
        
        # Vérifier que les versions sont sauvegardées
        assert config_manager.versions_file.exists()
        
        # Vérifier le contenu du fichier de versions
        with open(config_manager.versions_file, 'r') as f:
            versions_data = json.load(f)
        
        assert isinstance(versions_data, list)
        assert len(versions_data) >= len(changes)
        
        # Vérifier la structure des versions
        for version_data in versions_data:
            assert "version" in version_data
            assert "timestamp" in version_data
            assert "config_data" in version_data
            assert "hash" in version_data
    
    @pytest.mark.asyncio
    async def test_get_versions_limit(self, config_manager):
        """Test limitation du nombre de versions retournées"""
        # Créer plusieurs versions
        for i in range(15):
            await config_manager.set_config(f"test.value_{i}", i)
        
        # Test limite par défaut (10)
        versions_default = config_manager.get_versions()
        assert len(versions_default) <= 10
        
        # Test limite personnalisée
        versions_5 = config_manager.get_versions(limit=5)
        assert len(versions_5) <= 5
        
        # Vérifier l'ordre (plus récent en premier)
        if len(versions_default) > 1:
            assert versions_default[0].timestamp >= versions_default[1].timestamp


class TestConfigurationRollback:
    """Tests pour le rollback de configuration"""
    
    @pytest_asyncio.fixture
    async def config_manager_with_versions(self):
        """Gestionnaire avec plusieurs versions pour les tests de rollback"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('core.config.get_settings') as mock_settings:
                mock_config = MagicMock()
                mock_config.config_dir = Path(temp_dir) / "config"
                mock_settings.return_value = mock_config
                
                config_manager = DynamicConfigManager()
                await config_manager.initialize()
                
                # Créer quelques versions
                await config_manager.set_config("trading.max_position_size", 1000.0, "Initial value")
                await config_manager.set_config("trading.max_position_size", 1500.0, "First change")
                await config_manager.set_config("trading.max_position_size", 2000.0, "Second change")
                
                yield config_manager
                await config_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_rollback_to_existing_version(self, config_manager_with_versions):
        """Test rollback vers une version existante"""
        config_manager = config_manager_with_versions
        
        # Obtenir les versions disponibles
        versions = config_manager.get_versions()
        assert len(versions) >= 2
        
        # Sélectionner une version antérieure (pas la plus récente)
        target_version = versions[1].version  # Deuxième plus récente
        target_config = versions[1].config_data
        
        # Effectuer le rollback
        success = await config_manager.rollback_to_version(target_version)
        assert success is True
        
        # Vérifier que la configuration a été restaurée
        current_value = config_manager.get_config("trading.max_position_size")
        expected_value = target_config["trading"]["max_position_size"]
        assert current_value == expected_value
    
    @pytest.mark.asyncio
    async def test_rollback_to_nonexistent_version(self, config_manager_with_versions):
        """Test rollback vers une version inexistante"""
        config_manager = config_manager_with_versions
        
        # Tentative de rollback vers une version inexistante
        success = await config_manager.rollback_to_version("v9999")
        assert success is False
        
        # Vérifier que la configuration n'a pas changé
        current_config = config_manager.get_config()
        assert current_config is not None
    
    @pytest.mark.asyncio
    async def test_rollback_creates_new_version(self, config_manager_with_versions):
        """Test que le rollback crée une nouvelle version"""
        config_manager = config_manager_with_versions
        
        # Compter les versions avant rollback
        versions_before = len(config_manager.get_versions())
        
        # Effectuer un rollback
        versions = config_manager.get_versions()
        target_version = versions[1].version
        
        await config_manager.rollback_to_version(target_version)
        
        # Vérifier qu'une nouvelle version a été créée
        versions_after = len(config_manager.get_versions())
        assert versions_after > versions_before


class TestHotReload:
    """Tests pour le hot-reload de configuration"""
    
    @pytest_asyncio.fixture
    async def config_manager(self):
        """Gestionnaire de config pour les tests de hot-reload"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('core.config.get_settings') as mock_settings:
                mock_config = MagicMock()
                mock_config.config_dir = Path(temp_dir) / "config"
                mock_settings.return_value = mock_config
                
                config_manager = DynamicConfigManager()
                await config_manager.initialize()
                
                yield config_manager
                await config_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_hot_reload_from_file(self, config_manager):
        """Test hot-reload depuis un fichier modifié"""
        # Modifier directement le fichier de configuration
        new_config = config_manager.get_config()
        new_config["trading"]["max_position_size"] = 3000.0
        new_config["test_hot_reload"] = True
        
        # Écrire dans le fichier
        with open(config_manager.config_file, 'w') as f:
            json.dump(new_config, f, indent=2)
        
        # Effectuer le hot-reload
        success = await config_manager.reload_config()
        assert success is True
        
        # Vérifier que les changements ont été appliqués
        reloaded_value = config_manager.get_config("trading.max_position_size")
        assert reloaded_value == 3000.0
        
        test_value = config_manager.get_config("test_hot_reload")
        assert test_value is True
    
    @pytest.mark.asyncio
    async def test_hot_reload_no_change(self, config_manager):
        """Test hot-reload sans changement de fichier"""
        # Obtenir la configuration actuelle
        original_config = config_manager.get_config()
        
        # Hot-reload sans modification
        success = await config_manager.reload_config()
        assert success is True
        
        # Vérifier que la configuration n'a pas changé
        current_config = config_manager.get_config()
        assert current_config == original_config
    
    @pytest.mark.asyncio
    async def test_hot_reload_invalid_json(self, config_manager):
        """Test hot-reload avec JSON invalide"""
        # Écrire du JSON invalide dans le fichier
        with open(config_manager.config_file, 'w') as f:
            f.write("{ invalid json content }")
        
        # Le hot-reload devrait gérer l'erreur gracieusement
        success = await config_manager.reload_config()
        # Peut être False ou True selon la gestion d'erreur, mais ne doit pas crasher
        assert isinstance(success, bool)


class TestConfigurationObservers:
    """Tests pour le système d'observateurs"""
    
    @pytest_asyncio.fixture
    async def config_manager(self):
        """Gestionnaire de config pour les tests d'observateurs"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('core.config.get_settings') as mock_settings:
                mock_config = MagicMock()
                mock_config.config_dir = Path(temp_dir) / "config"
                mock_settings.return_value = mock_config
                
                config_manager = DynamicConfigManager()
                await config_manager.initialize()
                
                yield config_manager
                await config_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_add_remove_observer(self, config_manager):
        """Test ajout et suppression d'observateurs"""
        observer_calls = []
        
        def test_observer(config):
            observer_calls.append(config.copy())
        
        # Ajouter l'observateur
        config_manager.add_observer(test_observer)
        
        # Modifier la configuration
        await config_manager.set_config("trading.max_position_size", 1200.0)
        
        # Attendre que l'observateur soit appelé
        await asyncio.sleep(0.1)
        
        # Vérifier que l'observateur a été appelé
        assert len(observer_calls) > 0
        
        # Vérifier que la nouvelle configuration a été passée
        last_config = observer_calls[-1]
        assert last_config["trading"]["max_position_size"] == 1200.0
        
        # Supprimer l'observateur
        config_manager.remove_observer(test_observer)
        
        # Modifier à nouveau
        calls_before = len(observer_calls)
        await config_manager.set_config("trading.max_position_size", 1300.0)
        await asyncio.sleep(0.1)
        
        # L'observateur ne devrait plus être appelé
        assert len(observer_calls) == calls_before
    
    @pytest.mark.asyncio
    async def test_multiple_observers(self, config_manager):
        """Test avec plusieurs observateurs"""
        observer1_calls = []
        observer2_calls = []
        
        def observer1(config):
            observer1_calls.append("observer1")
        
        def observer2(config):
            observer2_calls.append("observer2")
        
        # Ajouter les deux observateurs
        config_manager.add_observer(observer1)
        config_manager.add_observer(observer2)
        
        # Modifier la configuration
        await config_manager.set_config("trading.enabled", False)
        await asyncio.sleep(0.1)
        
        # Vérifier que les deux observateurs ont été appelés
        assert len(observer1_calls) > 0
        assert len(observer2_calls) > 0
    
    @pytest.mark.asyncio
    async def test_observer_exception_handling(self, config_manager):
        """Test gestion des exceptions dans les observateurs"""
        def failing_observer(config):
            raise Exception("Observer error")
        
        def working_observer(config):
            working_observer.called = True
        working_observer.called = False
        
        # Ajouter les deux observateurs
        config_manager.add_observer(failing_observer)
        config_manager.add_observer(working_observer)
        
        # Modifier la configuration
        await config_manager.set_config("test.value", "test")
        await asyncio.sleep(0.1)
        
        # Vérifier que l'observateur fonctionnel a quand même été appelé
        assert working_observer.called is True


class TestConfigurationExportImport:
    """Tests pour l'export et import de configuration"""
    
    @pytest_asyncio.fixture
    async def config_manager_with_data(self):
        """Gestionnaire avec données pour les tests d'export/import"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('core.config.get_settings') as mock_settings:
                mock_config = MagicMock()
                mock_config.config_dir = Path(temp_dir) / "config"
                mock_settings.return_value = mock_config
                
                config_manager = DynamicConfigManager()
                await config_manager.initialize()
                
                # Ajouter quelques données
                await config_manager.set_config("trading.max_position_size", 1500.0)
                await config_manager.set_config("custom.test_value", "exported")
                
                yield config_manager, temp_dir
                await config_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_export_configuration(self, config_manager_with_data):
        """Test export de la configuration"""
        config_manager, temp_dir = config_manager_with_data
        
        export_file = Path(temp_dir) / "config_export.json"
        
        # Exporter la configuration
        success = await config_manager.export_config(export_file)
        assert success is True
        assert export_file.exists()
        
        # Vérifier le contenu de l'export
        with open(export_file, 'r') as f:
            export_data = json.load(f)
        
        assert "export_timestamp" in export_data
        assert "current_config" in export_data
        assert "versions" in export_data
        
        # Vérifier que la configuration actuelle est présente
        current_config = export_data["current_config"]
        assert current_config["trading"]["max_position_size"] == 1500.0
        assert current_config["custom"]["test_value"] == "exported"
    
    @pytest.mark.asyncio
    async def test_export_to_invalid_path(self, config_manager_with_data):
        """Test export vers un chemin invalide"""
        config_manager, temp_dir = config_manager_with_data
        
        # Chemin invalide (répertoire inexistant)
        invalid_path = Path("/invalid/nonexistent/path/config.json")
        
        success = await config_manager.export_config(invalid_path)
        assert success is False


class TestConfigurationStatus:
    """Tests pour le statut du gestionnaire de configuration"""
    
    @pytest_asyncio.fixture
    async def config_manager(self):
        """Gestionnaire de config pour les tests de statut"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('core.config.get_settings') as mock_settings:
                mock_config = MagicMock()
                mock_config.config_dir = Path(temp_dir) / "config"
                mock_settings.return_value = mock_config
                
                config_manager = DynamicConfigManager()
                await config_manager.initialize()
                
                yield config_manager
                await config_manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_get_status_complete(self, config_manager):
        """Test récupération du statut complet"""
        status = await config_manager.get_status()
        
        # Vérifier tous les champs de statut
        required_fields = [
            "initialized", "config_file", "config_exists", 
            "versions_count", "latest_version", "observers_count",
            "file_watching_active", "current_config_hash"
        ]
        
        for field in required_fields:
            assert field in status, f"Missing status field: {field}"
        
        # Vérifier les types et valeurs
        assert isinstance(status["initialized"], bool)
        assert isinstance(status["config_file"], str)
        assert isinstance(status["config_exists"], bool)
        assert isinstance(status["versions_count"], int)
        assert isinstance(status["observers_count"], int)
        assert isinstance(status["file_watching_active"], bool)
        assert isinstance(status["current_config_hash"], str)
        
        # Vérifier les valeurs cohérentes
        assert status["initialized"] is True
        assert status["config_exists"] is True
        assert status["versions_count"] >= 0
        assert len(status["current_config_hash"]) > 0


@pytest.mark.asyncio
async def test_global_config_manager_singleton():
    """Test que le gestionnaire global est bien un singleton"""
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch('core.config.get_settings') as mock_settings:
            mock_config = MagicMock()
            mock_config.config_dir = Path(temp_dir) / "config"
            mock_settings.return_value = mock_config
            
            # Obtenir deux instances
            manager1 = await get_dynamic_config_manager()
            manager2 = await get_config_manager()
            
            # Vérifier que c'est la même instance
            assert manager1 is manager2
            
            # Cleanup
            await manager1.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])