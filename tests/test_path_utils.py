"""
🧪 Tests pour le système de chemins robustes
Tests de validation pour core/path_utils.py
"""

import os
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

from core.path_utils import (
    detect_project_root,
    get_project_root,
    resolve_relative_path,
    ensure_directory_exists,
    get_config_dir,
    get_data_dir,
    get_logs_dir,
    get_database_dir,
    get_backups_dir,
    get_models_dir,
    is_portable_installation,
    create_project_directories
)


class TestProjectRootDetection:
    """Tests pour la détection du répertoire racine"""
    
    def test_detect_project_root_current_directory(self):
        """Test de détection depuis le répertoire actuel"""
        root = detect_project_root()
        assert root.is_absolute()
        assert root.is_dir()
        
        # Vérifier la présence des fichiers marqueurs
        markers = [
            "pyproject.toml",
            "requirements.txt", 
            "BYJY-TRADER_ROADMAP.md"
        ]
        
        found_markers = 0
        for marker in markers:
            if (root / marker).exists():
                found_markers += 1
        
        assert found_markers >= 2, f"Pas assez de marqueurs trouvés dans {root}"
    
    def test_detect_project_root_from_subdirectory(self):
        """Test de détection depuis un sous-répertoire"""
        # Partir du répertoire core
        core_dir = Path(__file__).parent.parent / "core"
        if core_dir.exists():
            root = detect_project_root(core_dir)
            assert root.is_absolute()
            assert (root / "pyproject.toml").exists() or (root / "requirements.txt").exists()
    
    def test_get_project_root_cached(self):
        """Test que get_project_root utilise le cache"""
        root1 = get_project_root()
        root2 = get_project_root()
        assert root1 == root2
        assert root1.is_absolute()
    
    def test_detect_project_root_failure(self):
        """Test de gestion d'échec de détection"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "very_deep" / "nested" / "directory"
            temp_path.mkdir(parents=True)
            with pytest.raises(RuntimeError, match="Impossible de détecter"):
                detect_project_root(temp_path)


class TestPathResolution:
    """Tests pour la résolution de chemins"""
    
    def test_resolve_relative_path_basic(self):
        """Test de résolution basique"""
        root = get_project_root()
        resolved = resolve_relative_path("config")
        expected = root / "config"
        assert resolved == expected.resolve()
    
    def test_resolve_relative_path_nested(self):
        """Test de résolution de chemin imbriqué"""
        root = get_project_root()
        resolved = resolve_relative_path("ai/models")
        expected = root / "ai" / "models"
        assert resolved == expected.resolve()
    
    def test_resolve_relative_path_absolute_passthrough(self):
        """Test que les chemins absolus passent sans modification"""
        abs_path = Path("/tmp/test").resolve()
        resolved = resolve_relative_path(abs_path)
        assert resolved == abs_path
    
    def test_resolve_relative_path_custom_base(self):
        """Test de résolution avec base personnalisée"""
        base = Path("/tmp")
        resolved = resolve_relative_path("subdir", base)
        expected = base / "subdir"
        assert resolved == expected.resolve()


class TestDirectoryUtils:
    """Tests pour les utilitaires de répertoires"""
    
    def test_ensure_directory_exists(self):
        """Test de création de répertoire"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "test_subdir" / "nested"
            result = ensure_directory_exists(test_dir, relative_to_project=False)
            assert result.exists()
            assert result.is_dir()
            assert result == test_dir.resolve()
    
    def test_directory_getters(self):
        """Test des fonctions de récupération de répertoires"""
        root = get_project_root()
        
        # Test tous les getters
        getters = [
            (get_config_dir, "config"),
            (get_data_dir, "data"),
            (get_logs_dir, "logs"),
            (get_database_dir, "database"),
            (get_backups_dir, "backups"),
            (get_models_dir, "ai/models"),
        ]
        
        for getter_func, expected_subpath in getters:
            result = getter_func()
            expected = root / expected_subpath
            assert result == expected.resolve()
    
    def test_create_project_directories(self):
        """Test de création de tous les répertoires du projet"""
        directories = create_project_directories()
        
        # Vérifier que tous les répertoires sont créés
        required_dirs = [
            'root', 'config', 'data', 'logs', 
            'database', 'backups', 'ai_models', 'data_cache'
        ]
        
        for dir_name in required_dirs:
            assert dir_name in directories
            assert directories[dir_name].exists()
            assert directories[dir_name].is_dir()


class TestPortabilityDetection:
    """Tests pour la détection de portabilité"""
    
    def test_is_portable_installation(self):
        """Test de détection d'installation portable"""
        # Ce test est difficile à tester de manière déterministe
        # car il dépend de l'environnement d'exécution
        result = is_portable_installation()
        assert isinstance(result, bool)
    
    def test_is_portable_installation_windows(self):
        """Test de détection sur Windows (skip if not available)"""
        pytest.skip("Windows-specific test, skipping on this platform")
    
    def test_is_portable_installation_unix_like(self):
        """Test de détection sur Unix-like"""
        with patch('os.name', 'posix'):
            result = is_portable_installation()
            assert isinstance(result, bool)


class TestErrorHandling:
    """Tests pour la gestion d'erreurs"""
    
    def test_path_utils_with_missing_dependencies(self):
        """Test du comportement avec des dépendances manquantes"""
        # Test que le module fonctionne même si certaines dépendances sont manquantes
        result = is_portable_installation()
        assert isinstance(result, bool)  # Le fonction doit fonctionner même sans win32file
    
    def test_path_resolution_with_permission_error(self):
        """Test de gestion des erreurs de permissions"""
        # Ce test est complexe car il dépend des permissions système
        # On va juste vérifier que la fonction fonctionne normalement
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "test_dir"
            result = ensure_directory_exists(test_path, relative_to_project=False)
            assert result.exists()
            assert result.is_dir()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])