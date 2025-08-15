"""
🔍 Path Utilities
Utilitaires pour la gestion robuste des chemins dans BYJY-Trader
Permet de détecter automatiquement le répertoire racine du projet
"""

import os
import sys
from pathlib import Path
from typing import Optional, Union


def detect_project_root(start_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Détecte automatiquement le répertoire racine du projet BYJY-Trader.
    
    La détection se base sur la présence de fichiers marqueurs spécifiques :
    - pyproject.toml
    - requirements.txt
    - BYJY-TRADER_ROADMAP.md
    - launcher/main.py
    
    Args:
        start_path: Chemin de départ pour la recherche (par défaut: répertoire du fichier actuel)
        
    Returns:
        Path: Chemin absolu vers le répertoire racine du projet
        
    Raises:
        RuntimeError: Si le répertoire racine n'est pas trouvé
    """
    if start_path is None:
        start_path = Path(__file__).parent
    else:
        start_path = Path(start_path)
    
    # Fichiers marqueurs qui indiquent la racine du projet
    project_markers = [
        "pyproject.toml",
        "requirements.txt", 
        "BYJY-TRADER_ROADMAP.md",
        "launcher/main.py"
    ]
    
    current_path = start_path.resolve()
    
    # Remonter dans l'arborescence jusqu'à trouver les marqueurs
    max_levels = 10  # Limite de sécurité
    levels_checked = 0
    
    while levels_checked < max_levels:
        # Vérifier si tous les marqueurs sont présents
        markers_found = 0
        for marker in project_markers:
            marker_path = current_path / marker
            if marker_path.exists():
                markers_found += 1
        
        # Si au moins 3 marqueurs sur 4 sont trouvés, c'est probablement la racine
        if markers_found >= 3:
            return current_path
        
        # Remonter d'un niveau
        parent = current_path.parent
        if parent == current_path:  # On a atteint la racine du système
            break
        current_path = parent
        levels_checked += 1
    
    # Si on n'a pas trouvé, essayer quelques chemins communs
    fallback_paths = [
        Path.cwd(),  # Répertoire de travail actuel
        Path(__file__).parent.parent,  # Parent du répertoire core
        Path(sys.argv[0]).parent if sys.argv else None,  # Répertoire du script principal
    ]
    
    for fallback_path in fallback_paths:
        if fallback_path is None:
            continue
            
        fallback_path = fallback_path.resolve()
        markers_found = 0
        for marker in project_markers:
            if (fallback_path / marker).exists():
                markers_found += 1
        
        if markers_found >= 2:  # Critère moins strict pour fallback
            return fallback_path
    
    raise RuntimeError(
        f"Impossible de détecter le répertoire racine du projet BYJY-Trader. "
        f"Assurez-vous que les fichiers marqueurs sont présents : {project_markers}"
    )


def get_project_root() -> Path:
    """
    Retourne le répertoire racine du projet (version cachée).
    
    Returns:
        Path: Chemin absolu vers le répertoire racine
    """
    return _get_cached_project_root()


def _get_cached_project_root() -> Path:
    """Cache interne pour éviter de recalculer le chemin racine"""
    if not hasattr(_get_cached_project_root, "_cached_root"):
        _get_cached_project_root._cached_root = detect_project_root()
    return _get_cached_project_root._cached_root


def resolve_relative_path(relative_path: Union[str, Path], base_path: Optional[Path] = None) -> Path:
    """
    Résout un chemin relatif par rapport à la racine du projet ou à un chemin de base.
    
    Args:
        relative_path: Chemin relatif à résoudre
        base_path: Chemin de base (par défaut: racine du projet)
        
    Returns:
        Path: Chemin absolu résolu
    """
    if base_path is None:
        base_path = get_project_root()
    
    relative_path = Path(relative_path)
    
    # Si le chemin est déjà absolu, le retourner tel quel
    if relative_path.is_absolute():
        return relative_path
    
    # Résoudre par rapport au chemin de base
    return (base_path / relative_path).resolve()


def ensure_directory_exists(path: Union[str, Path], relative_to_project: bool = True) -> Path:
    """
    S'assure qu'un répertoire existe, le crée si nécessaire.
    
    Args:
        path: Chemin du répertoire
        relative_to_project: Si True, résout le chemin par rapport à la racine du projet
        
    Returns:
        Path: Chemin absolu du répertoire
    """
    if relative_to_project:
        path = resolve_relative_path(path)
    else:
        path = Path(path).resolve()
    
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_config_dir() -> Path:
    """Retourne le répertoire de configuration"""
    return resolve_relative_path("config")


def get_data_dir() -> Path:
    """Retourne le répertoire de données"""
    return resolve_relative_path("data")


def get_logs_dir() -> Path:
    """Retourne le répertoire des logs"""
    return resolve_relative_path("logs")


def get_database_dir() -> Path:
    """Retourne le répertoire de la base de données"""
    return resolve_relative_path("database")


def get_backups_dir() -> Path:
    """Retourne le répertoire des sauvegardes"""
    return resolve_relative_path("backups")


def get_models_dir() -> Path:
    """Retourne le répertoire des modèles IA"""
    return resolve_relative_path("ai/models")


def get_frontend_dir() -> Path:
    """Retourne le répertoire frontend"""
    return resolve_relative_path("frontend")


def is_portable_installation() -> bool:
    """
    Vérifie si l'installation est portable (ex: sur clé USB).
    
    Returns:
        bool: True si l'installation semble portable
    """
    project_root = get_project_root()
    
    # Vérifier si on est sur un support amovible
    try:
        drive_type = None
        if os.name == 'nt':  # Windows
            import win32file
            drive = os.path.splitdrive(str(project_root))[0] + "\\"
            drive_type = win32file.GetDriveType(drive)
            # DRIVE_REMOVABLE = 2, DRIVE_FIXED = 3
            return drive_type == 2
        else:  # Unix-like
            # Vérifier si le point de montage contient des mots-clés de supports amovibles
            mount_info = os.statvfs(str(project_root))
            # Heuristique simple : si c'est dans /media, /mnt, ou contient 'usb'
            project_str = str(project_root).lower()
            portable_indicators = ['/media/', '/mnt/', '/run/media/', 'usb', 'removable']
            return any(indicator in project_str for indicator in portable_indicators)
    except ImportError:
        # Si win32file n'est pas disponible, utiliser des heuristiques
        pass
    except Exception:
        # En cas d'erreur, considérer comme non portable
        pass
    
    return False


def create_project_directories() -> dict:
    """
    Crée tous les répertoires nécessaires au projet.
    
    Returns:
        dict: Dictionnaire avec les chemins créés
    """
    directories = {
        'root': get_project_root(),
        'config': ensure_directory_exists("config"),
        'data': ensure_directory_exists("data"),
        'logs': ensure_directory_exists("logs"),
        'database': ensure_directory_exists("database"),
        'backups': ensure_directory_exists("backups"),
        'ai_models': ensure_directory_exists("ai/models"),
        'data_cache': ensure_directory_exists("data/cache"),
        'frontend_build': ensure_directory_exists("frontend/build"),
    }
    
    return directories


# Test de détection au chargement du module
try:
    _project_root = get_project_root()
    print(f"✅ BYJY-Trader project root detected: {_project_root}")
except RuntimeError as e:
    print(f"⚠️ Warning: {e}")