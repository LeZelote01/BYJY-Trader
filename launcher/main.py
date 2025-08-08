#!/usr/bin/env python3
"""
🚀 BYJY-Trader Main Launcher
Point d'entrée principal du bot de trading portable
"""

import sys
import os
import asyncio
import signal
from pathlib import Path
from typing import Optional

# Ajouter le répertoire racine au PYTHONPATH - Système robuste
try:
    from pathlib import Path
    import sys
    
    # Détection automatique du répertoire racine
    def detect_project_root_fallback():
        """Détection basique pour le launcher"""
        current = Path(__file__).parent
        for _ in range(5):  # Limite de sécurité
            if (current / "pyproject.toml").exists() and (current / "core").exists():
                return current
            current = current.parent
        return Path(__file__).parent.parent
    
    ROOT_DIR = detect_project_root_fallback().absolute()
    sys.path.insert(0, str(ROOT_DIR))
    
except Exception:
    # Fallback en cas d'erreur
    ROOT_DIR = Path(__file__).parent.parent.absolute()
    sys.path.insert(0, str(ROOT_DIR))

try:
    import typer
    import uvicorn
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from dotenv import load_dotenv
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Run: pip install -r requirements.txt")
    sys.exit(1)

# Configuration
console = Console()
app = typer.Typer(help="🤖 BYJY-Trader - Bot de Trading Personnel Avancé")

# Global state
_running = False
_api_server = None


def load_environment() -> bool:
    """Charge les variables d'environnement avec chemins robustes"""
    from core.path_utils import get_project_root, get_config_dir
    
    try:
        project_root = get_project_root()
        config_dir = get_config_dir()
    except Exception:
        # Fallback vers l'ancienne méthode
        project_root = ROOT_DIR
        config_dir = ROOT_DIR / "config"
    
    env_files = [
        project_root / ".env",
        config_dir / ".env",
    ]
    
    for env_file in env_files:
        if env_file.exists():
            load_dotenv(env_file, override=True)
            console.print(f"✅ Loaded environment: {env_file}")
            return True
    
    console.print("⚠️  No .env file found, using defaults")
    return False


def setup_directories():
    """Crée les répertoires nécessaires avec système robuste"""
    from core.path_utils import create_project_directories
    
    try:
        directories = create_project_directories()
        for name, path in directories.items():
            console.print(f"📁 Directory ready: {name} -> {path}")
    except Exception as e:
        console.print(f"⚠️ Error creating directories: {e}")
        # Fallback vers l'ancienne méthode
        directories = [
            "logs", "database", "backups", "config", "ai/models", "data/cache"
        ]
        
        for directory in directories:
            dir_path = ROOT_DIR / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            console.print(f"📁 Directory ready: {directory}")


def signal_handler(signum, frame):
    """Gestionnaire de signaux pour arrêt propre"""
    global _running
    console.print("\n🛑 Shutdown signal received...")
    _running = False


def display_banner():
    """Affiche la bannière de démarrage"""
    banner = Text.assemble(
        ("🤖 ", "bold blue"),
        ("BYJY-TRADER", "bold white"),
        (" v0.1.0\n", "dim"),
        ("Bot de Trading Personnel Avancé", "italic")
    )
    
    panel = Panel(
        banner,
        title="[bold green]STARTUP[/bold green]",
        border_style="blue",
        padding=(1, 2)
    )
    
    console.print(panel)


def check_requirements() -> bool:
    """Vérifie que tous les requirements sont satisfaits"""
    try:
        # Test des imports critiques
        import fastapi
        import sqlalchemy
        import pandas
        import numpy
        
        console.print("✅ All critical requirements satisfied")
        return True
    except ImportError as e:
        console.print(f"❌ Missing requirement: {e}")
        console.print("💡 Run: pip install -r requirements.txt")
        return False


@app.command()
def start(
    host: str = typer.Option("0.0.0.0", help="Host to bind"),
    port: int = typer.Option(8000, help="Port to bind"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
    workers: int = typer.Option(1, help="Number of workers"),
):
    """🚀 Démarre le bot BYJY-Trader"""
    global _running, _api_server
    
    display_banner()
    
    # Setup
    if not check_requirements():
        raise typer.Exit(1)
    
    load_environment()
    setup_directories()
    
    # Override with env vars if available
    host = os.getenv("API_HOST", host)
    port = int(os.getenv("API_PORT", port))
    reload = os.getenv("API_RELOAD", "false").lower() == "true" or reload
    
    console.print(f"🌐 Starting API server on {host}:{port}")
    console.print(f"🔄 Auto-reload: {reload}")
    console.print(f"👷 Workers: {workers}")
    
    # Signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        _running = True
        
        # Démarrage du serveur API
        config = uvicorn.Config(
            "api.main:app",
            host=host,
            port=port,
            reload=reload,
            workers=workers if not reload else 1,
            log_level="info",
            access_log=True,
        )
        
        server = uvicorn.Server(config)
        asyncio.run(server.serve())
        
    except KeyboardInterrupt:
        console.print("\n🛑 Shutdown requested by user")
    except Exception as e:
        console.print(f"❌ Error: {e}")
        raise typer.Exit(1)
    finally:
        console.print("👋 BYJY-Trader stopped")


@app.command()
def init():
    """🔧 Initialise le projet BYJY-Trader"""
    display_banner()
    
    console.print("🔧 Initializing BYJY-Trader...")
    
    # Setup directories
    setup_directories()
    
    # Create .env if not exists - avec chemins robustes
    try:
        from core.path_utils import get_project_root
        project_root = get_project_root()
    except Exception:
        project_root = ROOT_DIR
    
    env_file = project_root / ".env"
    if not env_file.exists():
        example_file = project_root / ".env.example"
        if example_file.exists():
            import shutil
            shutil.copy(example_file, env_file)
            console.print("✅ Created .env from template")
        else:
            console.print("⚠️  .env.example not found")
    
    # Test database connection
    try:
        from core.database import DatabaseManager
        db = DatabaseManager()
        console.print("✅ Database connection test passed")
    except Exception as e:
        console.print(f"⚠️  Database test failed: {e}")
    
    console.print("🎉 BYJY-Trader initialization complete!")


@app.command()
def test():
    """🧪 Lance les tests"""
    console.print("🧪 Running BYJY-Trader tests...")
    
    import subprocess
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/", "-v", "--cov=core"
        ], cwd=ROOT_DIR)
        
        if result.returncode == 0:
            console.print("✅ All tests passed!")
        else:
            console.print("❌ Some tests failed")
            raise typer.Exit(1)
            
    except FileNotFoundError:
        console.print("❌ pytest not found. Run: pip install pytest")
        raise typer.Exit(1)


@app.command()
def version():
    """📋 Affiche la version"""
    console.print("🤖 BYJY-Trader v0.1.0")
    console.print("🔧 Phase 1 - Development")


if __name__ == "__main__":
    app()