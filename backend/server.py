#!/usr/bin/env python3
"""
🚀 BYJY-Trader Backend Server
Point d'entrée pour supervisor - Redirection vers l'API principale
"""

import sys
from pathlib import Path

# Ajouter le répertoire racine au PYTHONPATH
ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

# Import de l'application principale
from api.main import app

# Export pour uvicorn
__all__ = ['app']

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )