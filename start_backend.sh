#!/bin/bash
# Script de lancement du backend BYJY-Trader avec venv local

cd /app
export PYTHONPATH="/app"
export PATH="/app/venv/bin:$PATH"

# Démarrer avec le venv et sans reload pour éviter les problèmes de file watching
exec /app/venv/bin/uvicorn server:app --host 0.0.0.0 --port 8001 --workers 1