#!/usr/bin/env python3
"""
Simple FastAPI server pour test
"""

import sys
from pathlib import Path

# Ajouter le répertoire racine au PYTHONPATH
ROOT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI

app = FastAPI(title="BYJY-Trader Test Server")

@app.get("/")
async def root():
    return {"message": "BYJY-Trader Test Server is running", "status": "ok"}

@app.get("/api/health/")
async def health():
    return {"status": "healthy", "message": "Test server is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("simple_server:app", host="0.0.0.0", port=8001, reload=False)