"""
🌐 BYJY-Trader FastAPI Main Application
Point d'entrée principal de l'API REST et WebSocket
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from core.config import get_settings
from core.database import get_database_manager
from core.logger import get_logger
from .routes import health, system, trading, auth, data_collection, ai_predictions, connectors, backtesting, strategies, risk_management, ensemble_predictions, sentiment_analysis, rl_trading, optimization
from .websocket import websocket_router

# Logger
logger = get_logger("byjy.api.main")
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire du cycle de vie de l'application"""
    logger.info("🚀 Starting BYJY-Trader API...")
    
    # Initialisation
    try:
        # Initialiser la base de données
        db_manager = get_database_manager()
        await db_manager.initialize_async()
        await db_manager.create_tables_async()
        logger.info("✅ Database initialized successfully")
        
        # Initialize data collection components
        try:
            from .routes.data_collection import initialize_data_components
            await initialize_data_components()
            logger.info("✅ Data collection components initialized successfully")
        except Exception as e:
            logger.error(f"⚠️ Failed to initialize data collection components: {e}")
        
        # Autres initialisations futures (Redis, etc.)
        logger.info("✅ BYJY-Trader API started successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to start API: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("🛑 Shutting down BYJY-Trader API...")
    try:
        db_manager = get_database_manager()
        await db_manager.close()
        
        # Cleanup data collection components
        try:
            from .routes.data_collection import collectors
            for collector in collectors.values():
                await collector.disconnect()
        except Exception as e:
            logger.error(f"⚠️ Error disconnecting collectors: {e}")
        
        logger.info("✅ API shutdown complete")
    except Exception as e:
        logger.error(f"⚠️ Error during shutdown: {e}")


# Création de l'application FastAPI
app = FastAPI(
    title="BYJY-Trader API",
    description="🤖 Bot de Trading Personnel Avancé avec IA",
    version="0.1.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None,
    lifespan=lifespan
)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permet toutes les origines pour Emergent
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["*", "Authorization", "Content-Type"],
)

# Middleware sécurité
if settings.is_production():
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1"]
    )

# Routes principales
app.include_router(health.router, prefix="/api/health", tags=["Health"])
app.include_router(system.router, prefix="/api/system", tags=["System"])
# Routes de trading et stratégies
app.include_router(trading.router, prefix="/api/trading", tags=["trading"])
app.include_router(strategies.router, prefix="/api/strategies", tags=["strategies"])
app.include_router(risk_management.router, prefix="/api/risk", tags=["risk_management"])
app.include_router(backtesting.router, prefix="/api/backtesting", tags=["Backtesting"])
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(data_collection.router, prefix="/api/data", tags=["Data Collection"])
app.include_router(ai_predictions.router, prefix="/api", tags=["AI Predictions"])
app.include_router(ensemble_predictions.router, tags=["Ensemble Predictions"])  # 🧠 Phase 3.1 - Ensemble Models
app.include_router(connectors.router, prefix="/api/connectors", tags=["Connectors"])  # 🔌 Phase 2.4 - Connecteurs Exchange
app.include_router(sentiment_analysis.router, tags=["Sentiment Analysis"])  # 🗣️ Phase 3.2 - Sentiment Analysis
app.include_router(rl_trading.router, prefix="/api", tags=["RL Trading"])  # 🤖 Phase 3.3 - Reinforcement Learning
app.include_router(optimization.router, tags=["Optimization"])  # 🧬 Phase 3.4 - Genetic Optimization



# WebSocket
app.include_router(websocket_router, prefix="/api/ws")

# Route racine
@app.get("/")
async def root():
    """Endpoint racine avec informations système"""
    return {
        "service": "BYJY-Trader API",
        "version": "0.1.0",
        "status": "running",
        "environment": settings.environment,
        "docs_url": "/docs" if settings.debug else "disabled"
    }

# Gestionnaire d'erreur global
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Gestionnaire d'erreurs HTTP personnalisé"""
    logger.error(f"HTTP {exc.status_code}: {exc.detail} - Path: {request.url.path}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url.path)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Gestionnaire d'erreurs générales"""
    logger.error(f"Unhandled exception: {exc} - Path: {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "path": str(request.url.path)
        }
    )

# Pour exécution directe
if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower()
    )