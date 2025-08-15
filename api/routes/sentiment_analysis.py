"""
🗣️ API Routes - Sentiment Analysis (Phase 3.2)
Endpoints REST pour système d'analyse sentiment
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import asyncio

from ai.sentiment.sentiment_manager import get_sentiment_manager
from core.logger import get_logger

logger = get_logger("byjy.api.sentiment")
router = APIRouter(prefix="/api/sentiment")


# === MODELS PYDANTIC ===

class SentimentRequest(BaseModel):
    """Requête analyse sentiment"""
    symbols: List[str] = Field(..., description="Liste des symboles à analyser")
    period_days: int = Field(default=30, ge=1, le=365, description="Période d'analyse en jours")
    include_correlation: bool = Field(default=True, description="Inclure analyse corrélation")
    

class CollectionRequest(BaseModel):
    """Requête collecte données sentiment"""
    symbols: List[str] = Field(..., description="Symboles à collecter")
    news_limit: int = Field(default=50, ge=1, le=500, description="Limite news")
    social_limit: int = Field(default=100, ge=1, le=500, description="Limite posts sociaux")
    analyze_sentiment: bool = Field(default=True, description="Analyser sentiment")


class CorrelationRequest(BaseModel):
    """Requête analyse corrélation"""
    symbol: str = Field(..., description="Symbole à analyser")
    period_days: int = Field(default=30, ge=7, le=365, description="Période en jours")


# === DEPENDENCY ===

async def get_sentiment_system():
    """Dépendance pour obtenir le système sentiment"""
    sentiment_manager = get_sentiment_manager()
    
    if not sentiment_manager.is_initialized:
        # Initialiser si nécessaire
        initialized = await sentiment_manager.initialize()
        if not initialized:
            raise HTTPException(
                status_code=503, 
                detail="Sentiment analysis system not available"
            )
    
    return sentiment_manager


# === HEALTH & STATUS ENDPOINTS ===

@router.get("/health")
async def sentiment_health(
    sentiment_manager: Any = Depends(get_sentiment_system)
) -> Dict[str, Any]:
    """
    Santé du système sentiment
    
    Returns:
        Statut de santé complet des composants
    """
    try:
        status = await sentiment_manager.get_system_status()
        
        # Déterminer santé globale
        all_healthy = (
            status['system']['initialized'] and
            all(
                comp.get('status', 'inactive') in ['healthy', 'active']
                for comp in status['components'].values()
            )
        )
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "timestamp": datetime.now(),
            "system": status['system'],
            "components": status['components'],
            "stats": status['stats']
        }
        
    except Exception as e:
        logger.error(f"❌ Sentiment health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/status")
async def sentiment_status(
    sentiment_manager: Any = Depends(get_sentiment_system)
) -> Dict[str, Any]:
    """
    Status complet du système sentiment
    
    Returns:
        État détaillé de tous les composants
    """
    try:
        return await sentiment_manager.get_system_status()
        
    except Exception as e:
        logger.error(f"❌ Sentiment status failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


# === SENTIMENT ANALYSIS ENDPOINTS ===

@router.get("/current/{symbol}")
async def get_current_sentiment(
    symbol: str,
    sentiment_manager: Any = Depends(get_sentiment_system)
) -> Dict[str, Any]:
    """
    Sentiment actuel pour un symbole
    
    Args:
        symbol: Symbole à analyser (BTC, ETH, etc.)
        
    Returns:
        Sentiment score, label, confiance, volume
    """
    try:
        symbol = symbol.upper()
        sentiment_data = await sentiment_manager.get_current_sentiment(symbol)
        
        return {
            "success": True,
            "data": sentiment_data,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"❌ Current sentiment failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")


@router.post("/analyze")
async def analyze_sentiment(
    request: SentimentRequest,
    sentiment_manager: Any = Depends(get_sentiment_system)
) -> Dict[str, Any]:
    """
    Analyse sentiment pour multiple symboles
    
    Args:
        request: Configuration analyse sentiment
        
    Returns:
        Résultats sentiment + corrélations optionnelles
    """
    try:
        results = {}
        
        # Analyse sentiment pour chaque symbole
        for symbol in request.symbols:
            symbol = symbol.upper()
            sentiment_data = await sentiment_manager.get_current_sentiment(symbol)
            results[symbol] = sentiment_data
            
            # Ajouter corrélation si demandée
            if request.include_correlation:
                correlation_data = await sentiment_manager.get_sentiment_correlation(
                    symbol, request.period_days
                )
                results[symbol]['correlation'] = correlation_data
        
        return {
            "success": True,
            "data": results,
            "request": request.dict(),
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"❌ Sentiment analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/collect")
async def collect_sentiment_data(
    request: CollectionRequest,
    background_tasks: BackgroundTasks,
    sentiment_manager: Any = Depends(get_sentiment_system)
) -> Dict[str, Any]:
    """
    Déclencher collecte données sentiment
    
    Args:
        request: Configuration collecte
        background_tasks: Tâches en arrière-plan
        
    Returns:
        Confirmation démarrage collecte
    """
    try:
        # Lancer collecte en arrière-plan
        async def collect_task():
            try:
                logger.info(f"🔄 Starting sentiment collection for {request.symbols}")
                
                # Collecte news et social en parallèle
                news_data = await sentiment_manager.news_collector.collect_data(
                    request.symbols, request.news_limit
                )
                social_data = await sentiment_manager.social_collector.collect_data(
                    request.symbols, request.social_limit
                )
                
                collected_count = len(news_data) + len(social_data)
                logger.info(f"✅ Collection completed: {collected_count} items")
                
                # Analyser sentiment si demandé
                if request.analyze_sentiment and collected_count > 0:
                    all_texts = []
                    all_texts.extend([f"{item['title']} {item['content']}" for item in news_data])
                    all_texts.extend([item['content'] for item in social_data])
                    
                    if all_texts:
                        results = await sentiment_manager.sentiment_analyzer.analyze_batch(all_texts)
                        logger.info(f"✅ Analyzed sentiment for {len(results)} items")
                
            except Exception as e:
                logger.error(f"❌ Background collection failed: {e}")
        
        background_tasks.add_task(collect_task)
        
        return {
            "success": True,
            "message": "Sentiment collection started",
            "symbols": request.symbols,
            "limits": {
                "news": request.news_limit,
                "social": request.social_limit
            },
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"❌ Collection request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Collection failed: {str(e)}")


# === CORRELATION ANALYSIS ENDPOINTS ===

@router.post("/correlation")
async def analyze_correlation(
    request: CorrelationRequest,
    sentiment_manager: Any = Depends(get_sentiment_system)
) -> Dict[str, Any]:
    """
    Analyse corrélation sentiment-prix
    
    Args:
        request: Configuration analyse corrélation
        
    Returns:
        Résultat corrélation avec métriques
    """
    try:
        correlation_data = await sentiment_manager.get_sentiment_correlation(
            request.symbol.upper(), request.period_days
        )
        
        return {
            "success": True,
            "data": correlation_data,
            "request": request.dict(),
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"❌ Correlation analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Correlation failed: {str(e)}")


@router.get("/correlation/{symbol}")
async def get_symbol_correlation(
    symbol: str,
    period_days: int = Query(default=30, ge=7, le=365, description="Période en jours"),
    sentiment_manager: Any = Depends(get_sentiment_system)
) -> Dict[str, Any]:
    """
    Corrélation sentiment-prix pour un symbole
    
    Args:
        symbol: Symbole à analyser
        period_days: Période d'analyse en jours
        
    Returns:
        Données corrélation avec historique
    """
    try:
        symbol = symbol.upper()
        
        # Corrélation principale
        correlation_data = await sentiment_manager.get_sentiment_correlation(symbol, period_days)
        
        # Tendance corrélation
        correlation_trend = await sentiment_manager.correlation_analyzer.get_correlation_trend(symbol, 7)
        
        return {
            "success": True,
            "data": {
                "current": correlation_data,
                "trend": correlation_trend,
                "meets_target": correlation_data.get('meets_target', False),
                "target_threshold": 0.6
            },
            "symbol": symbol,
            "period_days": period_days,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"❌ Symbol correlation failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Correlation failed: {str(e)}")


@router.get("/correlation/summary")
async def get_correlation_summary(
    period_days: int = Query(default=30, ge=7, le=365, description="Période d'analyse"),
    sentiment_manager: Any = Depends(get_sentiment_system)
) -> Dict[str, Any]:
    """
    Résumé corrélations système
    
    Args:
        period_days: Période d'analyse en jours
        
    Returns:
        Vue d'ensemble corrélations tous symboles
    """
    try:
        summary = await sentiment_manager.correlation_analyzer.get_correlation_summary(period_days)
        
        return {
            "success": True,
            "data": summary,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"❌ Correlation summary failed: {e}")
        raise HTTPException(status_code=500, detail=f"Summary failed: {str(e)}")


# === MANAGEMENT ENDPOINTS ===

@router.post("/start")
async def start_sentiment_collection(
    sentiment_manager: Any = Depends(get_sentiment_system)
) -> Dict[str, Any]:
    """
    Démarrer collecte automatique sentiment
    
    Returns:
        Confirmation démarrage
    """
    try:
        if sentiment_manager.is_running:
            return {
                "success": True,
                "message": "Sentiment collection already running",
                "status": "running",
                "timestamp": datetime.now()
            }
        
        success = await sentiment_manager.start_collection()
        
        if success:
            return {
                "success": True,
                "message": "Sentiment collection started",
                "status": "running",
                "timestamp": datetime.now()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to start collection")
        
    except Exception as e:
        logger.error(f"❌ Start collection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Start failed: {str(e)}")


@router.post("/stop")
async def stop_sentiment_collection(
    sentiment_manager: Any = Depends(get_sentiment_system)
) -> Dict[str, Any]:
    """
    Arrêter collecte sentiment
    
    Returns:
        Confirmation arrêt
    """
    try:
        await sentiment_manager.stop_collection()
        
        return {
            "success": True,
            "message": "Sentiment collection stopped",
            "status": "stopped",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"❌ Stop collection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Stop failed: {str(e)}")


@router.delete("/cache")
async def clear_sentiment_cache(
    sentiment_manager: Any = Depends(get_sentiment_system)
) -> Dict[str, Any]:
    """
    Vider cache sentiment
    
    Returns:
        Confirmation nettoyage
    """
    try:
        await sentiment_manager.correlation_analyzer.clear_cache()
        
        return {
            "success": True,
            "message": "Sentiment cache cleared",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"❌ Cache clear failed: {e}")
        raise HTTPException(status_code=500, detail=f"Clear failed: {str(e)}")


# === DATA ENDPOINTS ===

@router.get("/data/news/{symbol}")
async def get_news_data(
    symbol: str,
    limit: int = Query(default=20, ge=1, le=100, description="Nombre d'articles"),
    sentiment_manager: Any = Depends(get_sentiment_system)
) -> Dict[str, Any]:
    """
    Données news récentes pour un symbole
    
    Args:
        symbol: Symbole à rechercher
        limit: Nombre max d'articles
        
    Returns:
        Articles de news avec métadonnées
    """
    try:
        symbol = symbol.upper()
        
        news_data = await sentiment_manager.news_collector.collect_data([symbol], limit)
        
        return {
            "success": True,
            "data": {
                "articles": news_data,
                "count": len(news_data),
                "symbol": symbol
            },
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"❌ News data failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"News fetch failed: {str(e)}")


@router.get("/data/social/{symbol}")
async def get_social_data(
    symbol: str,
    limit: int = Query(default=50, ge=1, le=200, description="Nombre de posts"),
    sentiment_manager: Any = Depends(get_sentiment_system)
) -> Dict[str, Any]:
    """
    Données réseaux sociaux pour un symbole
    
    Args:
        symbol: Symbole à rechercher
        limit: Nombre max de posts
        
    Returns:
        Posts sociaux avec sentiment
    """
    try:
        symbol = symbol.upper()
        
        social_data = await sentiment_manager.social_collector.collect_data([symbol], limit)
        
        return {
            "success": True,
            "data": {
                "posts": social_data,
                "count": len(social_data),
                "symbol": symbol
            },
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"❌ Social data failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Social fetch failed: {str(e)}")


# === METRICS ENDPOINTS ===

@router.get("/metrics/{symbol}")
async def get_sentiment_metrics(
    symbol: str,
    days: int = Query(default=7, ge=1, le=30, description="Nombre de jours"),
    sentiment_manager: Any = Depends(get_sentiment_system)
) -> Dict[str, Any]:
    """
    Métriques sentiment historiques
    
    Args:
        symbol: Symbole à analyser
        days: Période en jours
        
    Returns:
        Historique sentiment avec tendances
    """
    try:
        symbol = symbol.upper()
        
        # TODO: Récupérer métriques depuis base de données
        # Pour l'instant, données simulées
        metrics = {
            "symbol": symbol,
            "period_days": days,
            "avg_sentiment": 0.15,
            "sentiment_trend": "positive",
            "volume_avg": 156,
            "data_points": 25,
            "last_update": datetime.now()
        }
        
        return {
            "success": True,
            "data": metrics,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"❌ Sentiment metrics failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics failed: {str(e)}")


@router.get("/metrics/system")
async def get_system_metrics(
    sentiment_manager: Any = Depends(get_sentiment_system)
) -> Dict[str, Any]:
    """
    Métriques système sentiment
    
    Returns:
        Performance et statistiques système
    """
    try:
        status = await sentiment_manager.get_system_status()
        
        # Agrégation métriques système
        system_metrics = {
            "collection": {
                "total_collected": status['stats']['total_collected'],
                "total_analyzed": status['stats']['total_analyzed'],
                "last_collection": status['stats']['last_collection'],
                "errors": status['stats']['errors']
            },
            "correlations": status['correlations'],
            "components": {
                name: comp.get('status', 'unknown')
                for name, comp in status['components'].items()
            }
        }
        
        return {
            "success": True,
            "data": system_metrics,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"❌ System metrics failed: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics failed: {str(e)}")