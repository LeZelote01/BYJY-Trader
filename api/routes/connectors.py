"""
🔌 API Routes Connecteurs - Phase 2.4

Endpoints API REST pour gestion des connecteurs d'exchange.
Support Binance, Coinbase, Kraken, Bybit.

Features:
- Connexion/déconnexion exchanges
- Configuration connecteurs  
- Status et health check
- Order management via connecteurs
- WebSocket feed management
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
from decimal import Decimal

from core.logger import get_logger
from connectors import BinanceConnector, CoinbaseConnector
from connectors.base.order_manager import OrderManager, RiskLevel
from connectors.base.feed_manager import FeedManager
from connectors.base.exchange_config import ExchangeConfig
from connectors.base.base_connector import OrderType, OrderSide

logger = get_logger(__name__)

# Router pour connecteurs
router = APIRouter(tags=["Connecteurs Exchange"])

# Instances globales (à améliorer avec DI en production)
active_connectors: Dict[str, Any] = {}
order_manager = OrderManager()
feed_manager = FeedManager()


# === MODÈLES PYDANTIC ===

class ConnectorConfig(BaseModel):
    """Configuration connecteur"""
    exchange: str = Field(..., description="Nom exchange (binance, coinbase, etc.)")
    api_key: Optional[str] = Field(None, description="Clé API")
    api_secret: Optional[str] = Field(None, description="Secret API")  
    sandbox: bool = Field(True, description="Mode sandbox/testnet")
    
class OrderRequest(BaseModel):
    """Requête placement ordre"""
    exchange: str = Field(..., description="Exchange à utiliser")
    symbol: str = Field(..., description="Symbole trading")
    order_type: str = Field(..., description="Type ordre (market, limit, stop_loss)")
    side: str = Field(..., description="Côté (buy, sell)")
    quantity: float = Field(..., gt=0, description="Quantité")
    price: Optional[float] = Field(None, description="Prix limite")
    stop_price: Optional[float] = Field(None, description="Prix stop")
    risk_level: str = Field("medium", description="Niveau risque (low, medium, high)")

class FeedSubscription(BaseModel):
    """Abonnement feed données"""
    exchange: str = Field(..., description="Exchange")
    feed_type: str = Field(..., description="Type feed (ticker, order_book, trades)")
    symbol: str = Field(..., description="Symbole")

# === ENDPOINTS CONNECTEURS ===

@router.get("/supported", response_model=Dict[str, Any])
async def get_supported_exchanges():
    """
    Liste des exchanges supportés
    
    Returns:
        Dict avec exchanges supportés et leurs fonctionnalités
    """
    try:
        supported = ExchangeConfig.get_supported_exchanges()
        
        exchanges_info = {}
        for exchange_name, config in supported.items():
            limits = ExchangeConfig.get_limits(exchange_name)
            exchanges_info[exchange_name] = {
                "name": config["name"],
                "features": {
                    "supports_spot": config["supports_spot"],
                    "supports_futures": config["supports_futures"],
                    "supports_websocket": config["supports_websocket"],
                    "order_types": config["order_types"],
                    "time_in_force": config["time_in_force"]
                },
                "limits": {
                    "requests_per_minute": limits.requests_per_minute,
                    "orders_per_second": limits.requests_per_second,
                    "max_order_size": float(limits.max_order_size),
                    "min_order_size": float(limits.min_order_size)
                }
            }
        
        return {
            "success": True,
            "supported_exchanges": exchanges_info,
            "count": len(supported)
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération exchanges supportés: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/connect", response_model=Dict[str, Any])
async def connect_exchange(config: ConnectorConfig):
    """
    Se connecter à un exchange
    
    Args:
        config: Configuration connecteur
        
    Returns:
        Dict avec status connexion
    """
    try:
        exchange = config.exchange.lower()
        
        # Vérifier si exchange supporté
        if exchange not in ExchangeConfig.get_supported_exchanges():
            raise HTTPException(
                status_code=400, 
                detail=f"Exchange '{exchange}' non supporté"
            )
        
        # Créer connecteur selon exchange
        connector = None
        if exchange == "binance":
            connector = BinanceConnector(
                api_key=config.api_key,
                api_secret=config.api_secret,
                sandbox=config.sandbox
            )
        elif exchange == "coinbase":
            connector = CoinbaseConnector(
                api_key=config.api_key,
                api_secret=config.api_secret,
                passphrase=getattr(config, 'passphrase', None),  # Pour legacy Coinbase Pro
                sandbox=config.sandbox
            )
        # TODO: Ajouter Kraken et Bybit
        else:
            raise HTTPException(
                status_code=501,
                detail=f"Connecteur {exchange} pas encore implémenté"
            )
        
        # Tester connexion
        connection_result = await connector.test_connection()
        if not connection_result["success"]:
            raise HTTPException(
                status_code=400,
                detail=f"Échec connexion {exchange}: {connection_result['error']}"
            )
        
        # Connecter
        connected = await connector.connect()
        if not connected:
            raise HTTPException(
                status_code=500,
                detail=f"Impossible de se connecter à {exchange}"
            )
        
        # Stocker connecteur actif
        active_connectors[exchange] = connector
        
        logger.info(f"✅ Connecteur {exchange} connecté avec succès")
        
        return {
            "success": True,
            "exchange": exchange,
            "connected_at": datetime.now().isoformat(),
            "connection_info": connection_result,
            "sandbox_mode": config.sandbox
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur connexion exchange: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/disconnect/{exchange}", response_model=Dict[str, Any])
async def disconnect_exchange(exchange: str):
    """
    Se déconnecter d'un exchange
    
    Args:
        exchange: Nom exchange
        
    Returns:
        Dict avec status déconnexion
    """
    try:
        exchange = exchange.lower()
        
        if exchange not in active_connectors:
            raise HTTPException(
                status_code=404,
                detail=f"Connecteur {exchange} non actif"
            )
        
        connector = active_connectors[exchange]
        
        # Déconnecter
        disconnected = await connector.disconnect()
        if disconnected:
            del active_connectors[exchange]
            logger.info(f"✅ Connecteur {exchange} déconnecté")
        
        return {
            "success": disconnected,
            "exchange": exchange,
            "disconnected_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur déconnexion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status", response_model=Dict[str, Any])
async def get_connectors_status():
    """
    Status de tous les connecteurs actifs
    
    Returns:
        Dict avec status de chaque connecteur
    """
    try:
        status_info = {}
        
        for exchange, connector in active_connectors.items():
            health_check = await connector.health_check()
            
            # Ajouter infos WebSocket si disponible
            ws_status = None
            if hasattr(connector, 'ws') and connector.ws:
                ws_status = connector.ws.get_status()
            
            status_info[exchange] = {
                "health": health_check,
                "websocket": ws_status
            }
        
        # Ajouter métriques order manager
        risk_metrics = order_manager.get_risk_metrics()
        
        # Ajouter status feed manager
        feed_status = feed_manager.get_feed_status()
        
        return {
            "success": True,
            "active_connectors": status_info,
            "active_count": len(active_connectors),
            "order_manager": risk_metrics,
            "feed_manager": feed_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{exchange}/health", response_model=Dict[str, Any])
async def get_exchange_health(exchange: str):
    """
    Health check d'un exchange spécifique
    
    Args:
        exchange: Nom exchange
        
    Returns:
        Dict avec health check détaillé
    """
    try:
        exchange = exchange.lower()
        
        if exchange not in active_connectors:
            raise HTTPException(
                status_code=404,
                detail=f"Connecteur {exchange} non actif"
            )
        
        connector = active_connectors[exchange]
        
        # Test connexion complet
        health_result = await connector.test_connection()
        
        return {
            "success": True,
            "exchange": exchange,
            "health_check": health_result,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === ENDPOINTS TRADING ===

@router.post("/order/place", response_model=Dict[str, Any])
async def place_order(order_request: OrderRequest):
    """
    Placer un ordre via connecteur
    
    Args:
        order_request: Détails de l'ordre
        
    Returns:
        Dict avec résultat placement
    """
    try:
        exchange = order_request.exchange.lower()
        
        if exchange not in active_connectors:
            raise HTTPException(
                status_code=404,
                detail=f"Connecteur {exchange} non connecté"
            )
        
        connector = active_connectors[exchange]
        
        # Convertir types
        order_type = OrderType(order_request.order_type)
        side = OrderSide(order_request.side)
        quantity = Decimal(str(order_request.quantity))
        price = Decimal(str(order_request.price)) if order_request.price else None
        stop_price = Decimal(str(order_request.stop_price)) if order_request.stop_price else None
        risk_level = RiskLevel(order_request.risk_level)
        
        # Placer ordre via order manager
        result = await order_manager.place_order(
            connector=connector,
            symbol=order_request.symbol,
            order_type=order_type,
            side=side,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            risk_level=risk_level
        )
        
        if result["success"]:
            logger.info(f"✅ Ordre placé: {result['order_id']}")
        else:
            logger.warning(f"⚠️ Échec ordre: {result['error']}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur placement ordre: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/order/{exchange}/{symbol}/{order_id}", response_model=Dict[str, Any])
async def cancel_order(exchange: str, symbol: str, order_id: str):
    """
    Annuler un ordre
    
    Args:
        exchange: Exchange
        symbol: Symbole
        order_id: ID ordre
        
    Returns:
        Dict avec résultat annulation
    """
    try:
        exchange = exchange.lower()
        
        if exchange not in active_connectors:
            raise HTTPException(
                status_code=404,
                detail=f"Connecteur {exchange} non connecté"
            )
        
        connector = active_connectors[exchange]
        
        # Annuler ordre via order manager
        result = await order_manager.cancel_order(
            connector=connector,
            symbol=symbol,
            order_id=order_id
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur annulation ordre: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/orders/active", response_model=Dict[str, Any])
async def get_active_orders():
    """
    Récupérer tous les ordres actifs
    
    Returns:
        Dict avec ordres actifs
    """
    try:
        active_orders = order_manager.get_active_orders()
        
        return {
            "success": True,
            "active_orders": active_orders,
            "count": len(active_orders),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération ordres actifs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/orders/history", response_model=Dict[str, Any])
async def get_orders_history(limit: int = 50):
    """
    Récupérer historique ordres
    
    Args:
        limit: Nombre max ordres
        
    Returns:
        Dict avec historique
    """
    try:
        history = order_manager.get_order_history(limit)
        
        return {
            "success": True,
            "order_history": history,
            "count": len(history),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération historique: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === ENDPOINTS MARKET DATA ===

@router.get("/{exchange}/ticker/{symbol}", response_model=Dict[str, Any])
async def get_ticker(exchange: str, symbol: str):
    """
    Récupérer ticker d'un symbole
    
    Args:
        exchange: Exchange
        symbol: Symbole
        
    Returns:
        Dict avec données ticker
    """
    try:
        exchange = exchange.lower()
        
        if exchange not in active_connectors:
            raise HTTPException(
                status_code=404,
                detail=f"Connecteur {exchange} non connecté"
            )
        
        connector = active_connectors[exchange]
        result = await connector.get_ticker(symbol)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur récupération ticker: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{exchange}/orderbook/{symbol}", response_model=Dict[str, Any])
async def get_order_book(exchange: str, symbol: str, depth: int = 20):
    """
    Récupérer order book
    
    Args:
        exchange: Exchange
        symbol: Symbole
        depth: Profondeur
        
    Returns:
        Dict avec order book
    """
    try:
        exchange = exchange.lower()
        
        if exchange not in active_connectors:
            raise HTTPException(
                status_code=404,
                detail=f"Connecteur {exchange} non connecté"
            )
        
        connector = active_connectors[exchange]
        result = await connector.get_order_book(symbol, depth)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur récupération order book: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === ENDPOINTS WEBSOCKET FEEDS ===

@router.post("/feeds/subscribe", response_model=Dict[str, Any])
async def subscribe_feed(subscription: FeedSubscription):
    """
    S'abonner à un feed de données
    
    Args:
        subscription: Détails abonnement
        
    Returns:
        Dict avec status abonnement
    """
    try:
        exchange = subscription.exchange.lower()
        
        if exchange not in active_connectors:
            raise HTTPException(
                status_code=404,
                detail=f"Connecteur {exchange} non connecté"
            )
        
        connector = active_connectors[exchange]
        
        # Démarrer feed via feed manager
        success = await feed_manager.start_feed(
            connector=connector,
            feed_type=subscription.feed_type,
            symbol=subscription.symbol
        )
        
        if success:
            logger.info(f"✅ Feed {subscription.feed_type} {subscription.symbol} démarré")
        
        return {
            "success": success,
            "exchange": exchange,
            "feed_type": subscription.feed_type,
            "symbol": subscription.symbol,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur abonnement feed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/feeds/{exchange}/{feed_type}/{symbol}", response_model=Dict[str, Any])
async def unsubscribe_feed(exchange: str, feed_type: str, symbol: str):
    """
    Se désabonner d'un feed
    
    Args:
        exchange: Exchange
        feed_type: Type feed
        symbol: Symbole
        
    Returns:
        Dict avec status désabonnement
    """
    try:
        exchange = exchange.lower()
        
        if exchange not in active_connectors:
            raise HTTPException(
                status_code=404,
                detail=f"Connecteur {exchange} non connecté"
            )
        
        connector = active_connectors[exchange]
        
        # Arrêter feed
        success = await feed_manager.stop_feed(
            connector=connector,
            feed_type=feed_type,
            symbol=symbol
        )
        
        return {
            "success": success,
            "exchange": exchange,
            "feed_type": feed_type,
            "symbol": symbol,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur désabonnement feed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/feeds/status", response_model=Dict[str, Any])
async def get_feeds_status():
    """
    Status de tous les feeds actifs
    
    Returns:
        Dict avec status feeds
    """
    try:
        status = feed_manager.get_feed_status()
        
        return {
            "success": True,
            "feeds_status": status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur status feeds: {e}")
        raise HTTPException(status_code=500, detail=str(e))