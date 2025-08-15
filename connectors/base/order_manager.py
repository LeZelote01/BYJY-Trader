"""
🛠️ Order Manager Universal - Phase 2.4

Gestionnaire d'ordres universel pour tous les exchanges.
Normalise les interfaces et assure la cohérence.

Features:
- Validation ordres avant envoi
- Tracking automatique status ordres
- Risk controls intégrés
- Retry logic pour échecs temporaires
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from decimal import Decimal
from enum import Enum

from core.logger import get_logger
from .base_connector import BaseConnector, OrderType, OrderSide, OrderStatus

logger = get_logger(__name__)


class RiskLevel(Enum):
    """Niveaux de risque"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class OrderManager:
    """
    Gestionnaire d'ordres universel multi-exchange
    
    Centralise la logique d'ordre, validation, tracking et risk management
    pour tous les connecteurs d'exchange.
    """
    
    def __init__(self, max_position_size: Decimal = Decimal("10000")):
        """
        Initialise l'order manager
        
        Args:
            max_position_size: Taille max position autorisée
        """
        self.max_position_size = max_position_size
        self.active_orders: Dict[str, Dict[str, Any]] = {}
        self.order_history: List[Dict[str, Any]] = []
        
        # Risk controls
        self.daily_loss_limit = Decimal("1000")  # $1000 max perte/jour
        self.daily_loss_current = Decimal("0")
        self.daily_reset_time = datetime.now().replace(hour=0, minute=0, second=0)
        
        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 1.0  # secondes
        
        logger.info("🛠️ OrderManager initialisé avec risk controls")
    
    async def place_order(
        self,
        connector: BaseConnector,
        symbol: str,
        order_type: OrderType,
        side: OrderSide,
        quantity: Decimal,
        price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None,
        risk_level: RiskLevel = RiskLevel.MEDIUM,
        time_in_force: str = "GTC"
    ) -> Dict[str, Any]:
        """
        Placer un ordre avec validation et risk controls
        
        Args:
            connector: Connecteur exchange à utiliser
            symbol: Symbole trading
            order_type: Type d'ordre
            side: Côté (buy/sell)
            quantity: Quantité
            price: Prix limite (optionnel)
            stop_price: Prix stop (optionnel)
            risk_level: Niveau de risque
            time_in_force: Validité ordre
            
        Returns:
            Dict avec résultat placement ordre
        """
        try:
            # 1. Validation ordre
            validation = await self._validate_order(
                symbol, order_type, side, quantity, price, stop_price, risk_level
            )
            
            if not validation["valid"]:
                logger.warning(f"⚠️ Ordre invalide: {validation['reason']}")
                return {
                    "success": False,
                    "error": validation["reason"],
                    "order_id": None
                }
            
            # 2. Générer client order ID unique
            client_order_id = f"BYJY_{uuid.uuid4().hex[:8]}_{int(datetime.now().timestamp())}"
            
            # 3. Placer ordre avec retry logic
            order_result = await self._place_order_with_retry(
                connector, symbol, order_type, side, quantity, 
                price, stop_price, time_in_force, client_order_id
            )
            
            # 4. Tracker l'ordre si placement réussi
            if order_result.get("success", False):
                await self._track_order(connector, order_result, {
                    "symbol": symbol,
                    "order_type": order_type,
                    "side": side,
                    "quantity": quantity,
                    "price": price,
                    "risk_level": risk_level,
                    "placed_at": datetime.now()
                })
            
            return order_result
            
        except Exception as e:
            logger.error(f"❌ Erreur placement ordre: {e}")
            return {
                "success": False,
                "error": str(e),
                "order_id": None
            }
    
    async def cancel_order(
        self,
        connector: BaseConnector,
        symbol: str,
        order_id: str = None,
        client_order_id: str = None
    ) -> Dict[str, Any]:
        """
        Annuler un ordre avec tracking
        
        Args:
            connector: Connecteur exchange
            symbol: Symbole trading
            order_id: ID ordre exchange
            client_order_id: ID ordre client
            
        Returns:
            Dict avec résultat annulation
        """
        try:
            # Annuler sur l'exchange
            cancel_result = await connector.cancel_order(symbol, order_id, client_order_id)
            
            # Mettre à jour tracking
            tracking_id = order_id or client_order_id
            if tracking_id in self.active_orders:
                self.active_orders[tracking_id]["status"] = OrderStatus.CANCELLED
                self.active_orders[tracking_id]["cancelled_at"] = datetime.now()
                
                # Déplacer vers historique
                self.order_history.append(self.active_orders[tracking_id])
                del self.active_orders[tracking_id]
                
                logger.info(f"✅ Ordre {tracking_id} annulé et retiré du tracking")
            
            return cancel_result
            
        except Exception as e:
            logger.error(f"❌ Erreur annulation ordre: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_order_status(
        self,
        connector: BaseConnector,
        symbol: str,
        order_id: str = None,
        client_order_id: str = None
    ) -> Dict[str, Any]:
        """
        Récupérer status ordre avec cache local
        
        Args:
            connector: Connecteur exchange
            symbol: Symbole trading
            order_id: ID ordre exchange  
            client_order_id: ID ordre client
            
        Returns:
            Dict avec status ordre
        """
        try:
            # Vérifier cache local d'abord
            tracking_id = order_id or client_order_id
            if tracking_id in self.active_orders:
                local_order = self.active_orders[tracking_id]
                
                # Si ordre récent, retourner cache
                if datetime.now() - local_order["last_updated"] < timedelta(seconds=30):
                    return {
                        "success": True,
                        "order": local_order,
                        "source": "cache"
                    }
            
            # Sinon, requête exchange
            exchange_result = await connector.get_order_status(symbol, order_id, client_order_id)
            
            # Mettre à jour cache
            if exchange_result.get("success") and tracking_id in self.active_orders:
                self.active_orders[tracking_id].update(exchange_result.get("order", {}))
                self.active_orders[tracking_id]["last_updated"] = datetime.now()
            
            return exchange_result
            
        except Exception as e:
            logger.error(f"❌ Erreur récupération status ordre: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _validate_order(
        self,
        symbol: str,
        order_type: OrderType,
        side: OrderSide,
        quantity: Decimal,
        price: Optional[Decimal],
        stop_price: Optional[Decimal],
        risk_level: RiskLevel
    ) -> Dict[str, Any]:
        """
        Valider un ordre avant placement
        
        Returns:
            Dict avec validation results
        """
        # Validation quantité
        if quantity <= 0:
            return {"valid": False, "reason": "Quantité doit être positive"}
        
        # Validation taille position
        estimated_value = quantity * (price or Decimal("50000"))  # Estimation avec BTC price
        if estimated_value > self.max_position_size:
            return {"valid": False, "reason": f"Position trop grande: ${estimated_value} > ${self.max_position_size}"}
        
        # Validation prix pour ordres limit
        if order_type == OrderType.LIMIT and price is None:
            return {"valid": False, "reason": "Prix requis pour ordre limit"}
        
        # Validation stop price pour stop orders
        if order_type in [OrderType.STOP_LOSS, OrderType.TAKE_PROFIT] and stop_price is None:
            return {"valid": False, "reason": "Stop price requis pour ordre stop"}
        
        # Risk control - limite perte journalière
        if self._check_daily_reset():
            if estimated_value > (self.daily_loss_limit - self.daily_loss_current):
                return {"valid": False, "reason": "Limite perte journalière dépassée"}
        
        # Validation risk level vs ordre size
        risk_limits = {
            RiskLevel.LOW: Decimal("1000"),
            RiskLevel.MEDIUM: Decimal("5000"), 
            RiskLevel.HIGH: Decimal("20000"),
            RiskLevel.EXTREME: self.max_position_size
        }
        
        if estimated_value > risk_limits[risk_level]:
            return {"valid": False, "reason": f"Ordre trop risqué pour niveau {risk_level.value}"}
        
        return {"valid": True, "reason": "Ordre valide"}
    
    async def _place_order_with_retry(
        self,
        connector: BaseConnector,
        symbol: str,
        order_type: OrderType,
        side: OrderSide,
        quantity: Decimal,
        price: Optional[Decimal],
        stop_price: Optional[Decimal],
        time_in_force: str,
        client_order_id: str
    ) -> Dict[str, Any]:
        """
        Placer ordre avec retry automatique
        """
        for attempt in range(self.max_retries + 1):
            try:
                result = await connector.place_order(
                    symbol, order_type, side, quantity,
                    price, stop_price, time_in_force, client_order_id
                )
                
                if result.get("success", False):
                    logger.info(f"✅ Ordre placé avec succès (tentative {attempt + 1})")
                    return result
                else:
                    logger.warning(f"⚠️ Échec placement ordre (tentative {attempt + 1}): {result.get('error', 'Unknown')}")
                    
            except Exception as e:
                logger.error(f"❌ Erreur placement ordre (tentative {attempt + 1}): {e}")
                
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                    continue
                else:
                    return {
                        "success": False,
                        "error": f"Échec après {self.max_retries + 1} tentatives: {str(e)}",
                        "order_id": None
                    }
        
        return {
            "success": False,
            "error": f"Max retries ({self.max_retries}) dépassé",
            "order_id": None
        }
    
    async def _track_order(
        self,
        connector: BaseConnector,
        order_result: Dict[str, Any],
        order_details: Dict[str, Any]
    ):
        """
        Commencer le tracking d'un ordre
        """
        order_id = order_result.get("order_id")
        if not order_id:
            return
        
        tracking_data = {
            **order_details,
            **order_result,
            "connector": connector.exchange_name,
            "status": OrderStatus.OPEN,
            "last_updated": datetime.now(),
            "fill_quantity": Decimal("0"),
            "fill_price": None
        }
        
        self.active_orders[order_id] = tracking_data
        logger.debug(f"🔍 Tracking ordre {order_id} démarré")
    
    def _check_daily_reset(self) -> bool:
        """
        Vérifier si reset journalier nécessaire
        """
        now = datetime.now()
        if now.date() > self.daily_reset_time.date():
            self.daily_loss_current = Decimal("0")
            self.daily_reset_time = now.replace(hour=0, minute=0, second=0)
            logger.info("🔄 Reset limites journalières effectué")
            return True
        return False
    
    def get_active_orders(self) -> Dict[str, Dict[str, Any]]:
        """Récupérer tous les ordres actifs"""
        return self.active_orders.copy()
    
    def get_order_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Récupérer historique ordres"""
        return self.order_history[-limit:]
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Récupérer métriques de risque"""
        self._check_daily_reset()
        
        total_position_value = sum(
            order["quantity"] * (order.get("price") or Decimal("50000"))
            for order in self.active_orders.values()
        )
        
        return {
            "active_orders_count": len(self.active_orders),
            "total_position_value": float(total_position_value),
            "daily_loss_current": float(self.daily_loss_current),
            "daily_loss_limit": float(self.daily_loss_limit),
            "daily_loss_remaining": float(self.daily_loss_limit - self.daily_loss_current),
            "max_position_size": float(self.max_position_size),
            "position_utilization": float(total_position_value / self.max_position_size * 100)
        }