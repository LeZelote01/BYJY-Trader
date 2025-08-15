"""
🛑 Stop Loss Manager - Gestionnaire de Stop-Loss
Gestion avancée des stop-loss avec trailing, ATR, et autres méthodes
"""

from typing import Dict, Optional, Any, Tuple
from enum import Enum
import logging

from core.logger import get_logger

logger = get_logger(__name__)


class StopLossType(Enum):
    """Types de stop-loss"""
    FIXED = "fixed"
    PERCENTAGE = "percentage"
    TRAILING = "trailing"
    ATR_BASED = "atr_based"
    VOLATILITY_BASED = "volatility_based"
    TIME_BASED = "time_based"


class StopLossManager:
    """
    Gestionnaire avancé des stop-loss
    Implémente différentes stratégies de stop-loss
    """
    
    def __init__(self):
        # Configuration par défaut
        self.default_config = {
            # Stop-loss basique
            "default_stop_loss_pct": 2.0,      # 2% par défaut
            "max_stop_loss_pct": 5.0,          # Maximum 5%
            
            # Trailing stop
            "trailing_distance_pct": 2.0,       # Distance trailing 2%
            "trailing_activation_pct": 1.0,     # Activation après 1% de profit
            
            # Stop-loss ATR
            "atr_multiplier": 2.0,              # Multiplicateur ATR
            "atr_period": 14,                   # Période ATR
            
            # Stop-loss temporel
            "max_hold_hours": 24,               # Max 24h de détention
            "time_stop_enabled": False,          # Désactivé par défaut
            
            # Protection contre les gaps
            "gap_protection": True,
            "max_gap_pct": 3.0                  # Protection gap 3%
        }
        
        # État des trailing stops
        self.trailing_stops = {}  # position_id -> trailing_data
        
        logger.info("StopLossManager initialisé")
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        side: str,  # "long" ou "short"
        stop_type: StopLossType = StopLossType.PERCENTAGE,
        parameters: Optional[Dict] = None
    ) -> float:
        """
        Calcule le niveau de stop-loss selon la méthode choisie
        """
        try:
            parameters = parameters or {}
            side = side.lower()
            
            if entry_price <= 0:
                logger.warning("Prix d'entrée invalide pour calcul stop-loss")
                return 0.0
            
            # Calcul selon le type
            if stop_type == StopLossType.FIXED:
                stop_price = parameters.get("stop_price", entry_price * 0.98)
                
            elif stop_type == StopLossType.PERCENTAGE:
                stop_pct = parameters.get("stop_loss_pct", self.default_config["default_stop_loss_pct"])
                stop_price = self._calculate_percentage_stop(entry_price, side, stop_pct)
                
            elif stop_type == StopLossType.TRAILING:
                # Pour trailing, retourner le stop initial
                trailing_pct = parameters.get("trailing_distance_pct", self.default_config["trailing_distance_pct"])
                stop_price = self._calculate_percentage_stop(entry_price, side, trailing_pct)
                
            elif stop_type == StopLossType.ATR_BASED:
                atr = parameters.get("atr", entry_price * 0.02)  # 2% si ATR non fourni
                multiplier = parameters.get("atr_multiplier", self.default_config["atr_multiplier"])
                stop_price = self._calculate_atr_stop(entry_price, side, atr, multiplier)
                
            elif stop_type == StopLossType.VOLATILITY_BASED:
                volatility = parameters.get("volatility", 0.02)  # 2% si vol non fournie
                multiplier = parameters.get("vol_multiplier", 2.0)
                stop_price = self._calculate_volatility_stop(entry_price, side, volatility, multiplier)
                
            else:
                logger.warning(f"Type de stop-loss non reconnu: {stop_type}")
                stop_price = self._calculate_percentage_stop(entry_price, side, 2.0)
            
            # Validation du résultat
            stop_price = self._validate_stop_loss(entry_price, side, stop_price)
            
            logger.debug(f"Stop-loss calculé: {stop_price:.4f} (type: {stop_type.value})")
            return stop_price
            
        except Exception as e:
            logger.error(f"Erreur calcul stop-loss: {str(e)}")
            return 0.0
    
    def update_trailing_stop(self, position: Dict[str, Any], current_price: float) -> Optional[float]:
        """
        Met à jour le trailing stop pour une position
        """
        try:
            position_id = position.get("id")
            if not position_id:
                return position.get("stop_loss")
            
            entry_price = position.get("entry_price", 0)
            current_stop = position.get("stop_loss")
            side = position.get("side", "long").lower()
            
            if entry_price <= 0 or current_price <= 0:
                return current_stop
            
            # Initialiser le trailing stop si nécessaire
            if position_id not in self.trailing_stops:
                self.trailing_stops[position_id] = {
                    "highest_price": current_price if side == "long" else current_price,
                    "lowest_price": current_price if side == "short" else current_price,
                    "initial_stop": current_stop,
                    "is_activated": False,
                    "trailing_distance_pct": self.default_config["trailing_distance_pct"]
                }
            
            trailing_data = self.trailing_stops[position_id]
            
            # Mise à jour des prix extrêmes
            if side == "long":
                if current_price > trailing_data["highest_price"]:
                    trailing_data["highest_price"] = current_price
                    
                    # Vérifier activation
                    profit_pct = ((current_price - entry_price) / entry_price) * 100
                    if profit_pct >= self.default_config["trailing_activation_pct"]:
                        trailing_data["is_activated"] = True
            else:  # short
                if current_price < trailing_data["lowest_price"]:
                    trailing_data["lowest_price"] = current_price
                    
                    # Vérifier activation
                    profit_pct = ((entry_price - current_price) / entry_price) * 100
                    if profit_pct >= self.default_config["trailing_activation_pct"]:
                        trailing_data["is_activated"] = True
            
            # Calcul du nouveau stop si activé
            if trailing_data["is_activated"]:
                new_stop = self._calculate_trailing_stop_price(trailing_data, side)
                
                # Ne déplacer le stop que dans le sens favorable
                if side == "long" and new_stop > current_stop:
                    return new_stop
                elif side == "short" and new_stop < current_stop:
                    return new_stop
            
            return current_stop
            
        except Exception as e:
            logger.error(f"Erreur update trailing stop: {str(e)}")
            return position.get("stop_loss")
    
    def check_stop_loss_hit(self, position: Dict[str, Any], current_price: float) -> Tuple[bool, str]:
        """
        Vérifie si le stop-loss a été touché
        """
        try:
            stop_loss = position.get("stop_loss")
            side = position.get("side", "long").lower()
            
            if not stop_loss or current_price <= 0:
                return False, "Stop-loss non défini ou prix invalide"
            
            # Vérification selon le côté
            if side == "long":
                if current_price <= stop_loss:
                    return True, f"Stop-loss long déclenché: {current_price:.4f} <= {stop_loss:.4f}"
            elif side == "short":
                if current_price >= stop_loss:
                    return True, f"Stop-loss short déclenché: {current_price:.4f} >= {stop_loss:.4f}"
            
            return False, "Stop-loss non déclenché"
            
        except Exception as e:
            logger.error(f"Erreur vérification stop-loss: {str(e)}")
            return False, f"Erreur: {str(e)}"
    
    def apply_time_based_stop(self, position: Dict[str, Any]) -> bool:
        """
        Vérifie si le stop temporel doit être appliqué
        """
        try:
            if not self.default_config["time_stop_enabled"]:
                return False
            
            from datetime import datetime, timezone
            
            created_at = position.get("created_at")
            if not created_at:
                return False
            
            # Calculer la durée de détention
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            
            now = datetime.now(timezone.utc)
            hold_duration = (now - created_at).total_seconds() / 3600  # en heures
            
            if hold_duration >= self.default_config["max_hold_hours"]:
                logger.info(f"Stop temporel déclenché - Position détenue {hold_duration:.1f}h")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Erreur stop temporel: {str(e)}")
            return False
    
    def get_stop_loss_info(self, position: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retourne les informations détaillées sur le stop-loss d'une position
        """
        try:
            position_id = position.get("id")
            entry_price = position.get("entry_price", 0)
            current_price = position.get("current_price", 0) 
            stop_loss = position.get("stop_loss")
            side = position.get("side", "long").lower()
            
            info = {
                "position_id": position_id,
                "current_stop_loss": stop_loss,
                "entry_price": entry_price,
                "current_price": current_price,
                "side": side
            }
            
            if stop_loss and entry_price > 0:
                # Calcul du risque
                if side == "long":
                    risk_pct = ((entry_price - stop_loss) / entry_price) * 100
                else:
                    risk_pct = ((stop_loss - entry_price) / entry_price) * 100
                
                info["risk_percentage"] = risk_pct
                
                # Distance actuelle au stop
                if current_price > 0:
                    if side == "long":
                        distance_pct = ((current_price - stop_loss) / current_price) * 100
                    else:
                        distance_pct = ((stop_loss - current_price) / current_price) * 100
                    
                    info["distance_to_stop_pct"] = distance_pct
            
            # Informations trailing si disponible
            if position_id in self.trailing_stops:
                trailing_data = self.trailing_stops[position_id]
                info["trailing_stop"] = {
                    "is_activated": trailing_data["is_activated"],
                    "highest_price": trailing_data.get("highest_price", 0),
                    "lowest_price": trailing_data.get("lowest_price", 0),
                    "trailing_distance_pct": trailing_data.get("trailing_distance_pct", 0)
                }
            
            return info
            
        except Exception as e:
            logger.error(f"Erreur info stop-loss: {str(e)}")
            return {"error": str(e)}
    
    def remove_trailing_stop(self, position_id: str):
        """
        Supprime les données de trailing stop (position fermée)
        """
        if position_id in self.trailing_stops:
            del self.trailing_stops[position_id]
            logger.debug(f"Trailing stop supprimé pour position {position_id}")
    
    # Méthodes privées
    
    def _calculate_percentage_stop(self, entry_price: float, side: str, stop_pct: float) -> float:
        """
        Calcule un stop-loss basé sur un pourcentage
        """
        stop_pct = min(stop_pct, self.default_config["max_stop_loss_pct"])
        
        if side == "long":
            return entry_price * (1 - stop_pct / 100)
        else:  # short
            return entry_price * (1 + stop_pct / 100)
    
    def _calculate_atr_stop(self, entry_price: float, side: str, atr: float, multiplier: float) -> float:
        """
        Calcule un stop-loss basé sur l'ATR
        """
        stop_distance = atr * multiplier
        
        if side == "long":
            return entry_price - stop_distance
        else:  # short
            return entry_price + stop_distance
    
    def _calculate_volatility_stop(self, entry_price: float, side: str, volatility: float, multiplier: float) -> float:
        """
        Calcule un stop-loss basé sur la volatilité
        """
        stop_distance = entry_price * volatility * multiplier
        
        if side == "long":
            return entry_price - stop_distance
        else:  # short
            return entry_price + stop_distance
    
    def _calculate_trailing_stop_price(self, trailing_data: Dict, side: str) -> float:
        """
        Calcule le prix du trailing stop
        """
        distance_pct = trailing_data["trailing_distance_pct"]
        
        if side == "long":
            return trailing_data["highest_price"] * (1 - distance_pct / 100)
        else:  # short
            return trailing_data["lowest_price"] * (1 + distance_pct / 100)
    
    def _validate_stop_loss(self, entry_price: float, side: str, stop_price: float) -> float:
        """
        Valide et corrige le niveau de stop-loss si nécessaire
        """
        try:
            # Vérifier que le stop est dans la bonne direction
            if side == "long":
                if stop_price >= entry_price:
                    logger.warning("Stop-loss long >= prix d'entrée, correction appliquée")
                    stop_price = entry_price * 0.98  # 2% en dessous
            else:  # short
                if stop_price <= entry_price:
                    logger.warning("Stop-loss short <= prix d'entrée, correction appliquée")
                    stop_price = entry_price * 1.02  # 2% au-dessus
            
            # Vérifier les limites de risque
            risk_pct = abs((entry_price - stop_price) / entry_price) * 100
            if risk_pct > self.default_config["max_stop_loss_pct"]:
                logger.warning(f"Risque stop-loss trop élevé ({risk_pct:.1f}%), limitation appliquée")
                if side == "long":
                    stop_price = entry_price * (1 - self.default_config["max_stop_loss_pct"] / 100)
                else:
                    stop_price = entry_price * (1 + self.default_config["max_stop_loss_pct"] / 100)
            
            return round(stop_price, 6)
            
        except Exception as e:
            logger.error(f"Erreur validation stop-loss: {str(e)}")
            return stop_price
    
    def update_config(self, new_config: Dict[str, Any]):
        """Met à jour la configuration du stop-loss manager"""
        try:
            self.default_config.update(new_config)
            logger.info(f"Configuration StopLossManager mise à jour: {new_config}")
        except Exception as e:
            logger.error(f"Erreur mise à jour config: {str(e)}")