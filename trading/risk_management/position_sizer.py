"""
📏 Position Sizer - Calculateur de Taille de Position
Calcul intelligent de la taille des positions selon différentes méthodes
"""

from typing import Dict, Optional, Any
from enum import Enum
import math
import logging

from core.logger import get_logger

logger = get_logger(__name__)


class SizingMethod(Enum):
    """Méthodes de calcul de taille de position"""
    FIXED_AMOUNT = "fixed_amount"
    FIXED_PERCENTAGE = "fixed_percentage" 
    RISK_BASED = "risk_based"
    KELLY_CRITERION = "kelly_criterion"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    ATR_BASED = "atr_based"


class PositionSizer:
    """
    Calculateur avancé de taille de position
    Implémente plusieurs méthodes de dimensionnement
    """
    
    def __init__(self):
        # Configuration par défaut
        self.default_config = {
            "risk_per_trade_pct": 2.0,      # 2% de risque par trade par défaut
            "max_position_pct": 10.0,       # Max 10% du compte par position
            "min_position_value": 10.0,     # Position minimum $10
            "max_position_value": 10000.0,  # Position maximum $10k
            "volatility_window": 20,        # Fenêtre pour calcul volatilité
            "kelly_multiplier": 0.25        # Multiplicateur conservateur pour Kelly
        }
        
        logger.info("PositionSizer initialisé")
    
    def calculate_position_size(
        self,
        account_balance: float,
        risk_per_trade: float,
        entry_price: float,
        stop_loss: Optional[float] = None,
        confidence: float = 1.0,
        method: SizingMethod = SizingMethod.RISK_BASED,
        additional_params: Optional[Dict] = None
    ) -> float:
        """
        Calcule la taille de position selon la méthode choisie
        """
        try:
            additional_params = additional_params or {}
            
            # Validation des paramètres
            if account_balance <= 0:
                logger.warning("Balance invalide pour calcul position size")
                return 0.0
            
            if entry_price <= 0:
                logger.warning("Prix d'entrée invalide pour calcul position size")
                return 0.0
            
            # Calcul selon la méthode
            if method == SizingMethod.FIXED_AMOUNT:
                size = self._calculate_fixed_amount(additional_params.get("amount", 100))
                
            elif method == SizingMethod.FIXED_PERCENTAGE:
                percentage = additional_params.get("percentage", 5.0)
                size = self._calculate_fixed_percentage(account_balance, percentage)
                
            elif method == SizingMethod.RISK_BASED:
                size = self._calculate_risk_based(
                    account_balance, risk_per_trade, entry_price, stop_loss, confidence
                )
                
            elif method == SizingMethod.KELLY_CRITERION:
                win_rate = additional_params.get("win_rate", 0.5)
                avg_win = additional_params.get("avg_win", 100)
                avg_loss = additional_params.get("avg_loss", 50)
                size = self._calculate_kelly_criterion(
                    account_balance, entry_price, win_rate, avg_win, avg_loss
                )
                
            elif method == SizingMethod.VOLATILITY_ADJUSTED:
                volatility = additional_params.get("volatility", 0.02)
                size = self._calculate_volatility_adjusted(
                    account_balance, entry_price, volatility, risk_per_trade
                )
                
            elif method == SizingMethod.ATR_BASED:
                atr = additional_params.get("atr", entry_price * 0.02)
                size = self._calculate_atr_based(
                    account_balance, entry_price, atr, risk_per_trade
                )
                
            else:
                logger.warning(f"Méthode de sizing non reconnue: {method}")
                size = self._calculate_risk_based(
                    account_balance, risk_per_trade, entry_price, stop_loss, confidence
                )
            
            # Application des limites
            size = self._apply_size_limits(size, account_balance, entry_price)
            
            logger.debug(f"Position size calculée: {size:.4f} (méthode: {method.value})")
            return size
            
        except Exception as e:
            logger.error(f"Erreur calcul position size: {str(e)}")
            return 0.0
    
    def _calculate_fixed_amount(self, amount: float) -> float:
        """
        Calcul avec montant fixe (en unités de base)
        """
        return max(0, amount)
    
    def _calculate_fixed_percentage(self, account_balance: float, percentage: float) -> float:
        """
        Calcul avec pourcentage fixe du compte
        """
        return (account_balance * percentage) / 100
    
    def _calculate_risk_based(
        self,
        account_balance: float,
        risk_per_trade: float,
        entry_price: float,
        stop_loss: Optional[float],
        confidence: float
    ) -> float:
        """
        Calcul basé sur le risque (méthode principale)
        """
        try:
            # Capital à risquer
            risk_amount = (account_balance * risk_per_trade) / 100
            
            # Ajustement selon la confiance
            adjusted_risk = risk_amount * confidence
            
            if stop_loss is None:
                # Sans stop-loss, utiliser un pourcentage du prix d'entrée
                risk_per_share = entry_price * 0.02  # 2% par défaut
            else:
                # Avec stop-loss, calculer le risque par action
                risk_per_share = abs(entry_price - stop_loss)
            
            if risk_per_share <= 0:
                logger.warning("Risque par action <= 0, impossible de calculer la taille")
                return 0.0
            
            # Nombre d'actions = Capital à risquer / Risque par action
            position_size = adjusted_risk / risk_per_share
            
            return position_size
            
        except Exception as e:
            logger.error(f"Erreur calcul risk-based: {str(e)}")
            return 0.0
    
    def _calculate_kelly_criterion(
        self,
        account_balance: float,
        entry_price: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calcul selon le critère de Kelly (avec multiplicateur conservateur)
        """
        try:
            if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
                return 0.0
            
            # Formule Kelly: f = (bp - q) / b
            # où b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - win_rate
            
            kelly_fraction = (b * p - q) / b
            
            # Application du multiplicateur conservateur
            conservative_kelly = kelly_fraction * self.default_config["kelly_multiplier"]
            
            # Limitation à des valeurs raisonnables
            conservative_kelly = max(0, min(conservative_kelly, 0.1))  # Max 10%
            
            # Calcul de la position en dollars puis conversion en unités
            position_value = account_balance * conservative_kelly
            position_size = position_value / entry_price
            
            return position_size
            
        except Exception as e:
            logger.error(f"Erreur calcul Kelly: {str(e)}")
            return 0.0
    
    def _calculate_volatility_adjusted(
        self,
        account_balance: float,
        entry_price: float,
        volatility: float,
        base_risk_pct: float
    ) -> float:
        """
        Calcul ajusté selon la volatilité
        """
        try:
            # Plus la volatilité est élevée, plus la position est petite
            # Volatilité de référence: 2% quotidien
            reference_vol = 0.02
            vol_adjustment = reference_vol / max(volatility, 0.001)
            
            # Application de l'ajustement à la taille de base
            adjusted_risk = base_risk_pct * vol_adjustment
            
            # Calcul de la position
            position_value = (account_balance * adjusted_risk) / 100
            position_size = position_value / entry_price
            
            return position_size
            
        except Exception as e:
            logger.error(f"Erreur calcul volatility-adjusted: {str(e)}")
            return 0.0
    
    def _calculate_atr_based(
        self,
        account_balance: float,
        entry_price: float,
        atr: float,
        risk_per_trade: float
    ) -> float:
        """
        Calcul basé sur l'Average True Range (ATR)
        """
        try:
            if atr <= 0:
                return 0.0
            
            # Capital à risquer
            risk_amount = (account_balance * risk_per_trade) / 100
            
            # Utiliser l'ATR comme mesure de risque par action
            # Multiplier par un facteur (ex: 2) pour le stop-loss
            risk_per_share = atr * 2
            
            # Calcul de la taille
            position_size = risk_amount / risk_per_share
            
            return position_size
            
        except Exception as e:
            logger.error(f"Erreur calcul ATR-based: {str(e)}")
            return 0.0
    
    def _apply_size_limits(self, size: float, account_balance: float, entry_price: float) -> float:
        """
        Applique les limites de taille minimum et maximum
        """
        try:
            # Valeur de la position
            position_value = size * entry_price
            
            # Limite minimum
            min_value = self.default_config["min_position_value"]
            if position_value < min_value:
                size = min_value / entry_price
            
            # Limite maximum (en dollars)
            max_value = min(
                self.default_config["max_position_value"],
                (account_balance * self.default_config["max_position_pct"]) / 100
            )
            
            if position_value > max_value:
                size = max_value / entry_price
            
            # Arrondir à un nombre raisonnable de décimales
            size = round(size, 6)
            
            return max(0, size)
            
        except Exception as e:
            logger.error(f"Erreur application limites: {str(e)}")
            return 0.0
    
    def get_position_info(
        self,
        size: float,
        entry_price: float,
        account_balance: float,
        stop_loss: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Retourne les informations détaillées sur une position
        """
        try:
            position_value = size * entry_price
            position_pct = (position_value / account_balance) * 100 if account_balance > 0 else 0
            
            # Calcul du risque si stop-loss fourni
            risk_amount = 0.0
            risk_pct = 0.0
            if stop_loss:
                risk_per_share = abs(entry_price - stop_loss)
                risk_amount = size * risk_per_share
                risk_pct = (risk_amount / account_balance) * 100 if account_balance > 0 else 0
            
            return {
                "position_size": size,
                "position_value": position_value,
                "position_percentage": position_pct,
                "risk_amount": risk_amount,
                "risk_percentage": risk_pct,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "account_balance": account_balance
            }
            
        except Exception as e:
            logger.error(f"Erreur calcul position info: {str(e)}")
            return {}
    
    def optimize_position_size(
        self,
        account_balance: float,
        entry_price: float,
        historical_performance: Dict[str, Any],
        current_market_conditions: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Optimise la taille de position basée sur la performance historique
        """
        try:
            # Extraction des données historiques
            win_rate = historical_performance.get("win_rate", 50) / 100
            avg_win = historical_performance.get("avg_win", 100)
            avg_loss = abs(historical_performance.get("avg_loss", 50))
            volatility = historical_performance.get("volatility", 0.02)
            
            # Calcul avec différentes méthodes
            methods_results = {}
            
            # Risk-based (méthode de base)
            risk_based_size = self.calculate_position_size(
                account_balance, 2.0, entry_price, method=SizingMethod.RISK_BASED
            )
            methods_results["risk_based"] = risk_based_size
            
            # Kelly Criterion
            kelly_size = self.calculate_position_size(
                account_balance, 0, entry_price, method=SizingMethod.KELLY_CRITERION,
                additional_params={"win_rate": win_rate, "avg_win": avg_win, "avg_loss": avg_loss}
            )
            methods_results["kelly"] = kelly_size
            
            # Volatility Adjusted
            vol_adj_size = self.calculate_position_size(
                account_balance, 2.0, entry_price, method=SizingMethod.VOLATILITY_ADJUSTED,
                additional_params={"volatility": volatility}
            )
            methods_results["volatility_adjusted"] = vol_adj_size
            
            # Recommandation finale (moyenne pondérée)
            recommended_size = (
                risk_based_size * 0.4 +
                kelly_size * 0.3 +
                vol_adj_size * 0.3
            )
            
            return {
                "recommended_size": recommended_size,
                "methods_comparison": methods_results,
                "optimization_factors": {
                    "win_rate": win_rate,
                    "avg_win": avg_win,
                    "avg_loss": avg_loss,
                    "volatility": volatility
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur optimisation position: {str(e)}")
            return {"recommended_size": 0.0, "error": str(e)}
    
    def update_config(self, new_config: Dict[str, Any]):
        """Met à jour la configuration du position sizer"""
        try:
            self.default_config.update(new_config)
            logger.info(f"Configuration PositionSizer mise à jour: {new_config}")
        except Exception as e:
            logger.error(f"Erreur mise à jour config: {str(e)}")