"""
🛡️ Risk Manager - Gestionnaire de Risques Principal
Système central de gestion des risques pour le trading
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from core.logger import get_logger
from .position_sizer import PositionSizer
from .stop_loss_manager import StopLossManager

logger = get_logger(__name__)


class RiskLevel(Enum):
    """Niveaux de risque"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskAssessment:
    """Évaluation des risques"""
    overall_risk: RiskLevel
    risk_score: float  # 0-100
    portfolio_exposure: float
    max_drawdown_pct: float
    active_positions: int
    recommendations: List[str]
    warnings: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_risk": self.overall_risk.value,
            "risk_score": self.risk_score,
            "portfolio_exposure": self.portfolio_exposure,
            "max_drawdown_pct": self.max_drawdown_pct,
            "active_positions": self.active_positions,
            "recommendations": self.recommendations,
            "warnings": self.warnings,
            "assessed_at": datetime.now(timezone.utc).isoformat()
        }


class RiskManager:
    """
    Gestionnaire central des risques
    Supervise et contrôle tous les aspects de risque du trading
    """
    
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        
        # Composants spécialisés
        self.position_sizer = PositionSizer()
        self.stop_loss_manager = StopLossManager()
        
        # Configuration des limites de risque
        self.risk_config = {
            # Limites globales
            "max_portfolio_risk_pct": 10.0,  # Max 10% du portfolio en risque
            "max_position_risk_pct": 2.0,    # Max 2% par position
            "max_correlation_exposure": 50.0, # Max 50% dans des actifs corrélés
            "max_drawdown_limit": 20.0,      # Stop trading si DD > 20%
            
            # Limites par stratégie
            "max_strategy_allocation": 25.0,  # Max 25% par stratégie
            "max_concurrent_positions": 5,   # Max 5 positions simultanées
            
            # Limites temporelles
            "max_daily_loss_pct": 5.0,      # Max 5% de perte par jour
            "max_weekly_loss_pct": 10.0,    # Max 10% de perte par semaine
            
            # Limites de volatilité
            "max_position_volatility": 30.0, # Max 30% volatilité annualisée
            "vol_lookback_days": 20          # Période de calcul volatilité
        }
        
        # État du système
        self.risk_state = {
            "total_exposure": 0.0,
            "daily_pnl": 0.0,
            "weekly_pnl": 0.0,
            "current_drawdown": 0.0,
            "max_drawdown": 0.0,
            "last_risk_check": None,
            "emergency_stop": False
        }
        
        # Historique des évaluations
        self.risk_history: List[RiskAssessment] = []
        
        logger.info(f"RiskManager initialisé - Balance initiale: ${initial_balance:,.2f}")
    
    def assess_portfolio_risk(self, positions: List[Dict], market_data: Optional[Dict] = None) -> RiskAssessment:
        """
        Évaluation complète des risques du portfolio
        """
        try:
            # Calcul des métriques de base
            total_exposure = sum(abs(pos.get("market_value", 0)) for pos in positions)
            exposure_pct = (total_exposure / self.current_balance) * 100 if self.current_balance > 0 else 0
            
            # Calcul du drawdown actuel
            current_portfolio_value = self.current_balance + sum(pos.get("unrealized_pnl", 0) for pos in positions)
            peak_value = max(self.initial_balance, current_portfolio_value)
            current_drawdown = ((peak_value - current_portfolio_value) / peak_value) * 100 if peak_value > 0 else 0
            
            # Mise à jour de l'état
            self.risk_state["total_exposure"] = total_exposure
            self.risk_state["current_drawdown"] = current_drawdown
            self.risk_state["max_drawdown"] = max(self.risk_state["max_drawdown"], current_drawdown)
            
            # Calcul du score de risque
            risk_score = self._calculate_risk_score(positions, exposure_pct, current_drawdown)
            
            # Détermination du niveau de risque
            risk_level = self._determine_risk_level(risk_score)
            
            # Génération des recommandations et avertissements
            recommendations = self._generate_risk_recommendations(positions, exposure_pct, current_drawdown)
            warnings = self._generate_risk_warnings(positions, exposure_pct, current_drawdown)
            
            # Création de l'évaluation
            assessment = RiskAssessment(
                overall_risk=risk_level,
                risk_score=risk_score,
                portfolio_exposure=exposure_pct,
                max_drawdown_pct=current_drawdown,
                active_positions=len(positions),
                recommendations=recommendations,
                warnings=warnings
            )
            
            # Sauvegarde dans l'historique
            self.risk_history.append(assessment)
            self.risk_state["last_risk_check"] = datetime.now(timezone.utc)
            
            logger.info(f"Évaluation risque: {risk_level.value} (score: {risk_score:.1f})")
            
            return assessment
            
        except Exception as e:
            logger.error(f"Erreur évaluation risque portfolio: {str(e)}")
            return RiskAssessment(
                overall_risk=RiskLevel.CRITICAL,
                risk_score=100.0,
                portfolio_exposure=0.0,
                max_drawdown_pct=0.0,
                active_positions=0,
                recommendations=["Erreur d'évaluation des risques"],
                warnings=["Système de risque en panne"]
            )
    
    def validate_new_position(
        self, 
        symbol: str, 
        side: str, 
        quantity: float, 
        entry_price: float,
        stop_loss: Optional[float] = None,
        current_positions: Optional[List[Dict]] = None
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Valide si une nouvelle position respecte les limites de risque
        """
        try:
            current_positions = current_positions or []
            
            # 1. Vérification de l'arrêt d'urgence
            if self.risk_state["emergency_stop"]:
                return False, "Trading suspendu - Arrêt d'urgence activé", {}
            
            # 2. Vérification du nombre de positions
            if len(current_positions) >= self.risk_config["max_concurrent_positions"]:
                return False, f"Limite de positions simultanées atteinte ({self.risk_config['max_concurrent_positions']})", {}
            
            # 3. Calcul de la taille de position recommandée
            recommended_size = self.position_sizer.calculate_position_size(
                account_balance=self.current_balance,
                risk_per_trade=self.risk_config["max_position_risk_pct"],
                entry_price=entry_price,
                stop_loss=stop_loss,
                confidence=1.0
            )
            
            # 4. Vérification de la taille demandée vs recommandée
            position_value = quantity * entry_price
            max_position_value = (self.current_balance * self.risk_config["max_position_risk_pct"]) / 100
            
            if position_value > max_position_value:
                return False, f"Taille de position trop importante (${position_value:,.2f} > ${max_position_value:,.2f})", {
                    "recommended_size": recommended_size,
                    "max_position_value": max_position_value
                }
            
            # 5. Vérification de l'exposition totale
            current_exposure = sum(abs(pos.get("market_value", 0)) for pos in current_positions)
            new_exposure = current_exposure + position_value
            max_exposure = (self.current_balance * self.risk_config["max_portfolio_risk_pct"]) / 100
            
            if new_exposure > max_exposure:
                return False, f"Exposition portfolio trop élevée (${new_exposure:,.2f} > ${max_exposure:,.2f})", {
                    "current_exposure": current_exposure,
                    "max_exposure": max_exposure
                }
            
            # 6. Vérification de la corrélation (simplifiée)
            symbol_exposure = sum(pos.get("market_value", 0) for pos in current_positions if pos.get("symbol", "").startswith(symbol[:3]))
            total_symbol_exposure = symbol_exposure + position_value
            max_corr_exposure = (self.current_balance * self.risk_config["max_correlation_exposure"]) / 100
            
            if total_symbol_exposure > max_corr_exposure:
                return False, f"Exposition corrélée trop élevée pour {symbol[:3]}", {
                    "current_correlation_exposure": symbol_exposure,
                    "max_correlation_exposure": max_corr_exposure
                }
            
            # 7. Vérification du drawdown
            if self.risk_state["current_drawdown"] > self.risk_config["max_drawdown_limit"]:
                return False, f"Drawdown limite atteint ({self.risk_state['current_drawdown']:.1f}%)", {}
            
            # Position validée
            validation_data = {
                "recommended_size": recommended_size,
                "position_risk_pct": (position_value / self.current_balance) * 100,
                "new_total_exposure": new_exposure,
                "exposure_pct": (new_exposure / self.current_balance) * 100
            }
            
            logger.info(f"Position validée: {symbol} {side} ${position_value:,.2f}")
            return True, "Position autorisée", validation_data
            
        except Exception as e:
            logger.error(f"Erreur validation position: {str(e)}")
            return False, f"Erreur de validation: {str(e)}", {}
    
    def update_stop_losses(self, positions: List[Dict], current_prices: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Met à jour les stop-loss selon la gestion des risques
        """
        try:
            updates = []
            
            for position in positions:
                symbol = position.get("symbol")
                if not symbol or symbol not in current_prices:
                    continue
                
                current_price = current_prices[symbol]
                
                # Mise à jour via le StopLossManager
                new_stop_loss = self.stop_loss_manager.update_trailing_stop(
                    position=position,
                    current_price=current_price
                )
                
                if new_stop_loss != position.get("stop_loss"):
                    updates.append({
                        "position_id": position.get("id"),
                        "symbol": symbol,
                        "old_stop_loss": position.get("stop_loss"),
                        "new_stop_loss": new_stop_loss,
                        "reason": "Trailing stop update"
                    })
            
            logger.info(f"{len(updates)} stop-loss mis à jour")
            return updates
            
        except Exception as e:
            logger.error(f"Erreur mise à jour stop-loss: {str(e)}")
            return []
    
    def check_emergency_conditions(self, positions: List[Dict]) -> bool:
        """
        Vérifie les conditions d'arrêt d'urgence
        """
        try:
            # 1. Drawdown critique
            if self.risk_state["current_drawdown"] > self.risk_config["max_drawdown_limit"]:
                self._activate_emergency_stop("Drawdown critique atteint")
                return True
            
            # 2. Perte journalière excessive
            daily_loss_pct = abs(self.risk_state["daily_pnl"] / self.current_balance) * 100
            if self.risk_state["daily_pnl"] < 0 and daily_loss_pct > self.risk_config["max_daily_loss_pct"]:
                self._activate_emergency_stop("Perte journalière excessive")
                return True
            
            # 3. Exposition excessive
            total_exposure = sum(abs(pos.get("market_value", 0)) for pos in positions)
            exposure_pct = (total_exposure / self.current_balance) * 100
            if exposure_pct > self.risk_config["max_portfolio_risk_pct"] * 1.5:  # 150% de la limite
                self._activate_emergency_stop("Exposition portfolio critique")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Erreur vérification urgence: {str(e)}")
            return True  # En cas d'erreur, activer la sécurité
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """
        Retourne les métriques de risque actuelles
        """
        return {
            "risk_state": self.risk_state.copy(),
            "risk_config": self.risk_config.copy(),
            "current_balance": self.current_balance,
            "initial_balance": self.initial_balance,
            "risk_assessments_count": len(self.risk_history),
            "last_assessment": self.risk_history[-1].to_dict() if self.risk_history else None
        }
    
    def update_balance(self, new_balance: float):
        """Met à jour le balance courant"""
        self.current_balance = new_balance
        logger.debug(f"Balance mis à jour: ${new_balance:,.2f}")
    
    # Méthodes privées
    
    def _calculate_risk_score(self, positions: List[Dict], exposure_pct: float, drawdown_pct: float) -> float:
        """
        Calcule un score de risque global (0-100)
        """
        score = 0.0
        
        # Exposition (30% du score)
        exposure_score = min((exposure_pct / self.risk_config["max_portfolio_risk_pct"]) * 30, 30)
        score += exposure_score
        
        # Drawdown (40% du score)
        drawdown_score = min((drawdown_pct / self.risk_config["max_drawdown_limit"]) * 40, 40)
        score += drawdown_score
        
        # Nombre de positions (15% du score)
        position_score = min((len(positions) / self.risk_config["max_concurrent_positions"]) * 15, 15)
        score += position_score
        
        # Volatilité et autres facteurs (15% du score)
        volatility_score = 15  # Simplifié pour l'instant
        score += volatility_score
        
        return min(score, 100.0)
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Détermine le niveau de risque basé sur le score"""
        if risk_score < 25:
            return RiskLevel.LOW
        elif risk_score < 50:
            return RiskLevel.MODERATE  
        elif risk_score < 75:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def _generate_risk_recommendations(self, positions: List[Dict], exposure_pct: float, drawdown_pct: float) -> List[str]:
        """Génère des recommandations de gestion des risques"""
        recommendations = []
        
        if exposure_pct > 80:  # 80% de la limite
            recommendations.append("Considérer réduire l'exposition du portfolio")
        
        if drawdown_pct > 10:
            recommendations.append("Surveiller de près le drawdown")
            
        if len(positions) >= 4:  # 80% de la limite
            recommendations.append("Éviter d'ouvrir trop de positions simultanément")
        
        if not recommendations:
            recommendations.append("Gestion des risques appropriée")
        
        return recommendations
    
    def _generate_risk_warnings(self, positions: List[Dict], exposure_pct: float, drawdown_pct: float) -> List[str]:
        """Génère des avertissements de risque"""
        warnings = []
        
        if exposure_pct > self.risk_config["max_portfolio_risk_pct"]:
            warnings.append("🚨 EXPOSITION PORTFOLIO EXCESSIVE")
        
        if drawdown_pct > self.risk_config["max_drawdown_limit"] * 0.8:
            warnings.append("⚠️ Drawdown approchant la limite critique")
        
        if len(positions) >= self.risk_config["max_concurrent_positions"]:
            warnings.append("⚠️ Nombre maximum de positions atteint")
        
        return warnings
    
    def _activate_emergency_stop(self, reason: str):
        """Active l'arrêt d'urgence"""
        self.risk_state["emergency_stop"] = True
        logger.critical(f"🚨 ARRÊT D'URGENCE ACTIVÉ: {reason}")
    
    def deactivate_emergency_stop(self):
        """Désactive l'arrêt d'urgence (manuel)"""
        self.risk_state["emergency_stop"] = False
        logger.warning("Arrêt d'urgence désactivé manuellement")