"""
📊 Portfolio Risk - Gestion des Risques Portefeuille
Analyse et gestion des risques globaux du portefeuille
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import pandas as pd
import numpy as np

from core.logger import get_logger

logger = get_logger(__name__)


class RiskMetricType(Enum):
    """Types de métriques de risque"""
    VAR = "value_at_risk"
    CVAR = "conditional_var" 
    DRAWDOWN = "drawdown"
    SHARPE = "sharpe_ratio"
    SORTINO = "sortino_ratio"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"
    BETA = "beta"


@dataclass
class PortfolioMetrics:
    """Métriques de risque du portefeuille"""
    timestamp: datetime
    total_value: float
    daily_return: float
    cumulative_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    var_95: float  # Value at Risk 95%
    cvar_95: float  # Conditional VaR 95%
    correlation_risk: float
    concentration_risk: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_value": self.total_value,
            "daily_return": self.daily_return,
            "cumulative_return": self.cumulative_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "correlation_risk": self.correlation_risk,
            "concentration_risk": self.concentration_risk
        }


@dataclass
class RiskLimit:
    """Limite de risque"""
    metric_type: RiskMetricType
    threshold: float
    alert_threshold: float  # Seuil d'alerte (avant limite)
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_type": self.metric_type.value,
            "threshold": self.threshold,
            "alert_threshold": self.alert_threshold,
            "enabled": self.enabled
        }


class PortfolioRiskManager:
    """
    Gestionnaire de risques au niveau portefeuille
    Analyse et surveille les risques globaux
    """
    
    def __init__(self, initial_value: float = 10000.0):
        self.initial_value = initial_value
        
        # Historique des métriques
        self.metrics_history: List[PortfolioMetrics] = []
        self.returns_history: List[float] = []
        self.values_history: List[float] = []
        
        # Limites de risque configurables
        self.risk_limits = self._initialize_risk_limits()
        
        # Configuration
        self.config = {
            "var_confidence": 0.95,  # Niveau confiance VaR
            "lookback_period": 252,  # Jours pour calculs (1 an trading)
            "min_history_points": 30,  # Minimum pour calculs fiables
            "correlation_threshold": 0.7,  # Seuil corrélation élevée
            "concentration_threshold": 0.3,  # % max par position
            "stress_test_scenarios": 5,  # Nombre scénarios stress test
            "risk_free_rate": 0.02  # Taux sans risque annuel (2%)
        }
        
        # État des alertes
        self.active_alerts: List[Dict[str, Any]] = []
        self.alert_history: List[Dict[str, Any]] = []
        
        logger.info(f"PortfolioRiskManager initialisé - Valeur initiale: ${initial_value:,.2f}")
    
    def _initialize_risk_limits(self) -> Dict[RiskMetricType, RiskLimit]:
        """Initialise les limites de risque par défaut"""
        return {
            RiskMetricType.VAR: RiskLimit(
                metric_type=RiskMetricType.VAR,
                threshold=0.05,  # 5% VaR maximum
                alert_threshold=0.04
            ),
            RiskMetricType.DRAWDOWN: RiskLimit(
                metric_type=RiskMetricType.DRAWDOWN,
                threshold=0.20,  # 20% drawdown max
                alert_threshold=0.15
            ),
            RiskMetricType.VOLATILITY: RiskLimit(
                metric_type=RiskMetricType.VOLATILITY,
                threshold=0.30,  # 30% volatilité annuelle max
                alert_threshold=0.25
            ),
            RiskMetricType.CORRELATION: RiskLimit(
                metric_type=RiskMetricType.CORRELATION,
                threshold=0.80,  # 80% corrélation max
                alert_threshold=0.70
            )
        }
    
    def update_portfolio_metrics(self, positions: List[Dict[str, Any]], 
                               market_data: Optional[Dict[str, float]] = None) -> PortfolioMetrics:
        """
        Met à jour et calcule les métriques de risque du portefeuille
        """
        try:
            timestamp = datetime.now(timezone.utc)
            
            # Calcul valeur totale portefeuille
            total_value = self._calculate_total_portfolio_value(positions)
            
            # Ajout à l'historique
            self.values_history.append(total_value)
            if len(self.values_history) > self.config["lookback_period"]:
                self.values_history.pop(0)
            
            # Calcul return journalier
            daily_return = 0.0
            if len(self.values_history) > 1:
                daily_return = (total_value - self.values_history[-2]) / self.values_history[-2]
                self.returns_history.append(daily_return)
                
                if len(self.returns_history) > self.config["lookback_period"]:
                    self.returns_history.pop(0)
            
            # Calcul des métriques si suffisant d'historique
            if len(self.returns_history) >= self.config["min_history_points"]:
                volatility = self._calculate_volatility()
                sharpe_ratio = self._calculate_sharpe_ratio(volatility)
                sortino_ratio = self._calculate_sortino_ratio()
                max_drawdown = self._calculate_max_drawdown()
                var_95 = self._calculate_var()
                cvar_95 = self._calculate_cvar()
            else:
                volatility = sharpe_ratio = sortino_ratio = 0.0
                max_drawdown = var_95 = cvar_95 = 0.0
            
            # Métriques de concentration et corrélation
            correlation_risk = self._calculate_correlation_risk(positions)
            concentration_risk = self._calculate_concentration_risk(positions, total_value)
            
            # Création de la métrique complète
            metrics = PortfolioMetrics(
                timestamp=timestamp,
                total_value=total_value,
                daily_return=daily_return,
                cumulative_return=(total_value - self.initial_value) / self.initial_value,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                var_95=var_95,
                cvar_95=cvar_95,
                correlation_risk=correlation_risk,
                concentration_risk=concentration_risk
            )
            
            # Ajout à l'historique des métriques
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > self.config["lookback_period"]:
                self.metrics_history.pop(0)
            
            # Vérification des limites de risque
            self._check_risk_limits(metrics)
            
            logger.debug(f"Métriques portefeuille mises à jour - Valeur: ${total_value:,.2f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Erreur mise à jour métriques portefeuille: {str(e)}")
            raise
    
    def _calculate_total_portfolio_value(self, positions: List[Dict[str, Any]]) -> float:
        """Calcule la valeur totale du portefeuille"""
        try:
            total_value = 0.0
            
            # Somme des valeurs des positions
            for position in positions:
                market_value = position.get("market_value", 0.0)
                unrealized_pnl = position.get("unrealized_pnl", 0.0)
                total_value += market_value + unrealized_pnl
            
            return max(total_value, 0.0)
            
        except Exception as e:
            logger.error(f"Erreur calcul valeur portefeuille: {str(e)}")
            return 0.0
    
    def _calculate_volatility(self) -> float:
        """Calcule la volatilité annualisée"""
        try:
            if len(self.returns_history) < 2:
                return 0.0
            
            daily_vol = np.std(self.returns_history, ddof=1)
            annual_vol = daily_vol * np.sqrt(252)  # Annualisation
            
            return float(annual_vol)
            
        except Exception as e:
            logger.error(f"Erreur calcul volatilité: {str(e)}")
            return 0.0
    
    def _calculate_sharpe_ratio(self, volatility: float) -> float:
        """Calcule le ratio de Sharpe"""
        try:
            if volatility == 0 or len(self.returns_history) < 2:
                return 0.0
            
            mean_return = np.mean(self.returns_history) * 252  # Annualisation
            risk_free_rate = self.config["risk_free_rate"]
            
            sharpe = (mean_return - risk_free_rate) / volatility
            return float(sharpe)
            
        except Exception as e:
            logger.error(f"Erreur calcul Sharpe ratio: {str(e)}")
            return 0.0
    
    def _calculate_sortino_ratio(self) -> float:
        """Calcule le ratio de Sortino (volatilité downside seulement)"""
        try:
            if len(self.returns_history) < 2:
                return 0.0
            
            mean_return = np.mean(self.returns_history) * 252
            risk_free_rate = self.config["risk_free_rate"]
            
            # Volatilité downside seulement (returns négatifs)
            negative_returns = [r for r in self.returns_history if r < 0]
            if not negative_returns:
                return float('inf') if mean_return > risk_free_rate else 0.0
            
            downside_vol = np.std(negative_returns, ddof=1) * np.sqrt(252)
            
            sortino = (mean_return - risk_free_rate) / downside_vol
            return float(sortino)
            
        except Exception as e:
            logger.error(f"Erreur calcul Sortino ratio: {str(e)}")
            return 0.0
    
    def _calculate_max_drawdown(self) -> float:
        """Calcule le drawdown maximum"""
        try:
            if len(self.values_history) < 2:
                return 0.0
            
            values = np.array(self.values_history)
            
            # Calcul des pics et drawdowns
            peak = np.maximum.accumulate(values)
            drawdown = (values - peak) / peak
            
            max_dd = float(np.min(drawdown))
            return abs(max_dd)  # Retourner valeur positive
            
        except Exception as e:
            logger.error(f"Erreur calcul max drawdown: {str(e)}")
            return 0.0
    
    def _calculate_var(self, confidence: float = None) -> float:
        """Calcule la Value at Risk"""
        try:
            if len(self.returns_history) < self.config["min_history_points"]:
                return 0.0
            
            confidence = confidence or self.config["var_confidence"]
            percentile = (1 - confidence) * 100
            
            var = np.percentile(self.returns_history, percentile)
            return abs(float(var))  # Valeur positive
            
        except Exception as e:
            logger.error(f"Erreur calcul VaR: {str(e)}")
            return 0.0
    
    def _calculate_cvar(self, confidence: float = None) -> float:
        """Calcule la Conditional Value at Risk (Expected Shortfall)"""
        try:
            if len(self.returns_history) < self.config["min_history_points"]:
                return 0.0
            
            confidence = confidence or self.config["var_confidence"]
            var_value = -self._calculate_var(confidence)  # VaR négatif pour calcul
            
            # Moyenne des returns inférieurs au VaR
            tail_returns = [r for r in self.returns_history if r <= var_value]
            
            if not tail_returns:
                return self._calculate_var(confidence)
            
            cvar = np.mean(tail_returns)
            return abs(float(cvar))
            
        except Exception as e:
            logger.error(f"Erreur calcul CVaR: {str(e)}")
            return 0.0
    
    def _calculate_correlation_risk(self, positions: List[Dict[str, Any]]) -> float:
        """Calcule le risque de corrélation entre positions"""
        try:
            if len(positions) < 2:
                return 0.0
            
            # Simulation corrélation basée sur types d'actifs
            crypto_positions = []
            stock_positions = []
            
            for position in positions:
                symbol = position.get("symbol", "")
                if symbol.endswith("USDT") or symbol.endswith("BTC"):
                    crypto_positions.append(position)
                else:
                    stock_positions.append(position)
            
            # Calcul risque corrélation
            total_crypto_value = sum(pos.get("market_value", 0) for pos in crypto_positions)
            total_stock_value = sum(pos.get("market_value", 0) for pos in stock_positions)
            total_value = total_crypto_value + total_stock_value
            
            if total_value == 0:
                return 0.0
            
            # Concentration dans une catégorie = risque corrélation
            crypto_concentration = total_crypto_value / total_value
            stock_concentration = total_stock_value / total_value
            
            max_concentration = max(crypto_concentration, stock_concentration)
            
            return float(max_concentration)
            
        except Exception as e:
            logger.error(f"Erreur calcul risque corrélation: {str(e)}")
            return 0.0
    
    def _calculate_concentration_risk(self, positions: List[Dict[str, Any]], total_value: float) -> float:
        """Calcule le risque de concentration"""
        try:
            if not positions or total_value == 0:
                return 0.0
            
            # Calcul de la concentration maximum par position
            max_position_pct = 0.0
            
            for position in positions:
                position_value = abs(position.get("market_value", 0))
                position_pct = position_value / total_value
                max_position_pct = max(max_position_pct, position_pct)
            
            return float(max_position_pct)
            
        except Exception as e:
            logger.error(f"Erreur calcul risque concentration: {str(e)}")
            return 0.0
    
    def _check_risk_limits(self, metrics: PortfolioMetrics):
        """Vérifie les limites de risque et génère des alertes"""
        try:
            current_alerts = []
            
            # Vérification de chaque limite configurée
            limits_to_check = {
                RiskMetricType.VAR: metrics.var_95,
                RiskMetricType.DRAWDOWN: metrics.max_drawdown,
                RiskMetricType.VOLATILITY: metrics.volatility,
                RiskMetricType.CORRELATION: metrics.correlation_risk
            }
            
            for risk_type, current_value in limits_to_check.items():
                if risk_type not in self.risk_limits or not self.risk_limits[risk_type].enabled:
                    continue
                
                limit = self.risk_limits[risk_type]
                
                # Vérification alerte
                if current_value >= limit.alert_threshold:
                    alert = {
                        "timestamp": metrics.timestamp,
                        "risk_type": risk_type.value,
                        "current_value": current_value,
                        "alert_threshold": limit.alert_threshold,
                        "limit_threshold": limit.threshold,
                        "severity": "WARNING" if current_value < limit.threshold else "CRITICAL"
                    }
                    current_alerts.append(alert)
                    
                    # Log approprié selon sévérité
                    if alert["severity"] == "CRITICAL":
                        logger.critical(f"🚨 LIMITE RISQUE DÉPASSÉE: {risk_type.value} = {current_value:.4f} > {limit.threshold}")
                    else:
                        logger.warning(f"⚠️ ALERTE RISQUE: {risk_type.value} = {current_value:.4f} > {limit.alert_threshold}")
            
            # Mise à jour des alertes actives
            self.active_alerts = current_alerts
            
            # Ajout à l'historique si nouvelles alertes
            if current_alerts:
                self.alert_history.extend(current_alerts)
                
                # Limitation taille historique
                if len(self.alert_history) > 1000:
                    self.alert_history = self.alert_history[-500:]
            
        except Exception as e:
            logger.error(f"Erreur vérification limites risque: {str(e)}")
    
    def run_stress_test(self, positions: List[Dict[str, Any]], 
                       scenarios: Optional[List[Dict[str, float]]] = None) -> Dict[str, Any]:
        """
        Exécute des tests de stress sur le portefeuille
        """
        try:
            scenarios = scenarios or self._get_default_stress_scenarios()
            results = {}
            
            current_value = self._calculate_total_portfolio_value(positions)
            
            for i, scenario in enumerate(scenarios):
                scenario_name = f"Scenario_{i+1}"
                scenario_value = self._apply_stress_scenario(positions, scenario)
                
                loss_pct = (current_value - scenario_value) / current_value if current_value > 0 else 0
                
                results[scenario_name] = {
                    "description": scenario.get("description", f"Stress test {i+1}"),
                    "market_shock": scenario,
                    "portfolio_value_before": current_value,
                    "portfolio_value_after": scenario_value,
                    "absolute_loss": current_value - scenario_value,
                    "loss_percentage": loss_pct,
                    "severity": "LOW" if loss_pct < 0.1 else "MEDIUM" if loss_pct < 0.2 else "HIGH"
                }
                
                logger.info(f"Stress test {scenario_name}: {loss_pct*100:.1f}% perte")
            
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "scenarios_tested": len(scenarios),
                "results": results,
                "worst_case_loss": max(result["loss_percentage"] for result in results.values()),
                "average_loss": sum(result["loss_percentage"] for result in results.values()) / len(results)
            }
            
        except Exception as e:
            logger.error(f"Erreur stress test: {str(e)}")
            return {"error": str(e)}
    
    def _get_default_stress_scenarios(self) -> List[Dict[str, Any]]:
        """Génère des scénarios de stress par défaut"""
        return [
            {
                "description": "Market crash -30%",
                "crypto_shock": -0.30,
                "stock_shock": -0.30
            },
            {
                "description": "Crypto crash -50%",
                "crypto_shock": -0.50,
                "stock_shock": -0.10
            },
            {
                "description": "Stock crash -40%", 
                "crypto_shock": -0.20,
                "stock_shock": -0.40
            },
            {
                "description": "Black swan -60%",
                "crypto_shock": -0.60,
                "stock_shock": -0.60
            },
            {
                "description": "Volatility spike +100%",
                "crypto_shock": -0.25,
                "stock_shock": -0.25,
                "volatility_multiplier": 2.0
            }
        ]
    
    def _apply_stress_scenario(self, positions: List[Dict[str, Any]], 
                             scenario: Dict[str, float]) -> float:
        """Applique un scénario de stress aux positions"""
        try:
            stressed_value = 0.0
            
            for position in positions:
                symbol = position.get("symbol", "")
                current_value = position.get("market_value", 0.0)
                
                # Détermination du choc selon type actif
                if symbol.endswith("USDT") or symbol.endswith("BTC"):
                    shock = scenario.get("crypto_shock", 0.0)
                else:
                    shock = scenario.get("stock_shock", 0.0)
                
                # Application du choc
                stressed_position_value = current_value * (1 + shock)
                stressed_value += max(stressed_position_value, 0)  # Pas de valeur négative
            
            return stressed_value
            
        except Exception as e:
            logger.error(f"Erreur application scénario stress: {str(e)}")
            return 0.0
    
    def get_risk_dashboard(self) -> Dict[str, Any]:
        """Retourne un dashboard complet des risques"""
        try:
            latest_metrics = self.metrics_history[-1] if self.metrics_history else None
            
            dashboard = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "portfolio_summary": {
                    "current_value": latest_metrics.total_value if latest_metrics else 0,
                    "initial_value": self.initial_value,
                    "total_return": latest_metrics.cumulative_return if latest_metrics else 0,
                    "daily_return": latest_metrics.daily_return if latest_metrics else 0
                } if latest_metrics else None,
                
                "risk_metrics": latest_metrics.to_dict() if latest_metrics else None,
                
                "risk_limits": {
                    risk_type.value: limit.to_dict() 
                    for risk_type, limit in self.risk_limits.items()
                },
                
                "active_alerts": self.active_alerts,
                "alert_count": len(self.active_alerts),
                
                "historical_summary": {
                    "metrics_count": len(self.metrics_history),
                    "tracking_days": len(self.returns_history),
                    "max_historical_drawdown": max(m.max_drawdown for m in self.metrics_history) if self.metrics_history else 0,
                    "best_daily_return": max(self.returns_history) if self.returns_history else 0,
                    "worst_daily_return": min(self.returns_history) if self.returns_history else 0
                }
            }
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Erreur génération dashboard risque: {str(e)}")
            return {"error": str(e)}
    
    def set_risk_limit(self, risk_type: RiskMetricType, threshold: float, 
                      alert_threshold: float = None):
        """Configure une limite de risque"""
        try:
            alert_threshold = alert_threshold or threshold * 0.8
            
            self.risk_limits[risk_type] = RiskLimit(
                metric_type=risk_type,
                threshold=threshold,
                alert_threshold=alert_threshold,
                enabled=True
            )
            
            logger.info(f"Limite risque configurée: {risk_type.value} = {threshold}")
            
        except Exception as e:
            logger.error(f"Erreur configuration limite risque: {str(e)}")
    
    def get_risk_recommendations(self) -> List[str]:
        """Génère des recommandations basées sur l'analyse des risques"""
        try:
            recommendations = []
            
            if not self.metrics_history:
                return ["Historique insuffisant pour recommandations"]
            
            latest = self.metrics_history[-1]
            
            # Recommandations basées sur les métriques
            if latest.volatility > 0.25:
                recommendations.append("📉 Volatilité élevée: Considérer réduction positions")
            
            if latest.max_drawdown > 0.15:
                recommendations.append("⚠️ Drawdown important: Réviser stops et taille positions")
            
            if latest.correlation_risk > 0.7:
                recommendations.append("🔗 Corrélation élevée: Diversifier davantage")
            
            if latest.concentration_risk > 0.3:
                recommendations.append("📊 Concentration excessive: Réduire position dominante")
            
            if latest.sharpe_ratio < 0.5:
                recommendations.append("📈 Ratio risque/rendement faible: Optimiser stratégies")
            
            if len(self.active_alerts) > 0:
                recommendations.append(f"🚨 {len(self.active_alerts)} alertes actives: Vérifier limites")
            
            if not recommendations:
                recommendations.append("✅ Profil de risque acceptable")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Erreur génération recommandations: {str(e)}")
            return [f"Erreur: {str(e)}"]