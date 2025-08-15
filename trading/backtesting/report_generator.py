"""
📊 Report Generator - Générateur de Rapports
Génération de rapports détaillés pour les résultats de backtesting
"""

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path

from core.logger import get_logger
from .performance_analyzer import PerformanceAnalyzer
from .metrics_calculator import MetricsCalculator

logger = get_logger(__name__)


class ReportGenerator:
    """
    Générateur de rapports complets pour les résultats de backtesting
    """
    
    def __init__(self):
        self.performance_analyzer = PerformanceAnalyzer()
        self.metrics_calculator = MetricsCalculator()
        self.reports_dir = Path("/app/reports/backtesting")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("ReportGenerator initialisé")
    
    def generate_full_report(self, backtest_result, save_to_file: bool = True) -> Dict[str, Any]:
        """
        Génère un rapport complet de backtesting
        """
        try:
            logger.info(f"Génération rapport complet pour backtest {backtest_result.backtest_id}")
            
            # Génération des différentes sections
            report = {
                "report_metadata": self._generate_metadata(backtest_result),
                "executive_summary": self._generate_executive_summary(backtest_result),
                "performance_analysis": self.performance_analyzer.analyze_backtest_performance(backtest_result),
                "detailed_metrics": self.metrics_calculator.calculate_all_metrics(backtest_result),
                "trade_analysis": self._generate_trade_analysis(backtest_result),
                "risk_analysis": self._generate_risk_analysis(backtest_result),
                "recommendations": self._generate_recommendations(backtest_result),
                "appendix": self._generate_appendix(backtest_result)
            }
            
            # Sauvegarde si demandé
            if save_to_file:
                report_path = self._save_report(backtest_result, report)
                report["report_metadata"]["file_path"] = str(report_path)
            
            logger.info(f"Rapport complet généré avec succès pour {backtest_result.backtest_id}")
            return report
            
        except Exception as e:
            logger.error(f"Erreur génération rapport: {str(e)}")
            return {"error": str(e)}
    
    def generate_summary_report(self, backtest_result) -> Dict[str, Any]:
        """
        Génère un rapport résumé (pour l'API/interface)
        """
        try:
            summary = {
                "backtest_id": backtest_result.backtest_id,
                "strategy_id": backtest_result.config.strategy_id,
                "symbol": backtest_result.config.symbol,
                "period": {
                    "start": backtest_result.config.start_date.isoformat(),
                    "end": backtest_result.config.end_date.isoformat(),
                    "duration_days": (backtest_result.config.end_date - backtest_result.config.start_date).days
                },
                "performance": {
                    "initial_balance": backtest_result.initial_balance,
                    "final_balance": backtest_result.final_balance,
                    "total_return": backtest_result.total_return,
                    "total_return_percent": backtest_result.total_return_percent,
                    "max_drawdown_percent": backtest_result.max_drawdown_percent
                },
                "trading": {
                    "total_trades": backtest_result.total_trades,
                    "winning_trades": backtest_result.winning_trades,
                    "losing_trades": backtest_result.losing_trades,
                    "win_rate": backtest_result.win_rate,
                    "profit_factor": backtest_result.profit_factor
                },
                "risk": {
                    "sharpe_ratio": backtest_result.sharpe_ratio,
                    "sortino_ratio": backtest_result.sortino_ratio,
                    "max_drawdown": backtest_result.max_drawdown,
                    "max_drawdown_percent": backtest_result.max_drawdown_percent
                },
                "status": backtest_result.status.value,
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Erreur génération summary: {str(e)}")
            return {"error": str(e)}
    
    def generate_comparison_report(self, backtest_results: List) -> Dict[str, Any]:
        """
        Génère un rapport de comparaison entre plusieurs backtests
        """
        try:
            if not backtest_results or len(backtest_results) < 2:
                return {"error": "Au moins 2 backtests requis pour la comparaison"}
            
            comparison = {
                "comparison_metadata": {
                    "backtest_count": len(backtest_results),
                    "backtest_ids": [bt.backtest_id for bt in backtest_results],
                    "generated_at": datetime.now(timezone.utc).isoformat()
                },
                "performance_comparison": self._compare_performance(backtest_results),
                "risk_comparison": self._compare_risk(backtest_results),
                "trading_comparison": self._compare_trading_metrics(backtest_results),
                "ranking": self._rank_strategies(backtest_results)
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Erreur génération comparison: {str(e)}")
            return {"error": str(e)}
    
    def _generate_metadata(self, result) -> Dict[str, Any]:
        """
        Génère les métadonnées du rapport
        """
        return {
            "report_id": f"RPT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "backtest_id": result.backtest_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generated_by": "BYJY-Trader Backtesting Engine",
            "version": "1.0.0",
            "config": result.config.to_dict()
        }
    
    def _generate_executive_summary(self, result) -> Dict[str, Any]:
        """
        Génère le résumé exécutif
        """
        try:
            # Calculs pour le résumé
            duration_days = (result.config.end_date - result.config.start_date).days
            
            # Classification de la performance
            performance_rating = self._classify_performance(result)
            risk_rating = self._classify_risk(result)
            
            # Points clés
            key_highlights = []
            
            if result.total_return_percent > 10:
                key_highlights.append(f"Performance positive de {result.total_return_percent:.1f}%")
            elif result.total_return_percent < -5:
                key_highlights.append(f"Perte de {abs(result.total_return_percent):.1f}%")
            else:
                key_highlights.append("Performance neutre")
            
            if result.win_rate > 60:
                key_highlights.append(f"Excellent taux de gain ({result.win_rate:.1f}%)")
            elif result.win_rate < 40:
                key_highlights.append(f"Faible taux de gain ({result.win_rate:.1f}%)")
            
            if result.max_drawdown_percent > 20:
                key_highlights.append("Risque de drawdown élevé")
            elif result.max_drawdown_percent < 5:
                key_highlights.append("Risque de drawdown faible")
            
            return {
                "strategy_id": result.config.strategy_id,
                "symbol": result.config.symbol,
                "test_period": f"{duration_days} jours",
                "overall_performance": performance_rating,
                "risk_assessment": risk_rating,
                "key_highlights": key_highlights,
                "final_verdict": self._generate_verdict(result),
                "recommendation": self._generate_recommendation_level(result)
            }
            
        except Exception as e:
            logger.error(f"Erreur génération executive summary: {str(e)}")
            return {"error": str(e)}
    
    def _generate_trade_analysis(self, result) -> Dict[str, Any]:
        """
        Génère l'analyse détaillée des trades
        """
        try:
            if not result.trades:
                return {"message": "Aucun trade à analyser"}
            
            trades = result.trades
            
            # Analyse par période
            hourly_analysis = self._analyze_trades_by_hour(trades)
            daily_analysis = self._analyze_trades_by_day(trades)
            monthly_analysis = self._analyze_trades_by_month(trades)
            
            # Analyse des séquences
            winning_streaks = self._analyze_winning_streaks(trades)
            losing_streaks = self._analyze_losing_streaks(trades)
            
            # Analyse de la taille des trades
            trade_size_analysis = self._analyze_trade_sizes(trades)
            
            return {
                "total_trades": len(trades),
                "trade_frequency": {
                    "hourly_distribution": hourly_analysis,
                    "daily_distribution": daily_analysis,
                    "monthly_distribution": monthly_analysis
                },
                "streak_analysis": {
                    "winning_streaks": winning_streaks,
                    "losing_streaks": losing_streaks
                },
                "trade_size_analysis": trade_size_analysis,
                "best_trades": sorted(trades, key=lambda t: t.pnl, reverse=True)[:5],
                "worst_trades": sorted(trades, key=lambda t: t.pnl)[:5]
            }
            
        except Exception as e:
            logger.error(f"Erreur génération trade analysis: {str(e)}")
            return {"error": str(e)}
    
    def _generate_risk_analysis(self, result) -> Dict[str, Any]:
        """
        Génère l'analyse de risque détaillée
        """
        try:
            risk_metrics = self.metrics_calculator.calculate_risk_metrics(result)
            
            # Classification du niveau de risque
            risk_level = self._assess_risk_level(result)
            
            # Recommandations de risque
            risk_recommendations = []
            
            if result.max_drawdown_percent > 15:
                risk_recommendations.append("Considérer réduire la taille des positions")
            
            if result.sharpe_ratio < 1.0:
                risk_recommendations.append("Performance ajustée au risque faible")
            
            if result.win_rate < 50:
                risk_recommendations.append("Améliorer la sélection des signaux")
            
            return {
                "risk_level": risk_level,
                "risk_score": self._calculate_risk_score(result),
                "key_risks": self._identify_key_risks(result),
                "risk_recommendations": risk_recommendations,
                "detailed_risk_metrics": risk_metrics.get("risk_metrics", {}),
                "var_analysis": self._analyze_var(result)
            }
            
        except Exception as e:
            logger.error(f"Erreur génération risk analysis: {str(e)}")
            return {"error": str(e)}
    
    def _generate_recommendations(self, result) -> Dict[str, Any]:
        """
        Génère les recommandations d'amélioration
        """
        try:
            recommendations = {
                "strategy_adjustments": [],
                "risk_management": [],
                "optimization_opportunities": [],
                "next_steps": []
            }
            
            # Recommandations basées sur les résultats
            if result.total_return_percent < 0:
                recommendations["strategy_adjustments"].append({
                    "type": "Performance",
                    "priority": "High", 
                    "action": "Revoir les signaux d'entrée et de sortie",
                    "details": "La stratégie montre des pertes, analyser les faux signaux"
                })
            
            if result.max_drawdown_percent > 10:
                recommendations["risk_management"].append({
                    "type": "Risk",
                    "priority": "High",
                    "action": "Implémenter un stop-loss plus strict",
                    "details": f"Drawdown max de {result.max_drawdown_percent:.1f}% trop élevé"
                })
            
            if result.win_rate < 50:
                recommendations["optimization_opportunities"].append({
                    "type": "Optimization",
                    "priority": "Medium",
                    "action": "Améliorer la précision des signaux",
                    "details": f"Taux de gain de {result.win_rate:.1f}% insuffisant"
                })
            
            if result.total_trades < 20:
                recommendations["next_steps"].append({
                    "type": "Testing",
                    "priority": "Medium",
                    "action": "Étendre la période de test",
                    "details": "Échantillon de trades trop petit pour conclusions fiables"
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Erreur génération recommendations: {str(e)}")
            return {"error": str(e)}
    
    def _generate_appendix(self, result) -> Dict[str, Any]:
        """
        Génère l'appendice avec données techniques
        """
        return {
            "technical_parameters": {
                "initial_balance": result.initial_balance,
                "commission_rate": result.config.commission_rate,
                "slippage_rate": result.config.slippage_rate,
                "timeframe": result.config.timeframe
            },
            "calculation_methods": {
                "sharpe_ratio": "Return excess / Standard deviation",
                "max_drawdown": "Maximum peak-to-trough decline",
                "profit_factor": "Gross profit / Gross loss"
            },
            "data_quality": {
                "total_data_points": len(result.equity_curve),
                "missing_data_pct": 0,  # TODO: Calculer réellement
                "data_source": "Historical market data"
            }
        }
    
    def _save_report(self, result, report: Dict[str, Any]) -> Path:
        """
        Sauvegarde le rapport dans un fichier
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_report_{result.backtest_id}_{timestamp}.json"
            filepath = self.reports_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Rapport sauvegardé: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde rapport: {str(e)}")
            raise
    
    # Méthodes utilitaires privées
    
    def _classify_performance(self, result) -> str:
        """Classifie la performance"""
        if result.total_return_percent > 20:
            return "Excellente"
        elif result.total_return_percent > 10:
            return "Bonne"
        elif result.total_return_percent > 0:
            return "Positive"
        elif result.total_return_percent > -10:
            return "Négative"
        else:
            return "Mauvaise"
    
    def _classify_risk(self, result) -> str:
        """Classifie le niveau de risque"""
        if result.max_drawdown_percent < 5:
            return "Faible"
        elif result.max_drawdown_percent < 15:
            return "Modéré"
        elif result.max_drawdown_percent < 25:
            return "Élevé"
        else:
            return "Très élevé"
    
    def _generate_verdict(self, result) -> str:
        """Génère un verdict global"""
        performance = result.total_return_percent
        risk = result.max_drawdown_percent
        win_rate = result.win_rate
        
        if performance > 10 and risk < 10 and win_rate > 55:
            return "Stratégie recommandée"
        elif performance > 0 and risk < 20:
            return "Stratégie prometteuse avec optimisations"
        elif performance < -5 or risk > 25:
            return "Stratégie non recommandée"
        else:
            return "Stratégie nécessite des améliorations"
    
    def _generate_recommendation_level(self, result) -> str:
        """Génère le niveau de recommandation"""
        if result.total_return_percent > 15 and result.max_drawdown_percent < 10:
            return "Fortement recommandée"
        elif result.total_return_percent > 5 and result.max_drawdown_percent < 15:
            return "Recommandée avec supervision"
        elif result.total_return_percent > 0:
            return "À considérer après optimisation"
        else:
            return "Non recommandée"
    
    def _analyze_trades_by_hour(self, trades: List) -> Dict:
        """Analyse les trades par heure"""
        hourly_pnl = {}
        for trade in trades:
            hour = trade.entry_time.hour
            hourly_pnl[hour] = hourly_pnl.get(hour, 0) + trade.pnl
        
        return {
            "best_hour": max(hourly_pnl.items(), key=lambda x: x[1]) if hourly_pnl else (0, 0),
            "worst_hour": min(hourly_pnl.items(), key=lambda x: x[1]) if hourly_pnl else (0, 0),
            "distribution": hourly_pnl
        }
    
    def _analyze_trades_by_day(self, trades: List) -> Dict:
        """Analyse les trades par jour de la semaine"""
        daily_pnl = {}
        days = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
        
        for trade in trades:
            day = trade.entry_time.weekday()
            daily_pnl[days[day]] = daily_pnl.get(days[day], 0) + trade.pnl
        
        return daily_pnl
    
    def _analyze_trades_by_month(self, trades: List) -> Dict:
        """Analyse les trades par mois"""
        monthly_pnl = {}
        for trade in trades:
            month = trade.entry_time.strftime("%Y-%m")
            monthly_pnl[month] = monthly_pnl.get(month, 0) + trade.pnl
        
        return monthly_pnl
    
    def _analyze_winning_streaks(self, trades: List) -> Dict:
        """Analyse les séquences de gains"""
        streaks = []
        current_streak = 0
        
        for trade in trades:
            if trade.pnl > 0:
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                    current_streak = 0
        
        if current_streak > 0:
            streaks.append(current_streak)
        
        return {
            "max_streak": max(streaks) if streaks else 0,
            "avg_streak": sum(streaks) / len(streaks) if streaks else 0,
            "streak_count": len(streaks)
        }
    
    def _analyze_losing_streaks(self, trades: List) -> Dict:
        """Analyse les séquences de pertes"""
        streaks = []
        current_streak = 0
        
        for trade in trades:
            if trade.pnl < 0:
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                    current_streak = 0
        
        if current_streak > 0:
            streaks.append(current_streak)
        
        return {
            "max_streak": max(streaks) if streaks else 0,
            "avg_streak": sum(streaks) / len(streaks) if streaks else 0,
            "streak_count": len(streaks)
        }
    
    def _analyze_trade_sizes(self, trades: List) -> Dict:
        """Analyse la taille des trades"""
        sizes = [trade.quantity for trade in trades]
        return {
            "avg_size": sum(sizes) / len(sizes) if sizes else 0,
            "min_size": min(sizes) if sizes else 0,
            "max_size": max(sizes) if sizes else 0
        }
    
    def _assess_risk_level(self, result) -> str:
        """Évalue le niveau de risque global"""
        risk_score = self._calculate_risk_score(result)
        
        if risk_score < 30:
            return "Faible"
        elif risk_score < 60:
            return "Modéré"
        elif risk_score < 80:
            return "Élevé"
        else:
            return "Très élevé"
    
    def _calculate_risk_score(self, result) -> float:
        """Calcule un score de risque (0-100)"""
        score = 0
        
        # Drawdown (40% du score)
        dd_score = min(result.max_drawdown_percent * 2, 40)
        score += dd_score
        
        # Volatility basée sur Sharpe ratio (30% du score)
        if result.sharpe_ratio < 0:
            vol_score = 30
        elif result.sharpe_ratio < 1:
            vol_score = 20
        else:
            vol_score = 10
        score += vol_score
        
        # Win rate (30% du score)
        win_rate_score = max(0, 30 - (result.win_rate * 0.6))
        score += win_rate_score
        
        return min(score, 100)
    
    def _identify_key_risks(self, result) -> List[str]:
        """Identifie les risques clés"""
        risks = []
        
        if result.max_drawdown_percent > 15:
            risks.append("Drawdown maximum élevé")
        
        if result.win_rate < 45:
            risks.append("Taux de gain faible")
        
        if result.sharpe_ratio < 0.5:
            risks.append("Ratio rendement/risque défavorable")
        
        if result.total_trades < 30:
            risks.append("Échantillon de trades insuffisant")
        
        return risks
    
    def _analyze_var(self, result) -> Dict:
        """Analyse Value at Risk"""
        # Simplifié pour l'instant
        return {
            "daily_var_95": result.max_drawdown * 0.1,
            "weekly_var_95": result.max_drawdown * 0.3,
            "monthly_var_95": result.max_drawdown * 0.7
        }
    
    def _compare_performance(self, results: List) -> Dict:
        """Compare les performances"""
        comparison = {}
        for result in results:
            comparison[result.backtest_id] = {
                "total_return_pct": result.total_return_percent,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown_pct": result.max_drawdown_percent,
                "win_rate": result.win_rate
            }
        return comparison
    
    def _compare_risk(self, results: List) -> Dict:
        """Compare les risques"""
        risk_comparison = {}
        for result in results:
            risk_comparison[result.backtest_id] = {
                "max_drawdown": result.max_drawdown_percent,
                "sharpe_ratio": result.sharpe_ratio,
                "risk_score": self._calculate_risk_score(result)
            }
        return risk_comparison
    
    def _compare_trading_metrics(self, results: List) -> Dict:
        """Compare les métriques de trading"""
        trading_comparison = {}
        for result in results:
            trading_comparison[result.backtest_id] = {
                "total_trades": result.total_trades,
                "win_rate": result.win_rate,
                "profit_factor": result.profit_factor,
                "avg_trade_return": result.total_return / result.total_trades if result.total_trades > 0 else 0
            }
        return trading_comparison
    
    def _rank_strategies(self, results: List) -> List[Dict]:
        """Classe les stratégies par performance"""
        ranked = []
        for result in results:
            score = self._calculate_strategy_score(result)
            ranked.append({
                "backtest_id": result.backtest_id,
                "strategy_id": result.config.strategy_id,
                "score": score,
                "total_return_pct": result.total_return_percent,
                "max_drawdown_pct": result.max_drawdown_percent,
                "sharpe_ratio": result.sharpe_ratio
            })
        
        return sorted(ranked, key=lambda x: x["score"], reverse=True)
    
    def _calculate_strategy_score(self, result) -> float:
        """Calcule un score global de stratégie"""
        # Score basé sur return, drawdown et sharpe ratio
        return_score = result.total_return_percent * 2
        drawdown_penalty = result.max_drawdown_percent * -3
        sharpe_bonus = result.sharpe_ratio * 10
        
        return return_score + drawdown_penalty + sharpe_bonus