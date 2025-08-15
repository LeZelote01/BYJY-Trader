"""
📈 Performance Analyzer - Analyseur de Performance
Analyse détaillée des performances de trading et backtesting
"""

import statistics
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import logging

from core.logger import get_logger

logger = get_logger(__name__)


class PerformanceAnalyzer:
    """
    Analyseur de performance pour les résultats de trading et backtesting
    """
    
    def __init__(self):
        logger.info("PerformanceAnalyzer initialisé")
    
    def analyze_backtest_performance(self, backtest_result) -> Dict[str, Any]:
        """
        Analyse complète des performances d'un backtest
        """
        try:
            analysis = {
                "summary": self._calculate_summary_metrics(backtest_result),
                "trading_metrics": self._calculate_trading_metrics(backtest_result),
                "risk_metrics": self._calculate_risk_metrics(backtest_result),
                "time_analysis": self._calculate_time_analysis(backtest_result),
                "drawdown_analysis": self._calculate_drawdown_analysis(backtest_result),
                "monthly_performance": self._calculate_monthly_performance(backtest_result),
                "trade_distribution": self._calculate_trade_distribution(backtest_result)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Erreur analyse performance: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_summary_metrics(self, result) -> Dict[str, Any]:
        """
        Calcule les métriques de résumé principal
        """
        try:
            duration_days = 0
            if result.config.start_date and result.config.end_date:
                duration = result.config.end_date - result.config.start_date
                duration_days = duration.days
            
            # CAGR (Compound Annual Growth Rate)
            if duration_days > 0 and result.initial_balance > 0:
                cagr = ((result.final_balance / result.initial_balance) ** (365 / duration_days) - 1) * 100
            else:
                cagr = 0.0
            
            return {
                "initial_balance": result.initial_balance,
                "final_balance": result.final_balance,
                "total_return": result.total_return,
                "total_return_percent": result.total_return_percent,
                "cagr": cagr,
                "duration_days": duration_days,
                "total_trades": result.total_trades,
                "win_rate": result.win_rate,
                "profit_factor": result.profit_factor,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown_percent": result.max_drawdown_percent
            }
            
        except Exception as e:
            logger.error(f"Erreur calcul summary metrics: {str(e)}")
            return {}
    
    def _calculate_trading_metrics(self, result) -> Dict[str, Any]:
        """
        Calcule les métriques de trading détaillées
        """
        try:
            if not result.trades:
                return {
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "win_rate": 0,
                    "loss_rate": 0,
                    "average_win": 0,
                    "average_loss": 0,
                    "largest_win": 0,
                    "largest_loss": 0,
                    "profit_factor": 0,
                    "payoff_ratio": 0,
                    "expectancy": 0
                }
            
            trades = result.trades
            
            # Classification des trades
            winning_trades = [t for t in trades if t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl < 0]
            breakeven_trades = [t for t in trades if t.pnl == 0]
            
            # Métriques de base
            total_trades = len(trades)
            win_count = len(winning_trades)
            loss_count = len(losing_trades)
            
            win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0
            loss_rate = (loss_count / total_trades) * 100 if total_trades > 0 else 0
            
            # PnL moyen
            avg_win = statistics.mean([t.pnl for t in winning_trades]) if winning_trades else 0
            avg_loss = statistics.mean([t.pnl for t in losing_trades]) if losing_trades else 0
            
            # Meilleurs/pires trades
            largest_win = max([t.pnl for t in trades]) if trades else 0
            largest_loss = min([t.pnl for t in trades]) if trades else 0
            
            # Profit Factor
            gross_profit = sum([t.pnl for t in winning_trades]) if winning_trades else 0
            gross_loss = abs(sum([t.pnl for t in losing_trades])) if losing_trades else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
            
            # Payoff Ratio (Average Win / Average Loss)
            payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            
            # Expectancy (espérance mathématique)
            expectancy = (win_rate/100 * avg_win) + (loss_rate/100 * avg_loss) if total_trades > 0 else 0
            
            # Métriques additionnelles
            consecutive_wins = self._calculate_consecutive_stats(trades, "win")
            consecutive_losses = self._calculate_consecutive_stats(trades, "loss")
            
            return {
                "total_trades": total_trades,
                "winning_trades": win_count,
                "losing_trades": loss_count,
                "breakeven_trades": len(breakeven_trades),
                "win_rate": win_rate,
                "loss_rate": loss_rate,
                "average_win": avg_win,
                "average_loss": avg_loss,
                "largest_win": largest_win,
                "largest_loss": largest_loss,
                "profit_factor": profit_factor,
                "payoff_ratio": payoff_ratio,
                "expectancy": expectancy,
                "gross_profit": gross_profit,
                "gross_loss": gross_loss,
                "max_consecutive_wins": consecutive_wins["max_consecutive"],
                "max_consecutive_losses": consecutive_losses["max_consecutive"],
                "avg_consecutive_wins": consecutive_wins["avg_consecutive"],
                "avg_consecutive_losses": consecutive_losses["avg_consecutive"]
            }
            
        except Exception as e:
            logger.error(f"Erreur calcul trading metrics: {str(e)}")
            return {}
    
    def _calculate_risk_metrics(self, result) -> Dict[str, Any]:
        """
        Calcule les métriques de risque
        """
        try:
            if not result.equity_curve or len(result.equity_curve) < 2:
                return {
                    "max_drawdown": result.max_drawdown,
                    "max_drawdown_percent": result.max_drawdown_percent,
                    "sharpe_ratio": result.sharpe_ratio,
                    "sortino_ratio": result.sortino_ratio,
                    "calmar_ratio": result.calmar_ratio,
                    "var_95": 0,
                    "var_99": 0,
                    "volatility": 0
                }
            
            # Calcul des returns
            returns = []
            for i in range(1, len(result.equity_curve)):
                prev_value = result.equity_curve[i-1]["portfolio_value"]
                curr_value = result.equity_curve[i]["portfolio_value"]
                
                if prev_value > 0:
                    ret = (curr_value - prev_value) / prev_value
                    returns.append(ret)
            
            if not returns:
                return {"error": "Pas assez de données pour calculer les métriques de risque"}
            
            # Volatilité (écart-type des returns)
            volatility = statistics.stdev(returns) * (252 ** 0.5) if len(returns) > 1 else 0  # Annualisée
            
            # VaR (Value at Risk)
            returns_sorted = sorted(returns)
            var_95 = returns_sorted[int(len(returns_sorted) * 0.05)] if returns_sorted else 0
            var_99 = returns_sorted[int(len(returns_sorted) * 0.01)] if returns_sorted else 0
            
            # Recovery Factor
            recovery_factor = abs(result.total_return / result.max_drawdown) if result.max_drawdown != 0 else 0
            
            # Ulcer Index (mesure alternative du drawdown)
            ulcer_index = self._calculate_ulcer_index(result.equity_curve)
            
            return {
                "max_drawdown": result.max_drawdown,
                "max_drawdown_percent": result.max_drawdown_percent,
                "sharpe_ratio": result.sharpe_ratio,
                "sortino_ratio": result.sortino_ratio,
                "calmar_ratio": result.calmar_ratio,
                "volatility": volatility,
                "var_95": var_95 * 100,  # En pourcentage
                "var_99": var_99 * 100,  # En pourcentage
                "recovery_factor": recovery_factor,
                "ulcer_index": ulcer_index,
                "returns_count": len(returns),
                "negative_returns": len([r for r in returns if r < 0]),
                "positive_returns": len([r for r in returns if r > 0])
            }
            
        except Exception as e:
            logger.error(f"Erreur calcul risk metrics: {str(e)}")
            return {}
    
    def _calculate_time_analysis(self, result) -> Dict[str, Any]:
        """
        Analyse des performances par période de temps
        """
        try:
            if not result.trades:
                return {"error": "Aucun trade pour analyser"}
            
            # Analyse par durée de trade
            durations = [t.duration_hours for t in result.trades]
            
            duration_stats = {
                "avg_duration_hours": statistics.mean(durations),
                "median_duration_hours": statistics.median(durations),
                "min_duration_hours": min(durations),
                "max_duration_hours": max(durations),
                "std_duration_hours": statistics.stdev(durations) if len(durations) > 1 else 0
            }
            
            # Analyse par heure de la journée (si timestamp disponible)
            hourly_analysis = self._analyze_hourly_performance(result.trades)
            
            # Analyse par jour de la semaine
            daily_analysis = self._analyze_daily_performance(result.trades)
            
            # Trades par mois
            monthly_trades = self._analyze_monthly_trades(result.trades)
            
            return {
                "duration_stats": duration_stats,
                "hourly_performance": hourly_analysis,
                "daily_performance": daily_analysis,
                "monthly_trades": monthly_trades
            }
            
        except Exception as e:
            logger.error(f"Erreur calcul time analysis: {str(e)}")
            return {}
    
    def _calculate_drawdown_analysis(self, result) -> Dict[str, Any]:
        """
        Analyse détaillée du drawdown
        """
        try:
            if not result.equity_curve:
                return {"error": "Pas de courbe d'equity pour analyser les drawdowns"}
            
            drawdowns = []
            peak_value = result.initial_balance
            current_drawdown = 0
            drawdown_duration = 0
            in_drawdown = False
            
            for point in result.equity_curve:
                value = point["portfolio_value"]
                
                if value > peak_value:
                    # Nouveau peak
                    if in_drawdown:
                        # Fin du drawdown
                        drawdowns.append({
                            "max_drawdown": current_drawdown,
                            "max_drawdown_percent": (current_drawdown / peak_value) * 100,
                            "duration": drawdown_duration,
                            "recovery_value": value
                        })
                        in_drawdown = False
                        current_drawdown = 0
                        drawdown_duration = 0
                    
                    peak_value = value
                else:
                    # En drawdown
                    current_drawdown = max(current_drawdown, peak_value - value)
                    in_drawdown = True
                    drawdown_duration += 1
            
            # Si on termine en drawdown
            if in_drawdown:
                drawdowns.append({
                    "max_drawdown": current_drawdown,
                    "max_drawdown_percent": (current_drawdown / peak_value) * 100,
                    "duration": drawdown_duration,
                    "recovery_value": None  # Pas encore récupéré
                })
            
            if drawdowns:
                avg_drawdown = statistics.mean([dd["max_drawdown"] for dd in drawdowns])
                avg_duration = statistics.mean([dd["duration"] for dd in drawdowns])
                max_duration = max([dd["duration"] for dd in drawdowns])
            else:
                avg_drawdown = 0
                avg_duration = 0
                max_duration = 0
            
            return {
                "drawdown_periods": len(drawdowns),
                "avg_drawdown": avg_drawdown,
                "avg_drawdown_duration": avg_duration,
                "max_drawdown_duration": max_duration,
                "current_in_drawdown": in_drawdown,
                "detailed_drawdowns": drawdowns[:10]  # Top 10 drawdowns
            }
            
        except Exception as e:
            logger.error(f"Erreur analyse drawdown: {str(e)}")
            return {}
    
    def _calculate_monthly_performance(self, result) -> Dict[str, Any]:
        """
        Calcule les performances mensuelles
        """
        try:
            if not result.trades:
                return {"error": "Aucun trade pour analyser"}
            
            monthly_pnl = {}
            monthly_trades = {}
            
            for trade in result.trades:
                # Extraire le mois (YYYY-MM)
                month_key = trade.entry_time.strftime("%Y-%m")
                
                if month_key not in monthly_pnl:
                    monthly_pnl[month_key] = 0
                    monthly_trades[month_key] = 0
                
                monthly_pnl[month_key] += trade.pnl
                monthly_trades[month_key] += 1
            
            # Calculer les stats par mois
            monthly_stats = []
            for month, pnl in monthly_pnl.items():
                monthly_stats.append({
                    "month": month,
                    "pnl": pnl,
                    "trades": monthly_trades[month],
                    "avg_pnl_per_trade": pnl / monthly_trades[month] if monthly_trades[month] > 0 else 0
                })
            
            # Trier par mois
            monthly_stats.sort(key=lambda x: x["month"])
            
            # Calculer les stats globales
            profitable_months = len([m for m in monthly_stats if m["pnl"] > 0])
            total_months = len(monthly_stats)
            
            return {
                "monthly_stats": monthly_stats,
                "total_months": total_months,
                "profitable_months": profitable_months,
                "profitable_months_rate": (profitable_months / total_months) * 100 if total_months > 0 else 0,
                "best_month": max(monthly_stats, key=lambda x: x["pnl"]) if monthly_stats else None,
                "worst_month": min(monthly_stats, key=lambda x: x["pnl"]) if monthly_stats else None
            }
            
        except Exception as e:
            logger.error(f"Erreur analyse mensuelle: {str(e)}")
            return {}
    
    def _calculate_trade_distribution(self, result) -> Dict[str, Any]:
        """
        Analyse de la distribution des trades
        """
        try:
            if not result.trades:
                return {"error": "Aucun trade pour analyser"}
            
            trades = result.trades
            pnls = [t.pnl for t in trades]
            
            # Distribution par plage de PnL
            ranges = {
                "large_wins": len([p for p in pnls if p > 100]),
                "medium_wins": len([p for p in pnls if 50 < p <= 100]),
                "small_wins": len([p for p in pnls if 0 < p <= 50]),
                "breakeven": len([p for p in pnls if p == 0]),
                "small_losses": len([p for p in pnls if -50 <= p < 0]),
                "medium_losses": len([p for p in pnls if -100 <= p < -50]),
                "large_losses": len([p for p in pnls if p < -100])
            }
            
            # Distribution par side
            long_trades = [t for t in trades if t.side == "long"]
            short_trades = [t for t in trades if t.side == "short"]
            
            side_analysis = {
                "long_trades": {
                    "count": len(long_trades),
                    "total_pnl": sum([t.pnl for t in long_trades]),
                    "avg_pnl": statistics.mean([t.pnl for t in long_trades]) if long_trades else 0,
                    "win_rate": (len([t for t in long_trades if t.pnl > 0]) / len(long_trades)) * 100 if long_trades else 0
                },
                "short_trades": {
                    "count": len(short_trades),
                    "total_pnl": sum([t.pnl for t in short_trades]),
                    "avg_pnl": statistics.mean([t.pnl for t in short_trades]) if short_trades else 0,
                    "win_rate": (len([t for t in short_trades if t.pnl > 0]) / len(short_trades)) * 100 if short_trades else 0
                }
            }
            
            # Distribution par durée
            durations = [t.duration_hours for t in trades]
            duration_ranges = {
                "very_short": len([d for d in durations if d < 1]),     # < 1h
                "short": len([d for d in durations if 1 <= d < 4]),     # 1-4h
                "medium": len([d for d in durations if 4 <= d < 24]),   # 4-24h
                "long": len([d for d in durations if 24 <= d < 168]),   # 1-7 jours
                "very_long": len([d for d in durations if d >= 168])    # > 1 semaine
            }
            
            return {
                "pnl_distribution": ranges,
                "side_analysis": side_analysis,
                "duration_distribution": duration_ranges,
                "trade_size_stats": {
                    "avg_quantity": statistics.mean([abs(t.quantity) for t in trades]),
                    "median_quantity": statistics.median([abs(t.quantity) for t in trades]),
                    "min_quantity": min([abs(t.quantity) for t in trades]),
                    "max_quantity": max([abs(t.quantity) for t in trades])
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur analyse distribution trades: {str(e)}")
            return {}
    
    # Méthodes utilitaires privées
    
    def _calculate_consecutive_stats(self, trades: List, trade_type: str) -> Dict[str, float]:
        """
        Calcule les statistiques de trades consécutifs (wins ou losses)
        """
        try:
            consecutive_counts = []
            current_count = 0
            
            for trade in trades:
                is_target_type = (trade_type == "win" and trade.pnl > 0) or (trade_type == "loss" and trade.pnl < 0)
                
                if is_target_type:
                    current_count += 1
                else:
                    if current_count > 0:
                        consecutive_counts.append(current_count)
                        current_count = 0
            
            # Ajouter le dernier streak s'il existe
            if current_count > 0:
                consecutive_counts.append(current_count)
            
            if consecutive_counts:
                return {
                    "max_consecutive": max(consecutive_counts),
                    "avg_consecutive": statistics.mean(consecutive_counts),
                    "streaks_count": len(consecutive_counts)
                }
            else:
                return {"max_consecutive": 0, "avg_consecutive": 0, "streaks_count": 0}
                
        except Exception as e:
            logger.error(f"Erreur calcul consecutive stats: {str(e)}")
            return {"max_consecutive": 0, "avg_consecutive": 0, "streaks_count": 0}
    
    def _calculate_ulcer_index(self, equity_curve: List[Dict]) -> float:
        """
        Calcule l'Ulcer Index (mesure alternative du drawdown)
        """
        try:
            if len(equity_curve) < 2:
                return 0
            
            peak_value = equity_curve[0]["portfolio_value"]
            squared_drawdowns = []
            
            for point in equity_curve:
                value = point["portfolio_value"]
                
                if value > peak_value:
                    peak_value = value
                
                drawdown_percent = ((peak_value - value) / peak_value) * 100 if peak_value > 0 else 0
                squared_drawdowns.append(drawdown_percent ** 2)
            
            if squared_drawdowns:
                return (sum(squared_drawdowns) / len(squared_drawdowns)) ** 0.5
            else:
                return 0
                
        except Exception as e:
            logger.error(f"Erreur calcul ulcer index: {str(e)}")
            return 0
    
    def _analyze_hourly_performance(self, trades: List) -> Dict[str, Any]:
        """
        Analyse les performances par heure de la journée
        """
        try:
            hourly_pnl = {}
            hourly_counts = {}
            
            for trade in trades:
                hour = trade.entry_time.hour
                
                if hour not in hourly_pnl:
                    hourly_pnl[hour] = 0
                    hourly_counts[hour] = 0
                
                hourly_pnl[hour] += trade.pnl
                hourly_counts[hour] += 1
            
            hourly_stats = []
            for hour in range(24):
                if hour in hourly_pnl:
                    hourly_stats.append({
                        "hour": hour,
                        "pnl": hourly_pnl[hour],
                        "trades": hourly_counts[hour],
                        "avg_pnl": hourly_pnl[hour] / hourly_counts[hour]
                    })
                else:
                    hourly_stats.append({
                        "hour": hour,
                        "pnl": 0,
                        "trades": 0,
                        "avg_pnl": 0
                    })
            
            return {
                "hourly_stats": hourly_stats,
                "best_hour": max([h for h in hourly_stats if h["trades"] > 0], 
                               key=lambda x: x["avg_pnl"], default={"hour": 0, "avg_pnl": 0}),
                "worst_hour": min([h for h in hourly_stats if h["trades"] > 0], 
                                key=lambda x: x["avg_pnl"], default={"hour": 0, "avg_pnl": 0})
            }
            
        except Exception as e:
            logger.error(f"Erreur analyse hourly: {str(e)}")
            return {}
    
    def _analyze_daily_performance(self, trades: List) -> Dict[str, Any]:
        """
        Analyse les performances par jour de la semaine
        """
        try:
            daily_pnl = {}
            daily_counts = {}
            
            days_names = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
            
            for trade in trades:
                day = trade.entry_time.weekday()  # 0=Lundi, 6=Dimanche
                
                if day not in daily_pnl:
                    daily_pnl[day] = 0
                    daily_counts[day] = 0
                
                daily_pnl[day] += trade.pnl
                daily_counts[day] += 1
            
            daily_stats = []
            for day in range(7):
                if day in daily_pnl:
                    daily_stats.append({
                        "day": day,
                        "day_name": days_names[day],
                        "pnl": daily_pnl[day],
                        "trades": daily_counts[day],
                        "avg_pnl": daily_pnl[day] / daily_counts[day]
                    })
                else:
                    daily_stats.append({
                        "day": day,
                        "day_name": days_names[day],
                        "pnl": 0,
                        "trades": 0,
                        "avg_pnl": 0
                    })
            
            return {
                "daily_stats": daily_stats,
                "best_day": max([d for d in daily_stats if d["trades"] > 0], 
                              key=lambda x: x["avg_pnl"], default={"day_name": "N/A", "avg_pnl": 0}),
                "worst_day": min([d for d in daily_stats if d["trades"] > 0], 
                               key=lambda x: x["avg_pnl"], default={"day_name": "N/A", "avg_pnl": 0})
            }
            
        except Exception as e:
            logger.error(f"Erreur analyse daily: {str(e)}")
            return {}
    
    def _analyze_monthly_trades(self, trades: List) -> Dict[str, Any]:
        """
        Analyse le nombre de trades par mois
        """
        try:
            monthly_counts = {}
            
            for trade in trades:
                month_key = trade.entry_time.strftime("%Y-%m")
                monthly_counts[month_key] = monthly_counts.get(month_key, 0) + 1
            
            if monthly_counts:
                avg_trades_per_month = statistics.mean(monthly_counts.values())
                max_trades_month = max(monthly_counts.values())
                min_trades_month = min(monthly_counts.values())
            else:
                avg_trades_per_month = 0
                max_trades_month = 0
                min_trades_month = 0
            
            return {
                "monthly_counts": monthly_counts,
                "avg_trades_per_month": avg_trades_per_month,
                "max_trades_month": max_trades_month,
                "min_trades_month": min_trades_month,
                "active_months": len(monthly_counts)
            }
            
        except Exception as e:
            logger.error(f"Erreur analyse monthly trades: {str(e)}")
            return {}