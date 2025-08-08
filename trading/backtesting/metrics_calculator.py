"""
📊 Metrics Calculator - Calculateur de Métriques
Calculs avancés des métriques de trading et backtesting
"""

import math
import statistics
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
import logging

from core.logger import get_logger

logger = get_logger(__name__)


class MetricsCalculator:
    """
    Calculateur de métriques avancées pour l'analyse de trading
    """
    
    def __init__(self):
        self.risk_free_rate = 0.02  # Taux sans risque (2% par défaut)
        logger.info("MetricsCalculator initialisé")
    
    def calculate_all_metrics(self, backtest_result) -> Dict[str, Any]:
        """
        Calcule toutes les métriques disponibles
        """
        try:
            metrics = {}
            
            # Métriques de base
            metrics.update(self.calculate_basic_metrics(backtest_result))
            
            # Métriques de risque
            metrics.update(self.calculate_risk_metrics(backtest_result))
            
            # Métriques de performance ajustées au risque
            metrics.update(self.calculate_risk_adjusted_metrics(backtest_result))
            
            # Métriques de trading
            metrics.update(self.calculate_trading_metrics(backtest_result))
            
            # Métriques de drawdown
            metrics.update(self.calculate_drawdown_metrics(backtest_result))
            
            # Métriques de distribution
            metrics.update(self.calculate_distribution_metrics(backtest_result))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Erreur calcul métriques complètes: {str(e)}")
            return {"error": str(e)}
    
    def calculate_basic_metrics(self, result) -> Dict[str, Any]:
        """
        Calcule les métriques de base
        """
        try:
            duration_days = 0
            if result.config.start_date and result.config.end_date:
                duration = result.config.end_date - result.config.start_date
                duration_days = max(1, duration.days)
            
            # Return metrics
            total_return = result.total_return
            total_return_pct = result.total_return_percent
            
            # Annualized return
            if duration_days > 0 and result.initial_balance > 0:
                annualized_return = ((result.final_balance / result.initial_balance) ** (365 / duration_days) - 1) * 100
            else:
                annualized_return = 0.0
            
            # Daily return average
            if duration_days > 0:
                daily_return_avg = total_return_pct / duration_days
            else:
                daily_return_avg = 0.0
            
            return {
                "basic_metrics": {
                    "initial_balance": result.initial_balance,
                    "final_balance": result.final_balance,
                    "total_return": total_return,
                    "total_return_percent": total_return_pct,
                    "annualized_return": annualized_return,
                    "daily_return_avg": daily_return_avg,
                    "duration_days": duration_days,
                    "duration_months": duration_days / 30.44 if duration_days > 0 else 0,
                    "duration_years": duration_days / 365.25 if duration_days > 0 else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur calcul basic metrics: {str(e)}")
            return {"basic_metrics": {}}
    
    def calculate_risk_metrics(self, result) -> Dict[str, Any]:
        """
        Calcule les métriques de risque
        """
        try:
            if not result.equity_curve or len(result.equity_curve) < 2:
                return {"risk_metrics": {"error": "Données insuffisantes"}}
            
            # Calcul des returns
            returns = self._calculate_returns(result.equity_curve)
            
            if not returns:
                return {"risk_metrics": {"error": "Impossible de calculer les returns"}}
            
            # Volatility (standard deviation of returns)
            daily_volatility = statistics.stdev(returns) if len(returns) > 1 else 0
            annualized_volatility = daily_volatility * (252 ** 0.5)  # Assuming daily data
            
            # Downside deviation (only negative returns)
            negative_returns = [r for r in returns if r < 0]
            downside_deviation = statistics.stdev(negative_returns) if len(negative_returns) > 1 else 0
            annualized_downside_deviation = downside_deviation * (252 ** 0.5)
            
            # Value at Risk (VaR)
            var_95 = self._calculate_var(returns, 0.95)
            var_99 = self._calculate_var(returns, 0.99)
            
            # Conditional Value at Risk (CVaR)
            cvar_95 = self._calculate_cvar(returns, 0.95)
            cvar_99 = self._calculate_cvar(returns, 0.99)
            
            # Maximum consecutive losses
            max_consecutive_losses = self._calculate_max_consecutive_losses(returns)
            
            # Beta (if benchmark returns were provided, simplified to 1.0)
            beta = 1.0
            
            return {
                "risk_metrics": {
                    "daily_volatility": daily_volatility,
                    "annualized_volatility": annualized_volatility,
                    "downside_deviation": downside_deviation,
                    "annualized_downside_deviation": annualized_downside_deviation,
                    "var_95": var_95,
                    "var_99": var_99,
                    "cvar_95": cvar_95,
                    "cvar_99": cvar_99,
                    "max_consecutive_losses": max_consecutive_losses,
                    "beta": beta,
                    "returns_count": len(returns),
                    "negative_returns_pct": (len(negative_returns) / len(returns)) * 100 if returns else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur calcul risk metrics: {str(e)}")
            return {"risk_metrics": {"error": str(e)}}
    
    def calculate_risk_adjusted_metrics(self, result) -> Dict[str, Any]:
        """
        Calcule les métriques ajustées au risque
        """
        try:
            if not result.equity_curve or len(result.equity_curve) < 2:
                return {"risk_adjusted_metrics": {"error": "Données insuffisantes"}}
            
            returns = self._calculate_returns(result.equity_curve)
            
            if not returns:
                return {"risk_adjusted_metrics": {"error": "Impossible de calculer les returns"}}
            
            mean_return = statistics.mean(returns)
            std_return = statistics.stdev(returns) if len(returns) > 1 else 0
            
            # Sharpe Ratio
            sharpe_ratio = (mean_return - self.risk_free_rate/252) / std_return if std_return > 0 else 0
            annualized_sharpe = sharpe_ratio * (252 ** 0.5)
            
            # Sortino Ratio (using downside deviation)
            negative_returns = [r for r in returns if r < self.risk_free_rate/252]
            downside_std = statistics.stdev(negative_returns) if len(negative_returns) > 1 else 0
            sortino_ratio = (mean_return - self.risk_free_rate/252) / downside_std if downside_std > 0 else 0
            annualized_sortino = sortino_ratio * (252 ** 0.5)
            
            # Calmar Ratio (Annual Return / Max Drawdown)
            annualized_return = mean_return * 252
            max_dd_pct = result.max_drawdown_percent / 100 if result.max_drawdown_percent > 0 else 0.001
            calmar_ratio = annualized_return / max_dd_pct if max_dd_pct > 0 else 0
            
            # Information Ratio (assuming benchmark return = risk-free rate)
            tracking_error = std_return
            information_ratio = (mean_return - self.risk_free_rate/252) / tracking_error if tracking_error > 0 else 0
            
            # Treynor Ratio (assuming beta = 1.0)
            beta = 1.0
            treynor_ratio = (mean_return - self.risk_free_rate/252) / beta if beta > 0 else 0
            
            # Jensen's Alpha (CAPM excess return)
            jensens_alpha = mean_return - (self.risk_free_rate/252 + beta * (mean_return - self.risk_free_rate/252))
            
            # Modigliani-Miller Ratio (M²)
            if std_return > 0:
                market_volatility = 0.16 / (252 ** 0.5)  # Assume market vol = 16% annually
                m2_ratio = (annualized_sharpe * market_volatility + self.risk_free_rate) - self.risk_free_rate
            else:
                m2_ratio = 0
            
            return {
                "risk_adjusted_metrics": {
                    "sharpe_ratio": sharpe_ratio,
                    "annualized_sharpe_ratio": annualized_sharpe,
                    "sortino_ratio": sortino_ratio,
                    "annualized_sortino_ratio": annualized_sortino,
                    "calmar_ratio": calmar_ratio,
                    "information_ratio": information_ratio,
                    "treynor_ratio": treynor_ratio,
                    "jensens_alpha": jensens_alpha,
                    "m2_ratio": m2_ratio,
                    "risk_free_rate_used": self.risk_free_rate
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur calcul risk adjusted metrics: {str(e)}")
            return {"risk_adjusted_metrics": {"error": str(e)}}
    
    def calculate_trading_metrics(self, result) -> Dict[str, Any]:
        """
        Calcule les métriques de trading spécialisées
        """
        try:
            if not result.trades:
                return {"trading_metrics": {"error": "Aucun trade disponible"}}
            
            trades = result.trades
            
            # Basic trading stats
            total_trades = len(trades)
            winning_trades = [t for t in trades if t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl < 0]
            
            win_count = len(winning_trades)
            loss_count = len(losing_trades)
            win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0
            
            # PnL statistics
            total_pnl = sum([t.pnl for t in trades])
            gross_profit = sum([t.pnl for t in winning_trades]) if winning_trades else 0
            gross_loss = abs(sum([t.pnl for t in losing_trades])) if losing_trades else 0
            
            # Average metrics
            avg_win = gross_profit / win_count if win_count > 0 else 0
            avg_loss = gross_loss / loss_count if loss_count > 0 else 0
            avg_trade = total_pnl / total_trades if total_trades > 0 else 0
            
            # Profit Factor
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
            
            # Payoff Ratio
            payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0
            
            # Expectancy
            expectancy = (win_rate/100 * avg_win) - ((100-win_rate)/100 * avg_loss)
            
            # Kelly Criterion
            if avg_loss > 0:
                kelly_f = (win_rate/100 * payoff_ratio - (1 - win_rate/100)) / payoff_ratio
            else:
                kelly_f = 0
            
            # Trade efficiency metrics
            largest_win = max([t.pnl for t in trades]) if trades else 0
            largest_loss = min([t.pnl for t in trades]) if trades else 0
            
            # System Quality Number (SQN)
            trade_pnls = [t.pnl for t in trades]
            if len(trade_pnls) > 1:
                sqn = (statistics.mean(trade_pnls) / statistics.stdev(trade_pnls)) * (len(trade_pnls) ** 0.5)
            else:
                sqn = 0
            
            # Recovery Factor
            recovery_factor = abs(total_pnl / result.max_drawdown) if result.max_drawdown != 0 else 0
            
            # Trade duration analysis
            durations = [t.duration_hours for t in trades]
            avg_duration = statistics.mean(durations) if durations else 0
            
            # Consistency metrics
            monthly_returns = self._calculate_monthly_consistency(trades)
            consistency_score = self._calculate_consistency_score(monthly_returns)
            
            return {
                "trading_metrics": {
                    "total_trades": total_trades,
                    "win_count": win_count,
                    "loss_count": loss_count,
                    "win_rate": win_rate,
                    "gross_profit": gross_profit,
                    "gross_loss": gross_loss,
                    "avg_win": avg_win,
                    "avg_loss": avg_loss,
                    "avg_trade": avg_trade,
                    "profit_factor": profit_factor,
                    "payoff_ratio": payoff_ratio,
                    "expectancy": expectancy,
                    "kelly_criterion": kelly_f,
                    "largest_win": largest_win,
                    "largest_loss": largest_loss,
                    "system_quality_number": sqn,
                    "recovery_factor": recovery_factor,
                    "avg_trade_duration": avg_duration,
                    "consistency_score": consistency_score
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur calcul trading metrics: {str(e)}")
            return {"trading_metrics": {"error": str(e)}}
    
    def calculate_drawdown_metrics(self, result) -> Dict[str, Any]:
        """
        Calcule les métriques de drawdown détaillées
        """
        try:
            if not result.equity_curve:
                return {"drawdown_metrics": {"error": "Pas de courbe d'equity"}}
            
            drawdowns = self._calculate_detailed_drawdowns(result.equity_curve)
            
            if not drawdowns:
                return {"drawdown_metrics": {"max_drawdown": 0, "avg_drawdown": 0}}
            
            # Max drawdown metrics
            max_dd = max([dd["max_drawdown_pct"] for dd in drawdowns])
            avg_dd = statistics.mean([dd["max_drawdown_pct"] for dd in drawdowns])
            
            # Duration metrics
            durations = [dd["duration"] for dd in drawdowns if dd["duration"] > 0]
            max_dd_duration = max(durations) if durations else 0
            avg_dd_duration = statistics.mean(durations) if durations else 0
            
            # Recovery metrics
            recoveries = [dd["recovery_time"] for dd in drawdowns if dd["recovery_time"] is not None]
            avg_recovery_time = statistics.mean(recoveries) if recoveries else 0
            max_recovery_time = max(recoveries) if recoveries else 0
            
            # Ulcer Index (pain index)
            ulcer_index = self._calculate_ulcer_index(result.equity_curve)
            
            # Pain Index (average drawdown)
            pain_index = statistics.mean([dd["max_drawdown_pct"] for dd in drawdowns])
            
            # Lake Ratio (time spent in drawdown)
            total_periods = len(result.equity_curve)
            periods_in_drawdown = sum([dd["duration"] for dd in drawdowns])
            lake_ratio = (periods_in_drawdown / total_periods) if total_periods > 0 else 0
            
            return {
                "drawdown_metrics": {
                    "max_drawdown_pct": max_dd,
                    "avg_drawdown_pct": avg_dd,
                    "drawdown_periods": len(drawdowns),
                    "max_drawdown_duration": max_dd_duration,
                    "avg_drawdown_duration": avg_dd_duration,
                    "avg_recovery_time": avg_recovery_time,
                    "max_recovery_time": max_recovery_time,
                    "ulcer_index": ulcer_index,
                    "pain_index": pain_index,
                    "lake_ratio": lake_ratio,
                    "drawdown_details": drawdowns[:5]  # Top 5 drawdowns
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur calcul drawdown metrics: {str(e)}")
            return {"drawdown_metrics": {"error": str(e)}}
    
    def calculate_distribution_metrics(self, result) -> Dict[str, Any]:
        """
        Calcule les métriques de distribution des returns
        """
        try:
            if not result.equity_curve or len(result.equity_curve) < 2:
                return {"distribution_metrics": {"error": "Données insuffisantes"}}
            
            returns = self._calculate_returns(result.equity_curve)
            
            if not returns or len(returns) < 2:
                return {"distribution_metrics": {"error": "Returns insuffisants"}}
            
            # Basic distribution stats
            mean_return = statistics.mean(returns)
            median_return = statistics.median(returns)
            std_return = statistics.stdev(returns)
            
            # Skewness (simplified calculation)
            skewness = self._calculate_skewness(returns, mean_return, std_return)
            
            # Kurtosis (simplified calculation)
            kurtosis = self._calculate_kurtosis(returns, mean_return, std_return)
            
            # Percentiles
            returns_sorted = sorted(returns)
            percentile_5 = returns_sorted[int(len(returns_sorted) * 0.05)]
            percentile_25 = returns_sorted[int(len(returns_sorted) * 0.25)]
            percentile_75 = returns_sorted[int(len(returns_sorted) * 0.75)]
            percentile_95 = returns_sorted[int(len(returns_sorted) * 0.95)]
            
            # Range metrics
            return_range = max(returns) - min(returns)
            interquartile_range = percentile_75 - percentile_25
            
            # Normality test (simplified)
            is_normal_dist = abs(skewness) < 0.5 and 2 < kurtosis < 4
            
            return {
                "distribution_metrics": {
                    "mean_return": mean_return,
                    "median_return": median_return,
                    "std_return": std_return,
                    "skewness": skewness,
                    "kurtosis": kurtosis,
                    "min_return": min(returns),
                    "max_return": max(returns),
                    "return_range": return_range,
                    "interquartile_range": interquartile_range,
                    "percentile_5": percentile_5,
                    "percentile_25": percentile_25,
                    "percentile_75": percentile_75,
                    "percentile_95": percentile_95,
                    "is_normal_distribution": is_normal_dist,
                    "positive_returns_pct": (len([r for r in returns if r > 0]) / len(returns)) * 100
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur calcul distribution metrics: {str(e)}")
            return {"distribution_metrics": {"error": str(e)}}
    
    # Méthodes utilitaires privées
    
    def _calculate_returns(self, equity_curve: List[Dict]) -> List[float]:
        """
        Calcule les returns à partir de la courbe d'equity
        """
        try:
            returns = []
            
            for i in range(1, len(equity_curve)):
                prev_value = equity_curve[i-1]["portfolio_value"]
                curr_value = equity_curve[i]["portfolio_value"]
                
                if prev_value > 0:
                    ret = (curr_value - prev_value) / prev_value
                    returns.append(ret)
            
            return returns
            
        except Exception as e:
            logger.error(f"Erreur calcul returns: {str(e)}")
            return []
    
    def _calculate_var(self, returns: List[float], confidence: float) -> float:
        """
        Calcule la Value at Risk
        """
        try:
            if not returns:
                return 0
            
            returns_sorted = sorted(returns)
            index = int(len(returns_sorted) * (1 - confidence))
            return returns_sorted[index] if index < len(returns_sorted) else returns_sorted[-1]
            
        except Exception as e:
            logger.error(f"Erreur calcul VaR: {str(e)}")
            return 0
    
    def _calculate_cvar(self, returns: List[float], confidence: float) -> float:
        """
        Calcule la Conditional Value at Risk (Expected Shortfall)
        """
        try:
            if not returns:
                return 0
            
            var = self._calculate_var(returns, confidence)
            tail_returns = [r for r in returns if r <= var]
            
            return statistics.mean(tail_returns) if tail_returns else 0
            
        except Exception as e:
            logger.error(f"Erreur calcul CVaR: {str(e)}")
            return 0
    
    def _calculate_max_consecutive_losses(self, returns: List[float]) -> int:
        """
        Calcule le nombre maximum de pertes consécutives
        """
        try:
            max_consecutive = 0
            current_consecutive = 0
            
            for ret in returns:
                if ret < 0:
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 0
            
            return max_consecutive
            
        except Exception as e:
            logger.error(f"Erreur calcul consecutive losses: {str(e)}")
            return 0
    
    def _calculate_detailed_drawdowns(self, equity_curve: List[Dict]) -> List[Dict]:
        """
        Calcule les drawdowns détaillés
        """
        try:
            drawdowns = []
            peak_value = equity_curve[0]["portfolio_value"]
            trough_value = peak_value
            peak_index = 0
            trough_index = 0
            in_drawdown = False
            
            for i, point in enumerate(equity_curve):
                value = point["portfolio_value"]
                
                if value > peak_value:
                    # Nouveau peak - fin du drawdown s'il y en avait un
                    if in_drawdown and trough_value < peak_value:
                        recovery_time = i - trough_index
                        drawdowns.append({
                            "peak_value": peak_value,
                            "trough_value": trough_value,
                            "recovery_value": value,
                            "max_drawdown": peak_value - trough_value,
                            "max_drawdown_pct": ((peak_value - trough_value) / peak_value) * 100,
                            "duration": trough_index - peak_index,
                            "recovery_time": recovery_time,
                            "peak_index": peak_index,
                            "trough_index": trough_index,
                            "recovery_index": i
                        })
                    
                    peak_value = value
                    peak_index = i
                    trough_value = value
                    trough_index = i
                    in_drawdown = False
                    
                elif value < trough_value:
                    trough_value = value
                    trough_index = i
                    in_drawdown = True
            
            # Si on termine en drawdown
            if in_drawdown and trough_value < peak_value:
                drawdowns.append({
                    "peak_value": peak_value,
                    "trough_value": trough_value,
                    "recovery_value": None,
                    "max_drawdown": peak_value - trough_value,
                    "max_drawdown_pct": ((peak_value - trough_value) / peak_value) * 100,
                    "duration": trough_index - peak_index,
                    "recovery_time": None,
                    "peak_index": peak_index,
                    "trough_index": trough_index,
                    "recovery_index": None
                })
            
            return drawdowns
            
        except Exception as e:
            logger.error(f"Erreur calcul detailed drawdowns: {str(e)}")
            return []
    
    def _calculate_ulcer_index(self, equity_curve: List[Dict]) -> float:
        """
        Calcule l'Ulcer Index
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
                
                if peak_value > 0:
                    drawdown_pct = ((peak_value - value) / peak_value) * 100
                    squared_drawdowns.append(drawdown_pct ** 2)
            
            if squared_drawdowns:
                return (sum(squared_drawdowns) / len(squared_drawdowns)) ** 0.5
            else:
                return 0
                
        except Exception as e:
            logger.error(f"Erreur calcul ulcer index: {str(e)}")
            return 0
    
    def _calculate_skewness(self, returns: List[float], mean: float, std: float) -> float:
        """
        Calcule la skewness (asymétrie)
        """
        try:
            if std == 0 or len(returns) < 3:
                return 0
            
            n = len(returns)
            skew_sum = sum(((r - mean) / std) ** 3 for r in returns)
            skewness = (n / ((n - 1) * (n - 2))) * skew_sum
            
            return skewness
            
        except Exception as e:
            logger.error(f"Erreur calcul skewness: {str(e)}")
            return 0
    
    def _calculate_kurtosis(self, returns: List[float], mean: float, std: float) -> float:
        """
        Calcule la kurtosis (aplatissement)
        """
        try:
            if std == 0 or len(returns) < 4:
                return 3  # Kurtosis normale
            
            n = len(returns)
            kurt_sum = sum(((r - mean) / std) ** 4 for r in returns)
            kurtosis = ((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))) * kurt_sum - (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
            
            return kurtosis + 3  # Excess kurtosis + 3
            
        except Exception as e:
            logger.error(f"Erreur calcul kurtosis: {str(e)}")
            return 3
    
    def _calculate_monthly_consistency(self, trades: List) -> List[float]:
        """
        Calcule les returns mensuels pour l'analyse de consistance
        """
        try:
            monthly_pnl = {}
            
            for trade in trades:
                month_key = trade.entry_time.strftime("%Y-%m")
                monthly_pnl[month_key] = monthly_pnl.get(month_key, 0) + trade.pnl
            
            return list(monthly_pnl.values())
            
        except Exception as e:
            logger.error(f"Erreur calcul monthly consistency: {str(e)}")
            return []
    
    def _calculate_consistency_score(self, monthly_returns: List[float]) -> float:
        """
        Calcule un score de consistance basé sur les returns mensuels
        """
        try:
            if not monthly_returns or len(monthly_returns) < 2:
                return 0
            
            positive_months = len([r for r in monthly_returns if r > 0])
            total_months = len(monthly_returns)
            
            # Pourcentage de mois positifs
            positive_ratio = positive_months / total_months
            
            # Coefficient de variation (std/mean)
            if statistics.mean(monthly_returns) != 0:
                cv = statistics.stdev(monthly_returns) / abs(statistics.mean(monthly_returns))
                consistency = positive_ratio * (1 / (1 + cv))
            else:
                consistency = positive_ratio
            
            return min(consistency, 1.0)  # Cap à 1.0
            
        except Exception as e:
            logger.error(f"Erreur calcul consistency score: {str(e)}")
            return 0
    
    def set_risk_free_rate(self, rate: float):
        """
        Définit le taux sans risque pour les calculs
        """
        self.risk_free_rate = max(0, rate)
        logger.info(f"Taux sans risque mis à jour: {self.risk_free_rate:.2%}")