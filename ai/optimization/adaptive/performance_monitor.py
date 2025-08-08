"""
📈 Performance Monitor Module
Continuous monitoring of strategy and system performance
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Continuous performance monitoring system
    Phase 3.4 - Adaptive Strategies Component
    """
    
    def __init__(self, 
                 monitoring_window: int = 100,
                 alert_threshold: float = -0.05,
                 benchmark_name: str = "BTC"):
        """
        Initialize Performance Monitor
        
        Args:
            monitoring_window: Number of periods to monitor
            alert_threshold: Threshold for performance alerts
            benchmark_name: Name of benchmark asset
        """
        self.monitoring_window = monitoring_window
        self.alert_threshold = alert_threshold
        self.benchmark_name = benchmark_name
        self.performance_history = []
        self.alerts = []
        self.benchmarks = {}
        
        logger.info("PerformanceMonitor initialized")
    
    def update_performance(self,
                          strategy_name: str,
                          performance_metrics: Dict[str, float],
                          market_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Update performance metrics for monitoring
        
        Args:
            strategy_name: Name of the strategy
            performance_metrics: Performance metrics to record
            market_data: Optional market data for context
            
        Returns:
            Updated monitoring status
        """
        monitoring_status = {
            'strategy': strategy_name,
            'timestamp': datetime.now(),
            'alerts_generated': [],
            'performance_summary': {},
            'trend_analysis': {}
        }
        
        try:
            # Record performance data
            performance_entry = {
                'timestamp': datetime.now(),
                'strategy': strategy_name,
                'metrics': performance_metrics,
                'market_data': market_data or {}
            }
            
            self.performance_history.append(performance_entry)
            
            # Keep only recent history
            if len(self.performance_history) > self.monitoring_window * 2:
                self.performance_history = self.performance_history[-self.monitoring_window:]
            
            # Check for performance alerts
            alerts = self._check_performance_alerts(strategy_name, performance_metrics)
            monitoring_status['alerts_generated'] = alerts
            
            # Generate performance summary
            summary = self._generate_performance_summary(strategy_name)
            monitoring_status['performance_summary'] = summary
            
            # Analyze trends
            trend_analysis = self._analyze_performance_trends(strategy_name)
            monitoring_status['trend_analysis'] = trend_analysis
            
            logger.info(f"Performance updated for {strategy_name}: {len(alerts)} alerts generated")
            
        except Exception as e:
            logger.error(f"Performance update failed: {e}")
            monitoring_status['error'] = str(e)
        
        return monitoring_status
    
    def compare_with_benchmark(self,
                             strategy_performance: List[float],
                             benchmark_data: List[float] = None) -> Dict[str, Any]:
        """
        Compare strategy performance with benchmark
        
        Args:
            strategy_performance: Strategy returns
            benchmark_data: Benchmark returns (optional)
            
        Returns:
            Comparison results
        """
        comparison_results = {
            'alpha': 0.0,
            'beta': 1.0,
            'correlation': 0.0,
            'excess_return': 0.0,
            'information_ratio': 0.0,
            'outperformance_periods': 0
        }
        
        if not strategy_performance:
            return comparison_results
        
        try:
            # Use default benchmark if not provided
            if benchmark_data is None:
                benchmark_data = self._get_default_benchmark(len(strategy_performance))
            
            if len(benchmark_data) != len(strategy_performance):
                # Align lengths
                min_length = min(len(strategy_performance), len(benchmark_data))
                strategy_performance = strategy_performance[-min_length:]
                benchmark_data = benchmark_data[-min_length:]
            
            if len(strategy_performance) < 5:
                return comparison_results
            
            # Calculate metrics
            strategy_returns = np.array(strategy_performance)
            benchmark_returns = np.array(benchmark_data)
            
            # Alpha and Beta calculation
            correlation = np.corrcoef(strategy_returns, benchmark_returns)[0, 1]
            
            if not np.isnan(correlation):
                beta = correlation * (np.std(strategy_returns) / np.std(benchmark_returns))
                alpha = np.mean(strategy_returns) - beta * np.mean(benchmark_returns)
                
                comparison_results.update({
                    'alpha': float(alpha),
                    'beta': float(beta),
                    'correlation': float(correlation)
                })
            
            # Excess return
            excess_returns = strategy_returns - benchmark_returns
            excess_return = np.mean(excess_returns)
            comparison_results['excess_return'] = float(excess_return)
            
            # Information ratio
            if np.std(excess_returns) > 0:
                information_ratio = excess_return / np.std(excess_returns)
                comparison_results['information_ratio'] = float(information_ratio)
            
            # Outperformance periods
            outperformance = np.sum(excess_returns > 0)
            comparison_results['outperformance_periods'] = int(outperformance)
            
            logger.info(f"Benchmark comparison: Alpha={alpha:.4f}, Beta={beta:.4f}")
            
        except Exception as e:
            logger.error(f"Benchmark comparison failed: {e}")
            comparison_results['error'] = str(e)
        
        return comparison_results
    
    def generate_performance_report(self,
                                   strategy_name: str = None,
                                   time_period: int = None) -> Dict[str, Any]:
        """
        Generate comprehensive performance report
        
        Args:
            strategy_name: Specific strategy to report on (None for all)
            time_period: Time period in days (None for all data)
            
        Returns:
            Performance report
        """
        report = {
            'report_timestamp': datetime.now(),
            'strategies_analyzed': [],
            'overall_performance': {},
            'detailed_metrics': {},
            'alerts_summary': {},
            'recommendations': []
        }
        
        try:
            # Filter performance history
            filtered_history = self._filter_performance_history(strategy_name, time_period)
            
            if not filtered_history:
                report['error'] = 'No performance data available'
                return report
            
            # Analyze each strategy
            strategies = set(entry['strategy'] for entry in filtered_history)
            report['strategies_analyzed'] = list(strategies)
            
            for strategy in strategies:
                strategy_data = [entry for entry in filtered_history if entry['strategy'] == strategy]
                
                metrics = self._calculate_detailed_metrics(strategy_data)
                report['detailed_metrics'][strategy] = metrics
            
            # Overall performance
            report['overall_performance'] = self._calculate_overall_performance(filtered_history)
            
            # Alerts summary
            report['alerts_summary'] = self._summarize_alerts(time_period)
            
            # Generate recommendations
            report['recommendations'] = self._generate_performance_recommendations(report)
            
            logger.info(f"Performance report generated for {len(strategies)} strategies")
            
        except Exception as e:
            logger.error(f"Performance report generation failed: {e}")
            report['error'] = str(e)
        
        return report
    
    def detect_performance_anomalies(self,
                                   lookback_window: int = 20) -> List[Dict[str, Any]]:
        """
        Detect performance anomalies using statistical methods
        
        Args:
            lookback_window: Window size for anomaly detection
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        if len(self.performance_history) < lookback_window:
            return anomalies
        
        try:
            # Group by strategy
            strategies = {}
            for entry in self.performance_history[-lookback_window:]:
                strategy = entry['strategy']
                if strategy not in strategies:
                    strategies[strategy] = []
                strategies[strategy].append(entry)
            
            # Detect anomalies for each strategy
            for strategy_name, strategy_data in strategies.items():
                if len(strategy_data) < 10:  # Minimum data for anomaly detection
                    continue
                
                returns = [entry['metrics'].get('return', 0.0) for entry in strategy_data]
                
                # Statistical anomaly detection (simple z-score method)
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                
                for i, return_val in enumerate(returns):
                    if std_return > 0:
                        z_score = abs(return_val - mean_return) / std_return
                        
                        if z_score > 2.5:  # Anomaly threshold
                            anomalies.append({
                                'strategy': strategy_name,
                                'timestamp': strategy_data[i]['timestamp'],
                                'return': return_val,
                                'z_score': float(z_score),
                                'severity': 'high' if z_score > 3.0 else 'medium',
                                'type': 'statistical_outlier'
                            })
            
            logger.info(f"Detected {len(anomalies)} performance anomalies")
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
        
        return anomalies
    
    def _check_performance_alerts(self,
                                strategy_name: str,
                                metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Check for performance alerts"""
        alerts = []
        
        # Return-based alerts
        current_return = metrics.get('return', 0.0)
        if current_return < self.alert_threshold:
            alert = {
                'type': 'poor_performance',
                'strategy': strategy_name,
                'severity': 'high' if current_return < self.alert_threshold * 2 else 'medium',
                'message': f"Poor performance: {current_return:.3f}",
                'timestamp': datetime.now(),
                'metric_value': current_return
            }
            alerts.append(alert)
            self.alerts.append(alert)
        
        # Volatility alerts
        volatility = metrics.get('volatility', 0.0)
        if volatility > 0.1:  # 10% volatility threshold
            alert = {
                'type': 'high_volatility',
                'strategy': strategy_name,
                'severity': 'medium',
                'message': f"High volatility detected: {volatility:.3f}",
                'timestamp': datetime.now(),
                'metric_value': volatility
            }
            alerts.append(alert)
            self.alerts.append(alert)
        
        # Drawdown alerts
        drawdown = metrics.get('max_drawdown', 0.0)
        if drawdown > 0.15:  # 15% drawdown threshold
            alert = {
                'type': 'high_drawdown',
                'strategy': strategy_name,
                'severity': 'high',
                'message': f"High drawdown: {drawdown:.3f}",
                'timestamp': datetime.now(),
                'metric_value': drawdown
            }
            alerts.append(alert)
            self.alerts.append(alert)
        
        return alerts
    
    def _generate_performance_summary(self, strategy_name: str) -> Dict[str, Any]:
        """Generate performance summary for strategy"""
        strategy_data = [
            entry for entry in self.performance_history[-self.monitoring_window:]
            if entry['strategy'] == strategy_name
        ]
        
        if not strategy_data:
            return {}
        
        returns = [entry['metrics'].get('return', 0.0) for entry in strategy_data]
        
        summary = {
            'total_periods': len(strategy_data),
            'avg_return': float(np.mean(returns)) if returns else 0.0,
            'volatility': float(np.std(returns)) if len(returns) > 1 else 0.0,
            'best_period': float(max(returns)) if returns else 0.0,
            'worst_period': float(min(returns)) if returns else 0.0,
            'positive_periods': sum(1 for r in returns if r > 0),
            'win_rate': (sum(1 for r in returns if r > 0) / len(returns)) if returns else 0.0
        }
        
        # Sharpe ratio approximation
        if summary['volatility'] > 0:
            summary['sharpe_ratio'] = summary['avg_return'] / summary['volatility']
        else:
            summary['sharpe_ratio'] = 0.0
        
        return summary
    
    def _analyze_performance_trends(self, strategy_name: str) -> Dict[str, Any]:
        """Analyze performance trends"""
        strategy_data = [
            entry for entry in self.performance_history[-self.monitoring_window:]
            if entry['strategy'] == strategy_name
        ]
        
        if len(strategy_data) < 5:
            return {'insufficient_data': True}
        
        returns = [entry['metrics'].get('return', 0.0) for entry in strategy_data]
        
        # Simple trend analysis
        recent_performance = np.mean(returns[-5:]) if len(returns) >= 5 else 0.0
        historical_performance = np.mean(returns[:-5]) if len(returns) > 5 else 0.0
        
        trend = 'improving' if recent_performance > historical_performance else 'declining'
        
        # Momentum calculation
        if len(returns) >= 3:
            momentum = np.mean(returns[-3:]) - np.mean(returns[-6:-3]) if len(returns) >= 6 else 0.0
        else:
            momentum = 0.0
        
        return {
            'trend_direction': trend,
            'recent_performance': float(recent_performance),
            'historical_performance': float(historical_performance),
            'momentum': float(momentum),
            'consistency': 1.0 - (np.std(returns) / max(abs(np.mean(returns)), 0.001))
        }
    
    def _get_default_benchmark(self, length: int) -> List[float]:
        """Get default benchmark data"""
        # Generate simple benchmark (random walk with slight upward bias)
        np.random.seed(42)  # For reproducibility
        benchmark = np.random.normal(0.001, 0.02, length)  # 0.1% mean, 2% std
        return benchmark.tolist()
    
    def _filter_performance_history(self,
                                   strategy_name: Optional[str],
                                   time_period: Optional[int]) -> List[Dict[str, Any]]:
        """Filter performance history based on criteria"""
        filtered_data = self.performance_history.copy()
        
        # Filter by strategy
        if strategy_name:
            filtered_data = [entry for entry in filtered_data if entry['strategy'] == strategy_name]
        
        # Filter by time period
        if time_period:
            cutoff_date = datetime.now() - timedelta(days=time_period)
            filtered_data = [entry for entry in filtered_data if entry['timestamp'] > cutoff_date]
        
        return filtered_data
    
    def _calculate_detailed_metrics(self, strategy_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate detailed performance metrics"""
        if not strategy_data:
            return {}
        
        returns = [entry['metrics'].get('return', 0.0) for entry in strategy_data]
        
        metrics = {
            'total_return': float(np.sum(returns)),
            'annualized_return': float(np.mean(returns) * 252) if returns else 0.0,  # Assuming daily data
            'volatility': float(np.std(returns) * np.sqrt(252)) if len(returns) > 1 else 0.0,
            'max_return': float(max(returns)) if returns else 0.0,
            'min_return': float(min(returns)) if returns else 0.0,
            'skewness': float(self._calculate_skewness(returns)) if len(returns) > 2 else 0.0,
            'kurtosis': float(self._calculate_kurtosis(returns)) if len(returns) > 3 else 0.0
        }
        
        # Drawdown calculation
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - running_max
        metrics['max_drawdown'] = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
        
        return metrics
    
    def _calculate_overall_performance(self, filtered_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall performance across all strategies"""
        if not filtered_history:
            return {}
        
        all_returns = [entry['metrics'].get('return', 0.0) for entry in filtered_history]
        
        return {
            'total_observations': len(filtered_history),
            'average_return': float(np.mean(all_returns)) if all_returns else 0.0,
            'overall_volatility': float(np.std(all_returns)) if len(all_returns) > 1 else 0.0,
            'best_performance': float(max(all_returns)) if all_returns else 0.0,
            'worst_performance': float(min(all_returns)) if all_returns else 0.0
        }
    
    def _summarize_alerts(self, time_period: Optional[int]) -> Dict[str, Any]:
        """Summarize alerts for the reporting period"""
        if time_period:
            cutoff_date = datetime.now() - timedelta(days=time_period)
            relevant_alerts = [alert for alert in self.alerts if alert['timestamp'] > cutoff_date]
        else:
            relevant_alerts = self.alerts
        
        alert_counts = {}
        for alert in relevant_alerts:
            alert_type = alert['type']
            alert_counts[alert_type] = alert_counts.get(alert_type, 0) + 1
        
        return {
            'total_alerts': len(relevant_alerts),
            'alert_types': alert_counts,
            'high_severity_alerts': sum(1 for alert in relevant_alerts if alert['severity'] == 'high')
        }
    
    def _generate_performance_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        detailed_metrics = report.get('detailed_metrics', {})
        alerts_summary = report.get('alerts_summary', {})
        
        # Check for poor performance
        for strategy, metrics in detailed_metrics.items():
            if metrics.get('annualized_return', 0.0) < 0:
                recommendations.append(f"Consider reviewing {strategy} strategy parameters - negative returns detected")
            
            if metrics.get('max_drawdown', 0.0) > 0.2:
                recommendations.append(f"Implement stronger risk management for {strategy} - high drawdown detected")
        
        # Check alert patterns
        high_severity_alerts = alerts_summary.get('high_severity_alerts', 0)
        if high_severity_alerts > 5:
            recommendations.append("Multiple high-severity alerts detected - consider reducing position sizes")
        
        if not recommendations:
            recommendations.append("Performance is within acceptable parameters - continue monitoring")
        
        return recommendations
    
    def _calculate_skewness(self, returns: List[float]) -> float:
        """Calculate skewness of returns"""
        if len(returns) < 3:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        skewness = np.mean([(r - mean_return) ** 3 for r in returns]) / (std_return ** 3)
        return skewness
    
    def _calculate_kurtosis(self, returns: List[float]) -> float:
        """Calculate kurtosis of returns"""
        if len(returns) < 4:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        kurtosis = np.mean([(r - mean_return) ** 4 for r in returns]) / (std_return ** 4) - 3
        return kurtosis