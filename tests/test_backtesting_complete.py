"""
🧪 Tests Complets Backtesting System
Tests pour le système de backtesting Phase 2.3
"""

import pytest
import asyncio
import pandas as pd
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

# Import des modules à tester
from trading.backtesting.backtest_engine import (
    BacktestEngine, BacktestConfig, BacktestStatus, BacktestResult
)
from trading.backtesting.performance_analyzer import PerformanceAnalyzer
from trading.backtesting.metrics_calculator import MetricsCalculator
from trading.strategies.trend_following import TrendFollowingStrategy
from trading.strategies.mean_reversion import MeanReversionStrategy


class TestBacktestEngine:
    """Tests pour BacktestEngine"""
    
    def setup_method(self):
        """Setup avant chaque test"""
        self.engine = BacktestEngine()
        self.sample_data = self.create_sample_data()
        self.config = BacktestConfig(
            strategy_id="test_strategy",
            symbol="BTCUSDT",
            start_date=datetime.now(timezone.utc) - timedelta(days=30),
            end_date=datetime.now(timezone.utc),
            initial_balance=10000.0,
            timeframe="1h",
            commission_rate=0.001,
            slippage_rate=0.0001
        )
    
    def create_sample_data(self) -> pd.DataFrame:
        """Crée des données de test"""
        dates = pd.date_range(
            start=datetime.now(timezone.utc) - timedelta(days=30),
            end=datetime.now(timezone.utc),
            freq='1H'
        )
        
        # Génération de données OHLCV simulées avec tendance
        base_price = 45000
        data = []
        
        for i, date in enumerate(dates):
            # Tendance haussière avec volatilité
            trend = i * 5  # Tendance progressive
            volatility = 200  # Volatilité
            
            price = base_price + trend + (i % 24 - 12) * volatility * 0.1
            
            data.append({
                'timestamp': date,
                'open': price + (i % 7 - 3) * 10,
                'high': price + abs(i % 11) * 15,
                'low': price - abs(i % 9) * 12,
                'close': price,
                'volume': 100 + (i % 50) * 10
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self):
        """Test l'initialisation du moteur"""
        assert len(self.engine.active_backtests) == 0
        assert len(self.engine.completed_backtests) == 0
        
        status = self.engine.get_engine_status()
        assert status["engine_status"] == "operational"
        assert status["active_backtests"] == 0
    
    @pytest.mark.asyncio
    async def test_backtest_execution_trend_following(self):
        """Test l'exécution d'un backtest avec stratégie trend following"""
        # Créer la stratégie
        strategy = TrendFollowingStrategy("BTCUSDT", "1h")
        
        # Lancer le backtest
        backtest_id = await self.engine.run_backtest(strategy, self.sample_data, self.config)
        
        # Vérifications
        assert backtest_id is not None
        assert backtest_id in self.engine.completed_backtests
        
        result = self.engine.get_backtest_result(backtest_id)
        assert result is not None
        assert result.status == BacktestStatus.COMPLETED
        assert result.initial_balance == 10000.0
        # Le final_balance peut être égal au initial si aucun signal n'est généré
        assert isinstance(result.final_balance, float)
    
    @pytest.mark.asyncio
    async def test_backtest_execution_mean_reversion(self):
        """Test l'exécution d'un backtest avec stratégie mean reversion"""
        # Créer la stratégie
        strategy = MeanReversionStrategy("BTCUSDT", "1h")
        
        # Lancer le backtest
        backtest_id = await self.engine.run_backtest(strategy, self.sample_data, self.config)
        
        # Vérifications
        result = self.engine.get_backtest_result(backtest_id)
        assert result is not None
        assert result.status == BacktestStatus.COMPLETED
        assert isinstance(result.total_trades, int)
        assert isinstance(result.win_rate, float)
    
    @pytest.mark.asyncio
    async def test_backtest_with_insufficient_data(self):
        """Test avec données insuffisantes"""
        # Données très limitées
        limited_data = self.sample_data.head(5)
        strategy = TrendFollowingStrategy("BTCUSDT", "1h")
        
        backtest_id = await self.engine.run_backtest(strategy, limited_data, self.config)
        
        result = self.engine.get_backtest_result(backtest_id)
        assert result is not None
        assert result.status == BacktestStatus.COMPLETED
        # Avec peu de données, peu ou pas de trades
    
    def test_backtest_config_validation(self):
        """Test la validation de la configuration"""
        config_dict = self.config.to_dict()
        
        assert config_dict["symbol"] == "BTCUSDT"
        assert config_dict["initial_balance"] == 10000.0
        assert config_dict["commission_rate"] == 0.001
        assert config_dict["slippage_rate"] == 0.0001
    
    @pytest.mark.asyncio
    async def test_backtest_cancellation(self):
        """Test l'annulation d'un backtest"""
        strategy = TrendFollowingStrategy("BTCUSDT", "1h")
        
        # Simuler un backtest en cours (ajout direct pour le test)
        result = BacktestResult(
            backtest_id="test_cancel_id",
            config=self.config,
            status=BacktestStatus.RUNNING,
            start_time=datetime.now(timezone.utc)
        )
        self.engine.active_backtests["test_cancel_id"] = result
        
        # Annuler
        success = self.engine.cancel_backtest("test_cancel_id")
        
        assert success is True
        assert "test_cancel_id" not in self.engine.active_backtests
        assert "test_cancel_id" in self.engine.completed_backtests
        
        cancelled_result = self.engine.completed_backtests["test_cancel_id"]
        assert cancelled_result.status == BacktestStatus.CANCELLED
    
    def test_backtest_history(self):
        """Test la récupération de l'historique"""
        # Ajouter quelques backtests complétés
        for i in range(3):
            result = BacktestResult(
                backtest_id=f"test_id_{i}",
                config=self.config,
                status=BacktestStatus.COMPLETED,
                start_time=datetime.now(timezone.utc),
                end_time=datetime.now(timezone.utc)
            )
            self.engine.completed_backtests[f"test_id_{i}"] = result
        
        history = self.engine.get_backtest_history(limit=2)
        
        assert len(history) == 2
        assert all("backtest_id" in bt for bt in history)
        assert all("status" in bt for bt in history)


class TestPerformanceAnalyzer:
    """Tests pour PerformanceAnalyzer"""
    
    def setup_method(self):
        """Setup avant chaque test"""
        self.analyzer = PerformanceAnalyzer()
        self.mock_result = self.create_mock_backtest_result()
    
    def create_mock_backtest_result(self):
        """Crée un résultat de backtest simulé"""
        from trading.backtesting.backtest_engine import BacktestTrade
        
        # Configuration mock
        config = BacktestConfig(
            strategy_id="test_strategy",
            symbol="BTCUSDT",
            start_date=datetime.now(timezone.utc) - timedelta(days=30),
            end_date=datetime.now(timezone.utc),
            initial_balance=10000.0
        )
        
        # Créer des trades simulés
        trades = []
        for i in range(20):
            pnl = 50 * (1 if i % 3 != 0 else -1) * (1 + i * 0.1)  # Mix gains/pertes
            trade = BacktestTrade(
                id=f"trade_{i}",
                symbol="BTCUSDT",
                side="long" if i % 2 == 0 else "short",
                entry_price=45000 + i * 100,
                exit_price=45000 + i * 100 + pnl,
                quantity=0.1,
                entry_time=datetime.now(timezone.utc) - timedelta(hours=i),
                exit_time=datetime.now(timezone.utc) - timedelta(hours=i-1),
                pnl=pnl,
                pnl_percent=(pnl / (45000 + i * 100)) * 100,
                commission=5.0,
                duration_hours=1.0,
                strategy_id="test_strategy"
            )
            trades.append(trade)
        
        # Courbe d'équité simulée
        equity_curve = []
        portfolio_value = 10000
        for i in range(100):
            portfolio_value += (i % 7 - 3) * 50  # Variation simulée
            equity_curve.append({
                "timestamp": (datetime.now(timezone.utc) - timedelta(hours=100-i)).isoformat(),
                "portfolio_value": portfolio_value,
                "cash_balance": portfolio_value * 0.8,
                "unrealized_pnl": portfolio_value * 0.1,
                "open_positions": 1 if i % 10 < 5 else 0
            })
        
        # Résultat mock
        result = BacktestResult(
            backtest_id="test_id",
            config=config,
            status=BacktestStatus.COMPLETED,
            start_time=datetime.now(timezone.utc) - timedelta(days=30),
            end_time=datetime.now(timezone.utc),
            initial_balance=10000.0,
            final_balance=11500.0,
            total_return=1500.0,
            total_return_percent=15.0
        )
        
        result.trades = trades
        result.equity_curve = equity_curve
        result.total_trades = len(trades)
        result.winning_trades = len([t for t in trades if t.pnl > 0])
        result.losing_trades = len([t for t in trades if t.pnl < 0])
        result.win_rate = (result.winning_trades / result.total_trades) * 100
        result.max_drawdown = 500.0
        result.max_drawdown_percent = 5.0
        
        return result
    
    def test_analyze_backtest_performance(self):
        """Test l'analyse complète des performances"""
        analysis = self.analyzer.analyze_backtest_performance(self.mock_result)
        
        # Vérifications des sections principales
        assert "summary" in analysis
        assert "trading_metrics" in analysis
        assert "risk_metrics" in analysis
        assert "time_analysis" in analysis
        assert "drawdown_analysis" in analysis
        assert "monthly_performance" in analysis
        assert "trade_distribution" in analysis
        
        # Vérifications summary
        summary = analysis["summary"]
        assert summary["initial_balance"] == 10000.0
        assert summary["final_balance"] == 11500.0
        assert summary["total_return_percent"] == 15.0
        assert "cagr" in summary
    
    def test_trading_metrics_calculation(self):
        """Test le calcul des métriques de trading"""
        analysis = self.analyzer.analyze_backtest_performance(self.mock_result)
        trading_metrics = analysis["trading_metrics"]
        
        assert trading_metrics["total_trades"] == 20
        assert "win_rate" in trading_metrics
        assert "profit_factor" in trading_metrics
        assert "average_win" in trading_metrics
        assert "average_loss" in trading_metrics
        assert "expectancy" in trading_metrics
    
    def test_risk_metrics_calculation(self):
        """Test le calcul des métriques de risque"""
        analysis = self.analyzer.analyze_backtest_performance(self.mock_result)
        risk_metrics = analysis["risk_metrics"]
        
        assert "max_drawdown" in risk_metrics
        assert "sharpe_ratio" in risk_metrics
        assert "volatility" in risk_metrics
        assert "var_95" in risk_metrics
        assert "var_99" in risk_metrics
    
    def test_time_analysis(self):
        """Test l'analyse temporelle"""
        analysis = self.analyzer.analyze_backtest_performance(self.mock_result)
        time_analysis = analysis["time_analysis"]
        
        assert "duration_stats" in time_analysis
        assert "hourly_performance" in time_analysis
        assert "daily_performance" in time_analysis
        assert "monthly_trades" in time_analysis
    
    def test_empty_trades_handling(self):
        """Test la gestion des résultats sans trades"""
        # Créer un résultat sans trades
        empty_result = self.mock_result
        empty_result.trades = []
        empty_result.total_trades = 0
        
        analysis = self.analyzer.analyze_backtest_performance(empty_result)
        
        # Doit gérer gracieusement l'absence de trades
        assert "trading_metrics" in analysis
        trading_metrics = analysis["trading_metrics"]
        assert trading_metrics["total_trades"] == 0
        assert trading_metrics["win_rate"] == 0


class TestMetricsCalculator:
    """Tests pour MetricsCalculator"""
    
    def setup_method(self):
        """Setup avant chaque test"""
        self.calculator = MetricsCalculator()
        self.mock_result = self.create_mock_result()
    
    def create_mock_result(self):
        """Crée un résultat mock pour les tests"""
        from trading.backtesting.backtest_engine import BacktestTrade
        
        config = BacktestConfig(
            strategy_id="test",
            symbol="BTCUSDT",
            start_date=datetime.now(timezone.utc) - timedelta(days=30),
            end_date=datetime.now(timezone.utc),
            initial_balance=10000.0
        )
        
        # Trades simulés
        trades = []
        for i in range(15):
            pnl = (100 if i % 3 == 0 else -50) * (1 + i * 0.1)
            trades.append(BacktestTrade(
                id=f"trade_{i}",
                symbol="BTCUSDT", 
                side="long",
                entry_price=45000,
                exit_price=45000 + pnl,
                quantity=0.1,
                entry_time=datetime.now(timezone.utc) - timedelta(hours=i),
                exit_time=datetime.now(timezone.utc) - timedelta(hours=i-1),
                pnl=pnl,
                pnl_percent=pnl/45000*100,
                commission=5,
                duration_hours=1,
                strategy_id="test"
            ))
        
        # Equity curve simulée
        equity_curve = []
        for i in range(50):
            value = 10000 + i * 30 + (i % 8 - 4) * 200  # Croissance avec variations
            equity_curve.append({
                "timestamp": (datetime.now(timezone.utc) - timedelta(hours=50-i)).isoformat(),
                "portfolio_value": value,
                "cash_balance": value * 0.9,
                "unrealized_pnl": 0,
                "open_positions": 0
            })
        
        result = BacktestResult(
            backtest_id="test",
            config=config,
            status=BacktestStatus.COMPLETED,
            start_time=datetime.now(timezone.utc) - timedelta(days=30),
            end_time=datetime.now(timezone.utc),
            initial_balance=10000.0,
            final_balance=11000.0,
            max_drawdown=400.0,
            max_drawdown_percent=4.0
        )
        
        result.trades = trades
        result.equity_curve = equity_curve
        
        return result
    
    def test_calculate_all_metrics(self):
        """Test le calcul de toutes les métriques"""
        metrics = self.calculator.calculate_all_metrics(self.mock_result)
        
        # Vérification des sections principales
        expected_sections = [
            "basic_metrics", "risk_metrics", "risk_adjusted_metrics",
            "trading_metrics", "drawdown_metrics", "distribution_metrics"
        ]
        
        for section in expected_sections:
            assert section in metrics, f"Section {section} manquante"
    
    def test_basic_metrics(self):
        """Test les métriques de base"""
        metrics = self.calculator.calculate_basic_metrics(self.mock_result)
        basic = metrics["basic_metrics"]
        
        assert basic["initial_balance"] == 10000.0
        assert basic["final_balance"] == 11000.0
        assert "total_return_percent" in basic
        assert "annualized_return" in basic
        assert "duration_days" in basic
    
    def test_risk_metrics(self):
        """Test les métriques de risque"""
        metrics = self.calculator.calculate_risk_metrics(self.mock_result)
        risk = metrics["risk_metrics"]
        
        assert "daily_volatility" in risk
        assert "var_95" in risk
        assert "var_99" in risk
        assert "downside_deviation" in risk
        assert "max_consecutive_losses" in risk
    
    def test_risk_adjusted_metrics(self):
        """Test les métriques ajustées au risque"""
        metrics = self.calculator.calculate_risk_adjusted_metrics(self.mock_result)
        risk_adj = metrics["risk_adjusted_metrics"]
        
        assert "sharpe_ratio" in risk_adj
        assert "sortino_ratio" in risk_adj
        assert "calmar_ratio" in risk_adj
        assert "information_ratio" in risk_adj
        assert "treynor_ratio" in risk_adj
    
    def test_trading_metrics(self):
        """Test les métriques de trading"""
        metrics = self.calculator.calculate_trading_metrics(self.mock_result)
        trading = metrics["trading_metrics"]
        
        assert "profit_factor" in trading
        assert "payoff_ratio" in trading
        assert "expectancy" in trading
        assert "kelly_criterion" in trading
        assert "system_quality_number" in trading
        assert "recovery_factor" in trading
    
    def test_drawdown_metrics(self):
        """Test les métriques de drawdown"""
        metrics = self.calculator.calculate_drawdown_metrics(self.mock_result)
        drawdown = metrics["drawdown_metrics"]
        
        assert "max_drawdown_pct" in drawdown
        assert "avg_drawdown_pct" in drawdown
        assert "ulcer_index" in drawdown
        assert "pain_index" in drawdown
        assert "lake_ratio" in drawdown
    
    def test_distribution_metrics(self):
        """Test les métriques de distribution"""
        metrics = self.calculator.calculate_distribution_metrics(self.mock_result)
        distribution = metrics["distribution_metrics"]
        
        assert "mean_return" in distribution
        assert "std_return" in distribution
        assert "skewness" in distribution
        assert "kurtosis" in distribution
        assert "percentile_95" in distribution
    
    def test_risk_free_rate_setting(self):
        """Test la configuration du taux sans risque"""
        original_rate = self.calculator.risk_free_rate
        
        # Changer le taux
        new_rate = 0.05
        self.calculator.set_risk_free_rate(new_rate)
        
        assert self.calculator.risk_free_rate == new_rate
        
        # Test avec taux négatif (doit être mis à 0)
        self.calculator.set_risk_free_rate(-0.01)
        assert self.calculator.risk_free_rate == 0
        
        # Restaurer
        self.calculator.set_risk_free_rate(original_rate)


class TestBacktestingIntegration:
    """Tests d'intégration pour le système complet"""
    
    def setup_method(self):
        """Setup pour tests d'intégration"""
        self.engine = BacktestEngine()
        self.analyzer = PerformanceAnalyzer()
        self.calculator = MetricsCalculator()
        self.sample_data = self.create_realistic_data()
    
    def create_realistic_data(self) -> pd.DataFrame:
        """Crée des données plus réalistes pour les tests"""
        import numpy as np
        
        # Générer 1000 points de données (environ 40 jours en 1H)
        timestamps = pd.date_range(
            start=datetime.now(timezone.utc) - timedelta(days=42),
            end=datetime.now(timezone.utc),
            freq='1H'
        )
        
        # Prix de départ
        base_price = 45000
        prices = [base_price]
        
        # Générer une série de prix avec random walk et tendance
        np.random.seed(42)  # Pour des résultats reproductibles
        
        for i in range(1, len(timestamps)):
            # Tendance légèrement haussière
            trend = 0.0001
            
            # Volatilité réaliste
            volatility = 0.02
            
            # Random walk
            change = np.random.normal(trend, volatility)
            new_price = prices[-1] * (1 + change)
            
            # Éviter les prix négatifs
            new_price = max(new_price, 1000)
            prices.append(new_price)
        
        # Créer OHLC à partir des prix de clôture
        data = []
        for i, (timestamp, close) in enumerate(zip(timestamps, prices)):
            # Générer OHLC réaliste
            volatility_range = close * 0.005  # 0.5% de range
            
            high = close + np.random.uniform(0, volatility_range)
            low = close - np.random.uniform(0, volatility_range)
            open_price = prices[i-1] if i > 0 else close
            
            # Assurer la cohérence OHLC
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            volume = np.random.uniform(50, 500)
            
            data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    @pytest.mark.asyncio
    async def test_full_backtesting_workflow(self):
        """Test complet du workflow de backtesting"""
        # Configuration
        config = BacktestConfig(
            strategy_id="integration_test",
            symbol="BTCUSDT",
            start_date=datetime.now(timezone.utc) - timedelta(days=40),
            end_date=datetime.now(timezone.utc),
            initial_balance=10000.0,
            commission_rate=0.001,
            slippage_rate=0.0001
        )
        
        # Créer et configurer la stratégie
        strategy = TrendFollowingStrategy("BTCUSDT", "1h", {
            "fast_ma_period": 10,
            "slow_ma_period": 20,
            "min_trend_strength": 0.2
        })
        
        # Exécuter le backtest
        backtest_id = await self.engine.run_backtest(strategy, self.sample_data, config)
        
        # Récupérer les résultats
        result = self.engine.get_backtest_result(backtest_id)
        assert result is not None
        assert result.status == BacktestStatus.COMPLETED
        
        # Analyser les performances
        analysis = self.analyzer.analyze_backtest_performance(result)
        assert "summary" in analysis
        assert "trading_metrics" in analysis
        
        # Calculer les métriques détaillées
        metrics = self.calculator.calculate_all_metrics(result)
        assert "basic_metrics" in metrics
        assert "risk_metrics" in metrics
        
        # Vérifications de cohérence
        summary = analysis["summary"]
        basic_metrics = metrics["basic_metrics"]
        
        # Les métriques de base doivent être cohérentes
        assert summary["initial_balance"] == basic_metrics["initial_balance"]
        assert summary["final_balance"] == basic_metrics["final_balance"]
        assert abs(summary["total_return_percent"] - basic_metrics["total_return_percent"]) < 0.01
    
    @pytest.mark.asyncio
    async def test_multiple_strategies_comparison(self):
        """Test de comparaison entre plusieurs stratégies"""
        config = BacktestConfig(
            strategy_id="comparison_test",
            symbol="BTCUSDT",
            start_date=datetime.now(timezone.utc) - timedelta(days=30),
            end_date=datetime.now(timezone.utc),
            initial_balance=10000.0
        )
        
        strategies = [
            TrendFollowingStrategy("BTCUSDT", "1h"),
            MeanReversionStrategy("BTCUSDT", "1h")
        ]
        
        results = []
        
        # Tester chaque stratégie
        for i, strategy in enumerate(strategies):
            config.strategy_id = f"strategy_{i}"
            backtest_id = await self.engine.run_backtest(strategy, self.sample_data, config)
            result = self.engine.get_backtest_result(backtest_id)
            results.append(result)
        
        # Vérifier que tous les backtests ont réussi
        for result in results:
            assert result.status == BacktestStatus.COMPLETED
            assert result.initial_balance == 10000.0
        
        # Comparer les performances
        performances = []
        for result in results:
            analysis = self.analyzer.analyze_backtest_performance(result)
            performances.append(analysis["summary"])
        
        # Vérifier que nous avons des résultats différents
        returns = [p["total_return_percent"] for p in performances]
        assert len(set(returns)) > 1 or all(r == 0 for r in returns)  # Différents ou tous zéro
    
    @pytest.mark.asyncio
    async def test_backtesting_error_handling(self):
        """Test la gestion des erreurs dans le backtesting"""
        config = BacktestConfig(
            strategy_id="error_test",
            symbol="BTCUSDT",
            start_date=datetime.now(timezone.utc) - timedelta(days=30),
            end_date=datetime.now(timezone.utc),
            initial_balance=10000.0
        )
        
        # Données corrompues (colonnes manquantes)
        bad_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'price': [45000]  # Colonnes OHLCV manquantes
        })
        bad_data.set_index('timestamp', inplace=True)
        
        strategy = TrendFollowingStrategy("BTCUSDT", "1h")
        
        # Le backtest doit gérer l'erreur gracieusement
        try:
            backtest_id = await self.engine.run_backtest(strategy, bad_data, config)
            result = self.engine.get_backtest_result(backtest_id)
            
            # Soit il y a une erreur, soit aucun trade n'est exécuté
            assert result.status in [BacktestStatus.ERROR, BacktestStatus.COMPLETED]
            
        except Exception:
            # L'exception est acceptable pour des données invalides
            pass
    
    def test_performance_comparison_metrics(self):
        """Test les métriques de comparaison de performance"""
        # Créer deux résultats simulés pour comparaison
        config = BacktestConfig(
            strategy_id="test",
            symbol="BTCUSDT", 
            start_date=datetime.now(timezone.utc) - timedelta(days=30),
            end_date=datetime.now(timezone.utc),
            initial_balance=10000.0
        )
        
        # Résultat 1: Stratégie performante
        result1 = BacktestResult(
            backtest_id="perf1",
            config=config,
            status=BacktestStatus.COMPLETED,
            start_time=datetime.now(timezone.utc) - timedelta(days=30),
            end_time=datetime.now(timezone.utc),
            initial_balance=10000,
            final_balance=12000,
            total_return=2000,
            total_return_percent=20.0,
            max_drawdown_percent=5.0
        )
        
        # Résultat 2: Stratégie moins performante
        result2 = BacktestResult(
            backtest_id="perf2", 
            config=config,
            status=BacktestStatus.COMPLETED,
            start_time=datetime.now(timezone.utc) - timedelta(days=30),
            end_time=datetime.now(timezone.utc),
            initial_balance=10000,
            final_balance=10500,
            total_return=500,
            total_return_percent=5.0,
            max_drawdown_percent=8.0
        )
        
        # Ajouter des courbes d'équité minimales
        for result in [result1, result2]:
            result.equity_curve = [
                {"timestamp": "2023-01-01", "portfolio_value": result.initial_balance},
                {"timestamp": "2023-01-15", "portfolio_value": (result.initial_balance + result.final_balance) / 2},
                {"timestamp": "2023-01-30", "portfolio_value": result.final_balance}
            ]
        
        # Calculer les métriques pour comparaison
        metrics1 = self.calculator.calculate_basic_metrics(result1)
        metrics2 = self.calculator.calculate_basic_metrics(result2)
        
        # Vérifier que la première stratégie est meilleure
        assert metrics1["basic_metrics"]["total_return_percent"] > metrics2["basic_metrics"]["total_return_percent"]
        
        # Calculer les métriques de risque
        risk1 = self.calculator.calculate_risk_adjusted_metrics(result1)
        risk2 = self.calculator.calculate_risk_adjusted_metrics(result2)
        
        # Vérifier la présence des métriques de comparaison
        assert "risk_adjusted_metrics" in risk1
        assert "risk_adjusted_metrics" in risk2


if __name__ == "__main__":
    # Exécuter les tests
    pytest.main([__file__, "-v"])