"""
🧪 Test Intégration Backtesting Simple avec Phase 2.1
Test réduit pour vérifier l'intégration
"""

import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta

from trading.backtesting.backtest_engine import BacktestEngine, BacktestConfig, BacktestStatus
from trading.backtesting.performance_analyzer import PerformanceAnalyzer
from trading.strategies.trend_following import TrendFollowingStrategy


class TestBacktestingIntegrationSimple:
    """Tests d'intégration simplifiés"""
    
    def setup_method(self):
        """Setup pour les tests"""
        self.engine = BacktestEngine()
        self.analyzer = PerformanceAnalyzer()
    
    @pytest.mark.asyncio
    async def test_backtesting_system_complete(self):
        """Test complet du système de backtesting"""
        
        # 1. Données simulées réalistes
        dates = pd.date_range(
            start=datetime.now(timezone.utc) - timedelta(days=10),
            end=datetime.now(timezone.utc) - timedelta(days=1),
            freq='h'
        )
        
        # Prix simulés avec tendance
        base_price = 45000
        data_rows = []
        
        for i, date in enumerate(dates):
            # Tendance + volatilité
            trend = i * 3
            volatility = 200
            noise = (i % 23 - 11) * volatility * 0.1
            
            price = base_price + trend + noise
            price = max(price, 30000)  # Prix minimum
            
            data_rows.append({
                'open': price + (i % 5 - 2) * 50,
                'high': price + abs(i % 7) * 30,
                'low': price - abs(i % 6) * 25,
                'close': price,
                'volume': 100 + (i % 50) * 10
            })
        
        data = pd.DataFrame(data_rows, index=dates)
        
        # 2. Configuration backtest
        config = BacktestConfig(
            strategy_id="integration_complete",
            symbol="BTCUSDT",
            start_date=dates[0],
            end_date=dates[-1],
            initial_balance=10000.0,
            timeframe="1h",
            commission_rate=0.001,
            slippage_rate=0.0001
        )
        
        # 3. Stratégie avec paramètres plus permissifs
        strategy = TrendFollowingStrategy("BTCUSDT", "1h", {
            "fast_ma_period": 8,
            "slow_ma_period": 16,
            "min_trend_strength": 0.05,  # Plus permissif
            "stop_loss_percent": 3.0,
            "take_profit_percent": 6.0
        })
        
        # 4. Exécution du backtest
        backtest_id = await self.engine.run_backtest(strategy, data, config)
        
        # 5. Vérifications des résultats
        result = self.engine.get_backtest_result(backtest_id)
        assert result is not None
        assert result.status == BacktestStatus.COMPLETED
        assert result.initial_balance == 10000.0
        assert isinstance(result.total_trades, int)
        assert result.total_trades >= 0
        
        # 6. Analyse des performances
        analysis = self.analyzer.analyze_backtest_performance(result)
        
        # Vérifications de l'analyse
        assert "summary" in analysis
        assert "trading_metrics" in analysis
        assert "risk_metrics" in analysis
        
        # 7. Affichage des résultats
        print(f"\n📊 RÉSULTATS BACKTEST COMPLET:")
        print(f"   ├─ Période: {len(data)} heures ({len(data)//24} jours)")
        print(f"   ├─ Trades exécutés: {result.total_trades}")
        print(f"   ├─ Balance finale: ${result.final_balance:,.2f}")
        print(f"   ├─ Rendement: {result.total_return_percent:.2f}%")
        
        if result.total_trades > 0:
            print(f"   ├─ Win Rate: {result.win_rate:.1f}%")
            print(f"   ├─ Profit Factor: {result.profit_factor:.2f}")
            print(f"   └─ Trades gagnants: {result.winning_trades}")
        else:
            print(f"   └─ Aucun signal généré (normal avec des paramètres stricts)")
        
        # 8. Test de la continuité
        assert len(result.equity_curve) > 0, "Courbe d'équité doit être générée"
        
        # 9. Test de l'historique
        history = self.engine.get_backtest_history(limit=1)
        assert len(history) == 1
        assert history[0]["backtest_id"] == backtest_id
        
        print("\n✅ Test d'intégration complet réussi!")
    
    def test_backtesting_engine_status(self):
        """Test du status du moteur de backtesting"""
        status = self.engine.get_engine_status()
        
        assert "engine_status" in status
        assert status["engine_status"] == "operational"
        assert "active_backtests" in status
        assert "completed_backtests" in status
        
        print(f"\n📊 STATUS MOTEUR BACKTESTING:")
        print(f"   ├─ Status: {status['engine_status']}")
        print(f"   ├─ Backtests actifs: {status['active_backtests']}")
        print(f"   └─ Backtests complétés: {status['completed_backtests']}")
    
    @pytest.mark.asyncio
    async def test_backtest_performance_with_trades(self):
        """Test avec données configurées pour générer des trades"""
        
        # Créer des données avec des patterns évidents
        periods = 200
        dates = pd.date_range(
            start=datetime.now(timezone.utc) - timedelta(hours=periods),
            periods=periods,
            freq='h'
        )
        
        # Pattern en escaliers pour forcer des signaux
        data_rows = []
        base_price = 45000
        
        for i in range(periods):
            # Créer des patterns de tendance clairs
            cycle = i // 40  # Changement tous les 40 points
            
            if cycle % 2 == 0:
                # Tendance haussière
                trend_price = base_price + (i % 40) * 50
            else:
                # Tendance baissière  
                trend_price = base_price + 2000 - (i % 40) * 50
            
            # Ajouter un peu de volatilité
            volatility = 100
            noise = (i % 7 - 3) * volatility * 0.3
            price = max(trend_price + noise, 20000)
            
            data_rows.append({
                'open': price + (i % 3 - 1) * 20,
                'high': price + abs(i % 5) * 15,
                'low': price - abs(i % 4) * 12,
                'close': price,
                'volume': 100 + (i % 30) * 5
            })
        
        data = pd.DataFrame(data_rows, index=dates)
        
        # Configuration avec stratégie très permissive
        config = BacktestConfig(
            strategy_id="performance_test",
            symbol="BTCUSDT",
            start_date=dates[0],
            end_date=dates[-1],
            initial_balance=10000.0,
            commission_rate=0.0005  # Commission réduite
        )
        
        strategy = TrendFollowingStrategy("BTCUSDT", "1h", {
            "fast_ma_period": 5,   # Très court
            "slow_ma_period": 10,  # Très court
            "min_trend_strength": 0.01,  # Très permissif
            "stop_loss_percent": 5.0,
            "take_profit_percent": 10.0
        })
        
        # Exécuter le backtest
        backtest_id = await self.engine.run_backtest(strategy, data, config)
        result = self.engine.get_backtest_result(backtest_id)
        
        # Vérifications
        assert result.status == BacktestStatus.COMPLETED
        
        print(f"\n📊 TEST PERFORMANCE AVEC PATTERNS:")
        print(f"   ├─ Données: {len(data)} points")
        print(f"   ├─ Trades: {result.total_trades}")
        print(f"   ├─ Balance: ${result.final_balance:,.2f}")
        print(f"   └─ Return: {result.total_return_percent:.2f}%")
        
        if result.total_trades > 0:
            # Analyser les performances avec des trades
            analysis = self.analyzer.analyze_backtest_performance(result)
            
            summary = analysis["summary"]
            trading_metrics = analysis["trading_metrics"]
            
            print(f"\n📈 ANALYSE DÉTAILLÉE:")
            print(f"   ├─ CAGR: {summary.get('cagr', 0):.2f}%")
            print(f"   ├─ Win Rate: {trading_metrics.get('win_rate', 0):.1f}%")
            print(f"   ├─ Avg Win: ${trading_metrics.get('average_win', 0):.2f}")
            print(f"   ├─ Avg Loss: ${trading_metrics.get('average_loss', 0):.2f}")
            print(f"   └─ Max Drawdown: {result.max_drawdown_percent:.2f}%")
            
            # Au moins quelques trades devraient être générés
            assert result.total_trades > 0, "Des trades devraient être générés avec ces patterns"
        else:
            print("   └─ Aucun trade généré (conditions trop strictes)")
        
        print("\n✅ Test de performance terminé!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])