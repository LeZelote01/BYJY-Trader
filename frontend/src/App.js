import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  Activity, 
  BarChart3, 
  Settings, 
  Wallet, 
  TrendingUp, 
  AlertCircle,
  CheckCircle,
  Bot,
  RefreshCw,
  DollarSign,
  Brain,
  Zap
} from 'lucide-react';
import './App.css';

// Import AI Components
import AIDashboard from './components/ai/AIDashboard';
import EnsemblePredictions from './components/EnsemblePredictions';
import SentimentAnalysis from './components/SentimentAnalysis'; // 🆕 Phase 3.2
import RLTrading from './components/RLTrading'; // 🤖 Phase 3.3
import OptimizationDashboard from './components/OptimizationDashboard'; // 🧬 Phase 3.4

// Import Trading Components
import ExchangeConnector from './components/exchange/ExchangeConnector';
import TradingInterface from './components/trading/TradingInterface';
import PortfolioManager from './components/portfolio/PortfolioManager';
import ExchangeSettings from './components/settings/ExchangeSettings';

// Configuration API
const API_BASE_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
});

function App() {
  // États principaux
  const [currentView, setCurrentView] = useState('dashboard');
  const [systemHealth, setSystemHealth] = useState(null);
  const [tradingStatus, setTradingStatus] = useState(null);
  const [portfolio, setPortfolio] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // Charger les données initiales
  useEffect(() => {
    loadInitialData();
    const interval = setInterval(loadInitialData, 30000); // Refresh toutes les 30s
    return () => clearInterval(interval);
  }, []);

  const loadInitialData = async () => {
    try {
      console.log('🔄 Début du chargement des données...');
      setIsLoading(true);
      setError(null);

      // Charger les données en parallèle avec timeout
      console.log('📡 Appel des APIs...');
      const [healthRes, tradingRes, portfolioRes] = await Promise.all([
        api.get('/api/health/'),
        api.get('/api/trading/status'),
        api.get('/api/trading/portfolio')
      ]);

      console.log('✅ APIs répondues, mise à jour des états...');
      setSystemHealth(healthRes.data);
      setTradingStatus(tradingRes.data);
      setPortfolio(portfolioRes.data);
      
      console.log('✅ États mis à jour, fin du chargement');

    } catch (err) {
      console.error('❌ Erreur lors du chargement des données:', err);
      setError(err.message || 'Erreur de connexion');
    } finally {
      console.log('🏁 setIsLoading(false) appelé');
      setIsLoading(false);
    }
  };

  // Components internes
  const StatusCard = ({ title, status, icon: Icon, color = "blue" }) => (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600 dark:text-gray-400">{title}</p>
          <p className={`text-lg font-semibold text-${color}-600 dark:text-${color}-400 capitalize`}>
            {status}
          </p>
        </div>
        <Icon className={`w-8 h-8 text-${color}-500`} />
      </div>
    </div>
  );

  const MetricCard = ({ title, value, change, icon: Icon }) => (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600 dark:text-gray-400">{title}</p>
          <p className="text-2xl font-bold text-gray-900 dark:text-white">{value}</p>
          {change && (
            <p className={`text-sm ${change > 0 ? 'text-green-600' : 'text-red-600'}`}>
              {change > 0 ? '+' : ''}{change}%
            </p>
          )}
        </div>
        <Icon className="w-8 h-8 text-primary-500" />
      </div>
    </div>
  );

  const Sidebar = () => {
    const menuItems = [
      { id: 'dashboard', name: 'Dashboard', icon: BarChart3 },
      { id: 'ai', name: 'IA Trading', icon: Brain },
      { id: 'ensemble', name: 'IA Avancée', icon: Activity },
      { id: 'sentiment', name: 'Sentiment', icon: Activity }, // 🆕 Phase 3.2
      { id: 'rl', name: 'RL Trading', icon: Bot }, // 🤖 Phase 3.3
      { id: 'optimization', name: 'Optimisation', icon: Zap }, // 🧬 Phase 3.4
      { id: 'trading', name: 'Trading', icon: TrendingUp },
      { id: 'portfolio', name: 'Portfolio', icon: Wallet },
      { id: 'strategies', name: 'Exchanges', icon: Bot },
      { id: 'settings', name: 'Configuration', icon: Settings },
    ];

    return (
      <div className="bg-gray-900 text-white w-64 min-h-screen p-4">
        <div className="flex items-center mb-8">
          <Bot className="w-8 h-8 text-primary-500 mr-3" />
          <h1 className="text-xl font-bold">BYJY-Trader</h1>
        </div>
        
        <nav className="space-y-2">
          {menuItems.map((item) => (
            <button
              key={item.id}
              onClick={() => setCurrentView(item.id)}
              className={`w-full flex items-center px-4 py-3 rounded-lg transition-colors ${
                currentView === item.id 
                  ? 'bg-primary-600 text-white' 
                  : 'text-gray-300 hover:bg-gray-800'
              }`}
            >
              <item.icon className="w-5 h-5 mr-3" />
              {item.name}
            </button>
          ))}
        </nav>

        {/* Status système en bas */}
        <div className="absolute bottom-4 left-4 right-4">
          <div className="bg-gray-800 rounded-lg p-3">
            <div className="flex items-center justify-between text-sm">
              <span>Système</span>
              <div className="flex items-center">
                {systemHealth?.status === 'healthy' ? (
                  <CheckCircle className="w-4 h-4 text-green-500 mr-1" />
                ) : (
                  <AlertCircle className="w-4 h-4 text-red-500 mr-1" />
                )}
                <span className="capitalize">{systemHealth?.status || 'Inconnu'}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const Dashboard = () => (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Dashboard</h2>
        <button
          onClick={loadInitialData}
          className="flex items-center px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700"
        >
          <RefreshCw className="w-4 h-4 mr-2" />
          Actualiser
        </button>
      </div>

      {/* Métriques principales */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard 
          title="Valeur Portfolio"
          value={portfolio ? `$${parseFloat(portfolio.total_value).toLocaleString()}` : '$0'}
          change={portfolio ? parseFloat(portfolio.pnl_24h) : 0}
          icon={DollarSign}
        />
        <MetricCard 
          title="P&L Total"
          value={portfolio ? `$${parseFloat(portfolio.pnl_total).toLocaleString()}` : '$0'}
          icon={TrendingUp}
        />
        <MetricCard 
          title="Positions Actives"
          value={portfolio?.positions?.length || 0}
          icon={Wallet}
        />
        <MetricCard 
          title="Stratégies"
          value={tradingStatus?.active_strategies || 0}
          icon={Bot}
        />
      </div>

      {/* Status cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <StatusCard 
          title="Trading Engine"
          status={tradingStatus?.status || 'inconnu'}
          icon={Activity}
          color="green"
        />
        <StatusCard 
          title="Base de Données"
          status={systemHealth?.components?.database?.status || 'inconnu'}
          icon={BarChart3}
          color="blue"
        />
        <StatusCard 
          title="API"
          status={systemHealth?.status || 'inconnu'}
          icon={Settings}
          color="purple"
        />
      </div>

      {/* Positions actuelles */}
      {portfolio?.positions && portfolio.positions.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
          <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Positions Actives
            </h3>
          </div>
          <div className="p-6">
            <div className="overflow-x-auto">
              <table className="min-w-full">
                <thead>
                  <tr className="text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    <th className="pb-3">Symbole</th>
                    <th className="pb-3">Quantité</th>
                    <th className="pb-3">Prix d'entrée</th>
                    <th className="pb-3">Prix actuel</th>
                    <th className="pb-3">P&L</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                  {portfolio.positions.map((position, index) => (
                    <tr key={index}>
                      <td className="py-3 font-medium">{position.symbol}</td>
                      <td className="py-3">{parseFloat(position.quantity).toFixed(4)}</td>
                      <td className="py-3">${parseFloat(position.entry_price).toFixed(2)}</td>
                      <td className="py-3">${parseFloat(position.current_price).toFixed(2)}</td>
                      <td className={`py-3 font-semibold ${
                        parseFloat(position.unrealized_pnl) >= 0 ? 'text-green-600' : 'text-red-600'
                      }`}>
                        ${parseFloat(position.unrealized_pnl).toFixed(2)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );

  const LoadingSpinner = () => (
    <div className="flex items-center justify-center min-h-screen">
      <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-primary-600"></div>
    </div>
  );

  const ErrorDisplay = () => (
    <div className="min-h-screen flex items-center justify-center">
      <div className="text-center">
        <AlertCircle className="w-16 h-16 text-red-500 mx-auto mb-4" />
        <h2 className="text-xl font-semibold text-gray-900 mb-2">Erreur de connexion</h2>
        <p className="text-gray-600 mb-4">{error}</p>
        <button
          onClick={loadInitialData}
          className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700"
        >
          Réessayer
        </button>
      </div>
    </div>
  );

  // Rendu principal
  if (error && !systemHealth) {
    return <ErrorDisplay />;
  }

  if (isLoading && !systemHealth) {
    return <LoadingSpinner />;
  }

  return (
    <div className="flex min-h-screen bg-gray-100 dark:bg-gray-900">
      <Sidebar />
      
      <main className="flex-1 p-6">
        {currentView === 'dashboard' && <Dashboard />}
        {currentView === 'ai' && <AIDashboard />}
        {currentView === 'ensemble' && <EnsemblePredictions />}
        {currentView === 'sentiment' && <SentimentAnalysis />}
        {currentView === 'rl' && <RLTrading />}
        {currentView === 'optimization' && <OptimizationDashboard />}
        {currentView === 'trading' && <TradingInterface />}
        {currentView === 'portfolio' && <PortfolioManager />}
        {currentView === 'strategies' && <ExchangeConnector />}
        {currentView === 'settings' && <ExchangeSettings />}
      </main>
    </div>
  );
}

export default App;