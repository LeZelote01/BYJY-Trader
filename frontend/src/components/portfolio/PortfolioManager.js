import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  Wallet,
  TrendingUp,
  TrendingDown,
  DollarSign,
  RefreshCw,
  BarChart3,
  PieChart,
  ArrowUpRight,
  ArrowDownRight,
  Clock,
  Settings
} from 'lucide-react';

const API_BASE_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

const PortfolioManager = () => {
  const [selectedExchange, setSelectedExchange] = useState('binance');
  const [balance, setBalance] = useState(null);
  const [portfolio, setPortfolio] = useState(null);
  const [positions, setPositions] = useState([]);
  const [tradingHistory, setTradingHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    loadPortfolioData();
    
    // Refresh every 30 seconds
    const interval = setInterval(loadPortfolioData, 30000);
    return () => clearInterval(interval);
  }, [selectedExchange]);

  const loadPortfolioData = async () => {
    try {
      setIsLoading(true);
      
      const [balanceRes, portfolioRes, historyRes] = await Promise.all([
        axios.get(`${API_BASE_URL}/api/connectors/${selectedExchange}/balance`),
        axios.get(`${API_BASE_URL}/api/trading/portfolio`),
        axios.get(`${API_BASE_URL}/api/trading/history?limit=50`)
      ]);

      setBalance(balanceRes.data);
      setPortfolio(portfolioRes.data);
      setTradingHistory(historyRes.data.trades || []);
      setPositions(portfolioRes.data.positions || []);

    } catch (error) {
      console.error('Erreur chargement portfolio:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const formatCurrency = (amount, currency = 'USD') => {
    return new Intl.NumberFormat('fr-FR', {
      style: 'currency',
      currency: currency
    }).format(amount || 0);
  };

  const formatPercentage = (value) => {
    const num = parseFloat(value) || 0;
    return `${num >= 0 ? '+' : ''}${num.toFixed(2)}%`;
  };

  const getBalanceIcon = (asset) => {
    const icons = {
      'BTC': '₿',
      'ETH': 'Ξ',
      'USDT': '₮',
      'USD': '$',
      'EUR': '€',
    };
    return icons[asset] || '◦';
  };

  const BalanceCard = ({ asset, data }) => {
    const totalValue = data.total || 0;
    if (totalValue <= 0) return null;

    return (
      <div className="bg-white rounded-lg shadow p-6 border">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
              <span className="text-blue-600 font-bold">
                {getBalanceIcon(asset)}
              </span>
            </div>
            <div>
              <h3 className="font-semibold text-gray-900">{asset}</h3>
              <p className="text-sm text-gray-600">
                {parseFloat(data.total).toFixed(6)}
              </p>
            </div>
          </div>
          <div className="text-right">
            <p className="font-semibold text-gray-900">
              {parseFloat(data.free).toFixed(6)}
            </p>
            <p className="text-sm text-gray-600">Disponible</p>
            {data.locked > 0 && (
              <p className="text-xs text-yellow-600">
                {parseFloat(data.locked).toFixed(6)} verrouillé
              </p>
            )}
          </div>
        </div>
      </div>
    );
  };

  const PositionCard = ({ position }) => {
    const pnl = parseFloat(position.unrealized_pnl) || 0;
    const pnlPercent = parseFloat(position.pnl_percentage) || 0;

    return (
      <div className="bg-white rounded-lg shadow p-6 border">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="font-semibold text-gray-900">{position.symbol}</h3>
            <p className="text-sm text-gray-600 capitalize">{position.side} Position</p>
          </div>
          <div className={`px-3 py-1 rounded-full text-sm font-medium ${
            pnl >= 0 
              ? 'bg-green-100 text-green-800' 
              : 'bg-red-100 text-red-800'
          }`}>
            {pnl >= 0 ? <ArrowUpRight className="w-4 h-4 inline mr-1" /> : <ArrowDownRight className="w-4 h-4 inline mr-1" />}
            {formatCurrency(Math.abs(pnl))}
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-sm text-gray-600">Quantité</p>
            <p className="font-semibold">{parseFloat(position.quantity).toFixed(6)}</p>
          </div>
          <div>
            <p className="text-sm text-gray-600">Prix d'entrée</p>
            <p className="font-semibold">{formatCurrency(position.entry_price)}</p>
          </div>
          <div>
            <p className="text-sm text-gray-600">Prix actuel</p>
            <p className="font-semibold">{formatCurrency(position.current_price)}</p>
          </div>
          <div>
            <p className="text-sm text-gray-600">P&L %</p>
            <p className={`font-semibold ${pnlPercent >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {formatPercentage(pnlPercent)}
            </p>
          </div>
        </div>
      </div>
    );
  };

  const TradeHistoryRow = ({ trade }) => {
    return (
      <tr className="border-b border-gray-200">
        <td className="py-3 px-4">
          <div className="flex items-center">
            {trade.side === 'buy' ? 
              <ArrowUpRight className="w-4 h-4 text-green-500 mr-2" /> :
              <ArrowDownRight className="w-4 h-4 text-red-500 mr-2" />
            }
            <span className="font-medium">{trade.symbol}</span>
          </div>
        </td>
        <td className="py-3 px-4 capitalize">
          <span className={`px-2 py-1 rounded-full text-xs font-medium ${
            trade.side === 'buy' 
              ? 'bg-green-100 text-green-800' 
              : 'bg-red-100 text-red-800'
          }`}>
            {trade.side}
          </span>
        </td>
        <td className="py-3 px-4">{parseFloat(trade.quantity).toFixed(6)}</td>
        <td className="py-3 px-4">{formatCurrency(trade.price)}</td>
        <td className="py-3 px-4">{formatCurrency(trade.quantity * trade.price)}</td>
        <td className="py-3 px-4 text-sm text-gray-600">
          {new Date(trade.timestamp).toLocaleDateString('fr-FR')}
        </td>
        <td className="py-3 px-4">
          <span className={`px-2 py-1 rounded-full text-xs capitalize ${
            trade.status === 'filled' 
              ? 'bg-green-100 text-green-800' 
              : trade.status === 'cancelled'
              ? 'bg-gray-100 text-gray-800'
              : 'bg-yellow-100 text-yellow-800'
          }`}>
            {trade.status}
          </span>
        </td>
      </tr>
    );
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Gestion Portfolio</h2>
          <p className="text-gray-600 mt-1">Vue d'ensemble de vos positions et balances</p>
        </div>
        <div className="flex items-center space-x-4">
          <select
            value={selectedExchange}
            onChange={(e) => setSelectedExchange(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="binance">Binance</option>
            <option value="coinbase">Coinbase</option>
            <option value="kraken">Kraken</option>
            <option value="bybit">Bybit</option>
          </select>
          <button
            onClick={loadPortfolioData}
            disabled={isLoading}
            className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
            Actualiser
          </button>
        </div>
      </div>

      {/* Portfolio Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-gradient-to-r from-blue-500 to-blue-600 rounded-lg p-6 text-white">
          <div className="flex items-center">
            <DollarSign className="w-8 h-8 mr-3" />
            <div>
              <p className="text-blue-100">Valeur Totale</p>
              <p className="text-2xl font-bold">
                {portfolio ? formatCurrency(portfolio.total_value) : '$0.00'}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-gradient-to-r from-green-500 to-green-600 rounded-lg p-6 text-white">
          <div className="flex items-center">
            <TrendingUp className="w-8 h-8 mr-3" />
            <div>
              <p className="text-green-100">P&L Total</p>
              <p className="text-2xl font-bold">
                {portfolio ? formatCurrency(portfolio.pnl_total) : '$0.00'}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-gradient-to-r from-purple-500 to-purple-600 rounded-lg p-6 text-white">
          <div className="flex items-center">
            <Wallet className="w-8 h-8 mr-3" />
            <div>
              <p className="text-purple-100">Positions</p>
              <p className="text-2xl font-bold">{positions.length}</p>
            </div>
          </div>
        </div>

        <div className="bg-gradient-to-r from-orange-500 to-orange-600 rounded-lg p-6 text-white">
          <div className="flex items-center">
            <BarChart3 className="w-8 h-8 mr-3" />
            <div>
              <p className="text-orange-100">P&L 24h</p>
              <p className="text-2xl font-bold">
                {portfolio ? formatPercentage(portfolio.pnl_24h) : '0.00%'}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Balances */}
      <div className="bg-white rounded-lg shadow">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900 flex items-center">
            <Wallet className="w-5 h-5 mr-2" />
            Balances - {selectedExchange}
          </h3>
        </div>
        <div className="p-6">
          {balance?.balances ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {Object.entries(balance.balances)
                .filter(([_, data]) => data.total > 0)
                .map(([asset, data]) => (
                  <BalanceCard key={asset} asset={asset} data={data} />
                ))}
            </div>
          ) : (
            <div className="text-center py-8">
              <Wallet className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-600">Aucune balance disponible</p>
              <p className="text-sm text-gray-500">Connectez-vous à un exchange pour voir vos balances</p>
            </div>
          )}
        </div>
      </div>

      {/* Active Positions */}
      <div className="bg-white rounded-lg shadow">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900 flex items-center">
            <TrendingUp className="w-5 h-5 mr-2" />
            Positions Actives
          </h3>
        </div>
        <div className="p-6">
          {positions.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {positions.map((position, index) => (
                <PositionCard key={index} position={position} />
              ))}
            </div>
          ) : (
            <div className="text-center py-8">
              <PieChart className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-600">Aucune position active</p>
              <p className="text-sm text-gray-500">Vos positions ouvertes apparaîtront ici</p>
            </div>
          )}
        </div>
      </div>

      {/* Trading History */}
      <div className="bg-white rounded-lg shadow">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900 flex items-center">
            <Clock className="w-5 h-5 mr-2" />
            Historique des Trades
          </h3>
        </div>
        <div className="overflow-x-auto">
          {tradingHistory.length > 0 ? (
            <table className="min-w-full">
              <thead className="bg-gray-50">
                <tr>
                  <th className="py-3 px-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Symbole
                  </th>
                  <th className="py-3 px-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Côté
                  </th>
                  <th className="py-3 px-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Quantité
                  </th>
                  <th className="py-3 px-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Prix
                  </th>
                  <th className="py-3 px-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Total
                  </th>
                  <th className="py-3 px-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Date
                  </th>
                  <th className="py-3 px-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Status
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white">
                {tradingHistory.slice(0, 20).map((trade, index) => (
                  <TradeHistoryRow key={index} trade={trade} />
                ))}
              </tbody>
            </table>
          ) : (
            <div className="text-center py-8">
              <BarChart3 className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-600">Aucun historique de trading</p>
              <p className="text-sm text-gray-500">Vos trades récents apparaîtront ici</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PortfolioManager;