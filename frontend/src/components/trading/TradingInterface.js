import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  TrendingUp, 
  TrendingDown,
  DollarSign,
  BarChart3,
  RefreshCw,
  AlertCircle,
  CheckCircle,
  Clock,
  XCircle,
  Settings
} from 'lucide-react';

const API_BASE_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

const TradingInterface = () => {
  const [selectedExchange, setSelectedExchange] = useState('binance');
  const [selectedSymbol, setSelectedSymbol] = useState('BTCUSDT');
  const [orderType, setOrderType] = useState('market');
  const [orderSide, setOrderSide] = useState('buy');
  const [quantity, setQuantity] = useState('');
  const [price, setPrice] = useState('');
  
  const [ticker, setTicker] = useState(null);
  const [orderBook, setOrderBook] = useState(null);
  const [balance, setBalance] = useState(null);
  const [openOrders, setOpenOrders] = useState([]);
  const [recentTrades, setRecentTrades] = useState([]);
  
  const [isLoading, setIsLoading] = useState(false);
  const [orderHistory, setOrderHistory] = useState([]);

  useEffect(() => {
    loadMarketData();
    loadAccountData();
    
    // Refresh market data every 5 seconds
    const marketInterval = setInterval(loadMarketData, 5000);
    
    // Refresh account data every 15 seconds
    const accountInterval = setInterval(loadAccountData, 15000);
    
    return () => {
      clearInterval(marketInterval);
      clearInterval(accountInterval);
    };
  }, [selectedExchange, selectedSymbol]);

  const loadMarketData = async () => {
    try {
      const [tickerRes, bookRes, tradesRes] = await Promise.all([
        axios.get(`${API_BASE_URL}/api/connectors/${selectedExchange}/ticker/${selectedSymbol}`),
        axios.get(`${API_BASE_URL}/api/connectors/${selectedExchange}/orderbook/${selectedSymbol}?depth=20`),
        axios.get(`${API_BASE_URL}/api/connectors/${selectedExchange}/trades/${selectedSymbol}?limit=10`)
      ]);

      setTicker(tickerRes.data);
      setOrderBook(bookRes.data);
      setRecentTrades(tradesRes.data.trades || []);
    } catch (error) {
      console.error('Erreur chargement données marché:', error);
    }
  };

  const loadAccountData = async () => {
    try {
      const [balanceRes, ordersRes] = await Promise.all([
        axios.get(`${API_BASE_URL}/api/connectors/${selectedExchange}/balance`),
        axios.get(`${API_BASE_URL}/api/connectors/${selectedExchange}/orders`)
      ]);

      setBalance(balanceRes.data);
      setOpenOrders(ordersRes.data.orders || []);
    } catch (error) {
      console.error('Erreur chargement données compte:', error);
    }
  };

  const placeOrder = async () => {
    try {
      setIsLoading(true);
      
      const orderData = {
        symbol: selectedSymbol,
        type: orderType,
        side: orderSide,
        quantity: parseFloat(quantity),
        ...(orderType === 'limit' && { price: parseFloat(price) })
      };

      const response = await axios.post(
        `${API_BASE_URL}/api/connectors/${selectedExchange}/order`,
        orderData
      );

      if (response.data.success) {
        // Reset form
        setQuantity('');
        setPrice('');
        
        // Reload data
        await loadAccountData();
        
        // Add to history
        setOrderHistory(prev => [
          { ...response.data, timestamp: new Date() },
          ...prev.slice(0, 9)
        ]);
      }
    } catch (error) {
      console.error('Erreur placement ordre:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const cancelOrder = async (orderId) => {
    try {
      setIsLoading(true);
      
      await axios.delete(
        `${API_BASE_URL}/api/connectors/${selectedExchange}/order/${orderId}?symbol=${selectedSymbol}`
      );

      await loadAccountData();
    } catch (error) {
      console.error('Erreur annulation ordre:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const getOrderStatusIcon = (status) => {
    switch (status) {
      case 'filled':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'cancelled':
        return <XCircle className="w-4 h-4 text-red-500" />;
      case 'pending':
      case 'open':
        return <Clock className="w-4 h-4 text-yellow-500" />;
      default:
        return <AlertCircle className="w-4 h-4 text-gray-500" />;
    }
  };

  const formatPrice = (price) => {
    return parseFloat(price).toFixed(selectedSymbol.includes('USDT') ? 2 : 8);
  };

  const formatQuantity = (qty) => {
    return parseFloat(qty).toFixed(6);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Interface Trading</h2>
          <p className="text-gray-600 mt-1">Trading temps réel sur les exchanges connectés</p>
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
            onClick={loadMarketData}
            className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            Actualiser
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Trading Panel */}
        <div className="lg:col-span-1 space-y-6">
          {/* Symbol Selector */}
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Sélection Symbole</h3>
            <select
              value={selectedSymbol}
              onChange={(e) => setSelectedSymbol(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="BTCUSDT">BTC/USDT</option>
              <option value="ETHUSDT">ETH/USDT</option>
              <option value="ADAUSDT">ADA/USDT</option>
              <option value="DOTUSDT">DOT/USDT</option>
              <option value="LINKUSDT">LINK/USDT</option>
            </select>
          </div>

          {/* Price Info */}
          {ticker && (
            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">{selectedSymbol}</h3>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Prix actuel</span>
                  <span className="text-xl font-bold text-gray-900">
                    ${formatPrice(ticker.price || 0)}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Change 24h</span>
                  <span className={`font-semibold ${ticker.change_percent_24h >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {ticker.change_percent_24h >= 0 ? '+' : ''}{(ticker.change_percent_24h || 0).toFixed(2)}%
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Volume 24h</span>
                  <span className="text-gray-900">${(ticker.quote_volume || 0).toLocaleString()}</span>
                </div>
              </div>
            </div>
          )}

          {/* Order Form */}
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Placer un Ordre</h3>
            
            <div className="space-y-4">
              {/* Order Type */}
              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={() => setOrderSide('buy')}
                  className={`px-4 py-2 rounded-lg font-medium ${
                    orderSide === 'buy' 
                      ? 'bg-green-600 text-white' 
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  Acheter
                </button>
                <button
                  onClick={() => setOrderSide('sell')}
                  className={`px-4 py-2 rounded-lg font-medium ${
                    orderSide === 'sell' 
                      ? 'bg-red-600 text-white' 
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  Vendre
                </button>
              </div>

              {/* Market/Limit */}
              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={() => setOrderType('market')}
                  className={`px-4 py-2 rounded-lg font-medium ${
                    orderType === 'market' 
                      ? 'bg-blue-600 text-white' 
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  Market
                </button>
                <button
                  onClick={() => setOrderType('limit')}
                  className={`px-4 py-2 rounded-lg font-medium ${
                    orderType === 'limit' 
                      ? 'bg-blue-600 text-white' 
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  Limit
                </button>
              </div>

              {/* Quantity */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Quantité
                </label>
                <input
                  type="number"
                  value={quantity}
                  onChange={(e) => setQuantity(e.target.value)}
                  placeholder="0.00"
                  step="0.000001"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>

              {/* Price (pour limit orders) */}
              {orderType === 'limit' && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Prix
                  </label>
                  <input
                    type="number"
                    value={price}
                    onChange={(e) => setPrice(e.target.value)}
                    placeholder="0.00"
                    step="0.01"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
              )}

              {/* Place Order Button */}
              <button
                onClick={placeOrder}
                disabled={isLoading || !quantity}
                className={`w-full px-4 py-3 rounded-lg font-medium text-white ${
                  orderSide === 'buy' 
                    ? 'bg-green-600 hover:bg-green-700' 
                    : 'bg-red-600 hover:bg-red-700'
                } disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                {isLoading ? 'Placement...' : `${orderSide === 'buy' ? 'Acheter' : 'Vendre'} ${selectedSymbol}`}
              </button>
            </div>
          </div>
        </div>

        {/* Market Data */}
        <div className="lg:col-span-2 space-y-6">
          {/* Order Book */}
          {orderBook && (
            <div className="bg-white rounded-lg shadow">
              <div className="px-6 py-4 border-b border-gray-200">
                <h3 className="text-lg font-semibold text-gray-900">Order Book - {selectedSymbol}</h3>
              </div>
              <div className="p-6">
                <div className="grid grid-cols-2 gap-6">
                  {/* Asks (Sell orders) */}
                  <div>
                    <h4 className="text-sm font-medium text-red-600 mb-3 flex items-center">
                      <TrendingDown className="w-4 h-4 mr-1" />
                      Ventes (Ask)
                    </h4>
                    <div className="space-y-1">
                      {orderBook.asks?.slice(0, 10).map((ask, index) => (
                        <div key={index} className="flex justify-between text-sm">
                          <span className="text-red-600">{formatPrice(ask[0])}</span>
                          <span className="text-gray-600">{formatQuantity(ask[1])}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Bids (Buy orders) */}
                  <div>
                    <h4 className="text-sm font-medium text-green-600 mb-3 flex items-center">
                      <TrendingUp className="w-4 h-4 mr-1" />
                      Achats (Bid)
                    </h4>
                    <div className="space-y-1">
                      {orderBook.bids?.slice(0, 10).map((bid, index) => (
                        <div key={index} className="flex justify-between text-sm">
                          <span className="text-green-600">{formatPrice(bid[0])}</span>
                          <span className="text-gray-600">{formatQuantity(bid[1])}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Open Orders */}
          <div className="bg-white rounded-lg shadow">
            <div className="px-6 py-4 border-b border-gray-200">
              <h3 className="text-lg font-semibold text-gray-900">Ordres Ouverts</h3>
            </div>
            <div className="p-6">
              {openOrders.length > 0 ? (
                <div className="space-y-3">
                  {openOrders.map((order, index) => (
                    <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <div className="flex items-center space-x-3">
                        {getOrderStatusIcon(order.status)}
                        <div>
                          <div className="font-medium">{order.symbol}</div>
                          <div className="text-sm text-gray-600">
                            {order.side.toUpperCase()} • {order.type.toUpperCase()}
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="font-medium">{formatQuantity(order.quantity)}</div>
                        {order.price && (
                          <div className="text-sm text-gray-600">${formatPrice(order.price)}</div>
                        )}
                      </div>
                      <button
                        onClick={() => cancelOrder(order.order_id)}
                        disabled={isLoading}
                        className="px-3 py-1 text-sm text-red-600 hover:bg-red-50 rounded"
                      >
                        Annuler
                      </button>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  Aucun ordre ouvert
                </div>
              )}
            </div>
          </div>

          {/* Recent Trades */}
          <div className="bg-white rounded-lg shadow">
            <div className="px-6 py-4 border-b border-gray-200">
              <h3 className="text-lg font-semibold text-gray-900">Trades Récents - {selectedSymbol}</h3>
            </div>
            <div className="p-6">
              <div className="space-y-2">
                {recentTrades.map((trade, index) => (
                  <div key={index} className="flex items-center justify-between text-sm">
                    <span className={`font-medium ${trade.side === 'buy' ? 'text-green-600' : 'text-red-600'}`}>
                      {trade.side === 'buy' ? '↗' : '↘'} {formatPrice(trade.price)}
                    </span>
                    <span className="text-gray-600">{formatQuantity(trade.quantity)}</span>
                    <span className="text-gray-500 text-xs">
                      {new Date(trade.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TradingInterface;