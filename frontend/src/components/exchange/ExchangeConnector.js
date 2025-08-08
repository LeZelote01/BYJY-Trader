import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  Wifi, 
  WifiOff, 
  Settings, 
  CheckCircle, 
  XCircle, 
  Clock,
  AlertTriangle,
  Key,
  Eye,
  EyeOff,
  RefreshCw
} from 'lucide-react';

const API_BASE_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

const ExchangeConnector = () => {
  const [exchanges, setExchanges] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedExchange, setSelectedExchange] = useState(null);
  const [credentials, setCredentials] = useState({
    api_key: '',
    api_secret: '',
    sandbox: true
  });
  const [showCredentials, setShowCredentials] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState({});

  useEffect(() => {
    loadExchanges();
    loadConnectionStatus();
    
    // Refresh status every 30 seconds
    const interval = setInterval(loadConnectionStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const loadExchanges = async () => {
    try {
      setIsLoading(true);
      const response = await axios.get(`${API_BASE_URL}/api/connectors/`);
      setExchanges(response.data.exchanges || []);
    } catch (error) {
      console.error('Erreur chargement exchanges:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const loadConnectionStatus = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/connectors/status`);
      setConnectionStatus(response.data.status || {});
    } catch (error) {
      console.error('Erreur status connexions:', error);
    }
  };

  const handleConnect = async (exchange) => {
    try {
      setIsLoading(true);
      const response = await axios.post(`${API_BASE_URL}/api/connectors/${exchange}/connect`, {
        api_key: credentials.api_key || null,
        api_secret: credentials.api_secret || null,
        sandbox: credentials.sandbox
      });

      if (response.data.success) {
        await loadConnectionStatus();
        setSelectedExchange(null);
        setCredentials({ api_key: '', api_secret: '', sandbox: true });
      }
    } catch (error) {
      console.error(`Erreur connexion ${exchange}:`, error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDisconnect = async (exchange) => {
    try {
      setIsLoading(true);
      await axios.post(`${API_BASE_URL}/api/connectors/${exchange}/disconnect`);
      await loadConnectionStatus();
    } catch (error) {
      console.error(`Erreur déconnexion ${exchange}:`, error);
    } finally {
      setIsLoading(false);
    }
  };

  const testConnection = async (exchange) => {
    try {
      setIsLoading(true);
      const response = await axios.post(`${API_BASE_URL}/api/connectors/${exchange}/test`);
      return response.data;
    } catch (error) {
      console.error(`Erreur test connexion ${exchange}:`, error);
      return { success: false, error: error.message };
    } finally {
      setIsLoading(false);
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'connected':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'disconnected':
        return <XCircle className="w-5 h-5 text-red-500" />;
      case 'connecting':
        return <Clock className="w-5 h-5 text-yellow-500 animate-pulse" />;
      case 'error':
        return <AlertTriangle className="w-5 h-5 text-red-500" />;
      default:
        return <WifiOff className="w-5 h-5 text-gray-400" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'connected': return 'text-green-600 bg-green-50';
      case 'disconnected': return 'text-red-600 bg-red-50';
      case 'connecting': return 'text-yellow-600 bg-yellow-50';
      case 'error': return 'text-red-600 bg-red-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const ExchangeCard = ({ exchange }) => {
    const status = connectionStatus[exchange];
    const isConnected = status?.status === 'connected';

    return (
      <div className="bg-white rounded-lg shadow-md p-6 border">
        <div className="flex items-start justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
              <span className="text-blue-600 font-semibold text-sm uppercase">
                {exchange.slice(0, 2)}
              </span>
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 capitalize">{exchange}</h3>
              <div className="flex items-center space-x-2 mt-1">
                {getStatusIcon(status?.status)}
                <span className={`text-sm px-2 py-1 rounded-full capitalize ${getStatusColor(status?.status)}`}>
                  {status?.status || 'Déconnecté'}
                </span>
              </div>
            </div>
          </div>

          <div className="flex space-x-2">
            <button
              onClick={() => testConnection(exchange)}
              disabled={isLoading}
              className="p-2 text-gray-500 hover:text-blue-600 rounded-lg hover:bg-blue-50"
              title="Tester la connexion"
            >
              <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
            </button>
            
            <button
              onClick={() => setSelectedExchange(exchange)}
              className="p-2 text-gray-500 hover:text-blue-600 rounded-lg hover:bg-blue-50"
              title="Configurer"
            >
              <Settings className="w-4 h-4" />
            </button>

            {isConnected ? (
              <button
                onClick={() => handleDisconnect(exchange)}
                disabled={isLoading}
                className="px-3 py-1 bg-red-100 text-red-600 rounded-lg hover:bg-red-200 text-sm"
              >
                <WifiOff className="w-4 h-4 inline mr-1" />
                Déconnecter
              </button>
            ) : (
              <button
                onClick={() => setSelectedExchange(exchange)}
                disabled={isLoading}
                className="px-3 py-1 bg-blue-100 text-blue-600 rounded-lg hover:bg-blue-200 text-sm"
              >
                <Wifi className="w-4 h-4 inline mr-1" />
                Connecter
              </button>
            )}
          </div>
        </div>

        {status?.authenticated && (
          <div className="mt-4 p-3 bg-green-50 rounded-lg">
            <div className="flex items-center text-sm text-green-800">
              <CheckCircle className="w-4 h-4 mr-2" />
              Authentifié - {status.permissions?.join(', ') || 'Permissions inconnues'}
            </div>
          </div>
        )}

        {status?.error && (
          <div className="mt-4 p-3 bg-red-50 rounded-lg">
            <div className="flex items-center text-sm text-red-800">
              <AlertTriangle className="w-4 h-4 mr-2" />
              {status.error}
            </div>
          </div>
        )}

        {status?.latency_ms && (
          <div className="mt-4 text-sm text-gray-600">
            Latence: {status.latency_ms}ms
          </div>
        )}
      </div>
    );
  };

  const ConnectionModal = () => {
    if (!selectedExchange) return null;

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg shadow-xl p-6 w-full max-w-md">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-semibold text-gray-900 capitalize">
              Connecter {selectedExchange}
            </h2>
            <button
              onClick={() => setSelectedExchange(null)}
              className="text-gray-400 hover:text-gray-600"
            >
              ✕
            </button>
          </div>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Mode
              </label>
              <div className="flex space-x-4">
                <label className="flex items-center">
                  <input
                    type="radio"
                    checked={credentials.sandbox}
                    onChange={() => setCredentials({...credentials, sandbox: true})}
                    className="mr-2"
                  />
                  <span className="text-sm">Sandbox/Test</span>
                </label>
                <label className="flex items-center">
                  <input
                    type="radio"
                    checked={!credentials.sandbox}
                    onChange={() => setCredentials({...credentials, sandbox: false})}
                    className="mr-2"
                  />
                  <span className="text-sm">Production</span>
                </label>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                <Key className="w-4 h-4 inline mr-1" />
                Clé API (optionnel)
              </label>
              <input
                type="text"
                value={credentials.api_key}
                onChange={(e) => setCredentials({...credentials, api_key: e.target.value})}
                placeholder="Votre clé API"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Secret API (optionnel)
              </label>
              <div className="relative">
                <input
                  type={showCredentials ? "text" : "password"}
                  value={credentials.api_secret}
                  onChange={(e) => setCredentials({...credentials, api_secret: e.target.value})}
                  placeholder="Votre secret API"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <button
                  type="button"
                  onClick={() => setShowCredentials(!showCredentials)}
                  className="absolute right-3 top-2 text-gray-400 hover:text-gray-600"
                >
                  {showCredentials ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>

            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3">
              <div className="flex items-start">
                <AlertTriangle className="w-4 h-4 text-yellow-600 mt-0.5 mr-2" />
                <div className="text-sm text-yellow-800">
                  <p className="font-medium">Mode Sandbox activé</p>
                  <p>Utilisez des clés API de test pour éviter les transactions réelles.</p>
                </div>
              </div>
            </div>
          </div>

          <div className="flex space-x-3 mt-6">
            <button
              onClick={() => setSelectedExchange(null)}
              className="flex-1 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50"
            >
              Annuler
            </button>
            <button
              onClick={() => handleConnect(selectedExchange)}
              disabled={isLoading}
              className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
            >
              {isLoading ? 'Connexion...' : 'Se connecter'}
            </button>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Connecteurs Exchange</h2>
          <p className="text-gray-600 mt-1">
            Gérez vos connexions aux différentes plateformes d'échange
          </p>
        </div>
        <button
          onClick={loadConnectionStatus}
          disabled={isLoading}
          className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
        >
          <RefreshCw className={`w-4 h-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
          Actualiser
        </button>
      </div>

      {/* Overview des connexions */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-blue-50 rounded-lg p-4">
          <div className="flex items-center">
            <Wifi className="w-6 h-6 text-blue-600 mr-3" />
            <div>
              <p className="text-sm text-blue-600">Connectés</p>
              <p className="text-lg font-semibold text-blue-900">
                {Object.values(connectionStatus).filter(s => s?.status === 'connected').length}
              </p>
            </div>
          </div>
        </div>
        
        <div className="bg-gray-50 rounded-lg p-4">
          <div className="flex items-center">
            <WifiOff className="w-6 h-6 text-gray-600 mr-3" />
            <div>
              <p className="text-sm text-gray-600">Disponibles</p>
              <p className="text-lg font-semibold text-gray-900">{exchanges.length}</p>
            </div>
          </div>
        </div>

        <div className="bg-green-50 rounded-lg p-4">
          <div className="flex items-center">
            <CheckCircle className="w-6 h-6 text-green-600 mr-3" />
            <div>
              <p className="text-sm text-green-600">Authentifiés</p>
              <p className="text-lg font-semibold text-green-900">
                {Object.values(connectionStatus).filter(s => s?.authenticated).length}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-yellow-50 rounded-lg p-4">
          <div className="flex items-center">
            <Settings className="w-6 h-6 text-yellow-600 mr-3" />
            <div>
              <p className="text-sm text-yellow-600">Mode Sandbox</p>
              <p className="text-lg font-semibold text-yellow-900">Actif</p>
            </div>
          </div>
        </div>
      </div>

      {/* Liste des exchanges */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {exchanges.map((exchange) => (
          <ExchangeCard key={exchange} exchange={exchange} />
        ))}
      </div>

      {/* Modal de connexion */}
      <ConnectionModal />

      {isLoading && (
        <div className="fixed inset-0 bg-black bg-opacity-25 flex items-center justify-center z-40">
          <div className="bg-white rounded-lg p-4 flex items-center space-x-3">
            <RefreshCw className="w-5 h-5 animate-spin text-blue-600" />
            <span>Chargement...</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default ExchangeConnector;