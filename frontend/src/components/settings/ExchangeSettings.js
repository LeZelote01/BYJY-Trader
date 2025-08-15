import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  Settings,
  Key,
  Eye,
  EyeOff,
  Shield,
  AlertTriangle,
  CheckCircle,
  Save,
  RefreshCw,
  Info,
  Lock,
  Globe
} from 'lucide-react';

const API_BASE_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

const ExchangeSettings = () => {
  const [selectedExchange, setSelectedExchange] = useState('binance');
  const [credentials, setCredentials] = useState({
    api_key: '',
    api_secret: '',
    passphrase: '', // Pour Coinbase Pro
    sandbox: true
  });
  const [showCredentials, setShowCredentials] = useState({
    api_key: false,
    api_secret: false,
    passphrase: false
  });
  const [isLoading, setIsLoading] = useState(false);
  const [isSaved, setIsSaved] = useState(false);
  const [connectionTest, setConnectionTest] = useState(null);

  useEffect(() => {
    loadExchangeCredentials();
  }, [selectedExchange]);

  const loadExchangeCredentials = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/connectors/${selectedExchange}/config`);
      if (response.data.success && response.data.config) {
        setCredentials({
          api_key: response.data.config.api_key || '',
          api_secret: response.data.config.api_secret || '',
          passphrase: response.data.config.passphrase || '',
          sandbox: response.data.config.sandbox !== false
        });
      }
    } catch (error) {
      console.error('Erreur chargement credentials:', error);
    }
  };

  const saveCredentials = async () => {
    try {
      setIsLoading(true);
      setIsSaved(false);

      const response = await axios.post(`${API_BASE_URL}/api/connectors/${selectedExchange}/configure`, {
        api_key: credentials.api_key || null,
        api_secret: credentials.api_secret || null,
        passphrase: credentials.passphrase || null,
        sandbox: credentials.sandbox
      });

      if (response.data.success) {
        setIsSaved(true);
        setTimeout(() => setIsSaved(false), 3000);
      }
    } catch (error) {
      console.error('Erreur sauvegarde credentials:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const testConnection = async () => {
    try {
      setIsLoading(true);
      setConnectionTest(null);

      const response = await axios.post(`${API_BASE_URL}/api/connectors/${selectedExchange}/test`, {
        api_key: credentials.api_key || null,
        api_secret: credentials.api_secret || null,
        passphrase: credentials.passphrase || null,
        sandbox: credentials.sandbox
      });

      setConnectionTest(response.data);
    } catch (error) {
      console.error('Erreur test connexion:', error);
      setConnectionTest({
        success: false,
        error: error.message,
        authenticated: false
      });
    } finally {
      setIsLoading(false);
    }
  };

  const getExchangeConfig = (exchange) => {
    const configs = {
      binance: {
        name: 'Binance',
        fields: ['api_key', 'api_secret'],
        docs_url: 'https://www.binance.com/en/support/faq/how-to-create-api-key',
        sandbox_url: 'https://testnet.binance.vision/',
        permissions: ['Spot Trading', 'Futures Trading', 'Read'],
        note: 'Assurez-vous d\'activer "Enable Spot & Margin Trading" pour le trading.'
      },
      coinbase: {
        name: 'Coinbase Advanced',
        fields: ['api_key', 'api_secret'],
        docs_url: 'https://docs.cloud.coinbase.com/advanced-trade-api/docs/rest-api-auth',
        sandbox_url: 'https://api-public.sandbox.exchange.coinbase.com',
        permissions: ['View', 'Trade'],
        note: 'Utilisez les nouvelles clés API Coinbase Advanced Trade.'
      },
      kraken: {
        name: 'Kraken',
        fields: ['api_key', 'api_secret'],
        docs_url: 'https://docs.kraken.com/rest/#section/Authentication',
        sandbox_url: null,
        permissions: ['Query Funds', 'Query Open Orders', 'Query Closed Orders', 'Trade'],
        note: 'Kraken n\'a pas de mode sandbox, utilisez avec précaution.'
      },
      bybit: {
        name: 'Bybit',
        fields: ['api_key', 'api_secret'],
        docs_url: 'https://bybit-exchange.github.io/docs/v5/guide#authentication',
        sandbox_url: 'https://api-testnet.bybit.com',
        permissions: ['Contract', 'Spot', 'Wallet', 'Options', 'Derivatives'],
        note: 'Activez les permissions nécessaires selon votre type de trading.'
      }
    };
    return configs[exchange] || configs.binance;
  };

  const toggleCredentialVisibility = (field) => {
    setShowCredentials(prev => ({
      ...prev,
      [field]: !prev[field]
    }));
  };

  const config = getExchangeConfig(selectedExchange);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Configuration Exchange</h2>
          <p className="text-gray-600 mt-1">Configurez vos clés API pour les différents exchanges</p>
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
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Configuration Form */}
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center mb-6">
              <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center mr-3">
                <span className="text-blue-600 font-semibold text-sm uppercase">
                  {selectedExchange.slice(0, 2)}
                </span>
              </div>
              <div>
                <h3 className="text-lg font-semibold text-gray-900">{config.name}</h3>
                <p className="text-sm text-gray-600">Configuration des clés API</p>
              </div>
            </div>

            {/* Mode Selection */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 mb-3">
                Mode d'opération
              </label>
              <div className="grid grid-cols-2 gap-4">
                <label className={`border-2 rounded-lg p-4 cursor-pointer transition-colors ${
                  credentials.sandbox 
                    ? 'border-blue-500 bg-blue-50' 
                    : 'border-gray-200 hover:border-gray-300'
                }`}>
                  <input
                    type="radio"
                    checked={credentials.sandbox}
                    onChange={() => setCredentials({...credentials, sandbox: true})}
                    className="sr-only"
                  />
                  <div className="flex items-center">
                    <Shield className="w-5 h-5 text-yellow-600 mr-3" />
                    <div>
                      <p className="font-medium text-gray-900">Sandbox/Test</p>
                      <p className="text-sm text-gray-600">Mode sécurisé pour les tests</p>
                    </div>
                  </div>
                </label>

                <label className={`border-2 rounded-lg p-4 cursor-pointer transition-colors ${
                  !credentials.sandbox 
                    ? 'border-red-500 bg-red-50' 
                    : 'border-gray-200 hover:border-gray-300'
                }`}>
                  <input
                    type="radio"
                    checked={!credentials.sandbox}
                    onChange={() => setCredentials({...credentials, sandbox: false})}
                    className="sr-only"
                  />
                  <div className="flex items-center">
                    <Globe className="w-5 h-5 text-red-600 mr-3" />
                    <div>
                      <p className="font-medium text-gray-900">Production</p>
                      <p className="text-sm text-gray-600">Trading réel avec de l'argent</p>
                    </div>
                  </div>
                </label>
              </div>

              {!credentials.sandbox && (
                <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
                  <div className="flex items-start">
                    <AlertTriangle className="w-5 h-5 text-red-600 mt-0.5 mr-2" />
                    <div className="text-sm">
                      <p className="font-medium text-red-800">Attention - Mode Production</p>
                      <p className="text-red-700">
                        Ce mode utilise de l'argent réel. Assurez-vous de bien comprendre les risques.
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* API Credentials */}
            <div className="space-y-4">
              {/* API Key */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  <Key className="w-4 h-4 inline mr-1" />
                  Clé API
                </label>
                <div className="relative">
                  <input
                    type={showCredentials.api_key ? "text" : "password"}
                    value={credentials.api_key}
                    onChange={(e) => setCredentials({...credentials, api_key: e.target.value})}
                    placeholder="Votre clé API"
                    className="w-full px-3 py-2 pr-10 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                  <button
                    type="button"
                    onClick={() => toggleCredentialVisibility('api_key')}
                    className="absolute right-3 top-2.5 text-gray-400 hover:text-gray-600"
                  >
                    {showCredentials.api_key ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
              </div>

              {/* API Secret */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  <Lock className="w-4 h-4 inline mr-1" />
                  Secret API
                </label>
                <div className="relative">
                  <input
                    type={showCredentials.api_secret ? "text" : "password"}
                    value={credentials.api_secret}
                    onChange={(e) => setCredentials({...credentials, api_secret: e.target.value})}
                    placeholder="Votre secret API"
                    className="w-full px-3 py-2 pr-10 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                  <button
                    type="button"
                    onClick={() => toggleCredentialVisibility('api_secret')}
                    className="absolute right-3 top-2.5 text-gray-400 hover:text-gray-600"
                  >
                    {showCredentials.api_secret ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
              </div>

              {/* Passphrase (Coinbase only) */}
              {selectedExchange === 'coinbase' && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    <Key className="w-4 h-4 inline mr-1" />
                    Passphrase
                  </label>
                  <div className="relative">
                    <input
                      type={showCredentials.passphrase ? "text" : "password"}
                      value={credentials.passphrase}
                      onChange={(e) => setCredentials({...credentials, passphrase: e.target.value})}
                      placeholder="Votre passphrase"
                      className="w-full px-3 py-2 pr-10 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                    <button
                      type="button"
                      onClick={() => toggleCredentialVisibility('passphrase')}
                      className="absolute right-3 top-2.5 text-gray-400 hover:text-gray-600"
                    >
                      {showCredentials.passphrase ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* Action Buttons */}
            <div className="flex space-x-3 mt-6">
              <button
                onClick={saveCredentials}
                disabled={isLoading}
                className={`flex-1 flex items-center justify-center px-4 py-2 rounded-lg font-medium text-white transition-colors ${
                  isSaved 
                    ? 'bg-green-600 hover:bg-green-700' 
                    : 'bg-blue-600 hover:bg-blue-700'
                } disabled:opacity-50`}
              >
                {isLoading ? (
                  <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                ) : isSaved ? (
                  <CheckCircle className="w-4 h-4 mr-2" />
                ) : (
                  <Save className="w-4 h-4 mr-2" />
                )}
                {isSaved ? 'Sauvegardé' : 'Sauvegarder'}
              </button>

              <button
                onClick={testConnection}
                disabled={isLoading}
                className="flex-1 flex items-center justify-center px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 disabled:opacity-50"
              >
                <RefreshCw className={`w-4 h-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
                Tester Connexion
              </button>
            </div>

            {/* Connection Test Results */}
            {connectionTest && (
              <div className={`mt-4 p-4 rounded-lg ${
                connectionTest.success 
                  ? 'bg-green-50 border border-green-200' 
                  : 'bg-red-50 border border-red-200'
              }`}>
                <div className="flex items-start">
                  {connectionTest.success ? (
                    <CheckCircle className="w-5 h-5 text-green-600 mt-0.5 mr-2" />
                  ) : (
                    <AlertTriangle className="w-5 h-5 text-red-600 mt-0.5 mr-2" />
                  )}
                  <div className="text-sm">
                    <p className={`font-medium ${
                      connectionTest.success ? 'text-green-800' : 'text-red-800'
                    }`}>
                      {connectionTest.success ? 'Connexion réussie' : 'Échec de connexion'}
                    </p>
                    {connectionTest.success ? (
                      <div className="mt-2 space-y-1 text-green-700">
                        <p>Latence: {connectionTest.latency_ms}ms</p>
                        {connectionTest.authenticated && (
                          <p>Authentifié: {connectionTest.permissions?.join(', ') || 'Oui'}</p>
                        )}
                        <p>Mode: {connectionTest.sandbox_mode ? 'Sandbox' : 'Production'}</p>
                      </div>
                    ) : (
                      <p className="mt-1 text-red-700">{connectionTest.error}</p>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Documentation Sidebar */}
        <div className="space-y-6">
          {/* Info Card */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
            <div className="flex items-center mb-4">
              <Info className="w-5 h-5 text-blue-600 mr-2" />
              <h3 className="font-semibold text-blue-900">À propos de {config.name}</h3>
            </div>
            <p className="text-sm text-blue-800 mb-4">{config.note}</p>
            
            {config.sandbox_url && (
              <div className="mb-4">
                <p className="text-sm font-medium text-blue-900 mb-2">URL Sandbox:</p>
                <p className="text-xs text-blue-700 bg-blue-100 p-2 rounded font-mono break-all">
                  {config.sandbox_url}
                </p>
              </div>
            )}
            
            <a
              href={config.docs_url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center text-sm text-blue-600 hover:text-blue-700 font-medium"
            >
              Documentation API
              <Globe className="w-4 h-4 ml-1" />
            </a>
          </div>

          {/* Permissions Required */}
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="font-semibold text-gray-900 mb-4 flex items-center">
              <Shield className="w-5 h-5 mr-2" />
              Permissions Requises
            </h3>
            <ul className="space-y-2">
              {config.permissions.map((permission, index) => (
                <li key={index} className="flex items-center text-sm">
                  <CheckCircle className="w-4 h-4 text-green-500 mr-2 flex-shrink-0" />
                  <span className="text-gray-700">{permission}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Security Notice */}
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6">
            <div className="flex items-center mb-3">
              <Shield className="w-5 h-5 text-yellow-600 mr-2" />
              <h3 className="font-semibold text-yellow-900">Sécurité</h3>
            </div>
            <ul className="text-sm text-yellow-800 space-y-1">
              <li>• Ne partagez jamais vos clés API</li>
              <li>• Utilisez uniquement HTTPS</li>
              <li>• Limitez les permissions API</li>
              <li>• Testez en mode sandbox d'abord</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ExchangeSettings;