import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  TrendingUp, 
  TrendingDown, 
  BarChart3, 
  MessageSquare, 
  Newspaper,
  Users,
  RefreshCw,
  AlertCircle,
  CheckCircle,
  Activity,
  Zap
} from 'lucide-react';

const SentimentAnalysis = () => {
  // États principaux
  const [sentimentData, setSentimentData] = useState({});
  const [correlationData, setCorrelationData] = useState({});
  const [systemStatus, setSystemStatus] = useState({});
  const [selectedSymbol, setSelectedSymbol] = useState('BTC');
  const [isLoading, setIsLoading] = useState(false);
  const [lastUpdate, setLastUpdate] = useState(null);

  const symbols = ['BTC', 'ETH', 'ADA', 'SOL'];
  const backendUrl = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

  // Chargement initial
  useEffect(() => {
    fetchSentimentData();
    fetchSystemStatus();
    
    // Actualisation automatique toutes les 5 minutes
    const interval = setInterval(() => {
      fetchSentimentData();
    }, 300000);

    return () => clearInterval(interval);
  }, [selectedSymbol]);

  const fetchSentimentData = async () => {
    setIsLoading(true);
    try {
      // Sentiment actuel pour le symbole sélectionné
      const sentimentResponse = await axios.get(
        `${backendUrl}/api/sentiment/current/${selectedSymbol}`
      );
      
      if (sentimentResponse.data.success) {
        setSentimentData(prev => ({
          ...prev,
          [selectedSymbol]: sentimentResponse.data.data
        }));
      }

      // Corrélation sentiment-prix
      const correlationResponse = await axios.get(
        `${backendUrl}/api/sentiment/correlation/${selectedSymbol}`
      );
      
      if (correlationResponse.data.success) {
        setCorrelationData(prev => ({
          ...prev,
          [selectedSymbol]: correlationResponse.data.data
        }));
      }

      setLastUpdate(new Date());
    } catch (error) {
      console.error('Erreur lors du chargement des données sentiment:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchSystemStatus = async () => {
    try {
      const response = await axios.get(`${backendUrl}/api/sentiment/status`);
      if (response.data) {
        setSystemStatus(response.data);
      }
    } catch (error) {
      console.error('Erreur status système sentiment:', error);
    }
  };

  const startCollection = async () => {
    try {
      setIsLoading(true);
      const response = await axios.post(`${backendUrl}/api/sentiment/start`);
      if (response.data.success) {
        await fetchSystemStatus();
        setTimeout(fetchSentimentData, 2000); // Rafraîchir après 2s
      }
    } catch (error) {
      console.error('Erreur démarrage collecte:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const getSentimentColor = (score) => {
    if (score > 0.2) return 'text-green-500';
    if (score < -0.2) return 'text-red-500';
    return 'text-yellow-500';
  };

  const getSentimentIcon = (label) => {
    switch(label) {
      case 'positive': return <TrendingUp className="w-5 h-5 text-green-500" />;
      case 'negative': return <TrendingDown className="w-5 h-5 text-red-500" />;
      default: return <Activity className="w-5 h-5 text-yellow-500" />;
    }
  };

  const formatPercentage = (value) => {
    return `${(value * 100).toFixed(1)}%`;
  };

  const currentSentiment = sentimentData[selectedSymbol] || {};
  const currentCorrelation = correlationData[selectedSymbol] || {};
  const isSystemRunning = systemStatus?.system?.running || false;

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-900 to-indigo-900 p-6">
        <div className="max-w-7xl mx-auto">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-3xl font-bold text-white flex items-center gap-3">
                <MessageSquare className="w-8 h-8" />
                Analyse Sentiment
              </h1>
              <p className="text-gray-300 mt-2">
                Phase 3.2 - Sentiment Analysis temps réel avec corrélation prix
              </p>
            </div>
            
            <div className="flex items-center gap-4">
              <button
                onClick={fetchSentimentData}
                disabled={isLoading}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg flex items-center gap-2 transition-colors disabled:opacity-50"
              >
                <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
                Actualiser
              </button>
              
              {!isSystemRunning && (
                <button
                  onClick={startCollection}
                  disabled={isLoading}
                  className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg flex items-center gap-2 transition-colors disabled:opacity-50"
                >
                  <Zap className="w-4 h-4" />
                  Démarrer Collecte
                </button>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto p-6 space-y-6">
        {/* Sélecteur de symbole */}
        <div className="bg-gray-800 rounded-xl p-6">
          <h2 className="text-xl font-semibold mb-4">Symbole à Analyser</h2>
          <div className="flex gap-2">
            {symbols.map(symbol => (
              <button
                key={symbol}
                onClick={() => setSelectedSymbol(symbol)}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  selectedSymbol === symbol
                    ? 'bg-purple-600 text-white'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
              >
                {symbol}
              </button>
            ))}
          </div>
        </div>

        {/* Status système */}
        <div className="bg-gray-800 rounded-xl p-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Activity className="w-5 h-5" />
            Status Système Sentiment
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-gray-700 rounded-lg p-4">
              <div className="flex items-center gap-2">
                <CheckCircle className={`w-5 h-5 ${isSystemRunning ? 'text-green-500' : 'text-red-500'}`} />
                <span className="font-medium">Collecte</span>
              </div>
              <p className="text-sm text-gray-400 mt-1">
                {isSystemRunning ? 'Active' : 'Inactive'}
              </p>
            </div>
            
            <div className="bg-gray-700 rounded-lg p-4">
              <div className="flex items-center gap-2">
                <Newspaper className="w-5 h-5 text-blue-500" />
                <span className="font-medium">News</span>
              </div>
              <p className="text-sm text-gray-400 mt-1">
                {systemStatus?.components?.news_collector?.status || 'Unknown'}
              </p>
            </div>
            
            <div className="bg-gray-700 rounded-lg p-4">
              <div className="flex items-center gap-2">
                <Users className="w-5 h-5 text-green-500" />
                <span className="font-medium">Social</span>
              </div>
              <p className="text-sm text-gray-400 mt-1">
                {systemStatus?.components?.social_collector?.status || 'Unknown'}
              </p>
            </div>
            
            <div className="bg-gray-700 rounded-lg p-4">
              <div className="flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-purple-500" />
                <span className="font-medium">Analyse</span>
              </div>
              <p className="text-sm text-gray-400 mt-1">
                {systemStatus?.components?.sentiment_analyzer?.status || 'Unknown'}
              </p>
            </div>
          </div>
        </div>

        {/* Sentiment actuel */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-gray-800 rounded-xl p-6">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              {getSentimentIcon(currentSentiment.sentiment_label)}
              Sentiment Actuel - {selectedSymbol}
            </h2>
            
            {currentSentiment.sentiment_score !== undefined ? (
              <div className="space-y-4">
                <div className="bg-gray-700 rounded-lg p-4">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400">Score Sentiment</span>
                    <span className={`text-xl font-bold ${getSentimentColor(currentSentiment.sentiment_score)}`}>
                      {currentSentiment.sentiment_score.toFixed(3)}
                    </span>
                  </div>
                  
                  <div className="mt-2 bg-gray-600 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full transition-all duration-300 ${
                        currentSentiment.sentiment_score > 0 ? 'bg-green-500' : 'bg-red-500'
                      }`}
                      style={{
                        width: `${Math.min(Math.abs(currentSentiment.sentiment_score) * 100, 100)}%`
                      }}
                    />
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-gray-700 rounded-lg p-3">
                    <p className="text-sm text-gray-400">Label</p>
                    <p className="font-semibold capitalize">
                      {currentSentiment.sentiment_label}
                    </p>
                  </div>
                  
                  <div className="bg-gray-700 rounded-lg p-3">
                    <p className="text-sm text-gray-400">Confiance</p>
                    <p className="font-semibold">
                      {formatPercentage(currentSentiment.confidence || 0)}
                    </p>
                  </div>
                  
                  <div className="bg-gray-700 rounded-lg p-3">
                    <p className="text-sm text-gray-400">Volume</p>
                    <p className="font-semibold">
                      {currentSentiment.volume || 0} posts
                    </p>
                  </div>
                  
                  <div className="bg-gray-700 rounded-lg p-3">
                    <p className="text-sm text-gray-400">Index</p>
                    <p className={`font-semibold ${getSentimentColor(currentSentiment.sentiment_index / 100)}`}>
                      {currentSentiment.sentiment_index?.toFixed(1) || '0.0'}
                    </p>
                  </div>
                </div>
                
                {/* Distribution sentiment */}
                {currentSentiment.distribution && (
                  <div className="bg-gray-700 rounded-lg p-4">
                    <h3 className="font-medium mb-3">Distribution Sentiment</h3>
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-green-400">Positif</span>
                        <span>{currentSentiment.distribution.positive?.toFixed(1) || 0}%</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-gray-400">Neutre</span>
                        <span>{currentSentiment.distribution.neutral?.toFixed(1) || 0}%</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-red-400">Négatif</span>
                        <span>{currentSentiment.distribution.negative?.toFixed(1) || 0}%</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-8">
                <AlertCircle className="w-12 h-12 text-gray-500 mx-auto mb-4" />
                <p className="text-gray-400">
                  {isLoading ? 'Chargement...' : 'Aucune donnée sentiment disponible'}
                </p>
              </div>
            )}
          </div>

          {/* Corrélation sentiment-prix */}
          <div className="bg-gray-800 rounded-xl p-6">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <BarChart3 className="w-5 h-5" />
              Corrélation Sentiment-Prix
            </h2>
            
            {currentCorrelation.current?.correlation !== undefined ? (
              <div className="space-y-4">
                <div className="bg-gray-700 rounded-lg p-4">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400">Corrélation</span>
                    <span className={`text-xl font-bold ${
                      Math.abs(currentCorrelation.current.correlation) >= 0.6 ? 'text-green-500' : 
                      Math.abs(currentCorrelation.current.correlation) >= 0.3 ? 'text-yellow-500' : 'text-red-500'
                    }`}>
                      {currentCorrelation.current.correlation.toFixed(3)}
                    </span>
                  </div>
                  
                  <div className="mt-2">
                    <div className="flex justify-between text-sm text-gray-400 mb-1">
                      <span>-1.0</span>
                      <span>Objectif: 0.6</span>
                      <span>1.0</span>
                    </div>
                    <div className="bg-gray-600 rounded-full h-2 relative">
                      <div
                        className={`absolute top-0 h-2 rounded-full ${
                          currentCorrelation.current.correlation > 0 ? 'bg-green-500' : 'bg-red-500'
                        }`}
                        style={{
                          left: currentCorrelation.current.correlation > 0 ? '50%' : 
                                `${50 + (currentCorrelation.current.correlation * 50)}%`,
                          width: `${Math.abs(currentCorrelation.current.correlation) * 50}%`
                        }}
                      />
                      {/* Ligne objectif */}
                      <div className="absolute top-0 left-[80%] w-0.5 h-2 bg-yellow-400" />
                    </div>
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-gray-700 rounded-lg p-3">
                    <p className="text-sm text-gray-400">P-Value</p>
                    <p className="font-semibold">
                      {currentCorrelation.current.p_value.toFixed(4)}
                    </p>
                  </div>
                  
                  <div className="bg-gray-700 rounded-lg p-3">
                    <p className="text-sm text-gray-400">Échantillon</p>
                    <p className="font-semibold">
                      {currentCorrelation.current.sample_size} points
                    </p>
                  </div>
                  
                  <div className="bg-gray-700 rounded-lg p-3">
                    <p className="text-sm text-gray-400">Sentiment Moy.</p>
                    <p className={`font-semibold ${getSentimentColor(currentCorrelation.current.sentiment_avg)}`}>
                      {currentCorrelation.current.sentiment_avg.toFixed(3)}
                    </p>
                  </div>
                  
                  <div className="bg-gray-700 rounded-lg p-3">
                    <p className="text-sm text-gray-400">Variation Prix</p>
                    <p className={`font-semibold ${
                      currentCorrelation.current.price_change > 0 ? 'text-green-500' : 'text-red-500'
                    }`}>
                      {currentCorrelation.current.price_change.toFixed(2)}%
                    </p>
                  </div>
                </div>
                
                {/* Statuts validation */}
                <div className="space-y-2">
                  <div className={`flex items-center gap-2 p-2 rounded ${
                    currentCorrelation.current.meets_target ? 'bg-green-900/30' : 'bg-red-900/30'
                  }`}>
                    {currentCorrelation.current.meets_target ? 
                      <CheckCircle className="w-4 h-4 text-green-500" /> : 
                      <AlertCircle className="w-4 h-4 text-red-500" />
                    }
                    <span className="text-sm">
                      Objectif corrélation >0.6: {currentCorrelation.current.meets_target ? 'Atteint' : 'Non atteint'}
                    </span>
                  </div>
                  
                  <div className={`flex items-center gap-2 p-2 rounded ${
                    currentCorrelation.current.is_significant ? 'bg-green-900/30' : 'bg-yellow-900/30'
                  }`}>
                    {currentCorrelation.current.is_significant ? 
                      <CheckCircle className="w-4 h-4 text-green-500" /> : 
                      <AlertCircle className="w-4 h-4 text-yellow-500" />
                    }
                    <span className="text-sm">
                      Significativité statistique: {currentCorrelation.current.is_significant ? 'Validée' : 'Non validée'}
                    </span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center py-8">
                <BarChart3 className="w-12 h-12 text-gray-500 mx-auto mb-4" />
                <p className="text-gray-400">
                  {isLoading ? 'Calcul corrélation...' : 'Aucune corrélation calculée'}
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Sources de données */}
        {currentSentiment.sources && (
          <div className="bg-gray-800 rounded-xl p-6">
            <h2 className="text-xl font-semibold mb-4">Sources de Données</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-gray-700 rounded-lg p-4 flex items-center gap-3">
                <Newspaper className="w-6 h-6 text-blue-500" />
                <div>
                  <p className="font-medium">Articles News</p>
                  <p className="text-sm text-gray-400">
                    {currentSentiment.sources.news} articles collectés
                  </p>
                </div>
              </div>
              
              <div className="bg-gray-700 rounded-lg p-4 flex items-center gap-3">
                <Users className="w-6 h-6 text-green-500" />
                <div>
                  <p className="font-medium">Posts Sociaux</p>
                  <p className="text-sm text-gray-400">
                    {currentSentiment.sources.social} posts collectés
                  </p>
                </div>
              </div>
            </div>
            
            {lastUpdate && (
              <p className="text-sm text-gray-400 mt-4">
                Dernière mise à jour: {lastUpdate.toLocaleTimeString()}
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default SentimentAnalysis;