// AI Dashboard Component for BYJY-Trader
// Phase 2.2 - Main AI dashboard with predictions and signals

import React, { useState, useEffect } from 'react';
import { TrendingUp, Brain, Target, AlertTriangle, Activity, Zap } from 'lucide-react';
import PredictionWidget from './PredictionWidget';
import AIModelStatus from './AIModelStatus';
import SignalDashboard from './SignalDashboard';

const AIDashboard = () => {
  const [aiStatus, setAiStatus] = useState(null);
  const [topSymbols, setTopSymbols] = useState(['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT']);
  const [activeTab, setActiveTab] = useState('predictions');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchAIStatus();
  }, []);

  const fetchAIStatus = async () => {
    try {
      const response = await fetch(`${process.env.REACT_APP_BACKEND_URL}/api/ai/health`);
      if (response.ok) {
        const data = await response.json();
        setAiStatus(data);
      } else {
        console.error('Failed to fetch AI status');
      }
    } catch (error) {
      console.error('Error fetching AI status:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-50 p-6">
        <div className="max-w-7xl mx-auto">
          <div className="bg-white rounded-lg shadow-lg p-8">
            <div className="flex items-center justify-center">
              <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600"></div>
            </div>
            <p className="text-center mt-4 text-gray-600">Initialisation du système IA...</p>
          </div>
        </div>
      </div>
    );
  }

  const getStatusColor = (status) => {
    switch (status) {
      case 'healthy':
      case 'ready':
        return 'text-green-600';
      case 'warning':
        return 'text-yellow-600';
      case 'error':
        return 'text-red-600';
      default:
        return 'text-gray-600';
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg shadow-lg p-6 mb-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <Brain className="h-8 w-8 text-white mr-3" />
              <div>
                <h1 className="text-2xl font-bold text-white">IA Trading Dashboard</h1>
                <p className="text-blue-100">Prédictions avancées et signaux automatiques</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="text-right">
                <p className="text-blue-100 text-sm">Status Système IA</p>
                <p className={`font-bold ${getStatusColor(aiStatus?.status)} bg-white px-3 py-1 rounded-full text-sm`}>
                  {aiStatus?.status || 'Inconnu'}
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* AI System Status Cards */}
        {aiStatus && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center">
                <Activity className="h-8 w-8 text-blue-600 mr-3" />
                <div>
                  <p className="text-sm font-medium text-gray-600">Prédicteur IA</p>
                  <p className="text-2xl font-bold text-gray-900">
                    {aiStatus.ai_components?.predictor || 'N/A'}
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center">
                <Target className="h-8 w-8 text-green-600 mr-3" />
                <div>
                  <p className="text-sm font-medium text-gray-600">Signaux Trading</p>
                  <p className="text-2xl font-bold text-gray-900">
                    {aiStatus.ai_components?.signal_generator || 'N/A'}
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center">
                <Zap className="h-8 w-8 text-purple-600 mr-3" />
                <div>
                  <p className="text-sm font-medium text-gray-600">Modèles Chargés</p>
                  <p className="text-2xl font-bold text-gray-900">
                    {aiStatus.models ? Object.keys(aiStatus.models).length : 0}
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center">
                <TrendingUp className="h-8 w-8 text-orange-600 mr-3" />
                <div>
                  <p className="text-sm font-medium text-gray-600">Cache Prédictions</p>
                  <p className="text-2xl font-bold text-gray-900">
                    {aiStatus.cache_stats?.cached_predictions || 0}
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Navigation Tabs */}
        <div className="bg-white rounded-lg shadow-lg mb-6">
          <div className="border-b border-gray-200">
            <nav className="-mb-px flex space-x-8 px-6" aria-label="Tabs">
              {[
                { id: 'predictions', name: 'Prédictions', icon: TrendingUp },
                { id: 'signals', name: 'Signaux Trading', icon: Target },
                { id: 'models', name: 'Modèles IA', icon: Brain },
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm flex items-center`}
                >
                  <tab.icon className="h-5 w-5 mr-2" />
                  {tab.name}
                </button>
              ))}
            </nav>
          </div>

          {/* Tab Content */}
          <div className="p-6">
            {activeTab === 'predictions' && (
              <div>
                <h3 className="text-lg font-semibold text-gray-900 mb-4">
                  Prédictions Temps Réel - Top Cryptos
                </h3>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {topSymbols.map((symbol) => (
                    <PredictionWidget key={symbol} symbol={symbol} />
                  ))}
                </div>
              </div>
            )}

            {activeTab === 'signals' && (
              <div>
                <h3 className="text-lg font-semibold text-gray-900 mb-4">
                  Signaux de Trading Actifs
                </h3>
                <SignalDashboard symbols={topSymbols} />
              </div>
            )}

            {activeTab === 'models' && (
              <div>
                <h3 className="text-lg font-semibold text-gray-900 mb-4">
                  Status des Modèles IA
                </h3>
                <AIModelStatus aiStatus={aiStatus} />
              </div>
            )}
          </div>
        </div>

        {/* Quick Actions */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Actions Rapides</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <button
              onClick={fetchAIStatus}
              className="flex items-center justify-center px-4 py-3 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 transition-colors"
            >
              <Activity className="h-5 w-5 mr-2" />
              Actualiser Prédictions
            </button>
            
            <button
              onClick={fetchAIStatus}
              className="flex items-center justify-center px-4 py-3 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 transition-colors"
            >
              <Zap className="h-5 w-5 mr-2" />
              Vérifier Status IA
            </button>
            
            <button
              onClick={() => setActiveTab('models')}
              className="flex items-center justify-center px-4 py-3 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 transition-colors"
            >
              <Brain className="h-5 w-5 mr-2" />
              Gérer Modèles
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AIDashboard;