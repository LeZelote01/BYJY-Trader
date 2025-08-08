// Signal Dashboard Component for BYJY-Trader
// Phase 2.2 - Trading signals display and management

import React, { useState, useEffect } from 'react';
import { Target, TrendingUp, TrendingDown, AlertTriangle, Shield, Clock, Zap } from 'lucide-react';

const SignalDashboard = ({ symbols }) => {
  const [signals, setSignals] = useState({});
  const [loading, setLoading] = useState(false);
  const [lastUpdate, setLastUpdate] = useState(null);

  useEffect(() => {
    if (symbols && symbols.length > 0) {
      fetchSignals();
      const interval = setInterval(fetchSignals, 120000); // Update every 2 minutes
      return () => clearInterval(interval);
    }
  }, [symbols]);

  const fetchSignals = async () => {
    if (!symbols || symbols.length === 0) return;

    try {
      setLoading(true);
      const response = await fetch(
        `${process.env.REACT_APP_BACKEND_URL}/api/ai/signals/batch`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            symbols: symbols,
            horizons: ['1h', '4h', '1d'],
            model: 'lstm'
          }),
        }
      );

      if (response.ok) {
        const data = await response.json();
        setSignals(data.signals || {});
        setLastUpdate(new Date());
      } else {
        console.error('Failed to fetch signals');
      }
    } catch (error) {
      console.error('Error fetching signals:', error);
    } finally {
      setLoading(false);
    }
  };

  const getSignalColor = (signal) => {
    switch (signal?.toUpperCase()) {
      case 'STRONG_BUY':
        return 'bg-green-600 text-white';
      case 'BUY':
        return 'bg-green-400 text-white';
      case 'HOLD':
        return 'bg-yellow-400 text-black';
      case 'SELL':
        return 'bg-red-400 text-white';
      case 'STRONG_SELL':
        return 'bg-red-600 text-white';
      default:
        return 'bg-gray-400 text-white';
    }
  };

  const getSignalIcon = (signal) => {
    switch (signal?.toUpperCase()) {
      case 'STRONG_BUY':
      case 'BUY':
        return <TrendingUp className="h-5 w-5" />;
      case 'STRONG_SELL':
      case 'SELL':
        return <TrendingDown className="h-5 w-5" />;
      case 'HOLD':
        return <Target className="h-5 w-5" />;
      default:
        return <AlertTriangle className="h-5 w-5" />;
    }
  };

  const getRiskColor = (riskLevel) => {
    switch (riskLevel?.toUpperCase()) {
      case 'LOW':
        return 'text-green-600 bg-green-100';
      case 'MEDIUM':
        return 'text-yellow-600 bg-yellow-100';
      case 'HIGH':
        return 'text-red-600 bg-red-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  const getRiskIcon = (riskLevel) => {
    switch (riskLevel?.toUpperCase()) {
      case 'LOW':
        return <Shield className="h-4 w-4 text-green-600" />;
      case 'MEDIUM':
        return <AlertTriangle className="h-4 w-4 text-yellow-600" />;
      case 'HIGH':
        return <AlertTriangle className="h-4 w-4 text-red-600" />;
      default:
        return <Shield className="h-4 w-4 text-gray-600" />;
    }
  };

  const formatConfidence = (confidence) => {
    if (confidence === undefined || confidence === null) return 'N/A';
    return `${(confidence * 100).toFixed(1)}%`;
  };

  const formatStrength = (strength) => {
    if (strength === undefined || strength === null) return 'N/A';
    return `${strength.toFixed(1)}`;
  };

  if (loading && Object.keys(signals).length === 0) {
    return (
      <div className="space-y-4">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="bg-white border border-gray-200 rounded-lg p-6 animate-pulse">
            <div className="flex items-center justify-between mb-4">
              <div className="h-6 bg-gray-200 rounded w-20"></div>
              <div className="h-6 bg-gray-200 rounded w-24"></div>
            </div>
            <div className="space-y-2">
              <div className="h-4 bg-gray-200 rounded"></div>
              <div className="h-4 bg-gray-200 rounded w-3/4"></div>
            </div>
          </div>
        ))}
      </div>
    );
  }

  if (Object.keys(signals).length === 0) {
    return (
      <div className="bg-gray-50 rounded-lg p-8 text-center">
        <Target className="h-12 w-12 text-gray-400 mx-auto mb-4" />
        <p className="text-gray-600">Aucun signal disponible</p>
        <button
          onClick={fetchSignals}
          className="mt-4 px-4 py-2 text-sm font-medium text-blue-600 hover:text-blue-800"
        >
          Générer des signaux
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header with refresh */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Target className="h-5 w-5 text-gray-600" />
          <span className="text-sm text-gray-600">
            {lastUpdate && `Dernière mise à jour: ${lastUpdate.toLocaleTimeString('fr-FR')}`}
          </span>
        </div>
        <button
          onClick={fetchSignals}
          disabled={loading}
          className="flex items-center px-3 py-1 text-sm font-medium text-blue-600 hover:text-blue-800 disabled:opacity-50"
        >
          <Zap className={`h-4 w-4 mr-1 ${loading ? 'animate-spin' : ''}`} />
          Actualiser
        </button>
      </div>

      {/* Signals Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {Object.entries(signals).map(([symbol, signalData]) => (
          <div key={symbol} className="bg-white border border-gray-200 rounded-lg p-6">
            {signalData.error ? (
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-gray-900">{symbol}</h3>
                <div className="bg-red-100 text-red-700 px-3 py-1 rounded-full text-sm">
                  Erreur
                </div>
              </div>
            ) : (
              <>
                {/* Signal Header */}
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center">
                    <h3 className="text-lg font-semibold text-gray-900 mr-3">{symbol}</h3>
                    <div className={`flex items-center px-3 py-1 rounded-full text-sm font-medium ${getSignalColor(signalData.signal)}`}>
                      {getSignalIcon(signalData.signal)}
                      <span className="ml-2">{signalData.signal}</span>
                    </div>
                  </div>
                  
                  {/* Risk Level */}
                  <div className={`flex items-center px-2 py-1 rounded-full text-sm font-medium ${getRiskColor(signalData.risk_level)}`}>
                    {getRiskIcon(signalData.risk_level)}
                    <span className="ml-1">{signalData.risk_level}</span>
                  </div>
                </div>

                {/* Signal Metrics */}
                <div className="grid grid-cols-3 gap-4 mb-4">
                  <div className="text-center">
                    <p className="text-sm text-gray-600">Force</p>
                    <p className="text-lg font-semibold text-gray-900">
                      {formatStrength(signalData.signal_strength)}
                    </p>
                  </div>
                  <div className="text-center">
                    <p className="text-sm text-gray-600">Confiance</p>
                    <p className="text-lg font-semibold text-gray-900">
                      {formatConfidence(signalData.confidence)}
                    </p>
                  </div>
                  <div className="text-center">
                    <p className="text-sm text-gray-600">Risque</p>
                    <p className="text-lg font-semibold text-gray-900">
                      {signalData.risk_score ? `${(signalData.risk_score * 100).toFixed(0)}%` : 'N/A'}
                    </p>
                  </div>
                </div>

                {/* Recommendations */}
                {signalData.recommendations && (
                  <div className="bg-blue-50 rounded-lg p-4 mb-4">
                    <h4 className="text-sm font-medium text-blue-900 mb-2">Recommandations</h4>
                    <div className="space-y-2 text-sm text-blue-800">
                      <div className="flex justify-between">
                        <span>Taille position:</span>
                        <span className="font-medium capitalize">
                          {signalData.recommendations.position_size}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>Stop Loss:</span>
                        <span className="font-medium">
                          {signalData.recommendations.stop_loss_percent}%
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>Take Profit:</span>
                        <span className="font-medium">
                          {signalData.recommendations.take_profit_percent}%
                        </span>
                      </div>
                    </div>
                  </div>
                )}

                {/* Notes */}
                {signalData.recommendations?.notes && signalData.recommendations.notes.length > 0 && (
                  <div className="border-t pt-3">
                    <p className="text-sm text-gray-600 mb-1">Notes importantes:</p>
                    <ul className="text-xs text-gray-700 space-y-1">
                      {signalData.recommendations.notes.map((note, index) => (
                        <li key={index} className="flex items-start">
                          <span className="inline-block w-1 h-1 bg-gray-400 rounded-full mt-2 mr-2 flex-shrink-0"></span>
                          <span>{note}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Timestamp */}
                <div className="flex items-center justify-between mt-4 pt-3 border-t text-xs text-gray-500">
                  <div className="flex items-center">
                    <Clock className="h-3 w-3 mr-1" />
                    <span>
                      {signalData.timestamp ? 
                        new Date(signalData.timestamp).toLocaleString('fr-FR') : 
                        'Heure inconnue'
                      }
                    </span>
                  </div>
                  {signalData.valid_until && (
                    <span>
                      Valide jusqu'à {new Date(signalData.valid_until).toLocaleTimeString('fr-FR')}
                    </span>
                  )}
                </div>
              </>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default SignalDashboard;