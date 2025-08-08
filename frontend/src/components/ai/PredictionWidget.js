// Prediction Widget Component for BYJY-Trader
// Phase 2.2 - Real-time price predictions widget

import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, Clock, AlertCircle, Zap } from 'lucide-react';

const PredictionWidget = ({ symbol }) => {
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedHorizon, setSelectedHorizon] = useState('1h');

  const horizons = [
    { value: '15m', label: '15min' },
    { value: '1h', label: '1h' },
    { value: '4h', label: '4h' },
    { value: '1d', label: '1 jour' }
  ];

  useEffect(() => {
    fetchPredictions();
    const interval = setInterval(fetchPredictions, 60000); // Update every minute
    return () => clearInterval(interval);
  }, [symbol]);

  const fetchPredictions = async () => {
    try {
      setLoading(true);
      const response = await fetch(
        `${process.env.REACT_APP_BACKEND_URL}/api/ai/predictions/multi-horizon`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            symbol: symbol,
            horizons: ['15m', '1h', '4h', '1d'],
            model: 'lstm'
          }),
        }
      );

      if (response.ok) {
        const data = await response.json();
        setPredictions(data);
        setError(null);
      } else {
        const errorData = await response.json();
        setError(errorData.detail || 'Erreur de prédiction');
      }
    } catch (err) {
      setError('Erreur de connexion à l\'API IA');
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  const getPredictionForHorizon = (horizon) => {
    if (!predictions?.predictions?.[horizon]) return null;
    return predictions.predictions[horizon];
  };

  const formatPrice = (price) => {
    if (!price) return 'N/A';
    return new Intl.NumberFormat('fr-FR', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 6,
    }).format(price);
  };

  const formatPercent = (percent) => {
    if (percent === undefined || percent === null) return 'N/A';
    const sign = percent >= 0 ? '+' : '';
    return `${sign}${percent.toFixed(2)}%`;
  };

  const getPredictionTrend = (changePercent) => {
    if (!changePercent) return 'neutral';
    return changePercent > 0 ? 'up' : 'down';
  };

  const getTrendColor = (trend) => {
    switch (trend) {
      case 'up': return 'text-green-600';
      case 'down': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  const getTrendIcon = (trend) => {
    switch (trend) {
      case 'up': return <TrendingUp className="h-5 w-5" />;
      case 'down': return <TrendingDown className="h-5 w-5" />;
      default: return <Clock className="h-5 w-5" />;
    }
  };

  const getQualityColor = (quality) => {
    switch (quality?.toLowerCase()) {
      case 'high': return 'bg-green-100 text-green-800';
      case 'medium': return 'bg-yellow-100 text-yellow-800';
      case 'low': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="animate-pulse">
          <div className="flex items-center justify-between mb-4">
            <div className="h-4 bg-gray-200 rounded w-16"></div>
            <div className="h-6 bg-gray-200 rounded w-20"></div>
          </div>
          <div className="space-y-3">
            <div className="h-4 bg-gray-200 rounded"></div>
            <div className="h-4 bg-gray-200 rounded w-3/4"></div>
            <div className="h-4 bg-gray-200 rounded w-1/2"></div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">{symbol}</h3>
          <AlertCircle className="h-6 w-6 text-red-500" />
        </div>
        <div className="bg-red-50 border border-red-200 rounded-md p-4">
          <p className="text-red-700 text-sm">{error}</p>
          <button
            onClick={fetchPredictions}
            className="mt-2 text-red-600 hover:text-red-800 text-sm font-medium"
          >
            Réessayer
          </button>
        </div>
      </div>
    );
  }

  const currentPrediction = getPredictionForHorizon(selectedHorizon);
  const trend = getPredictionTrend(currentPrediction?.price_change_percent);

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center">
          <h3 className="text-lg font-semibold text-gray-900 mr-2">{symbol}</h3>
          {currentPrediction && (
            <span className={`px-2 py-1 rounded-full text-xs font-medium ${getQualityColor(currentPrediction.prediction_quality)}`}>
              {currentPrediction.prediction_quality || 'N/A'}
            </span>
          )}
        </div>
        <div className={`flex items-center ${getTrendColor(trend)}`}>
          {getTrendIcon(trend)}
          <Zap className="h-4 w-4 ml-1" />
        </div>
      </div>

      {/* Horizon Selection */}
      <div className="flex space-x-2 mb-4">
        {horizons.map((horizon) => (
          <button
            key={horizon.value}
            onClick={() => setSelectedHorizon(horizon.value)}
            className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${
              selectedHorizon === horizon.value
                ? 'bg-blue-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            {horizon.label}
          </button>
        ))}
      </div>

      {/* Current Prediction */}
      {currentPrediction && !currentPrediction.error ? (
        <div className="space-y-4">
          {/* Price Information */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-sm font-medium text-gray-600">Prix Actuel</p>
              <p className="text-xl font-bold text-gray-900">
                {formatPrice(currentPrediction.current_price)}
              </p>
            </div>
            <div>
              <p className="text-sm font-medium text-gray-600">Prix Prédit</p>
              <p className={`text-xl font-bold ${getTrendColor(trend)}`}>
                {formatPrice(currentPrediction.predicted_price)}
              </p>
            </div>
          </div>

          {/* Change Information */}
          <div className="border-t pt-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm font-medium text-gray-600">Variation</p>
                <p className={`text-lg font-semibold ${getTrendColor(trend)}`}>
                  {formatPercent(currentPrediction.price_change_percent)}
                </p>
              </div>
              <div>
                <p className="text-sm font-medium text-gray-600">Montant</p>
                <p className={`text-lg font-semibold ${getTrendColor(trend)}`}>
                  {formatPrice(currentPrediction.price_change)}
                </p>
              </div>
            </div>
          </div>

          {/* Confidence Interval */}
          {currentPrediction.confidence_interval && (
            <div className="border-t pt-4">
              <p className="text-sm font-medium text-gray-600 mb-2">Intervalle de Confiance</p>
              <div className="bg-gray-50 rounded-md p-3">
                <div className="flex justify-between text-sm">
                  <span className="text-red-600">
                    Min: {formatPrice(currentPrediction.confidence_interval.lower)}
                  </span>
                  <span className="text-green-600">
                    Max: {formatPrice(currentPrediction.confidence_interval.upper)}
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* Timestamp */}
          <div className="border-t pt-2">
            <p className="text-xs text-gray-500">
              Dernière mise à jour: {new Date(predictions.timestamp).toLocaleString('fr-FR')}
            </p>
          </div>
        </div>
      ) : (
        <div className="bg-yellow-50 border border-yellow-200 rounded-md p-4">
          <p className="text-yellow-700 text-sm">
            {currentPrediction?.error || 'Prédiction non disponible pour cet horizon'}
          </p>
        </div>
      )}

      {/* Quick Refresh */}
      <div className="mt-4 pt-4 border-t">
        <button
          onClick={fetchPredictions}
          className="w-full flex items-center justify-center px-3 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 transition-colors"
        >
          <Zap className="h-4 w-4 mr-2" />
          Actualiser Prédiction
        </button>
      </div>
    </div>
  );
};

export default PredictionWidget;