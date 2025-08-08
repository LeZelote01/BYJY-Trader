import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { AlertTriangle, TrendingUp, TrendingDown, BarChart3, Brain, Clock, Target, Zap } from 'lucide-react';

const EnsemblePredictions = () => {
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedSymbol, setSelectedSymbol] = useState('BTCUSDT');
  const [selectedHorizon, setSelectedHorizon] = useState('1h');
  const [modelPreference, setModelPreference] = useState('ensemble');
  const [ensembleStatus, setEnsembleStatus] = useState(null);

  const symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT'];
  const horizons = [
    { value: '15m', label: '15 minutes' },
    { value: '1h', label: '1 heure' },
    { value: '4h', label: '4 heures' },
    { value: '1d', label: '1 jour' },
    { value: '7d', label: '7 jours' }
  ];
  const models = [
    { value: 'ensemble', label: 'Ensemble (Recommandé)', icon: Brain },
    { value: 'transformer', label: 'Transformer', icon: Zap },
    { value: 'lstm', label: 'LSTM', icon: TrendingUp },
    { value: 'xgboost', label: 'XGBoost', icon: BarChart3 },
    { value: 'best', label: 'Meilleur Modèle', icon: Target }
  ];

  useEffect(() => {
    fetchEnsembleStatus();
  }, []);

  const fetchEnsembleStatus = async () => {
    try {
      const backendUrl = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';
      const response = await fetch(`${backendUrl}/api/ensemble/status`);
      
      if (response.ok) {
        const data = await response.json();
        setEnsembleStatus(data.data);
      }
    } catch (err) {
      console.error('Erreur lors de la récupération du statut ensemble:', err);
    }
  };

  const fetchPrediction = async () => {
    setLoading(true);
    setError(null);

    try {
      const backendUrl = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';
      const response = await fetch(`${backendUrl}/api/ensemble/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          symbol: selectedSymbol,
          horizon: selectedHorizon,
          model_preference: modelPreference,
          confidence_level: 0.95,
          include_individual: true
        }),
      });

      if (!response.ok) {
        throw new Error(`Erreur API: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.success) {
        setPredictions(data.data);
      } else {
        throw new Error(data.message || 'Erreur inconnue');
      }
    } catch (err) {
      setError(err.message);
      console.error('Erreur lors de la prédiction:', err);
    } finally {
      setLoading(false);
    }
  };

  const getPredictionColor = (changePercent) => {
    if (changePercent > 2) return 'text-green-600 bg-green-50';
    if (changePercent > 0) return 'text-green-500 bg-green-50';
    if (changePercent < -2) return 'text-red-600 bg-red-50';
    if (changePercent < 0) return 'text-red-500 bg-red-50';
    return 'text-gray-500 bg-gray-50';
  };

  const getQualityBadge = (quality) => {
    const variants = {
      'HIGH': 'bg-green-100 text-green-800',
      'MEDIUM': 'bg-yellow-100 text-yellow-800',
      'LOW': 'bg-red-100 text-red-800'
    };
    return variants[quality] || variants['MEDIUM'];
  };

  const formatPrice = (price) => {
    return new Intl.NumberFormat('fr-FR', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 6
    }).format(price);
  };

  const formatPercent = (percent) => {
    const sign = percent > 0 ? '+' : '';
    return `${sign}${percent.toFixed(2)}%`;
  };

  return (
    <div className="w-full max-w-7xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2 flex items-center justify-center gap-3">
          <Brain className="w-8 h-8 text-purple-600" />
          Prédictions IA Avancées
        </h1>
        <p className="text-gray-600">
          Système d'ensemble combinant LSTM, Transformer et XGBoost pour des prédictions optimales
        </p>
      </div>

      {/* Status Ensemble */}
      {ensembleStatus && (
        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Brain className="w-5 h-5" />
              Statut du Système d'Ensemble
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center">
                <Badge 
                  variant={ensembleStatus.ensemble_available ? "default" : "secondary"}
                  className={ensembleStatus.ensemble_available ? "bg-green-100 text-green-800" : ""}
                >
                  {ensembleStatus.ensemble_available ? "Ensemble Actif" : "Ensemble Inactif"}
                </Badge>
              </div>
              {Object.entries(ensembleStatus.individual_models || {}).map(([model, info]) => (
                <div key={model} className="text-center">
                  <p className="font-medium capitalize">{model}</p>
                  <Badge variant={info.trained ? "default" : "secondary"}>
                    {info.trained ? "Entraîné" : "Non Entraîné"}
                  </Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Configuration des Prédictions */}
      <Card>
        <CardHeader>
          <CardTitle>Configuration des Prédictions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
            {/* Sélection du Symbole */}
            <div>
              <label className="block text-sm font-medium mb-2">Cryptomonnaie</label>
              <select
                value={selectedSymbol}
                onChange={(e) => setSelectedSymbol(e.target.value)}
                className="w-full p-2 border rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                {symbols.map((symbol) => (
                  <option key={symbol} value={symbol}>{symbol}</option>
                ))}
              </select>
            </div>

            {/* Sélection de l'Horizon */}
            <div>
              <label className="block text-sm font-medium mb-2">Horizon de Prédiction</label>
              <select
                value={selectedHorizon}
                onChange={(e) => setSelectedHorizon(e.target.value)}
                className="w-full p-2 border rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                {horizons.map((horizon) => (
                  <option key={horizon.value} value={horizon.value}>
                    {horizon.label}
                  </option>
                ))}
              </select>
            </div>

            {/* Sélection du Modèle */}
            <div>
              <label className="block text-sm font-medium mb-2">Modèle IA</label>
              <select
                value={modelPreference}
                onChange={(e) => setModelPreference(e.target.value)}
                className="w-full p-2 border rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                {models.map((model) => (
                  <option key={model.value} value={model.value}>
                    {model.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <Button
            onClick={fetchPrediction}
            disabled={loading}
            className="w-full md:w-auto"
          >
            {loading ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                Analyse en cours...
              </>
            ) : (
              <>
                <Brain className="w-4 h-4 mr-2" />
                Générer la Prédiction IA
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      {/* Erreur */}
      {error && (
        <Card className="border-red-200 bg-red-50">
          <CardContent className="pt-6">
            <div className="flex items-center gap-2 text-red-800">
              <AlertTriangle className="w-5 h-5" />
              <p>Erreur: {error}</p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Résultats des Prédictions */}
      {predictions && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Prédiction Principale */}
          <Card className="lg:col-span-2">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Target className="w-5 h-5 text-blue-600" />
                Prédiction Principale - {predictions.primary_model?.toUpperCase()}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {/* Prix Actuel vs Prédiction */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="text-center p-4 bg-gray-50 rounded-lg">
                    <p className="text-sm text-gray-600 mb-1">Prix Actuel</p>
                    <p className="text-2xl font-bold text-gray-900">
                      ${formatPrice(predictions.current_price)}
                    </p>
                  </div>
                  <div className={`text-center p-4 rounded-lg ${getPredictionColor(predictions.price_change_percent)}`}>
                    <p className="text-sm mb-1">Prédiction</p>
                    <p className="text-2xl font-bold">
                      ${formatPrice(predictions.predicted_price)}
                    </p>
                    <p className="text-sm font-medium">
                      {formatPercent(predictions.price_change_percent)}
                    </p>
                  </div>
                </div>

                {/* Métriques de Qualité */}
                <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                  <div className="text-center">
                    <Badge className={getQualityBadge(predictions.prediction_quality)}>
                      {predictions.prediction_quality}
                    </Badge>
                    <p className="text-xs text-gray-600 mt-1">Qualité</p>
                  </div>
                  
                  {predictions.enhanced_metrics && (
                    <>
                      <div className="text-center">
                        <p className="font-bold text-lg">
                          {(predictions.enhanced_metrics.model_agreement * 100).toFixed(0)}%
                        </p>
                        <p className="text-xs text-gray-600">Consensus</p>
                      </div>
                      <div className="text-center">
                        <Badge variant="outline">
                          {predictions.enhanced_metrics.market_regime}
                        </Badge>
                        <p className="text-xs text-gray-600 mt-1">Régime</p>
                      </div>
                    </>
                  )}
                </div>

                {/* Intervalle de Confiance */}
                {predictions.confidence_interval && (
                  <div className="bg-gray-50 p-3 rounded-lg">
                    <p className="text-sm font-medium mb-2">Intervalle de Confiance (95%)</p>
                    <div className="flex justify-between items-center text-sm">
                      <span>Min: ${formatPrice(predictions.confidence_interval.lower)}</span>
                      <span>Max: ${formatPrice(predictions.confidence_interval.upper)}</span>
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Métadonnées */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Clock className="w-5 h-5 text-green-600" />
                Détails de l'Analyse
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3 text-sm">
                <div>
                  <p className="font-medium">Symbole:</p>
                  <p className="text-gray-600">{predictions.symbol}</p>
                </div>
                <div>
                  <p className="font-medium">Horizon:</p>
                  <p className="text-gray-600">{predictions.horizon} ({predictions.horizon_minutes} min)</p>
                </div>
                <div>
                  <p className="font-medium">Points de Données:</p>
                  <p className="text-gray-600">{predictions.data_points_used}</p>
                </div>
                <div>
                  <p className="font-medium">Version IA:</p>
                  <p className="text-gray-600">v{predictions.prediction_version}</p>
                </div>
                
                {predictions.ensemble_details && predictions.ensemble_details.available && (
                  <div className="border-t pt-3">
                    <p className="font-medium mb-2">Ensemble:</p>
                    <div className="space-y-1 text-xs">
                      <p>Méthode: {predictions.ensemble_details.fusion_method}</p>
                      {predictions.ensemble_details.model_weights && (
                        <div>
                          <p>Poids des Modèles:</p>
                          {Object.entries(predictions.ensemble_details.model_weights).map(([model, weight]) => (
                            <p key={model} className="ml-2">
                              {model}: {(weight * 100).toFixed(1)}%
                            </p>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Prédictions Individuelles */}
          {predictions.individual_predictions && (
            <Card className="lg:col-span-3">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BarChart3 className="w-5 h-5 text-purple-600" />
                  Comparaison des Modèles Individuels
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {Object.entries(predictions.individual_predictions).map(([modelName, pred]) => {
                    const ModelIcon = models.find(m => m.value === modelName)?.icon || Brain;
                    
                    return (
                      <div key={modelName} className="border rounded-lg p-4">
                        <div className="flex items-center gap-2 mb-3">
                          <ModelIcon className="w-4 h-4" />
                          <h4 className="font-medium capitalize">{modelName}</h4>
                          <Badge className={getQualityBadge(pred.prediction_quality)} size="sm">
                            {pred.prediction_quality}
                          </Badge>
                        </div>
                        
                        <div className="space-y-2">
                          <div className={`p-2 rounded text-center ${getPredictionColor(pred.price_change_percent)}`}>
                            <p className="font-bold">${formatPrice(pred.predicted_price)}</p>
                            <p className="text-sm">{formatPercent(pred.price_change_percent)}</p>
                          </div>
                          
                          {pred.confidence_interval && (
                            <div className="text-xs text-gray-600">
                              <p>Confiance: ${formatPrice(pred.confidence_interval.lower)} - ${formatPrice(pred.confidence_interval.upper)}</p>
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      )}
    </div>
  );
};

export default EnsemblePredictions;