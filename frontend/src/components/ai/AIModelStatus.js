// AI Model Status Component for BYJY-Trader
// Phase 2.2 - Display AI model status and performance

import React, { useState, useEffect } from 'react';
import { Brain, CheckCircle, XCircle, AlertTriangle, RefreshCw, Settings } from 'lucide-react';

const AIModelStatus = ({ aiStatus }) => {
  const [modelDetails, setModelDetails] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (aiStatus?.models) {
      setModelDetails(aiStatus.models);
    }
  }, [aiStatus]);

  const fetchModelStatus = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${process.env.REACT_APP_BACKEND_URL}/api/ai/models/status`);
      
      if (response.ok) {
        const data = await response.json();
        setModelDetails(data.models);
      }
    } catch (error) {
      console.error('Error fetching model status:', error);
    } finally {
      setLoading(false);
    }
  };

  const getStatusIcon = (isActive, isTrained) => {
    if (isActive && isTrained) {
      return <CheckCircle className="h-6 w-6 text-green-500" />;
    } else if (isActive && !isTrained) {
      return <AlertTriangle className="h-6 w-6 text-yellow-500" />;
    } else {
      return <XCircle className="h-6 w-6 text-red-500" />;
    }
  };

  const getStatusText = (isActive, isTrained) => {
    if (isActive && isTrained) {
      return { text: 'Opérationnel', color: 'text-green-600' };
    } else if (isActive && !isTrained) {
      return { text: 'Non Entraîné', color: 'text-yellow-600' };
    } else {
      return { text: 'Inactif', color: 'text-red-600' };
    }
  };

  if (!modelDetails) {
    return (
      <div className="bg-gray-50 rounded-lg p-8 text-center">
        <Brain className="h-12 w-12 text-gray-400 mx-auto mb-4" />
        <p className="text-gray-600">Aucune information sur les modèles disponible</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Refresh Button */}
      <div className="flex justify-between items-center">
        <h4 className="text-lg font-medium text-gray-900">Modèles IA Disponibles</h4>
        <button
          onClick={fetchModelStatus}
          disabled={loading}
          className="flex items-center px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50"
        >
          <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
          Actualiser
        </button>
      </div>

      {/* Models Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {Object.entries(modelDetails).map(([modelName, details]) => {
          const status = getStatusText(details.model_exists, details.is_trained);
          
          return (
            <div key={modelName} className="bg-white border border-gray-200 rounded-lg p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center">
                  <Brain className="h-8 w-8 text-blue-600 mr-3" />
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 capitalize">
                      {modelName.toUpperCase()} Model
                    </h3>
                    <p className="text-sm text-gray-600">Version {details.version}</p>
                  </div>
                </div>
                {getStatusIcon(details.model_exists, details.is_trained)}
              </div>

              {/* Status */}
              <div className="mb-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-600">Status:</span>
                  <span className={`text-sm font-semibold ${status.color}`}>
                    {status.text}
                  </span>
                </div>
              </div>

              {/* Model Details */}
              <div className="space-y-3">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-600">Entraîné:</span>
                  <span className={details.is_trained ? 'text-green-600' : 'text-red-600'}>
                    {details.is_trained ? 'Oui' : 'Non'}
                  </span>
                </div>

                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-600">Modèle chargé:</span>
                  <span className={details.model_exists ? 'text-green-600' : 'text-red-600'}>
                    {details.model_exists ? 'Oui' : 'Non'}
                  </span>
                </div>

                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-600">Historique disponible:</span>
                  <span className={details.history_available ? 'text-green-600' : 'text-gray-600'}>
                    {details.history_available ? 'Oui' : 'Non'}
                  </span>
                </div>
              </div>

              {/* Configuration Preview */}
              {details.config && Object.keys(details.config).length > 0 && (
                <div className="mt-4 pt-4 border-t">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-gray-600">Configuration:</span>
                    <Settings className="h-4 w-4 text-gray-400" />
                  </div>
                  <div className="bg-gray-50 rounded-md p-3">
                    <div className="space-y-1">
                      {Object.entries(details.config).slice(0, 3).map(([key, value]) => (
                        <div key={key} className="flex justify-between text-xs">
                          <span className="text-gray-600">{key}:</span>
                          <span className="text-gray-900 font-mono">
                            {typeof value === 'object' ? JSON.stringify(value).substring(0, 20) + '...' : String(value)}
                          </span>
                        </div>
                      ))}
                      {Object.keys(details.config).length > 3 && (
                        <div className="text-xs text-gray-500 text-center">
                          ... et {Object.keys(details.config).length - 3} autres paramètres
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* Action Buttons */}
              <div className="mt-4 pt-4 border-t flex space-x-3">
                {details.is_trained ? (
                  <button className="flex-1 flex items-center justify-center px-3 py-2 text-sm font-medium text-white bg-green-600 rounded-md hover:bg-green-700 transition-colors">
                    <CheckCircle className="h-4 w-4 mr-2" />
                    Opérationnel
                  </button>
                ) : (
                  <button
                    onClick={() => alert('Entraînement disponible dans une future version')}
                    className="flex-1 flex items-center justify-center px-3 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 transition-colors"
                  >
                    <Brain className="h-4 w-4 mr-2" />
                    Entraîner
                  </button>
                )}
                
                <button
                  onClick={() => alert('Configuration avancée disponible dans une future version')}
                  className="px-3 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 transition-colors"
                >
                  <Settings className="h-4 w-4" />
                </button>
              </div>

              {/* File Paths (Development Info) */}
              {details.model_path && (
                <div className="mt-3 pt-3 border-t">
                  <p className="text-xs text-gray-500 break-all">
                    Chemin: {details.model_path}
                  </p>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Cache Statistics */}
      {aiStatus?.cache_stats && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h4 className="text-sm font-medium text-blue-900 mb-2">Statistiques Cache</h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-blue-600">Prédictions en cache:</span>
              <span className="ml-2 font-semibold text-blue-900">
                {aiStatus.cache_stats.cached_predictions || 0}
              </span>
            </div>
            <div>
              <span className="text-blue-600">Taille cache:</span>
              <span className="ml-2 font-semibold text-blue-900">
                {((aiStatus.cache_stats.cache_size_mb || 0) * 1024 * 1024).toFixed(1)} MB
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AIModelStatus;