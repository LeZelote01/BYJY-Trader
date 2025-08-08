import React, { useState, useEffect } from 'react';
import { 
  Settings, 
  Play, 
  Square, 
  BarChart3, 
  Brain, 
  TrendingUp, 
  Zap, 
  RefreshCw,
  AlertCircle,
  CheckCircle,
  Clock
} from 'lucide-react';

const OptimizationDashboard = () => {
  const [optimizationJobs, setOptimizationJobs] = useState([]);
  const [serviceStatus, setServiceStatus] = useState(null);
  const [newOptimization, setNewOptimization] = useState({
    target_model: 'lstm',
    population_size: 50,
    num_generations: 100,
    crossover_prob: 0.8,
    mutation_prob: 0.1
  });
  const [isLoading, setIsLoading] = useState(true);

  // Load data on component mount
  useEffect(() => {
    loadOptimizationData();
    const interval = setInterval(loadOptimizationData, 10000); // Refresh every 10s
    return () => clearInterval(interval);
  }, []);

  const loadOptimizationData = async () => {
    try {
      // Load service status
      const statusResponse = await fetch(`${process.env.REACT_APP_BACKEND_URL}/api/optimization/status`);
      if (statusResponse.ok) {
        const statusData = await statusResponse.json();
        setServiceStatus(statusData);
      }

      // Load optimization history
      const historyResponse = await fetch(`${process.env.REACT_APP_BACKEND_URL}/api/optimization/genetic/history`);
      if (historyResponse.ok) {
        const historyData = await historyResponse.json();
        setOptimizationJobs(historyData.history || []);
      }

      setIsLoading(false);
    } catch (error) {
      console.error('Error loading optimization data:', error);
      setIsLoading(false);
    }
  };

  const startGeneticOptimization = async () => {
    try {
      const request = {
        parameter_space: {
          parameters: {
            learning_rate: { type: "float", min: 0.001, max: 0.1 },
            batch_size: { type: "int", min: 16, max: 128 },
            neurons: { type: "int", min: 32, max: 512 },
            dropout: { type: "float", min: 0.1, max: 0.5 }
          }
        },
        optimization_config: {
          population_size: newOptimization.population_size,
          num_generations: newOptimization.num_generations,
          crossover_prob: newOptimization.crossover_prob,
          mutation_prob: newOptimization.mutation_prob
        },
        target_model: newOptimization.target_model
      };

      const response = await fetch(`${process.env.REACT_APP_BACKEND_URL}/api/optimization/genetic/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(request)
      });

      if (response.ok) {
        const result = await response.json();
        console.log('Optimization started:', result);
        loadOptimizationData(); // Refresh data
      } else {
        console.error('Failed to start optimization');
      }
    } catch (error) {
      console.error('Error starting optimization:', error);
    }
  };

  const stopOptimization = async (jobId) => {
    try {
      const response = await fetch(`${process.env.REACT_APP_BACKEND_URL}/api/optimization/genetic/stop/${jobId}`, {
        method: 'POST'
      });

      if (response.ok) {
        console.log('Optimization stopped');
        loadOptimizationData(); // Refresh data
      }
    } catch (error) {
      console.error('Error stopping optimization:', error);
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'running':
        return <Clock className="h-4 w-4 text-blue-500" />;
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'failed':
        return <AlertCircle className="h-4 w-4 text-red-500" />;
      case 'stopped':
        return <Square className="h-4 w-4 text-gray-500" />;
      default:
        return <RefreshCw className="h-4 w-4 text-gray-400" />;
    }
  };

  const getStatusBadge = (status) => {
    const colors = {
      running: 'bg-blue-100 text-blue-800',
      completed: 'bg-green-100 text-green-800',
      failed: 'bg-red-100 text-red-800',
      stopped: 'bg-gray-100 text-gray-800'
    };

    return (
      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${colors[status] || 'bg-gray-100 text-gray-800'}`}>
        {getStatusIcon(status)}
        <span className="ml-1">{status.charAt(0).toUpperCase() + status.slice(1)}</span>
      </span>
    );
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <RefreshCw className="h-8 w-8 text-blue-500 animate-spin" />
        <span className="ml-3 text-gray-600">Chargement des optimisations...</span>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 flex items-center">
            <Settings className="h-6 w-6 mr-2 text-blue-600" />
            Optimisation Génétique & Meta-Learning
          </h1>
          <p className="text-gray-600 mt-1">Phase 3.4 - Auto-tuning paramètres modèles IA</p>
        </div>
        <button
          onClick={loadOptimizationData}
          className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
        >
          <RefreshCw className="h-4 w-4 mr-2" />
          Actualiser
        </button>
      </div>

      {/* Service Status */}
      {serviceStatus && (
        <div className="bg-white p-6 rounded-lg shadow border">
          <h2 className="text-lg font-semibold mb-4 flex items-center">
            <BarChart3 className="h-5 w-5 mr-2 text-green-600" />
            Statut Service Optimisation
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-green-50 p-4 rounded-lg">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-green-600 font-medium">Service</p>
                  <p className="text-lg font-semibold text-green-900">
                    {serviceStatus.status === 'healthy' ? 'Opérationnel' : serviceStatus.status}
                  </p>
                </div>
                <CheckCircle className="h-8 w-8 text-green-500" />
              </div>
            </div>

            <div className="bg-blue-50 p-4 rounded-lg">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-blue-600 font-medium">Jobs Actifs</p>
                  <p className="text-lg font-semibold text-blue-900">
                    {serviceStatus.job_statistics.running}
                  </p>
                </div>
                <RefreshCw className="h-8 w-8 text-blue-500" />
              </div>
            </div>

            <div className="bg-purple-50 p-4 rounded-lg">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-purple-600 font-medium">Jobs Terminés</p>
                  <p className="text-lg font-semibold text-purple-900">
                    {serviceStatus.job_statistics.completed}
                  </p>
                </div>
                <CheckCircle className="h-8 w-8 text-purple-500" />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* New Optimization */}
      <div className="bg-white p-6 rounded-lg shadow border">
        <h2 className="text-lg font-semibold mb-4 flex items-center">
          <Play className="h-5 w-5 mr-2 text-blue-600" />
          Nouvelle Optimisation Génétique
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              <Brain className="h-4 w-4 inline mr-1" />
              Modèle Cible
            </label>
            <select
              value={newOptimization.target_model}
              onChange={(e) => setNewOptimization({...newOptimization, target_model: e.target.value})}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="lstm">LSTM Model</option>
              <option value="transformer">Transformer</option>
              <option value="xgboost">XGBoost</option>
              <option value="ensemble">Ensemble Models</option>
              <option value="trading">Trading Strategies</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Taille Population
            </label>
            <input
              type="number"
              min="10"
              max="500"
              value={newOptimization.population_size}
              onChange={(e) => setNewOptimization({...newOptimization, population_size: parseInt(e.target.value)})}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Générations Max
            </label>
            <input
              type="number"
              min="10"
              max="1000"
              value={newOptimization.num_generations}
              onChange={(e) => setNewOptimization({...newOptimization, num_generations: parseInt(e.target.value)})}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Prob. Croisement
            </label>
            <input
              type="number"
              min="0"
              max="1"
              step="0.1"
              value={newOptimization.crossover_prob}
              onChange={(e) => setNewOptimization({...newOptimization, crossover_prob: parseFloat(e.target.value)})}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Prob. Mutation
            </label>
            <input
              type="number"
              min="0"
              max="1"
              step="0.01"
              value={newOptimization.mutation_prob}
              onChange={(e) => setNewOptimization({...newOptimization, mutation_prob: parseFloat(e.target.value)})}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>

        <button
          onClick={startGeneticOptimization}
          className="flex items-center px-6 py-2 bg-green-600 text-white rounded-md hover:bg-green-700"
        >
          <Play className="h-4 w-4 mr-2" />
          Démarrer Optimisation
        </button>
      </div>

      {/* Optimization Jobs */}
      <div className="bg-white p-6 rounded-lg shadow border">
        <h2 className="text-lg font-semibold mb-4 flex items-center">
          <TrendingUp className="h-5 w-5 mr-2 text-purple-600" />
          Historique Optimisations
        </h2>

        {optimizationJobs.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            <Zap className="h-12 w-12 mx-auto mb-4 text-gray-300" />
            <p>Aucune optimisation en cours</p>
            <p className="text-sm mt-1">Démarrez une nouvelle optimisation génétique</p>
          </div>
        ) : (
          <div className="space-y-4">
            {optimizationJobs.map((job, index) => (
              <div key={job.job_id} className="border border-gray-200 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-3">
                      <span className="font-medium text-gray-900">
                        {job.target_model.toUpperCase()} Optimization
                      </span>
                      {getStatusBadge(job.status)}
                    </div>
                    
                    <div className="mt-2 text-sm text-gray-600">
                      <span>Job ID: {job.job_id.substring(0, 8)}...</span>
                      <span className="ml-4">
                        Démarré: {new Date(job.started_at).toLocaleString()}
                      </span>
                      {job.progress !== undefined && (
                        <span className="ml-4">Progression: {job.progress.toFixed(1)}%</span>
                      )}
                    </div>
                  </div>

                  <div className="flex items-center space-x-2">
                    {job.status === 'running' && (
                      <button
                        onClick={() => stopOptimization(job.job_id)}
                        className="flex items-center px-3 py-1 bg-red-100 text-red-700 rounded hover:bg-red-200"
                      >
                        <Square className="h-4 w-4 mr-1" />
                        Arrêter
                      </button>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default OptimizationDashboard;