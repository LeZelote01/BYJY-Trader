import React, { useState, useEffect } from 'react';
import { 
  Bot, 
  Play, 
  Square, 
  Settings,
  TrendingUp,
  Activity,
  Monitor,
  Brain,
  Download,
  Upload,
  AlertCircle,
  CheckCircle,
  BarChart3,
  Zap,
  Target
} from 'lucide-react';

const RLTrading = () => {
  const [activeTrainings, setActiveTrainings] = useState({});
  const [deployedAgents, setDeployedAgents] = useState({});
  const [availableModels, setAvailableModels] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedAgent, setSelectedAgent] = useState('ppo');
  const [trainingConfig, setTrainingConfig] = useState({
    symbol: 'BTCUSDT',
    initial_balance: 10000.0,
    max_episodes: 2000,
    learning_rate: 0.0003,
    reward_function: 'profit_risk'
  });

  const API_BASE_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

  useEffect(() => {
    loadRLData();
    const interval = setInterval(loadRLData, 10000); // Refresh chaque 10s
    return () => clearInterval(interval);
  }, []);

  const loadRLData = async () => {
    try {
      const [trainingRes, agentsRes, modelsRes] = await Promise.all([
        fetch(`${API_BASE_URL}/api/rl/training/status`),
        fetch(`${API_BASE_URL}/api/rl/agents`),
        fetch(`${API_BASE_URL}/api/rl/models`)
      ]);

      if (trainingRes.ok) {
        const trainingData = await trainingRes.json();
        setActiveTrainings(trainingData.active_trainings || {});
      }

      if (agentsRes.ok) {
        const agentsData = await agentsRes.json();
        setDeployedAgents(agentsData.deployed_agents || {});
      }

      if (modelsRes.ok) {
        const modelsData = await modelsRes.json();
        setAvailableModels(modelsData.models || []);
      }
    } catch (error) {
      console.error('Erreur chargement données RL:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const startTraining = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/rl/training/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          agent_type: selectedAgent,
          ...trainingConfig
        }),
      });

      if (response.ok) {
        const result = await response.json();
        console.log('Entraînement démarré:', result);
        loadRLData();
      }
    } catch (error) {
      console.error('Erreur démarrage entraînement:', error);
    }
  };

  const stopTraining = async (trainingId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/rl/training/${trainingId}/stop`, {
        method: 'POST',
      });

      if (response.ok) {
        loadRLData();
      }
    } catch (error) {
      console.error('Erreur arrêt entraînement:', error);
    }
  };

  const deployAgent = async (modelPath, agentName) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/rl/agents/deploy`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          agent_name: agentName,
          model_path: modelPath,
          symbol: 'BTCUSDT',
          initial_balance: 10000.0,
          max_position_size: 1.0
        }),
      });

      if (response.ok) {
        loadRLData();
      }
    } catch (error) {
      console.error('Erreur déploiement agent:', error);
    }
  };

  const StatusBadge = ({ status }) => {
    const colors = {
      running: 'bg-blue-100 text-blue-800',
      active: 'bg-green-100 text-green-800',
      stopped: 'bg-gray-100 text-gray-800',
      error: 'bg-red-100 text-red-800'
    };
    
    return (
      <span className={`px-2 py-1 rounded-full text-xs font-medium ${colors[status] || colors.stopped}`}>
        {status}
      </span>
    );
  };

  const MetricCard = ({ title, value, icon: Icon, color = "blue" }) => (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className={`text-lg font-semibold text-${color}-600`}>{value}</p>
        </div>
        <Icon className={`w-8 h-8 text-${color}-500`} />
      </div>
    </div>
  );

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-96">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6 bg-gray-900 min-h-screen p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white flex items-center">
            <Bot className="w-8 h-8 mr-3 text-purple-500" />
            Reinforcement Learning Trading
          </h1>
          <p className="text-gray-400 mt-2">Phase 3.3 - Agents RL Autonomes</p>
        </div>
      </div>

      {/* Métriques Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-gray-800 rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">Entraînements Actifs</p>
              <p className="text-lg font-semibold text-blue-400">{Object.keys(activeTrainings).length}</p>
            </div>
            <Activity className="w-8 h-8 text-blue-500" />
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">Agents Déployés</p>
              <p className="text-lg font-semibold text-green-400">{Object.keys(deployedAgents).length}</p>
            </div>
            <Bot className="w-8 h-8 text-green-500" />
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">Modèles Disponibles</p>
              <p className="text-lg font-semibold text-purple-400">{availableModels.length}</p>
            </div>
            <Brain className="w-8 h-8 text-purple-500" />
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">Performance Totale</p>
              <p className="text-lg font-semibold text-yellow-400">
                {Object.values(deployedAgents).reduce((acc, agent) => acc + (agent.total_return || 0), 0).toFixed(2)}%
              </p>
            </div>
            <TrendingUp className="w-8 h-8 text-yellow-500" />
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Configuration d'Entraînement */}
        <div className="bg-gray-800 rounded-lg shadow">
          <div className="px-6 py-4 border-b border-gray-700">
            <h3 className="text-lg font-semibold text-white flex items-center">
              <Settings className="w-5 h-5 mr-2" />
              Configuration Entraînement
            </h3>
          </div>
          <div className="p-6 space-y-4">
            {/* Agent Type */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">Type d'Agent</label>
              <select
                value={selectedAgent}
                onChange={(e) => setSelectedAgent(e.target.value)}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:ring-2 focus:ring-purple-500 focus:border-transparent"
              >
                <option value="ppo">PPO (Proximal Policy Optimization)</option>
                <option value="a3c">A3C (Asynchronous Advantage Actor-Critic)</option>
              </select>
            </div>

            {/* Symbol */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">Symbole Trading</label>
              <select
                value={trainingConfig.symbol}
                onChange={(e) => setTrainingConfig({...trainingConfig, symbol: e.target.value})}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:ring-2 focus:ring-purple-500 focus:border-transparent"
              >
                <option value="BTCUSDT">BTC/USDT</option>
                <option value="ETHUSDT">ETH/USDT</option>
                <option value="ADAUSDT">ADA/USDT</option>
                <option value="SOLUSDT">SOL/USDT</option>
              </select>
            </div>

            {/* Max Episodes */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">Episodes Maximum</label>
              <input
                type="number"
                value={trainingConfig.max_episodes}
                onChange={(e) => setTrainingConfig({...trainingConfig, max_episodes: parseInt(e.target.value)})}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:ring-2 focus:ring-purple-500 focus:border-transparent"
              />
            </div>

            {/* Learning Rate */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">Taux d'Apprentissage</label>
              <input
                type="number"
                step="0.0001"
                value={trainingConfig.learning_rate}
                onChange={(e) => setTrainingConfig({...trainingConfig, learning_rate: parseFloat(e.target.value)})}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:ring-2 focus:ring-purple-500 focus:border-transparent"
              />
            </div>

            <button
              onClick={startTraining}
              className="w-full bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded-lg flex items-center justify-center"
            >
              <Play className="w-4 h-4 mr-2" />
              Démarrer Entraînement
            </button>
          </div>
        </div>

        {/* Entraînements Actifs */}
        <div className="bg-gray-800 rounded-lg shadow">
          <div className="px-6 py-4 border-b border-gray-700">
            <h3 className="text-lg font-semibold text-white flex items-center">
              <Monitor className="w-5 h-5 mr-2" />
              Entraînements Actifs
            </h3>
          </div>
          <div className="p-6">
            {Object.keys(activeTrainings).length === 0 ? (
              <div className="text-center text-gray-400 py-8">
                <Activity className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p>Aucun entraînement en cours</p>
              </div>
            ) : (
              <div className="space-y-4">
                {Object.entries(activeTrainings).map(([trainingId, training]) => (
                  <div key={trainingId} className="bg-gray-700 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-medium text-white">{trainingId}</h4>
                      <button
                        onClick={() => stopTraining(trainingId)}
                        className="bg-red-600 hover:bg-red-700 text-white px-3 py-1 rounded text-sm flex items-center"
                      >
                        <Square className="w-3 h-3 mr-1" />
                        Arrêter
                      </button>
                    </div>
                    <div className="text-sm text-gray-300">
                      <p>Status: <StatusBadge status={training.status} /></p>
                      {training.metrics && (
                        <div className="mt-2">
                          <p>Episodes: {training.metrics.episode_count || 0}</p>
                          <p>Récompense Moyenne: {(training.metrics.mean_reward || 0).toFixed(4)}</p>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Agents Déployés */}
      <div className="bg-gray-800 rounded-lg shadow">
        <div className="px-6 py-4 border-b border-gray-700">
          <h3 className="text-lg font-semibold text-white flex items-center">
            <Bot className="w-5 h-5 mr-2" />
            Agents Déployés
          </h3>
        </div>
        <div className="p-6">
          {Object.keys(deployedAgents).length === 0 ? (
            <div className="text-center text-gray-400 py-8">
              <Bot className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p>Aucun agent déployé</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {Object.entries(deployedAgents).map(([agentName, agent]) => (
                <div key={agentName} className="bg-gray-700 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-3">
                    <h4 className="font-medium text-white">{agentName}</h4>
                    <StatusBadge status={agent.status} />
                  </div>
                  
                  <div className="text-sm text-gray-300 space-y-1">
                    <p>Type: <span className="text-purple-400">{agent.agent_type}</span></p>
                    <p>Valeur Portfolio: <span className="text-green-400">${agent.portfolio_value?.toFixed(2)}</span></p>
                    <p>Retour Total: <span className={agent.total_return >= 0 ? 'text-green-400' : 'text-red-400'}>
                      {(agent.total_return || 0).toFixed(2)}%
                    </span></p>
                  </div>

                  {Object.keys(agent.positions || {}).length > 0 && (
                    <div className="mt-3">
                      <p className="text-xs font-medium text-gray-400 mb-1">Positions:</p>
                      <div className="text-xs text-gray-300">
                        {Object.entries(agent.positions).map(([symbol, size]) => (
                          <span key={symbol} className="inline-block bg-gray-600 px-2 py-1 rounded mr-2 mb-1">
                            {symbol}: {parseFloat(size).toFixed(4)}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Modèles Disponibles */}
      <div className="bg-gray-800 rounded-lg shadow">
        <div className="px-6 py-4 border-b border-gray-700">
          <h3 className="text-lg font-semibold text-white flex items-center">
            <Brain className="w-5 h-5 mr-2" />
            Modèles Entraînés ({availableModels.length})
          </h3>
        </div>
        <div className="p-6">
          {availableModels.length === 0 ? (
            <div className="text-center text-gray-400 py-8">
              <Brain className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p>Aucun modèle disponible</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full">
                <thead>
                  <tr className="text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                    <th className="pb-3">Type Agent</th>
                    <th className="pb-3">Nom Modèle</th>
                    <th className="pb-3">Taille</th>
                    <th className="pb-3">Créé</th>
                    <th className="pb-3">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-700">
                  {availableModels.map((model, index) => (
                    <tr key={index} className="text-gray-300">
                      <td className="py-3">
                        <span className={`px-2 py-1 rounded text-xs font-medium ${
                          model.agent_type === 'ppo' ? 'bg-purple-100 text-purple-800' : 'bg-blue-100 text-blue-800'
                        }`}>
                          {model.agent_type.toUpperCase()}
                        </span>
                      </td>
                      <td className="py-3 font-medium">{model.model_name}</td>
                      <td className="py-3 text-sm">{model.size_mb} MB</td>
                      <td className="py-3 text-sm">
                        {new Date(model.created).toLocaleDateString()}
                      </td>
                      <td className="py-3">
                        <button
                          onClick={() => deployAgent(model.model_path, `${model.agent_type}_${Date.now()}`)}
                          className="bg-green-600 hover:bg-green-700 text-white px-3 py-1 rounded text-sm flex items-center"
                        >
                          <Upload className="w-3 h-3 mr-1" />
                          Déployer
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default RLTrading;