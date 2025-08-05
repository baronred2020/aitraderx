import React, { useState, useEffect } from 'react';
import { useScalpingStrategies } from '../../hooks/useScalpingStrategies';

interface AIStrategiesProps {
  onStrategySelect?: (strategy: any) => void;
}

const AIStrategies: React.FC<AIStrategiesProps> = ({ onStrategySelect }) => {
  const {
    strategies,
    isLoading,
    error,
    createStrategy,
    startStrategy,
    stopStrategy,
    deleteStrategy,
    getAvailableTypes,
    refreshStrategies
  } = useScalpingStrategies();

  const [showCreateModal, setShowCreateModal] = useState(false);
  const [availableTypes, setAvailableTypes] = useState<any[]>([]);
  const [selectedType, setSelectedType] = useState('');
  const [config, setConfig] = useState({
    parameters: {},
    filters: {}
  });

  useEffect(() => {
    loadAvailableTypes();
  }, []);

  const loadAvailableTypes = async () => {
    const result = await getAvailableTypes();
    if (result.success) {
      setAvailableTypes(result.strategy_types || []);
    }
  };

  const handleCreateStrategy = async () => {
    if (!selectedType) {
      window.alert('Selecciona un tipo de estrategia');
      return;
    }

    const result = await createStrategy(selectedType, config);
    if (result.success) {
      setShowCreateModal(false);
      setSelectedType('');
      setConfig({ parameters: {}, filters: {} });
    } else {
      window.alert(`Error: ${result.message}`);
    }
  };

  const handleStartStrategy = async (strategyId: string) => {
    const result = await startStrategy(strategyId);
    if (!result.success) {
      window.alert(`Error: ${result.message}`);
    }
  };

  const handleStopStrategy = async (strategyId: string) => {
    const result = await stopStrategy(strategyId);
    if (!result.success) {
      window.alert(`Error: ${result.message}`);
    }
  };

  const handleDeleteStrategy = async (strategyId: string) => {
    if (window.confirm('¿Estás seguro de que quieres eliminar esta estrategia?')) {
      const result = await deleteStrategy(strategyId);
      if (!result.success) {
        window.alert(`Error: ${result.message}`);
      }
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'bg-green-100 text-green-800';
      case 'stopped':
        return 'bg-red-100 text-red-800';
      case 'created':
        return 'bg-yellow-100 text-yellow-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'active':
        return 'Activa';
      case 'stopped':
        return 'Detenida';
      case 'created':
        return 'Creada';
      default:
        return status;
    }
  };

  const getStrategyIcon = (type: string) => {
    switch (type) {
      case 'scalping_eurusd':
      case 'scalping_gbpusd':
        return '⚡';
      case 'day_trading':
        return '📈';
      case 'swing_trading':
        return '📊';
      case 'position_trading':
        return '🎯';
      default:
        return '🤖';
    }
  };

  const getStrategyCategory = (type: string) => {
    if (type.includes('scalping')) return 'Scalping';
    if (type.includes('day')) return 'Day Trading';
    if (type.includes('swing')) return 'Swing Trading';
    if (type.includes('position')) return 'Position Trading';
    return 'General';
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
        <span className="ml-2">Cargando estrategias...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <div className="flex">
          <div className="flex-shrink-0">
            <span className="text-red-400">⚠️</span>
          </div>
          <div className="ml-3">
            <h3 className="text-sm font-medium text-red-800">Error</h3>
            <div className="mt-2 text-sm text-red-700">
              <p>{error}</p>
            </div>
            <div className="mt-4">
              <button
                onClick={refreshStrategies}
                className="bg-red-100 text-red-800 px-3 py-1 rounded-md text-sm font-medium hover:bg-red-200"
              >
                Reintentar
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
                     <h2 className="text-2xl font-bold text-white">Estrategias de Inteligencia Artificial</h2>
           <p className="text-gray-300">Estrategias predefinidas de scalping que se conectan con MT4/MT5 para trading real</p>
        </div>
        <button
          onClick={() => setShowCreateModal(true)}
          className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg font-medium transition-colors"
        >
          ➕ Nueva Estrategia
        </button>
      </div>

      {/* Available Strategies Overview */}
      <div className="bg-gray-800 rounded-lg shadow p-6 border border-gray-700">
        <h3 className="text-lg font-medium text-white mb-4">Estrategias Disponibles</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {availableTypes.map((type) => (
            <div key={type} className="border border-gray-600 rounded-lg p-4 hover:shadow-md transition-shadow bg-gray-700">
              <div className="flex items-center mb-2">
                <span className="text-2xl mr-2">{getStrategyIcon(type)}</span>
                <div>
                  <h4 className="font-medium text-white">
                    {type.replace('_', ' ').toUpperCase()}
                  </h4>
                  <p className="text-sm text-gray-300">{getStrategyCategory(type)}</p>
                </div>
              </div>
              <p className="text-sm text-gray-300 mb-3">
                Estrategia optimizada para trading automático con IA
              </p>
              <button
                onClick={() => {
                  setSelectedType(type);
                  setShowCreateModal(true);
                }}
                className="w-full bg-green-500 hover:bg-green-600 text-white py-2 px-3 rounded text-sm font-medium"
              >
                🚀 Implementar
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-gray-800 p-4 rounded-lg shadow border border-gray-700">
          <div className="flex items-center">
            <div className="p-2 bg-blue-900 rounded-lg">
              <span className="text-blue-300 text-xl">📊</span>
            </div>
            <div className="ml-4">
              <p className="text-sm text-gray-300">Total Estrategias</p>
              <p className="text-2xl font-bold text-white">{strategies.length}</p>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 p-4 rounded-lg shadow border border-gray-700">
          <div className="flex items-center">
            <div className="p-2 bg-green-900 rounded-lg">
              <span className="text-green-300 text-xl">▶️</span>
            </div>
            <div className="ml-4">
              <p className="text-sm text-gray-300">Activas</p>
              <p className="text-2xl font-bold text-white">
                {strategies.filter((s: any) => s.status === 'active').length}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 p-4 rounded-lg shadow border border-gray-700">
          <div className="flex items-center">
            <div className="p-2 bg-purple-900 rounded-lg">
              <span className="text-purple-300 text-xl">💰</span>
            </div>
            <div className="ml-4">
              <p className="text-sm text-gray-300">P&L Total</p>
              <p className={`text-2xl font-bold ${
                strategies.reduce((sum: number, s: any) => sum + s.total_pnl, 0) >= 0 
                  ? 'text-green-400' 
                  : 'text-red-400'
              }`}>
                ${strategies.reduce((sum: number, s: any) => sum + s.total_pnl, 0).toFixed(2)}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 p-4 rounded-lg shadow border border-gray-700">
          <div className="flex items-center">
            <div className="p-2 bg-orange-900 rounded-lg">
              <span className="text-orange-300 text-xl">🎯</span>
            </div>
            <div className="ml-4">
              <p className="text-sm text-gray-300">Win Rate Promedio</p>
              <p className="text-2xl font-bold text-white">
                {strategies.length > 0 
                  ? Math.round(strategies.reduce((sum: number, s: any) => sum + s.win_rate, 0) / strategies.length)
                  : 0}%
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Active Strategies List */}
      <div className="bg-gray-800 rounded-lg shadow border border-gray-700">
        <div className="px-6 py-4 border-b border-gray-700">
          <h3 className="text-lg font-medium text-white">Estrategias Activas</h3>
        </div>
        
        {strategies.length === 0 ? (
          <div className="p-8 text-center">
            <div className="text-6xl mb-4">🤖</div>
            <h3 className="text-lg font-medium text-white mb-2">No hay estrategias activas</h3>
            <p className="text-gray-300 mb-4">
              Selecciona una estrategia de la lista superior para comenzar
            </p>
            <button
              onClick={() => setShowCreateModal(true)}
              className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg font-medium"
            >
              Ver Estrategias Disponibles
            </button>
          </div>
        ) : (
          <div className="divide-y divide-gray-700">
            {strategies.map((strategy: any) => (
              <div key={strategy.id} className="p-6">
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-3">
                      <span className="text-2xl">{getStrategyIcon(strategy.type)}</span>
                      <h4 className="text-lg font-medium text-white">{strategy.name}</h4>
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(strategy.status)}`}>
                        {getStatusText(strategy.status)}
                      </span>
                      <span className="px-2 py-1 rounded-full text-xs font-medium bg-blue-900 text-blue-300">
                        {getStrategyCategory(strategy.type)}
                      </span>
                    </div>
                    
                    <p className="text-gray-300 mt-1">{strategy.description}</p>
                    
                    <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div>
                        <p className="text-sm text-gray-400">Par</p>
                        <p className="font-medium text-white">{strategy.pair}</p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-400">P&L</p>
                        <p className={`font-medium ${strategy.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                          ${strategy.total_pnl.toFixed(2)}
                        </p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-400">Operaciones</p>
                        <p className="font-medium text-white">{strategy.total_trades}</p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-400">Win Rate</p>
                        <p className="font-medium text-white">{strategy.win_rate.toFixed(1)}%</p>
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex space-x-2 ml-4">
                    {strategy.status === 'active' ? (
                      <button
                        onClick={() => handleStopStrategy(strategy.id)}
                        className="bg-red-500 hover:bg-red-600 text-white px-3 py-1 rounded text-sm font-medium"
                      >
                        ⏸️ Detener
                      </button>
                    ) : (
                      <button
                        onClick={() => handleStartStrategy(strategy.id)}
                        className="bg-green-500 hover:bg-green-600 text-white px-3 py-1 rounded text-sm font-medium"
                      >
                        ▶️ Iniciar
                      </button>
                    )}
                    
                    <button
                      onClick={() => handleDeleteStrategy(strategy.id)}
                      className="bg-gray-500 hover:bg-gray-600 text-white px-3 py-1 rounded text-sm font-medium"
                    >
                      🗑️ Eliminar
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Create Strategy Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-gray-900 bg-opacity-75 overflow-y-auto h-full w-full z-50">
          <div className="relative top-20 mx-auto p-5 border border-gray-600 w-96 shadow-lg rounded-md bg-gray-800">
            <div className="mt-3">
              <h3 className="text-lg font-medium text-white mb-4">Implementar Estrategia IA</h3>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Estrategia IA Disponible
                  </label>
                  <select
                    value={selectedType}
                    onChange={(e) => setSelectedType(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 bg-gray-700 text-white"
                  >
                    <option value="">Selecciona una estrategia</option>
                    {availableTypes.map((type) => (
                      <option key={type} value={type}>
                        {getStrategyIcon(type)} {type.replace('_', ' ').toUpperCase()} - {getStrategyCategory(type)}
                      </option>
                    ))}
                  </select>
                </div>
                
                                 <div className="bg-green-900 bg-opacity-30 border border-green-500 rounded-lg p-3">
                   <div className="flex items-start space-x-2">
                     <span className="text-green-400 text-lg">🔗</span>
                     <div className="text-sm text-green-200">
                       <p className="font-medium mb-1">Conexión MT4/MT5</p>
                       <p>Esta estrategia se conectará con tu plataforma MT4/MT5 para ejecutar trades reales en tu cuenta demo o real.</p>
                     </div>
                   </div>
                 </div>
              </div>
              
              <div className="flex space-x-3 mt-6">
                <button
                  onClick={handleCreateStrategy}
                  className="flex-1 bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg font-medium"
                >
                  Crear Estrategia
                </button>
                <button
                  onClick={() => setShowCreateModal(false)}
                  className="flex-1 bg-gray-600 hover:bg-gray-500 text-white px-4 py-2 rounded-lg font-medium"
                >
                  Cancelar
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AIStrategies; 