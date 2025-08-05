import React, { useState, useEffect } from 'react';
import { usePortfolioStrategies } from '../../hooks/usePortfolioStrategies';
import MT4Connection from './MT4Connection';
import ActiveStrategies from './ActiveStrategies';
import StrategyConfig from './StrategyConfig';
import MT4Installer from './MT4Installer';

// Declaración de tipos para el componente
interface AutomatedTradingProps {}

const AutomatedTrading: React.FC = () => {
  const { 
    strategies, 
    isLoading, 
    error, 
    createStrategy, 
    startStrategy, 
    stopStrategy, 
    deleteStrategy
  } = usePortfolioStrategies();
  const [showConfigModal, setShowConfigModal] = useState(false);
  const [showInstaller, setShowInstaller] = useState(false);
  const [selectedStrategy, setSelectedStrategy] = useState<any>(null);
  const [showDetailsModal, setShowDetailsModal] = useState(false);
  const [connectionInfo, setConnectionInfo] = useState({
    connected: false,
    account: '',
    balance: 0,
    equity: 0
  });

  const handleInstallationComplete = () => {
    setShowInstaller(false);
    // Aquí podrías actualizar el estado de conexión
  };

  const handleStrategyDetails = (strategy: any) => {
    setSelectedStrategy(strategy);
    setShowDetailsModal(true);
  };

  const handleStrategyDelete = async (id: string) => {
    if (window.confirm('¿Estás seguro de que quieres eliminar esta estrategia? Esta acción no se puede deshacer.')) {
      await deleteStrategy(id);
    }
  };

  const totalPnL = strategies.reduce((sum: number, s: any) => sum + (s.totalPnL || 0), 0);
  const activeStrategies = strategies.filter((s: any) => s.status === 'active').length;

  if (showInstaller) {
    return <MT4Installer onInstallationComplete={handleInstallationComplete} />;
  }

  return (
    <div className="space-y-6">
      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-gray-800 p-4 rounded-lg shadow border border-gray-700">
          <div className="flex items-center">
            <div className="p-2 bg-blue-900 rounded-lg">
              <span className="text-blue-300 text-xl">📊</span>
            </div>
            <div className="ml-4">
              <p className="text-sm text-gray-300">Estrategias Activas</p>
              <p className="text-2xl font-bold text-white">{activeStrategies}</p>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 p-4 rounded-lg shadow border border-gray-700">
          <div className="flex items-center">
            <div className="p-2 bg-green-900 rounded-lg">
              <span className="text-green-300 text-xl">💰</span>
            </div>
            <div className="ml-4">
              <p className="text-sm text-gray-300">P&L Total</p>
              <p className={`text-2xl font-bold ${totalPnL >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                ${totalPnL.toFixed(2)}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 p-4 rounded-lg shadow border border-gray-700">
          <div className="flex items-center">
            <div className="p-2 bg-purple-900 rounded-lg">
              <span className="text-purple-300 text-xl">🎯</span>
            </div>
            <div className="ml-4">
              <p className="text-sm text-gray-300">Win Rate</p>
              <p className="text-2xl font-bold text-white">
                {strategies.length > 0 
                  ? Math.round(strategies.reduce((sum: number, s: any) => sum + (s.winningTrades / Math.max(s.totalTrades, 1) * 100), 0) / strategies.length)
                  : 0}%
              </p>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 p-4 rounded-lg shadow border border-gray-700">
          <div className="flex items-center">
            <div className="p-2 bg-orange-900 rounded-lg">
              <span className="text-orange-300 text-xl">⚡</span>
            </div>
            <div className="ml-4">
              <p className="text-sm text-gray-300">Señales Hoy</p>
              <p className="text-2xl font-bold text-white">
                {strategies.reduce((sum: number, s: any) => sum + (s.lastSignal ? 1 : 0), 0)}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* MT4 Connection & Installer */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <MT4Connection />
        
        <div className="bg-gray-800 p-6 rounded-lg shadow border border-gray-700">
          <h3 className="text-lg font-semibold text-white mb-4">🤖 Configuración MT4</h3>
          <p className="text-gray-300 mb-4">
            Configura tu MetaTrader 4/5 para trading automático
          </p>
          <button
            onClick={() => setShowInstaller(true)}
            className="w-full bg-gradient-to-r from-blue-500 to-purple-600 text-white py-3 px-4 rounded-lg hover:from-blue-600 hover:to-purple-700 transition-all duration-200 font-medium"
          >
            🚀 Configurar Trading Automático
          </button>
        </div>
      </div>

      {/* Active Strategies */}
      <ActiveStrategies 
        strategies={strategies}
        onStart={startStrategy}
        onStop={stopStrategy}
        isLoading={isLoading}
        onDetails={handleStrategyDetails}
        onDelete={handleStrategyDelete}
      />

      {/* Create Strategy Button */}
      <div className="text-center">
        <button
          onClick={() => setShowConfigModal(true)}
          className="bg-green-500 hover:bg-green-600 text-white px-6 py-3 rounded-lg font-medium transition-colors"
        >
          ➕ Crear Nueva Estrategia
        </button>
      </div>

      {/* Strategy Config Modal */}
      {showConfigModal && (
        <StrategyConfig
          onClose={() => setShowConfigModal(false)}
          onCreate={createStrategy}
          isLoading={isLoading}
        />
      )}

      {/* Strategy Details Modal */}
      {showDetailsModal && selectedStrategy && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-gray-800 rounded-lg p-6 max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-bold text-white">📊 Detalles de la Estrategia</h3>
              <button
                onClick={() => setShowDetailsModal(false)}
                className="text-gray-400 hover:text-white text-2xl"
              >
                ×
              </button>
            </div>
            
            <div className="space-y-4">
              {/* Basic Info */}
              <div className="bg-gray-700 rounded-lg p-4">
                <h4 className="text-lg font-medium text-white mb-3">Información Básica</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <span className="text-gray-400">Nombre:</span>
                    <div className="text-white font-medium">{selectedStrategy.name}</div>
                  </div>
                  <div>
                    <span className="text-gray-400">Par:</span>
                    <div className="text-white font-medium">{selectedStrategy.pair}</div>
                  </div>
                  <div>
                    <span className="text-gray-400">Tipo:</span>
                    <div className="text-white font-medium">{selectedStrategy.type}</div>
                  </div>
                  <div>
                    <span className="text-gray-400">Estado:</span>
                    <div className={`font-medium ${selectedStrategy.status === 'active' ? 'text-green-400' : 'text-red-400'}`}>
                      {selectedStrategy.status}
                    </div>
                  </div>
                </div>
              </div>

              {/* Performance Stats */}
              <div className="bg-gray-700 rounded-lg p-4">
                <h4 className="text-lg font-medium text-white mb-3">Estadísticas de Rendimiento</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <span className="text-gray-400">P&L Total:</span>
                    <div className={`font-medium ${selectedStrategy.totalPnL >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      ${selectedStrategy.totalPnL.toFixed(2)}
                    </div>
                  </div>
                  <div>
                    <span className="text-gray-400">Total Trades:</span>
                    <div className="text-white font-medium">{selectedStrategy.totalTrades}</div>
                  </div>
                  <div>
                    <span className="text-gray-400">Trades Ganadores:</span>
                    <div className="text-green-400 font-medium">{selectedStrategy.winningTrades}</div>
                  </div>
                  <div>
                    <span className="text-gray-400">Win Rate:</span>
                    <div className="text-white font-medium">
                      {selectedStrategy.totalTrades > 0 
                        ? ((selectedStrategy.winningTrades / selectedStrategy.totalTrades) * 100).toFixed(1)
                        : '0.0'}%
                    </div>
                  </div>
                </div>
              </div>

              {/* Last Signal */}
              {selectedStrategy.lastSignal && (
                <div className="bg-gray-700 rounded-lg p-4">
                  <h4 className="text-lg font-medium text-white mb-3">Última Señal</h4>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <span className="text-gray-400">Tipo:</span>
                      <div className="text-white font-medium">{selectedStrategy.lastSignal.type}</div>
                    </div>
                    <div>
                      <span className="text-gray-400">Precio:</span>
                      <div className="text-white font-medium">{selectedStrategy.lastSignal.price}</div>
                    </div>
                    <div>
                      <span className="text-gray-400">Fecha:</span>
                      <div className="text-white font-medium">
                        {new Date(selectedStrategy.lastSignal.time).toLocaleString()}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Timestamps */}
              <div className="bg-gray-700 rounded-lg p-4">
                <h4 className="text-lg font-medium text-white mb-3">Información Temporal</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <span className="text-gray-400">Creada:</span>
                    <div className="text-white font-medium">
                      {new Date(selectedStrategy.createdAt).toLocaleString()}
                    </div>
                  </div>
                  <div>
                    <span className="text-gray-400">Actualizada:</span>
                    <div className="text-white font-medium">
                      {new Date(selectedStrategy.updatedAt).toLocaleString()}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="flex justify-end space-x-3 mt-6">
              <button
                onClick={() => setShowDetailsModal(false)}
                className="bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-lg transition-colors"
              >
                Cerrar
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AutomatedTrading;

// Asegurar que TypeScript reconozca este archivo como un módulo
export {}; 