import React, { useState, useEffect } from 'react';
import { useAutomatedTrading } from '../../hooks/useAutomatedTrading';
import MT4Connection from './MT4Connection';
import ActiveStrategies from './ActiveStrategies';
import StrategyConfig from './StrategyConfig';
import MT4Installer from './MT4Installer';

// Declaración de tipos para el componente
interface AutomatedTradingProps {}

const AutomatedTrading: React.FC = () => {
  const { strategies, isLoading, error, createStrategy, startStrategy, stopStrategy, deleteStrategy } = useAutomatedTrading();
  const [showConfigModal, setShowConfigModal] = useState(false);
  const [showInstaller, setShowInstaller] = useState(false);
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
    </div>
  );
};

export default AutomatedTrading;

// Asegurar que TypeScript reconozca este archivo como un módulo
export {}; 