import React, { useState, useEffect } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { useAutomatedTrading } from '../../hooks/useAutomatedTrading';

interface MobileTradingProps {
  onBack: () => void;
}

const MobileTrading: React.FC<MobileTradingProps> = ({ onBack }) => {
  const { user, subscription } = useAuth();
  const { strategies, downloadEA } = useAutomatedTrading();
  const [activeTab, setActiveTab] = useState<'overview' | 'strategies' | 'mt4' | 'settings'>('overview');
  const [showQuickActions, setShowQuickActions] = useState(false);

//   const isMobile = window.innerWidth <= 768;

  // Quick Actions para móvil
  const quickActions = [
    {
      id: 'download-ea',
      title: 'Descargar EA',
      icon: '📥',
      action: () => downloadEA(),
      color: 'bg-blue-500'
    },
    {
      id: 'connect-mt4',
      title: 'Conectar MT4',
      icon: '🔗',
      action: () => setActiveTab('mt4'),
      color: 'bg-green-500'
    },
    {
      id: 'new-strategy',
      title: 'Nueva Estrategia',
      icon: '➕',
      action: () => setActiveTab('strategies'),
      color: 'bg-purple-500'
    },
    {
      id: 'settings',
      title: 'Configuración',
      icon: '⚙️',
      action: () => setActiveTab('settings'),
      color: 'bg-gray-500'
    }
  ];

  const totalPnL = strategies.reduce((sum: number, s: any) => sum + (s.totalPnL || 0), 0);
  const activeStrategies = strategies.filter((s: any) => s.status === 'active').length;

  return (
    <div className="min-h-screen bg-gray-900">
      {/* Header Móvil */}
      <div className="bg-gray-800 shadow-sm border-b border-gray-700">
        <div className="flex items-center justify-between p-4">
          <button
            onClick={onBack}
            className="p-2 rounded-lg bg-gray-700 hover:bg-gray-600 text-white"
          >
            ← Volver
          </button>
          <h1 className="text-lg font-semibold text-white">
            Trading Móvil
          </h1>
          <button
            onClick={() => setShowQuickActions(!showQuickActions)}
            className="p-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-white"
          >
            ⚡
          </button>
        </div>
      </div>

      {/* Quick Actions Panel */}
      {showQuickActions && (
        <div className="bg-gray-800 border-b border-gray-700 p-4">
          <h3 className="text-sm font-medium text-white mb-3">Acciones Rápidas</h3>
          <div className="grid grid-cols-2 gap-3">
            {quickActions.map((action) => (
              <button
                key={action.id}
                onClick={action.action}
                className={`${action.color} text-white p-3 rounded-lg flex flex-col items-center space-y-1`}
              >
                <span className="text-xl">{action.icon}</span>
                <span className="text-xs font-medium">{action.title}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Navigation Tabs */}
      <div className="bg-gray-800 border-b border-gray-700">
        <div className="flex space-x-1 p-2">
          {[
            { id: 'overview', label: 'Resumen', icon: '📊' },
            { id: 'strategies', label: 'Estrategias', icon: '🤖' },
            { id: 'mt4', label: 'MT4', icon: '💻' },
            { id: 'settings', label: 'Ajustes', icon: '⚙️' }
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex-1 py-2 px-3 rounded-lg text-sm font-medium transition-colors ${
                activeTab === tab.id
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              <div className="flex flex-col items-center space-y-1">
                <span>{tab.icon}</span>
                <span>{tab.label}</span>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Content Area */}
      <div className="p-4">
        {activeTab === 'overview' && (
          <div className="space-y-4">
            {/* Quick Stats Cards */}
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-gray-800 p-4 rounded-lg shadow-sm border border-gray-700">
                <div className="flex items-center space-x-2">
                  <span className="text-2xl">📊</span>
                  <div>
                    <p className="text-xs text-gray-400">Activas</p>
                    <p className="text-lg font-bold text-white">{activeStrategies}</p>
                  </div>
                </div>
              </div>
              
              <div className="bg-gray-800 p-4 rounded-lg shadow-sm border border-gray-700">
                <div className="flex items-center space-x-2">
                  <span className="text-2xl">💰</span>
                  <div>
                    <p className="text-xs text-gray-400">P&L Total</p>
                    <p className={`text-lg font-bold ${totalPnL >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      ${totalPnL.toFixed(2)}
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Recent Activity */}
            <div className="bg-gray-800 rounded-lg shadow-sm p-4 border border-gray-700">
              <h3 className="text-sm font-medium text-white mb-3">Actividad Reciente</h3>
              <div className="space-y-2">
                {strategies.slice(0, 3).map((strategy: any) => (
                  <div key={strategy.id} className="flex items-center justify-between p-2 bg-gray-700 rounded">
                    <div>
                      <p className="text-sm font-medium text-white">{strategy.name}</p>
                      <p className="text-xs text-gray-400">{strategy.pair}</p>
                    </div>
                    <div className="text-right">
                      <p className={`text-sm font-medium ${strategy.totalPnL >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        ${strategy.totalPnL.toFixed(2)}
                      </p>
                      <p className="text-xs text-gray-400">{strategy.status}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'strategies' && (
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <h3 className="text-lg font-medium text-white">Estrategias</h3>
              <button className="bg-blue-500 text-white px-4 py-2 rounded-lg text-sm">
                ➕ Nueva
              </button>
            </div>
            
            <div className="space-y-3">
              {strategies.map((strategy: any) => (
                <div key={strategy.id} className="bg-gray-800 rounded-lg shadow-sm p-4 border border-gray-700">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-medium text-white">{strategy.name}</h4>
                    <span className={`px-2 py-1 rounded-full text-xs ${
                      strategy.status === 'active' ? 'bg-green-900 text-green-300' :
                      strategy.status === 'paused' ? 'bg-yellow-900 text-yellow-300' :
                      'bg-red-900 text-red-300'
                    }`}>
                      {strategy.status}
                    </span>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div>
                      <p className="text-gray-400">Par</p>
                      <p className="font-medium text-white">{strategy.pair}</p>
                    </div>
                    <div>
                      <p className="text-gray-400">P&L</p>
                      <p className={`font-medium ${strategy.totalPnL >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        ${strategy.totalPnL.toFixed(2)}
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex space-x-2 mt-3">
                    <button className="flex-1 bg-green-500 text-white py-2 rounded text-sm">
                      ▶️ Iniciar
                    </button>
                    <button className="flex-1 bg-red-500 text-white py-2 rounded text-sm">
                      ⏸️ Pausar
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'mt4' && (
          <div className="space-y-4">
            <div className="bg-gray-800 rounded-lg shadow-sm p-4 border border-gray-700">
              <h3 className="text-lg font-medium text-white mb-4">Conexión MT4</h3>
              
              <div className="space-y-3">
                <div className="flex items-center justify-between p-3 bg-gray-700 rounded">
                  <div className="flex items-center space-x-3">
                    <span className="text-2xl">💻</span>
                    <div>
                      <p className="font-medium text-white">MetaTrader 4</p>
                      <p className="text-sm text-gray-400">Estado de conexión</p>
                    </div>
                  </div>
                  <span className="text-green-400">✅ Conectado</span>
                </div>
                
                <button className="w-full bg-blue-500 text-white py-3 rounded-lg font-medium">
                  📥 Descargar Expert Advisor
                </button>
                
                <button className="w-full bg-green-500 text-white py-3 rounded-lg font-medium">
                  🔗 Conectar MT4
                </button>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'settings' && (
          <div className="space-y-4">
            <div className="bg-gray-800 rounded-lg shadow-sm p-4 border border-gray-700">
              <h3 className="text-lg font-medium text-white mb-4">Configuración</h3>
              
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Notificaciones Push</span>
                  <button className="w-12 h-6 bg-blue-500 rounded-full relative">
                    <div className="w-4 h-4 bg-white rounded-full absolute right-1 top-1"></div>
                  </button>
                </div>
                
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Trading Automático</span>
                  <button className="w-12 h-6 bg-gray-600 rounded-full relative">
                    <div className="w-4 h-4 bg-white rounded-full absolute left-1 top-1"></div>
                  </button>
                </div>
                
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Sonidos</span>
                  <button className="w-12 h-6 bg-blue-500 rounded-full relative">
                    <div className="w-4 h-4 bg-white rounded-full absolute right-1 top-1"></div>
                  </button>
                </div>
              </div>
            </div>
            
            <div className="bg-gray-800 rounded-lg shadow-sm p-4 border border-gray-700">
              <h4 className="font-medium text-white mb-3">Información de Cuenta</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Plan:</span>
                  <span className="font-medium text-white">{subscription?.planType || 'Starter'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Usuario:</span>
                  <span className="font-medium text-white">{user?.email || 'N/A'}</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Floating Action Button */}
      <div className="fixed bottom-6 right-6">
        <button
          onClick={() => setShowQuickActions(!showQuickActions)}
          className="w-14 h-14 bg-blue-500 text-white rounded-full shadow-lg flex items-center justify-center text-2xl hover:bg-blue-600 transition-colors"
        >
          ⚡
        </button>
      </div>
    </div>
  );
};

export default MobileTrading; 