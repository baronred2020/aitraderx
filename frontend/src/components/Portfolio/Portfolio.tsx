import React, { useState } from 'react';
import AutomatedTrading from './AutomatedTrading';
import TradingHistory from './TradingHistory';
import PortfolioStats from './PortfolioStats';
import AIStrategies from './AIStrategies';

interface PortfolioProps {}

const Portfolio: React.FC<PortfolioProps> = () => {
  const [activeTab, setActiveTab] = useState<'automated' | 'scalping' | 'history' | 'stats'>('automated');

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">📊 Portfolio</h1>
          <p className="text-gray-400">Gestión de trading automático y análisis de rendimiento</p>
        </div>

        {/* Navigation Tabs */}
        <div className="flex space-x-1 bg-gray-800 p-1 rounded-lg mb-6">
          <button
            onClick={() => setActiveTab('automated')}
            className={`flex-1 py-3 px-4 rounded-md font-medium transition-colors ${
              activeTab === 'automated'
                ? 'bg-blue-600 text-white'
                : 'text-gray-400 hover:text-white hover:bg-gray-700'
            }`}
          >
            🤖 Trading Automático
          </button>
          <button
            onClick={() => setActiveTab('scalping')}
            className={`flex-1 py-3 px-4 rounded-md font-medium transition-colors ${
              activeTab === 'scalping'
                ? 'bg-blue-600 text-white'
                : 'text-gray-400 hover:text-white hover:bg-gray-700'
            }`}
          >
            🤖 Estrategias IA
          </button>
          <button
            onClick={() => setActiveTab('history')}
            className={`flex-1 py-3 px-4 rounded-md font-medium transition-colors ${
              activeTab === 'history'
                ? 'bg-blue-600 text-white'
                : 'text-gray-400 hover:text-white hover:bg-gray-700'
            }`}
          >
            📈 Historial
          </button>
          <button
            onClick={() => setActiveTab('stats')}
            className={`flex-1 py-3 px-4 rounded-md font-medium transition-colors ${
              activeTab === 'stats'
                ? 'bg-blue-600 text-white'
                : 'text-gray-400 hover:text-white hover:bg-gray-700'
            }`}
          >
            📊 Estadísticas
          </button>
        </div>

        {/* Content */}
        <div className="space-y-6">
          {activeTab === 'automated' && <AutomatedTrading />}
          {activeTab === 'scalping' && <AIStrategies />}
          {activeTab === 'history' && <TradingHistory />}
          {activeTab === 'stats' && <PortfolioStats />}
        </div>
      </div>
    </div>
  );
};

export default Portfolio; 