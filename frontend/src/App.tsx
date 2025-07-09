import React, { useState, useEffect } from 'react';
import { Brain, Clock, Settings, BarChart3, Target, PieChart as PieChartIcon, Bell, Zap } from 'lucide-react';
import { RLDashboard } from './components/RL/RLDashboard';
import { MobileNav } from './components/Common/MobileNav';
import './index.css';

// Componentes básicos (se implementarán después)
const Dashboard = () => (
  <div className="p-3 sm:p-6">
    <h2 className="text-xl sm:text-2xl font-bold mb-4">AITRADERX Dashboard</h2>
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-6">
      <div className="bg-white rounded-lg shadow p-4 sm:p-6">
        <h3 className="text-sm sm:text-lg font-semibold">Valor Total</h3>
        <p className="text-2xl sm:text-3xl font-bold text-blue-600">$125,430</p>
        <p className="text-green-600 text-sm sm:text-base">+2.3%</p>
      </div>
      <div className="bg-white rounded-lg shadow p-4 sm:p-6">
        <h3 className="text-sm sm:text-lg font-semibold">P&L Diario</h3>
        <p className="text-2xl sm:text-3xl font-bold text-green-600">+$1,234</p>
        <p className="text-green-600 text-sm sm:text-base">+0.98%</p>
      </div>
      <div className="bg-white rounded-lg shadow p-4 sm:p-6">
        <h3 className="text-sm sm:text-lg font-semibold">Precisión IA</h3>
        <p className="text-2xl sm:text-3xl font-bold text-purple-600">78.3%</p>
        <p className="text-green-600 text-sm sm:text-base">+1.2%</p>
      </div>
      <div className="bg-white rounded-lg shadow p-4 sm:p-6">
        <h3 className="text-sm sm:text-lg font-semibold">Sharpe Ratio</h3>
        <p className="text-2xl sm:text-3xl font-bold text-indigo-600">1.45</p>
        <p className="text-green-600 text-sm sm:text-base">+0.12</p>
      </div>
    </div>
    <div className="mt-4 sm:mt-6 bg-white rounded-lg shadow p-4 sm:p-6">
      <h3 className="text-lg font-semibold mb-4">AITRADERX - Sistema de IA</h3>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 sm:gap-4">
        <div className="bg-green-50 p-3 sm:p-4 rounded">
          <h4 className="font-medium text-sm sm:text-base">IA Tradicional</h4>
          <p className="text-green-600 text-sm sm:text-base">✅ Funcionando</p>
          <p className="text-xs sm:text-sm text-gray-600">AITRADERX - Random Forest + LSTM</p>
        </div>
        <div className="bg-blue-50 p-3 sm:p-4 rounded">
          <h4 className="font-medium text-sm sm:text-base">Reinforcement Learning</h4>
          <p className="text-blue-600 text-sm sm:text-base">🤖 Entrenando</p>
          <p className="text-xs sm:text-sm text-gray-600">AITRADERX - DQN Agent</p>
        </div>
        <div className="bg-purple-50 p-3 sm:p-4 rounded">
          <h4 className="font-medium text-sm sm:text-base">Auto-entrenamiento</h4>
          <p className="text-purple-600 text-sm sm:text-base">🔄 Monitoreando</p>
          <p className="text-xs sm:text-sm text-gray-600">AITRADERX - Detección de drift</p>
        </div>
      </div>
    </div>
  </div>
);

const Analysis = () => (
  <div className="p-6">
    <h2 className="text-2xl font-bold mb-4">AITRADERX - Análisis de IA</h2>
    <p>Componente de análisis - En desarrollo</p>
  </div>
);

const Portfolio = () => (
  <div className="p-6">
    <h2 className="text-2xl font-bold mb-4">AITRADERX - Portafolio</h2>
    <p>Componente de portafolio - En desarrollo</p>
  </div>
);

const AIMonitor = () => (
  <div className="p-6">
    <h2 className="text-2xl font-bold mb-4">AITRADERX - Monitor de IA</h2>
    <p>Componente de monitoreo - En desarrollo</p>
  </div>
);

const RLDashboardComponent = () => (
  <div className="p-6">
    <h2 className="text-2xl font-bold mb-4">AITRADERX - Reinforcement Learning</h2>
    <p>Dashboard de RL - En desarrollo</p>
  </div>
);

const Alerts = () => (
  <div className="p-6">
    <h2 className="text-2xl font-bold mb-4">AITRADERX - Alertas</h2>
    <p>Sistema de alertas - En desarrollo</p>
  </div>
);

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [lastUpdate, setLastUpdate] = useState(new Date());
  const [isConnected, setIsConnected] = useState(true);

  useEffect(() => {
    // Simular actualizaciones en tiempo real
    const interval = setInterval(() => {
      setLastUpdate(new Date());
    }, 30000);

    return () => clearInterval(interval);
  }, []);

  const tabs = [
    { id: 'dashboard', name: 'Dashboard', icon: BarChart3 },
    { id: 'analysis', name: 'Análisis', icon: Target },
    { id: 'portfolio', name: 'Portafolio', icon: PieChartIcon },
    { id: 'ai-monitor', name: 'Monitor IA', icon: Brain },
    { id: 'rl', name: 'Reinforcement Learning', icon: Zap },
    { id: 'alerts', name: 'Alertas', icon: Bell }
  ];

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <Brain className="w-6 h-6 sm:w-8 sm:h-8 text-blue-600 mr-2 sm:mr-3" />
              <h1 className="text-lg sm:text-xl font-bold text-gray-900">AITRADERX</h1>
              <span className="ml-2 sm:ml-3 px-1.5 sm:px-2 py-0.5 sm:py-1 bg-green-100 text-green-800 text-xs rounded-full">
                v1.0.0
              </span>
            </div>
            <div className="flex items-center space-x-2 sm:space-x-4">
              {/* Desktop Status */}
              <div className="hidden sm:flex items-center text-sm text-gray-600">
                <Clock className="w-4 h-4 mr-1" />
                <span>Última actualización: {lastUpdate.toLocaleTimeString()}</span>
              </div>
              
              {/* Mobile Status */}
              <div className="sm:hidden flex items-center text-xs text-gray-600">
                <Clock className="w-3 h-3 mr-1" />
                <span>{lastUpdate.toLocaleTimeString()}</span>
              </div>
              
              <div className="flex items-center">
                <div className={`w-2 h-2 sm:w-3 sm:h-3 rounded-full ${isConnected ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`}></div>
                <span className={`text-xs sm:text-sm ml-1 sm:ml-2 font-medium ${isConnected ? 'text-green-600' : 'text-red-600'} hidden sm:inline`}>
                  {isConnected ? 'En vivo' : 'Desconectado'}
                </span>
              </div>
              <button className="p-1.5 sm:p-2 rounded-full hover:bg-gray-100">
                <Settings className="w-4 h-4 sm:w-5 sm:h-5 text-gray-600" />
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          {/* Desktop Navigation */}
          <div className="hidden md:flex space-x-8">
            {tabs.map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`py-4 px-1 border-b-2 font-medium text-sm flex items-center space-x-2 ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700'
                }`}
              >
                <tab.icon className="w-4 h-4" />
                <span>{tab.name}</span>
              </button>
            ))}
          </div>
          
          {/* Mobile Navigation */}
          <div className="md:hidden">
            <div className="flex overflow-x-auto space-x-4 pb-2">
              {tabs.map(tab => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex-shrink-0 px-3 py-2 rounded-lg text-xs font-medium flex items-center space-x-1 ${
                    activeTab === tab.id
                      ? 'bg-blue-500 text-white'
                      : 'bg-gray-100 text-gray-600'
                  }`}
                >
                  <tab.icon className="w-3 h-3" />
                  <span className="hidden sm:inline">{tab.name}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8 pb-20 md:pb-6">
        <div className="px-4 py-6 sm:px-0">
          {activeTab === 'dashboard' && <Dashboard />}
          {activeTab === 'analysis' && <Analysis />}
          {activeTab === 'portfolio' && <Portfolio />}
          {activeTab === 'ai-monitor' && <AIMonitor />}
          {activeTab === 'rl' && <RLDashboardComponent />}
          {activeTab === 'alerts' && <Alerts />}
        </div>
      </main>

      {/* Mobile Navigation */}
      <MobileNav activeTab={activeTab} onTabChange={setActiveTab} />
    </div>
  );
}

export default App; 