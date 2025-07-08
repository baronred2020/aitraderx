import React, { useState, useEffect } from 'react';
import { Brain, Clock, Settings, BarChart3, Target, PieChart as PieChartIcon, Bell, Zap } from 'lucide-react';
import './index.css';

// Componentes básicos (se implementarán después)
const Dashboard = () => (
  <div className="p-6">
    <h2 className="text-2xl font-bold mb-4">Dashboard Principal</h2>
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold">Valor Total</h3>
        <p className="text-3xl font-bold text-blue-600">$125,430</p>
        <p className="text-green-600">+2.3%</p>
      </div>
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold">P&L Diario</h3>
        <p className="text-3xl font-bold text-green-600">+$1,234</p>
        <p className="text-green-600">+0.98%</p>
      </div>
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold">Precisión IA</h3>
        <p className="text-3xl font-bold text-purple-600">78.3%</p>
        <p className="text-green-600">+1.2%</p>
      </div>
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold">Sharpe Ratio</h3>
        <p className="text-3xl font-bold text-indigo-600">1.45</p>
        <p className="text-green-600">+0.12</p>
      </div>
    </div>
    <div className="mt-6 bg-white rounded-lg shadow p-6">
      <h3 className="text-lg font-semibold mb-4">Sistema de IA</h3>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-green-50 p-4 rounded">
          <h4 className="font-medium">IA Tradicional</h4>
          <p className="text-green-600">✅ Funcionando</p>
          <p className="text-sm text-gray-600">Random Forest + LSTM</p>
        </div>
        <div className="bg-blue-50 p-4 rounded">
          <h4 className="font-medium">Reinforcement Learning</h4>
          <p className="text-blue-600">🤖 Entrenando</p>
          <p className="text-sm text-gray-600">DQN Agent</p>
        </div>
        <div className="bg-purple-50 p-4 rounded">
          <h4 className="font-medium">Auto-entrenamiento</h4>
          <p className="text-purple-600">🔄 Monitoreando</p>
          <p className="text-sm text-gray-600">Detección de drift</p>
        </div>
      </div>
    </div>
  </div>
);

const Analysis = () => (
  <div className="p-6">
    <h2 className="text-2xl font-bold mb-4">Análisis de IA</h2>
    <p>Componente de análisis - En desarrollo</p>
  </div>
);

const Portfolio = () => (
  <div className="p-6">
    <h2 className="text-2xl font-bold mb-4">Portafolio</h2>
    <p>Componente de portafolio - En desarrollo</p>
  </div>
);

const AIMonitor = () => (
  <div className="p-6">
    <h2 className="text-2xl font-bold mb-4">Monitor de IA</h2>
    <p>Componente de monitoreo - En desarrollo</p>
  </div>
);

const RLDashboard = () => (
  <div className="p-6">
    <h2 className="text-2xl font-bold mb-4">Reinforcement Learning</h2>
    <p>Dashboard de RL - En desarrollo</p>
  </div>
);

const Alerts = () => (
  <div className="p-6">
    <h2 className="text-2xl font-bold mb-4">Alertas</h2>
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
              <Brain className="w-8 h-8 text-blue-600 mr-3" />
              <h1 className="text-xl font-bold text-gray-900">AI Trading Pro</h1>
              <span className="ml-3 px-2 py-1 bg-green-100 text-green-800 text-xs rounded-full">
                v1.0.0
              </span>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center text-sm text-gray-600">
                <Clock className="w-4 h-4 mr-1" />
                <span>Última actualización: {lastUpdate.toLocaleTimeString()}</span>
              </div>
              <div className="flex items-center">
                <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`}></div>
                <span className={`text-sm ml-2 font-medium ${isConnected ? 'text-green-600' : 'text-red-600'}`}>
                  {isConnected ? 'En vivo' : 'Desconectado'}
                </span>
              </div>
              <button className="p-2 rounded-full hover:bg-gray-100">
                <Settings className="w-5 h-5 text-gray-600" />
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
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
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        <div className="px-4 py-6 sm:px-0">
          {activeTab === 'dashboard' && <Dashboard />}
          {activeTab === 'analysis' && <Analysis />}
          {activeTab === 'portfolio' && <Portfolio />}
          {activeTab === 'ai-monitor' && <AIMonitor />}
          {activeTab === 'rl' && <RLDashboard />}
          {activeTab === 'alerts' && <Alerts />}
        </div>
      </main>
    </div>
  );
}

export default App; 