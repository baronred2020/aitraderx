import React, { useState, useEffect } from 'react';
import { Layout } from './components/Common/Layout';
import { Dashboard } from './components/Dashboard/Dashboard';
import { TradingView } from './components/Trading/TradingView';
import { RLDashboard } from './components/RL/RLDashboard';
import './index.css';

// Componentes temporales para las nuevas secciones
const Portfolio = () => (
  <div className="p-6">
    <div className="trading-card p-6">
      <h2 className="text-2xl font-bold text-white mb-4">Portfolio</h2>
      <div className="bg-gray-800/50 rounded-lg p-8 text-center">
        <div className="w-16 h-16 bg-gradient-to-r from-green-500 to-emerald-500 rounded-full flex items-center justify-center mx-auto mb-4">
          <span className="text-2xl">💼</span>
        </div>
        <h3 className="text-lg font-semibold text-white mb-2">Gestión de Portfolio</h3>
        <p className="text-gray-400 mb-4">Análisis detallado de posiciones y rendimiento</p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-gray-700/50 rounded-lg p-4">
            <div className="text-xl font-bold text-white">$125,430</div>
            <div className="text-sm text-gray-400">Balance Total</div>
          </div>
          <div className="bg-gray-700/50 rounded-lg p-4">
            <div className="text-xl font-bold text-green-400">+$1,234</div>
            <div className="text-sm text-gray-400">P&L Diario</div>
          </div>
          <div className="bg-gray-700/50 rounded-lg p-4">
            <div className="text-xl font-bold text-blue-400">78.3%</div>
            <div className="text-sm text-gray-400">Precisión IA</div>
          </div>
        </div>
      </div>
    </div>
  </div>
);

const Analysis = () => (
  <div className="p-6">
    <div className="trading-card p-6">
      <h2 className="text-2xl font-bold text-white mb-4">Análisis Avanzado</h2>
      <div className="bg-gray-800/50 rounded-lg p-8 text-center">
        <div className="w-16 h-16 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center mx-auto mb-4">
          <span className="text-2xl">📊</span>
        </div>
        <h3 className="text-lg font-semibold text-white mb-2">Análisis Técnico y Fundamental</h3>
        <p className="text-gray-400 mb-4">Herramientas avanzadas de análisis de mercado</p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-gray-700/50 rounded-lg p-4">
            <div className="text-lg font-semibold text-white mb-2">Análisis Técnico</div>
            <div className="text-sm text-gray-400">Indicadores, patrones y señales</div>
          </div>
          <div className="bg-gray-700/50 rounded-lg p-4">
            <div className="text-lg font-semibold text-white mb-2">Análisis Fundamental</div>
            <div className="text-sm text-gray-400">Noticias, eventos y datos económicos</div>
          </div>
        </div>
      </div>
    </div>
  </div>
);

const AIMonitor = () => (
  <div className="p-6">
    <div className="trading-card p-6">
      <h2 className="text-2xl font-bold text-white mb-4">Monitor de IA</h2>
      <div className="bg-gray-800/50 rounded-lg p-8 text-center">
        <div className="w-16 h-16 bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full flex items-center justify-center mx-auto mb-4">
          <span className="text-2xl">🤖</span>
        </div>
        <h3 className="text-lg font-semibold text-white mb-2">Monitoreo de Inteligencia Artificial</h3>
        <p className="text-gray-400 mb-4">Seguimiento en tiempo real del rendimiento de los modelos IA</p>
        <div className="space-y-4">
          <div className="flex items-center justify-between bg-gray-700/50 rounded-lg p-4">
            <div>
              <div className="font-semibold text-white">IA Tradicional</div>
              <div className="text-sm text-gray-400">Random Forest + LSTM</div>
            </div>
            <div className="text-right">
              <div className="text-green-400 font-semibold">78.3%</div>
              <div className="text-sm text-gray-400">Precisión</div>
            </div>
          </div>
          <div className="flex items-center justify-between bg-gray-700/50 rounded-lg p-4">
            <div>
              <div className="font-semibold text-white">Reinforcement Learning</div>
              <div className="text-sm text-gray-400">DQN Agent</div>
            </div>
            <div className="text-right">
              <div className="text-blue-400 font-semibold">Entrenando</div>
              <div className="text-sm text-gray-400">Epoch 1,234</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
);

const RLDashboardComponent = () => (
  <div className="p-6">
    <div className="trading-card p-6">
      <h2 className="text-2xl font-bold text-white mb-4">Reinforcement Learning</h2>
      <div className="bg-gray-800/50 rounded-lg p-8 text-center">
        <div className="w-16 h-16 bg-gradient-to-r from-orange-500 to-red-500 rounded-full flex items-center justify-center mx-auto mb-4">
          <span className="text-2xl">⚡</span>
        </div>
        <h3 className="text-lg font-semibold text-white mb-2">Agente de Aprendizaje por Refuerzo</h3>
        <p className="text-gray-400 mb-4">Entrenamiento y optimización de estrategias de trading</p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-gray-700/50 rounded-lg p-4">
            <div className="text-xl font-bold text-white">1,234</div>
            <div className="text-sm text-gray-400">Epocas</div>
          </div>
          <div className="bg-gray-700/50 rounded-lg p-4">
            <div className="text-xl font-bold text-green-400">0.85</div>
            <div className="text-sm text-gray-400">Reward</div>
          </div>
          <div className="bg-gray-700/50 rounded-lg p-4">
            <div className="text-xl font-bold text-blue-400">72.1%</div>
            <div className="text-sm text-gray-400">Win Rate</div>
          </div>
        </div>
      </div>
    </div>
  </div>
);

const Alerts = () => (
  <div className="p-6">
    <div className="trading-card p-6">
      <h2 className="text-2xl font-bold text-white mb-4">Sistema de Alertas</h2>
      <div className="bg-gray-800/50 rounded-lg p-8 text-center">
        <div className="w-16 h-16 bg-gradient-to-r from-red-500 to-pink-500 rounded-full flex items-center justify-center mx-auto mb-4">
          <span className="text-2xl">🔔</span>
        </div>
        <h3 className="text-lg font-semibold text-white mb-2">Notificaciones Inteligentes</h3>
        <p className="text-gray-400 mb-4">Alertas personalizadas para oportunidades de trading</p>
        <div className="space-y-3">
          <div className="flex items-center justify-between bg-gray-700/50 rounded-lg p-4">
            <div className="flex items-center space-x-3">
              <div className="w-3 h-3 bg-green-400 rounded-full"></div>
              <div>
                <div className="font-semibold text-white">Señal de Compra</div>
                <div className="text-sm text-gray-400">EUR/USD - Nivel de soporte alcanzado</div>
              </div>
            </div>
            <div className="text-sm text-gray-400">2 min</div>
          </div>
          <div className="flex items-center justify-between bg-gray-700/50 rounded-lg p-4">
            <div className="flex items-center space-x-3">
              <div className="w-3 h-3 bg-red-400 rounded-full"></div>
              <div>
                <div className="font-semibold text-white">Stop Loss</div>
                <div className="text-sm text-gray-400">GBP/USD - Posición cerrada</div>
              </div>
            </div>
            <div className="text-sm text-gray-400">5 min</div>
          </div>
        </div>
      </div>
    </div>
  </div>
);

const Reports = () => (
  <div className="p-6">
    <div className="trading-card p-6">
      <h2 className="text-2xl font-bold text-white mb-4">Reportes y Análisis</h2>
      <div className="bg-gray-800/50 rounded-lg p-8 text-center">
        <div className="w-16 h-16 bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full flex items-center justify-center mx-auto mb-4">
          <span className="text-2xl">📋</span>
        </div>
        <h3 className="text-lg font-semibold text-white mb-2">Reportes Detallados</h3>
        <p className="text-gray-400 mb-4">Análisis completo de rendimiento y estrategias</p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-gray-700/50 rounded-lg p-4">
            <div className="text-lg font-semibold text-white mb-2">Reporte Diario</div>
            <div className="text-sm text-gray-400">Resumen de operaciones y P&L</div>
          </div>
          <div className="bg-gray-700/50 rounded-lg p-4">
            <div className="text-lg font-semibold text-white mb-2">Reporte Semanal</div>
            <div className="text-sm text-gray-400">Análisis de tendencias y patrones</div>
          </div>
        </div>
      </div>
    </div>
  </div>
);

const Community = () => (
  <div className="p-6">
    <div className="trading-card p-6">
      <h2 className="text-2xl font-bold text-white mb-4">Comunidad de Traders</h2>
      <div className="bg-gray-800/50 rounded-lg p-8 text-center">
        <div className="w-16 h-16 bg-gradient-to-r from-green-500 to-emerald-500 rounded-full flex items-center justify-center mx-auto mb-4">
          <span className="text-2xl">👥</span>
        </div>
        <h3 className="text-lg font-semibold text-white mb-2">Conecta con Otros Traders</h3>
        <p className="text-gray-400 mb-4">Comparte estrategias y aprende de la comunidad</p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-gray-700/50 rounded-lg p-4">
            <div className="text-xl font-bold text-white">1,247</div>
            <div className="text-sm text-gray-400">Miembros</div>
          </div>
          <div className="bg-gray-700/50 rounded-lg p-4">
            <div className="text-xl font-bold text-blue-400">89</div>
            <div className="text-sm text-gray-400">Estrategias</div>
          </div>
          <div className="bg-gray-700/50 rounded-lg p-4">
            <div className="text-xl font-bold text-green-400">24/7</div>
            <div className="text-sm text-gray-400">Soporte</div>
          </div>
        </div>
      </div>
    </div>
  </div>
);

const Help = () => (
  <div className="p-6">
    <div className="trading-card p-6">
      <h2 className="text-2xl font-bold text-white mb-4">Centro de Ayuda</h2>
      <div className="bg-gray-800/50 rounded-lg p-8 text-center">
        <div className="w-16 h-16 bg-gradient-to-r from-yellow-500 to-orange-500 rounded-full flex items-center justify-center mx-auto mb-4">
          <span className="text-2xl">❓</span>
        </div>
        <h3 className="text-lg font-semibold text-white mb-2">Soporte y Documentación</h3>
        <p className="text-gray-400 mb-4">Recursos para maximizar tu experiencia de trading</p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-gray-700/50 rounded-lg p-4">
            <div className="text-lg font-semibold text-white mb-2">Documentación</div>
            <div className="text-sm text-gray-400">Guías y tutoriales completos</div>
          </div>
          <div className="bg-gray-700/50 rounded-lg p-4">
            <div className="text-lg font-semibold text-white mb-2">Soporte Técnico</div>
            <div className="text-sm text-gray-400">Asistencia personalizada</div>
          </div>
        </div>
      </div>
    </div>
  </div>
);

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');

  const renderContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return <Dashboard />;
      case 'trading':
        return <TradingView />;
      case 'portfolio':
        return <Portfolio />;
      case 'analysis':
        return <Analysis />;
      case 'ai-monitor':
        return <AIMonitor />;
      case 'rl':
        return <RLDashboardComponent />;
      case 'alerts':
        return <Alerts />;
      case 'reports':
        return <Reports />;
      case 'community':
        return <Community />;
      case 'help':
        return <Help />;
      default:
        return <Dashboard />;
    }
  };

  return (
    <Layout activeTab={activeTab} onTabChange={setActiveTab}>
      {renderContent()}
    </Layout>
  );
}

export default App; 