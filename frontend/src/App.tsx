import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { Layout } from './components/Common/Layout';
import { Dashboard } from './components/Dashboard/Dashboard';
import { TradingView } from './components/Trading/TradingView';
import { Analysis } from './components/Analysis/Analysis';
import { Login } from './components/Auth/Login';
import { Register } from './components/Auth/Register';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { useFeatureAccess } from './hooks/useFeatureAccess';
import { UpgradeModal } from './components/Common/UpgradeModal';
import { ErrorBoundary } from './components/Common/ErrorBoundary';
import { Brain } from 'lucide-react';
import './index.css';
import AIMonitorPage from './components/AIMonitor';

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

function AppContent() {
  const { user, isLoading } = useAuth();
  const { requireAccess, showUpgradeModal, upgradeInfo, closeUpgradeModal } = useFeatureAccess();
  const [activeTab, setActiveTab] = useState('dashboard');
  const [currentRoute, setCurrentRoute] = useState('login');

  // Detectar la ruta actual
  useEffect(() => {
    const path = window.location.pathname;
    if (path === '/register') {
      setCurrentRoute('register');
    } else if (path === '/login') {
      setCurrentRoute('login');
    }
  }, []);

  // Memoizar el handler de cambio de tab
  const handleTabChange = useCallback((tab: string) => {
    setActiveTab(tab);
  }, []);

  // Memoizar el contenido para evitar re-renders infinitos
  const content = useMemo(() => {
    // Verificar acceso antes de renderizar cada sección
    switch (activeTab) {
      case 'dashboard':
        return <Dashboard />;
      case 'trading':
        if (requireAccess('trading')) {
          return <TradingView />;
        }
        return null;
      case 'portfolio':
        if (requireAccess('portfolio')) {
          return <Portfolio />;
        }
        return null;
      case 'analysis':
        if (requireAccess('analysis')) {
          return <Analysis />;
        }
        return null;
      case 'ai-monitor':
        if (requireAccess('ai-monitor')) {
          return <AIMonitorPage />;
        }
        return null;
      case 'rl':
        if (requireAccess('rl')) {
          return <RLDashboardComponent />;
        }
        return null;
      case 'alerts':
        if (requireAccess('alerts')) {
          return <Alerts />;
        }
        return null;
      case 'reports':
        if (requireAccess('reports')) {
          return <Reports />;
        }
        return null;
      case 'community':
        if (requireAccess('community')) {
          return <Community />;
        }
        return null;
      case 'help':
        return <Help />;
      default:
        return <Dashboard />;
    }
  }, [activeTab, requireAccess]);

  // Mostrar loading mientras se verifica la autenticación
  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 flex items-center justify-center p-4">
        <div className="text-center max-w-md w-full">
          <div className="flex items-center justify-center mb-8">
            <div className="w-24 h-24 bg-gradient-to-r from-blue-500 to-teal-500 rounded-3xl flex items-center justify-center animate-pulse shadow-2xl">
              <Brain className="w-12 h-12 text-white" />
            </div>
          </div>
          <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-400 to-teal-400 bg-clip-text text-transparent mb-4 animate-pulse">
            AITRADERX
          </h1>
          <p className="text-gray-400 text-xl mb-8">AI Trading Platform</p>
          <div className="space-y-4">
            <div className="w-48 h-3 bg-gray-700 rounded-full mx-auto overflow-hidden">
              <div className="h-full bg-gradient-to-r from-blue-500 to-teal-500 rounded-full animate-pulse transition-all duration-1000" style={{width: '70%'}}></div>
            </div>
            <p className="text-gray-500 text-base">Inicializando sistema...</p>
          </div>
        </div>
      </div>
    );
  }

  // Si no hay usuario autenticado, mostrar login o registro según la ruta
  if (!user) {
    if (currentRoute === 'register') {
      return <Register />;
    }
    return <Login />;
  }

  return (
    <>
      <Layout activeTab={activeTab} onTabChange={handleTabChange}>
        {content}
      </Layout>
      {upgradeInfo && (
        <UpgradeModal
          isOpen={showUpgradeModal}
          onClose={closeUpgradeModal}
          currentPlan={upgradeInfo.currentPlan}
          requiredPlan={upgradeInfo.requiredPlan}
          feature={upgradeInfo.feature}
        />
      )}
    </>
  );
}

function App() {
  return (
    <ErrorBoundary>
      <AuthProvider>
        <AppContent />
      </AuthProvider>
    </ErrorBoundary>
  );
}

export default App; 