import React, { useState } from 'react';
import { 
  BarChart3, 
  Target, 
  PieChart, 
  Brain, 
  Zap, 
  Bell, 
  Settings, 
  Menu, 
  X,
  TrendingUp,
  DollarSign,
  Activity,
  Shield,
  Users,
  FileText,
  HelpCircle
} from 'lucide-react';

interface LayoutProps {
  children: React.ReactNode;
  activeTab: string;
  onTabChange: (tab: string) => void;
}

export const Layout: React.FC<LayoutProps> = ({ children, activeTab, onTabChange }) => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [lastUpdate, setLastUpdate] = useState(new Date());
  const [isConnected, setIsConnected] = useState(true);

  // Simular actualizaciones en tiempo real
  React.useEffect(() => {
    const interval = setInterval(() => {
      setLastUpdate(new Date());
    }, 30000);
    return () => clearInterval(interval);
  }, []);

  const navigationItems = [
    { id: 'dashboard', name: 'Dashboard', icon: BarChart3, badge: null },
    { id: 'trading', name: 'Trading', icon: Target, badge: 'LIVE' },
    { id: 'portfolio', name: 'Portfolio', icon: PieChart, badge: null },
    { id: 'analysis', name: 'Análisis', icon: TrendingUp, badge: null },
    { id: 'ai-monitor', name: 'Monitor IA', icon: Brain, badge: 'AI' },
    { id: 'rl', name: 'Reinforcement Learning', icon: Zap, badge: 'RL' },
    { id: 'alerts', name: 'Alertas', icon: Bell, badge: '3' },
    { id: 'reports', name: 'Reportes', icon: FileText, badge: null },
    { id: 'community', name: 'Comunidad', icon: Users, badge: null },
    { id: 'help', name: 'Ayuda', icon: HelpCircle, badge: null },
  ];

  const quickStats = [
    { label: 'Balance Total', value: '$125,430', change: '+2.3%', positive: true },
    { label: 'P&L Diario', value: '+$1,234', change: '+0.98%', positive: true },
    { label: 'Precisión IA', value: '78.3%', change: '+1.2%', positive: true },
    { label: 'Sharpe Ratio', value: '1.45', change: '+0.12', positive: true },
  ];

  return (
    <div className="min-h-screen w-full bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900">
      {/* Header */}
      <header className="glass-effect border-b border-gray-700/50 sticky top-0 z-50">
        <div className="flex items-center justify-between px-4 py-3">
          {/* Logo y título */}
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="lg:hidden p-2 rounded-lg hover:bg-gray-700/50 transition-colors"
            >
              {sidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
            
            <div className="flex items-center space-x-3">
              <div className="relative">
                <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-teal-500 rounded-lg flex items-center justify-center">
                  <Brain className="w-5 h-5 text-white" />
                </div>
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-400 rounded-full animate-pulse-slow"></div>
              </div>
              <div>
                <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-teal-400 bg-clip-text text-transparent">
                  AITRADERX
                </h1>
                <p className="text-xs text-gray-400">AI Trading Platform</p>
              </div>
            </div>
          </div>

          {/* Stats rápidos */}
          <div className="hidden md:flex items-center space-x-6">
            {quickStats.map((stat, index) => (
              <div key={index} className="text-center">
                <p className="text-xs text-gray-400">{stat.label}</p>
                <p className="text-sm font-semibold text-white">{stat.value}</p>
                <p className={`text-xs ${stat.positive ? 'text-green-400' : 'text-red-400'}`}>
                  {stat.change}
                </p>
              </div>
            ))}
          </div>

          {/* Status y controles */}
          <div className="flex items-center space-x-4">
            {/* Status de conexión */}
            <div className="hidden sm:flex items-center space-x-2">
              <div className={`status-indicator ${isConnected ? 'status-online' : 'status-offline'}`}></div>
              <span className="text-xs text-gray-400">
                {isConnected ? 'Conectado' : 'Desconectado'}
              </span>
            </div>

            {/* Última actualización */}
            <div className="hidden lg:flex items-center space-x-2 text-xs text-gray-400">
              <Activity className="w-4 h-4" />
              <span>{lastUpdate.toLocaleTimeString()}</span>
            </div>

            {/* Notificaciones */}
            <button className="relative p-2 rounded-lg hover:bg-gray-700/50 transition-colors">
              <Bell className="w-5 h-5" />
              <span className="absolute -top-1 -right-1 w-4 h-4 bg-red-500 text-xs rounded-full flex items-center justify-center">
                3
              </span>
            </button>

            {/* Configuración */}
            <button className="p-2 rounded-lg hover:bg-gray-700/50 transition-colors">
              <Settings className="w-5 h-5" />
            </button>
          </div>
        </div>
      </header>

      <div className="flex w-full h-full">
        {/* Sidebar */}
        <aside className={`
          fixed lg:static inset-y-0 left-0 z-40 w-64 bg-gray-900/95 backdrop-blur-xl border-r border-gray-700/50
          transform transition-transform duration-300 ease-in-out
          ${sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
        `}>
          <div className="flex flex-col h-full">
            {/* Logo en sidebar */}
            <div className="p-6 border-b border-gray-700/50">
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-teal-500 rounded-lg flex items-center justify-center">
                  <Brain className="w-5 h-5 text-white" />
                </div>
                <span className="text-lg font-bold text-white">AITRADERX</span>
              </div>
            </div>

            {/* Navegación */}
            <nav className="flex-1 px-4 py-6 space-y-2">
              {navigationItems.map((item) => (
                <button
                  key={item.id}
                  onClick={() => {
                    onTabChange(item.id);
                    setSidebarOpen(false);
                  }}
                  className={`
                    w-full flex items-center justify-between px-4 py-3 rounded-lg transition-all duration-200
                    ${activeTab === item.id
                      ? 'bg-gradient-to-r from-blue-600/20 to-teal-600/20 border border-blue-500/30 text-blue-400'
                      : 'text-gray-300 hover:bg-gray-800/50 hover:text-white'
                    }
                  `}
                >
                  <div className="flex items-center space-x-3">
                    <item.icon className="w-5 h-5" />
                    <span className="font-medium">{item.name}</span>
                  </div>
                  {item.badge && (
                    <span className={`
                      px-2 py-1 text-xs font-semibold rounded-full
                      ${item.badge === 'LIVE' ? 'bg-red-500/20 text-red-400' :
                        item.badge === 'AI' ? 'bg-purple-500/20 text-purple-400' :
                        item.badge === 'RL' ? 'bg-orange-500/20 text-orange-400' :
                        'bg-blue-500/20 text-blue-400'}
                    `}>
                      {item.badge}
                    </span>
                  )}
                </button>
              ))}
            </nav>

            {/* Footer del sidebar */}
            <div className="p-4 border-t border-gray-700/50">
              <div className="bg-gray-800/50 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-300">Estado del Sistema</span>
                  <div className="status-indicator status-online"></div>
                </div>
                <div className="space-y-1 text-xs text-gray-400">
                  <div className="flex justify-between">
                    <span>IA Tradicional</span>
                    <span className="text-green-400">✅ Activo</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Reinforcement Learning</span>
                    <span className="text-blue-400">🔄 Entrenando</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Auto-entrenamiento</span>
                    <span className="text-purple-400">📊 Monitoreando</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </aside>

        {/* Overlay para móvil */}
        {sidebarOpen && (
          <div
            className="fixed inset-0 bg-black/50 z-30 lg:hidden"
            onClick={() => setSidebarOpen(false)}
          />
        )}

        {/* Contenido principal */}
        <main className="flex-1 min-h-screen w-full h-full flex flex-col">
          <div className="flex-1 w-full h-full animate-fade-in">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}; 