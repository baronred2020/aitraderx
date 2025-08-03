import React, { useState, useEffect } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { 
  BarChart3, 
  Target, 
  PieChart, 
  TrendingUp, 
  Brain, 
  Crown, 
  Zap, 
  Bell, 
  FileText, 
  CreditCard, 
  Users, 
  HelpCircle,
  Menu,
  X,
  User,
  Activity,
  Settings,
  LogOut
} from 'lucide-react';

interface LayoutProps {
  children: React.ReactNode;
  activeTab: string;
  onTabChange: (tab: string) => void;
  onShowUpgradeModal?: (upgradeInfo: {
    currentPlan: string;
    requiredPlan: string;
    feature: string;
  }) => void;
}

export const Layout: React.FC<LayoutProps> = ({ children, activeTab, onTabChange, onShowUpgradeModal }) => {
  const { user, subscription, logout, canAccess, isLoading } = useAuth();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [lastUpdate, setLastUpdate] = useState(new Date());
  const [isConnected, setIsConnected] = useState(true);
  const [showUserMenu, setShowUserMenu] = useState(false);

  // Debug: Monitorear estado de carga
  useEffect(() => {
    console.log('🔍 Layout: Estado de carga - isLoading:', isLoading);
    console.log('🔍 Layout: Estado de suscripción - subscription:', subscription);
  }, [isLoading, subscription]);

  // Cerrar menú de usuario al hacer click fuera
  React.useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as Element;
      if (!target.closest('.user-menu-container')) {
        setShowUserMenu(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

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
    { id: 'brain-trader', name: 'Brain Trader', icon: Brain, badge: 'AI' },
    { id: 'mega-mind', name: 'Mega Mind', icon: Crown, badge: 'INSTITUTIONAL' },
    { id: 'ai-monitor', name: 'Monitor IA', icon: Brain, badge: 'AI' },
    { id: 'rl', name: 'Reinforcement Learning', icon: Zap, badge: 'RL' },
    { id: 'alerts', name: 'Alertas', icon: Bell, badge: '3' },
    { id: 'reports', name: 'Reportes', icon: FileText, badge: null },
    { id: 'subscriptions', name: 'Suscripciones', icon: CreditCard, badge: null },
    { id: 'community', name: 'Comunidad', icon: Users, badge: null },
    { id: 'help', name: 'Ayuda', icon: HelpCircle, badge: null },
  ];

  const quickStats = [
    { label: 'Balance Total', value: '$125,430', change: '+2.3%', positive: true },
    { label: 'P&L Diario', value: '+$1,234', change: '+0.98%', positive: true },
    { label: 'Precisión IA', value: '78.3%', change: '+1.2%', positive: true },
    { label: 'Sharpe Ratio', value: '1.45', change: '+0.12', positive: true },
  ];

  const handleTabClick = (tabId: string) => {
    console.log(`🔍 Layout: handleTabClick llamado con tabId: ${tabId}`);
    
    // Verificar si el usuario tiene acceso a esta sección
    const hasAccess = canAccess(tabId);
    console.log(`🔍 Layout: canAccess(${tabId}) retornó: ${hasAccess}`);
    
    if (hasAccess) {
      console.log(`✅ Layout: Navegando a ${tabId}`);
      onTabChange(tabId);
      setSidebarOpen(false);
    } else {
      console.log(`🚫 Layout: Acceso denegado a ${tabId}, mostrando modal de upgrade`);
      
      // Buscar la configuración de la característica
      const featureConfig = {
        'ai-monitor': { requiredPlan: 'expert', feature: 'ai-monitor' },
        'rl': { requiredPlan: 'premium', feature: 'reinforcement_learning' },
        'reports': { requiredPlan: 'expert', feature: 'reports' },
        'alerts': { requiredPlan: 'trader', feature: 'alerts' },
        'mt4_integration': { requiredPlan: 'expert', feature: 'mt4_integration' },
        'api_access': { requiredPlan: 'premium', feature: 'api_access' },
        'custom_models': { requiredPlan: 'premium', feature: 'custom_models' },
        'brain-trader': { requiredPlan: 'starter', feature: 'brain_trader_basic' },
        'mega-mind': { requiredPlan: 'premium', feature: 'mega_mind_institutional' },
        'subscriptions': { requiredPlan: 'starter', feature: 'subscription_management' },
        'community': { requiredPlan: 'trader', feature: 'community_access' },
        'help': { requiredPlan: 'starter', feature: 'help_support' },
        'monitoring_agents': { requiredPlan: 'trader', feature: 'monitoring_agents' },
        'monitoring_alerts': { requiredPlan: 'trader', feature: 'monitoring_alerts' },
        'monitoring_config': { requiredPlan: 'expert', feature: 'monitoring_config' }
      };
      
      const config = featureConfig[tabId as keyof typeof featureConfig];
      if (config && onShowUpgradeModal) {
        console.log(`🔍 Layout: Configurando modal para ${tabId}:`, config);
        onShowUpgradeModal({
          currentPlan: subscription?.planType || 'starter',
          requiredPlan: config.requiredPlan,
          feature: config.feature
        });
      }
    }
  };

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

            {/* Información del usuario */}
            <div className="relative user-menu-container">
              <button 
                onClick={() => setShowUserMenu(!showUserMenu)}
                className="flex items-center space-x-2 p-2 rounded-lg hover:bg-gray-700/50 transition-colors"
              >
                <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-teal-500 rounded-full flex items-center justify-center">
                  <User className="w-4 h-4 text-white" />
                </div>
                <div className="hidden sm:block text-left">
                  <p className="text-sm font-medium text-white">{user?.username}</p>
                  <div className="flex items-center space-x-1">
                    {subscription?.planType === 'premium' && <Crown className="w-3 h-3 text-yellow-400" />}
                    {subscription?.planType === 'expert' && <Brain className="w-3 h-3 text-purple-400" />}
                    {subscription?.planType === 'trader' && <Zap className="w-3 h-3 text-blue-400" />}
                    <span className="text-xs text-gray-400 capitalize">
                      {subscription?.planType || 'starter'}
                    </span>
                  </div>
                </div>
              </button>

              {/* Menú desplegable del usuario */}
              {showUserMenu && (
                <div className="absolute right-0 top-full mt-2 w-64 glass-effect rounded-lg border border-gray-700/50 p-4 z-50">
                  <div className="space-y-4">
                    {/* Información del usuario */}
                    <div className="border-b border-gray-700/50 pb-3">
                      <div className="flex items-center space-x-3">
                        <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-teal-500 rounded-full flex items-center justify-center">
                          <User className="w-5 h-5 text-white" />
                        </div>
                        <div>
                          <p className="font-medium text-white">{user?.username}</p>
                          <p className="text-sm text-gray-400">{user?.email}</p>
                        </div>
                      </div>
                    </div>

                    {/* Información de suscripción */}
                    <div className="border-b border-gray-700/50 pb-3">
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-400">Plan Actual:</span>
                        <div className="flex items-center space-x-1">
                          {subscription?.planType === 'premium' && <Crown className="w-3 h-3 text-yellow-400" />}
                          {subscription?.planType === 'expert' && <Brain className="w-3 h-3 text-purple-400" />}
                          {subscription?.planType === 'trader' && <Zap className="w-3 h-3 text-blue-400" />}
                          <span className="text-sm font-medium text-white capitalize">
                            {subscription?.planType || 'starter'}
                          </span>
                        </div>
                      </div>
                      <div className="flex items-center justify-between mt-1">
                        <span className="text-sm text-gray-400">Estado:</span>
                        <span className={`text-sm font-medium ${
                          subscription?.status === 'active' ? 'text-green-400' : 'text-red-400'
                        }`}>
                          {subscription?.status === 'active' ? 'Activo' : 'Inactivo'}
                        </span>
                      </div>
                    </div>

                    {/* Acciones */}
                    <div className="space-y-2">
                      <button className="w-full flex items-center space-x-2 px-3 py-2 rounded-lg hover:bg-gray-700/50 transition-colors text-left">
                        <Settings className="w-4 h-4 text-gray-400" />
                        <span className="text-sm text-gray-300">Configuración</span>
                      </button>
                      <button 
                        onClick={logout}
                        className="w-full flex items-center space-x-2 px-3 py-2 rounded-lg hover:bg-red-500/20 transition-colors text-left"
                      >
                        <LogOut className="w-4 h-4 text-red-400" />
                        <span className="text-sm text-red-400">Cerrar Sesión</span>
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>
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
              {isLoading ? (
                // Mostrar loading mientras se carga la suscripción
                <div className="space-y-2">
                  {navigationItems.map((item) => (
                    <div
                      key={item.id}
                      className="w-full flex items-center justify-between px-4 py-3 rounded-lg bg-gray-800/30 animate-pulse"
                    >
                      <div className="flex items-center space-x-3">
                        <div className="w-5 h-5 bg-gray-600 rounded"></div>
                        <div className="h-4 bg-gray-600 rounded w-20"></div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                // Mostrar menú normal cuando la suscripción está cargada
                navigationItems.map((item) => {
                  const hasAccess = canAccess(item.id);
                  return (
                    <button
                      key={item.id}
                      onClick={() => handleTabClick(item.id)}
                      className={`
                        w-full flex items-center justify-between px-4 py-3 rounded-lg transition-all duration-200
                        ${activeTab === item.id
                          ? 'bg-gradient-to-r from-blue-600/20 to-teal-600/20 border border-blue-500/30 text-blue-400'
                          : hasAccess 
                            ? 'text-gray-300 hover:bg-gray-800/50 hover:text-white'
                            : 'text-gray-500 hover:bg-gray-800/30 hover:text-gray-400'
                        }
                      `}
                    >
                      <div className="flex items-center space-x-3">
                        <item.icon className={`w-5 h-5 ${!hasAccess ? 'opacity-50' : ''}`} />
                        <span className="font-medium">{item.name}</span>
                      </div>
                      {item.badge && hasAccess && (
                        <span className={`
                          px-2 py-1 text-xs font-semibold rounded-full
                          ${item.badge === 'LIVE' ? 'bg-red-500/20 text-red-400' :
                            item.badge === 'AI' ? 'bg-purple-500/20 text-purple-400' :
                            item.badge === 'RL' ? 'bg-orange-500/20 text-orange-400' :
                            item.badge === 'INSTITUTIONAL' ? 'bg-gradient-to-r from-purple-500/20 to-pink-500/20 text-purple-400 border border-purple-500/30' :
                            'bg-blue-500/20 text-blue-400'}
                        `}>
                          {item.badge}
                        </span>
                      )}
                    </button>
                  );
                })
              )}
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