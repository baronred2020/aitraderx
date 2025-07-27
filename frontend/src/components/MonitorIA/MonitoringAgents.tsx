import React, { useState, useEffect } from 'react';
import { 
  AlertTriangle, 
  Activity, 
  Shield, 
  Clock, 
  BarChart3, 
  Settings, 
  Play, 
  Pause, 
  RefreshCw,
  Eye,
  EyeOff,
  Bell,
  BellOff,
  CheckCircle,
  XCircle,
  Info,
  Zap,
  TrendingUp,
  TrendingDown,
  AlertCircle,
  Star,
  Users,
  Target,
  Gauge,
  Database,
  Cpu,
  Network,
  HardDrive,
  Wifi,
  WifiOff
} from 'lucide-react';
import { useMonitoringAgents } from '../../hooks/useMonitoringAgents';
import { useAuth } from '../../contexts/AuthContext';
import { useFeatureAccess } from '../../hooks/useFeatureAccess';
import { MonitoringAlert, MonitoringAgentStatus } from '../../services/api';

interface MonitoringAgentsProps {
  currentPair?: string;
  currentBrain?: string;
}

const MonitoringAgents: React.FC<MonitoringAgentsProps> = ({ 
  currentPair = 'EURUSD', 
  currentBrain = 'brain_max' 
}) => {
  const { subscription } = useAuth();
  const { checkFeature } = useFeatureAccess();
  const {
    alerts,
    systemStatus,
    config,
    agentsStatus,
    loading,
    errors,
    loadAlerts,
    loadSystemStatus,
    loadConfig,
    markAlertAsRead,
    startMonitoring,
    stopMonitoring,
    updateConfig,
    getAlertsBySeverity,
    getAlertsByAgent,
    getUnreadAlertsCount,
    getCriticalAlertsCount,
    isMonitoringAvailable,
    getSubscriptionLimits
  } = useMonitoringAgents();

  const [selectedAgent, setSelectedAgent] = useState<string>('all');
  const [selectedSeverity, setSelectedSeverity] = useState<string>('all');
  const [showConfig, setShowConfig] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);

  // Auto-refresh cada 30 segundos
  useEffect(() => {
    if (!autoRefresh || !isMonitoringAvailable()) return;

    const interval = setInterval(() => {
      loadAlerts();
      loadSystemStatus();
    }, 30000);

    return () => clearInterval(interval);
  }, [autoRefresh, isMonitoringAvailable, loadAlerts, loadSystemStatus]);

  // Verificar si el monitoreo está disponible según la suscripción
  if (!isMonitoringAvailable()) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-center text-gray-400">
          <AlertTriangle className="w-6 h-6 mr-2" />
          <span>Los Agentes de Monitoreo no están disponibles en tu plan actual</span>
        </div>
      </div>
    );
  }

  const getAgentIcon = (agentType: string) => {
    switch (agentType) {
      case 'technical': return <BarChart3 className="w-5 h-5" />;
      case 'ai': return <Cpu className="w-5 h-5" />;
      case 'risk': return <Shield className="w-5 h-5" />;
      case 'temporal': return <Clock className="w-5 h-5" />;
      case 'fundamental': return <Database className="w-5 h-5" />;
      default: return <Activity className="w-5 h-5" />;
    }
  };

  const getAgentColor = (agentType: string) => {
    switch (agentType) {
      case 'technical': return 'text-blue-400';
      case 'ai': return 'text-purple-400';
      case 'risk': return 'text-red-400';
      case 'temporal': return 'text-green-400';
      case 'fundamental': return 'text-yellow-400';
      default: return 'text-gray-400';
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'text-red-500 bg-red-500/10 border-red-500/20';
      case 'high': return 'text-orange-500 bg-orange-500/10 border-orange-500/20';
      case 'medium': return 'text-yellow-500 bg-yellow-500/10 border-yellow-500/20';
      case 'low': return 'text-blue-500 bg-blue-500/10 border-blue-500/20';
      default: return 'text-gray-500 bg-gray-500/10 border-gray-500/20';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'monitoring': return 'text-green-500';
      case 'idle': return 'text-gray-500';
      case 'error': return 'text-red-500';
      case 'maintenance': return 'text-yellow-500';
      default: return 'text-gray-500';
    }
  };

  const filteredAlerts = alerts.filter(alert => {
    if (selectedAgent !== 'all' && alert.agent_type !== selectedAgent) return false;
    if (selectedSeverity !== 'all' && alert.severity !== selectedSeverity) return false;
    return true;
  });

  const subscriptionLimits = getSubscriptionLimits();

  return (
    <div className="space-y-6">
      {/* Header con estadísticas */}
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-blue-500/20 rounded-lg">
              <Activity className="w-6 h-6 text-blue-400" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-white">Agentes de Monitoreo</h3>
              <p className="text-sm text-gray-400">Monitoreo inteligente en tiempo real</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`p-2 rounded-lg transition-colors ${
                autoRefresh 
                  ? 'bg-green-500/20 text-green-400' 
                  : 'bg-gray-600 text-gray-400'
              }`}
              title={autoRefresh ? 'Auto-refresh activado' : 'Auto-refresh desactivado'}
            >
              {autoRefresh ? <RefreshCw className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
            </button>
            
            <button
              onClick={() => {
                loadAlerts();
                loadSystemStatus();
              }}
              disabled={loading.alerts || loading.systemStatus}
              className="p-2 bg-blue-500/20 text-blue-400 rounded-lg hover:bg-blue-500/30 transition-colors disabled:opacity-50"
              title="Actualizar datos"
            >
              <RefreshCw className={`w-4 h-4 ${loading.alerts || loading.systemStatus ? 'animate-spin' : ''}`} />
            </button>
            
            {checkFeature('monitoring_config') && (
              <button
                onClick={() => setShowConfig(!showConfig)}
                className="p-2 bg-gray-600 text-gray-400 rounded-lg hover:bg-gray-500 transition-colors"
                title="Configuración"
              >
                <Settings className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>

        {/* Estadísticas del sistema */}
        {systemStatus && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-gray-700/50 rounded-lg p-3">
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${
                  systemStatus.overall_status === 'healthy' ? 'bg-green-500' :
                  systemStatus.overall_status === 'warning' ? 'bg-yellow-500' : 'bg-red-500'
                }`} />
                <span className="text-sm text-gray-400">Estado</span>
              </div>
              <p className="text-lg font-semibold text-white capitalize">
                {systemStatus.overall_status}
              </p>
            </div>
            
            <div className="bg-gray-700/50 rounded-lg p-3">
              <div className="flex items-center space-x-2">
                <Users className="w-4 h-4 text-blue-400" />
                <span className="text-sm text-gray-400">Agentes Activos</span>
              </div>
              <p className="text-lg font-semibold text-white">
                {systemStatus.active_agents}/5
              </p>
            </div>
            
            <div className="bg-gray-700/50 rounded-lg p-3">
              <div className="flex items-center space-x-2">
                <Bell className="w-4 h-4 text-yellow-400" />
                <span className="text-sm text-gray-400">Alertas</span>
              </div>
              <p className="text-lg font-semibold text-white">
                {systemStatus.total_alerts}
              </p>
            </div>
            
            <div className="bg-gray-700/50 rounded-lg p-3">
              <div className="flex items-center space-x-2">
                <AlertCircle className="w-4 h-4 text-red-400" />
                <span className="text-sm text-gray-400">Críticas</span>
              </div>
              <p className="text-lg font-semibold text-white">
                {systemStatus.critical_alerts}
              </p>
            </div>
          </div>
        )}

        {/* Límites de suscripción */}
        {subscriptionLimits && (
          <div className="mt-4 p-3 bg-blue-500/10 border border-blue-500/20 rounded-lg">
            <div className="flex items-center space-x-2 text-sm text-blue-400">
              <Info className="w-4 h-4" />
              <span>
                Límites de tu plan: {subscriptionLimits.max_alerts} alertas, {subscriptionLimits.max_agents} agentes
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Estado de los agentes */}
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h4 className="text-md font-semibold text-white mb-4">Estado de los Agentes</h4>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
          {agentsStatus.map((agent) => (
            <div key={agent.agent_type} className="bg-gray-700/50 rounded-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <div className={`flex items-center space-x-2 ${getAgentColor(agent.agent_type)}`}>
                  {getAgentIcon(agent.agent_type)}
                  <span className="text-sm font-medium capitalize">
                    {agent.agent_type.replace('_', ' ')}
                  </span>
                </div>
                
                <div className={`w-2 h-2 rounded-full ${
                  agent.is_active ? 'bg-green-500' : 'bg-gray-500'
                }`} />
              </div>
              
              <div className="space-y-2">
                <div className="flex justify-between text-xs">
                  <span className="text-gray-400">Estado:</span>
                  <span className={`${getStatusColor(agent.status)} capitalize`}>
                    {agent.status}
                  </span>
                </div>
                
                <div className="flex justify-between text-xs">
                  <span className="text-gray-400">Alertas:</span>
                  <span className="text-white">{agent.alerts_count}</span>
                </div>
                
                <div className="flex justify-between text-xs">
                  <span className="text-gray-400">Rendimiento:</span>
                  <span className="text-white">{agent.performance_score}%</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Controles de monitoreo */}
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h4 className="text-md font-semibold text-white mb-4">Controles de Monitoreo</h4>
        
        <div className="flex flex-wrap gap-4">
          <button
            onClick={() => startMonitoring(currentPair, currentBrain)}
            disabled={loading.startMonitoring}
            className="flex items-center space-x-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors disabled:opacity-50"
          >
            <Play className="w-4 h-4" />
            <span>Iniciar Monitoreo</span>
          </button>
          
          <button
            onClick={() => stopMonitoring(currentPair)}
            disabled={loading.stopMonitoring}
            className="flex items-center space-x-2 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors disabled:opacity-50"
          >
            <Pause className="w-4 h-4" />
            <span>Detener Monitoreo</span>
          </button>
        </div>
        
        {errors.startMonitoring && (
          <div className="mt-3 p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
            <p className="text-sm text-red-400">{errors.startMonitoring}</p>
          </div>
        )}
        
        {errors.stopMonitoring && (
          <div className="mt-3 p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
            <p className="text-sm text-red-400">{errors.stopMonitoring}</p>
          </div>
        )}
      </div>

      {/* Filtros de alertas */}
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-4">
          <h4 className="text-md font-semibold text-white">Alertas de Monitoreo</h4>
          <div className="flex items-center space-x-2 text-sm text-gray-400">
            <span>{filteredAlerts.length} alertas</span>
            <span>•</span>
            <span>{getUnreadAlertsCount()} sin leer</span>
          </div>
        </div>
        
        {/* Filtros */}
        <div className="flex flex-wrap gap-4 mb-4">
          <select
            value={selectedAgent}
            onChange={(e) => setSelectedAgent(e.target.value)}
            className="px-3 py-2 bg-gray-700 text-white rounded-lg border border-gray-600 focus:border-blue-500 focus:outline-none"
          >
            <option value="all">Todos los agentes</option>
            <option value="technical">Técnico</option>
            <option value="ai">IA</option>
            <option value="risk">Riesgo</option>
            <option value="temporal">Temporal</option>
            <option value="fundamental">Fundamental</option>
          </select>
          
          <select
            value={selectedSeverity}
            onChange={(e) => setSelectedSeverity(e.target.value)}
            className="px-3 py-2 bg-gray-700 text-white rounded-lg border border-gray-600 focus:border-blue-500 focus:outline-none"
          >
            <option value="all">Todas las severidades</option>
            <option value="critical">Crítica</option>
            <option value="high">Alta</option>
            <option value="medium">Media</option>
            <option value="low">Baja</option>
          </select>
        </div>
        
        {/* Lista de alertas */}
        <div className="space-y-3 max-h-96 overflow-y-auto">
          {loading.alerts ? (
            <div className="flex items-center justify-center py-8">
              <RefreshCw className="w-6 h-6 animate-spin text-blue-400" />
              <span className="ml-2 text-gray-400">Cargando alertas...</span>
            </div>
          ) : filteredAlerts.length === 0 ? (
            <div className="flex items-center justify-center py-8 text-gray-400">
              <CheckCircle className="w-6 h-6 mr-2" />
              <span>No hay alertas para mostrar</span>
            </div>
          ) : (
            filteredAlerts.map((alert) => (
              <div
                key={alert.id}
                className={`p-4 rounded-lg border ${
                  alert.is_read ? 'bg-gray-700/50' : 'bg-blue-500/10'
                } ${getSeverityColor(alert.severity)}`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-2">
                      <div className={`flex items-center space-x-1 ${getAgentColor(alert.agent_type)}`}>
                        {getAgentIcon(alert.agent_type)}
                        <span className="text-sm font-medium capitalize">
                          {alert.agent_type.replace('_', ' ')}
                        </span>
                      </div>
                      
                      <span className={`px-2 py-1 rounded-full text-xs font-medium capitalize ${
                        alert.severity === 'critical' ? 'bg-red-500/20 text-red-400' :
                        alert.severity === 'high' ? 'bg-orange-500/20 text-orange-400' :
                        alert.severity === 'medium' ? 'bg-yellow-500/20 text-yellow-400' :
                        'bg-blue-500/20 text-blue-400'
                      }`}>
                        {alert.severity}
                      </span>
                      
                      {!alert.is_read && (
                        <div className="w-2 h-2 bg-blue-400 rounded-full" />
                      )}
                    </div>
                    
                    <h5 className="font-medium text-white mb-1">{alert.title}</h5>
                    <p className="text-sm text-gray-300 mb-2">{alert.description}</p>
                    
                    <div className="flex items-center space-x-4 text-xs text-gray-400">
                      {alert.pair && (
                        <span>Par: {alert.pair}</span>
                      )}
                      {alert.brain_type && (
                        <span>Cerebro: {alert.brain_type}</span>
                      )}
                      <span>{new Date(alert.timestamp).toLocaleString()}</span>
                    </div>
                  </div>
                  
                  <button
                    onClick={() => markAlertAsRead(alert.id)}
                    className="ml-4 p-1 text-gray-400 hover:text-white transition-colors"
                    title={alert.is_read ? 'Ya leída' : 'Marcar como leída'}
                  >
                    {alert.is_read ? <CheckCircle className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Configuración (solo para Expert y superior) */}
      {showConfig && checkFeature('monitoring_config') && config && (
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h4 className="text-md font-semibold text-white mb-4">Configuración de Monitoreo</h4>
          
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-gray-300">Monitoreo habilitado</span>
              <button
                onClick={() => updateConfig({ enabled: !config.enabled })}
                disabled={loading.config}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  config.enabled ? 'bg-blue-600' : 'bg-gray-600'
                }`}
              >
                <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  config.enabled ? 'translate-x-6' : 'translate-x-1'
                }`} />
              </button>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Intervalo de verificación (segundos)
              </label>
              <input
                type="number"
                value={config.check_interval}
                onChange={(e) => updateConfig({ check_interval: parseInt(e.target.value) })}
                disabled={loading.config}
                className="w-full px-3 py-2 bg-gray-700 text-white rounded-lg border border-gray-600 focus:border-blue-500 focus:outline-none"
                min="10"
                max="300"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Retención de alertas (días)
              </label>
              <input
                type="number"
                value={config.alert_retention_days}
                onChange={(e) => updateConfig({ alert_retention_days: parseInt(e.target.value) })}
                disabled={loading.config}
                className="w-full px-3 py-2 bg-gray-700 text-white rounded-lg border border-gray-600 focus:border-blue-500 focus:outline-none"
                min="1"
                max="90"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MonitoringAgents; 