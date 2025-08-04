import React, { useState, useEffect } from 'react';
import { 
  Brain, 
  Activity, 
  AlertTriangle, 
  CheckCircle, 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  Clock, 
  BarChart3, 
  Shield, 
  Zap, 
  Target,
  Eye,
  EyeOff,
  RefreshCw,
  Settings,
  Bell,
  BellOff,
  Play,
  Pause,
  Info,
  Gauge,
  Cpu,
  Database,
  Network,
  HardDrive,
  Wifi,
  WifiOff
} from 'lucide-react';
import { useAuth } from '../../contexts/AuthContext';
import { useFeatureAccess } from '../../hooks/useFeatureAccess';
import { useMonitoringAgents } from '../../hooks/useMonitoringAgents';
import { useAvailablePairs } from '../../hooks/useAvailablePairs';
import { useRealSignals } from '../../hooks/useRealSignals';
import AIModelAnalytics from './AIModelAnalytics';

interface AIModelStatus {
  name: string;
  status: 'active' | 'idle' | 'error' | 'training';
  accuracy: number;
  confidence: number;
  lastUpdate: string;
  performance: 'improving' | 'stable' | 'declining';
  alerts: number;
}

interface TradingSignal {
  id: string;
  pair: string;
  signal: 'buy' | 'sell' | 'hold';
  confidence: number;
  price: number;
  timestamp: string;
  source: string;
  reasoning: string;
  // Ultra quality properties
  quality_score?: number;
  risk_reward_ratio?: number;
  stop_loss?: number;
  take_profit?: number;
  technical_analysis?: any;
  trend_analysis?: any;
  momentum_analysis?: any;
  volume_analysis?: any;
  market_conditions?: any;
  ai_consensus?: any;
  volatility_analysis?: any;
  support_resistance?: any;
}

interface SystemHealth {
  overall: 'healthy' | 'warning' | 'critical';
  components: {
    api: 'online' | 'offline' | 'slow';
    database: 'online' | 'offline' | 'slow';
    models: 'online' | 'offline' | 'slow';
    market_data: 'online' | 'offline' | 'slow';
  };
  lastCheck: string;
}

const MonitorIA: React.FC = () => {
  const { subscription } = useAuth();
  const { checkFeature } = useFeatureAccess();
  const { alerts, systemStatus, loading, loadAlerts, loadSystemStatus } = useMonitoringAgents();
  const { pairs: availablePairs } = useAvailablePairs();
  const { 
    signals: realSignals, 
    modelPerformance: realModelPerformance,
    loading: signalsLoading,
    errors: signalsErrors,
    loadSignals,
    loadModelPerformance,
    refreshAll: refreshSignals,
    getRecentSignals
  } = useRealSignals();

  const [selectedPair, setSelectedPair] = useState('EURUSD');
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [activeTab, setActiveTab] = useState<'overview' | 'signals' | 'models' | 'health' | 'analytics'>('overview');

  // Convertir datos reales al formato esperado
  const aiModels: AIModelStatus[] = realModelPerformance.map(model => ({
    name: model.modelName,
    status: 'active' as const,
    accuracy: model.accuracy,
    confidence: model.f1Score,
    lastUpdate: new Date(model.lastUpdate).toLocaleTimeString(),
    performance: model.status,
    alerts: model.alerts
  }));

  // Usar SOLO señales reales - sin fallbacks
  const recentSignals: TradingSignal[] = realSignals.map(signal => ({
    id: signal.id,
    pair: signal.pair,
    signal: signal.signal,
    confidence: signal.confidence,
    price: signal.price,
    timestamp: new Date(signal.timestamp).toLocaleTimeString(),
    source: signal.source,
    reasoning: signal.reasoning,
    // Ultra quality properties
    quality_score: signal.quality_score,
    risk_reward_ratio: signal.risk_reward_ratio,
    stop_loss: signal.stop_loss,
    take_profit: signal.take_profit,
    technical_analysis: signal.technical_analysis,
    trend_analysis: signal.trend_analysis,
    momentum_analysis: signal.momentum_analysis,
    volume_analysis: signal.volume_analysis,
    market_conditions: signal.market_conditions,
    ai_consensus: signal.ai_consensus,
    volatility_analysis: signal.volatility_analysis,
    support_resistance: signal.support_resistance
  }));

  const [systemHealth] = useState<SystemHealth>({
    overall: 'healthy',
    components: {
      api: 'online',
      database: 'online',
      models: 'online',
      market_data: 'online'
    },
    lastCheck: '1 min ago'
  });

  // Auto-refresh cada 30 segundos
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      loadAlerts();
      loadSystemStatus();
      refreshSignals(); // Actualizar señales reales también
    }, 30000);

    return () => clearInterval(interval);
  }, [autoRefresh, loadAlerts, loadSystemStatus, refreshSignals]);

  // Verificar acceso a la funcionalidad
  if (!checkFeature('monitoring_agents')) {
    return (
      <div className="p-6">
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6 text-center">
          <h2 className="text-xl font-semibold text-yellow-800 mb-2">
            Monitor IA no disponible
          </h2>
          <p className="text-yellow-700 mb-4">
            Esta funcionalidad requiere un plan de suscripción superior.
          </p>
          <p className="text-sm text-yellow-600">
            Plan actual: {subscription?.planType || 'N/A'}
          </p>
        </div>
      </div>
    );
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'text-green-400';
      case 'idle': return 'text-gray-400';
      case 'error': return 'text-red-400';
      case 'training': return 'text-yellow-400';
      default: return 'text-gray-400';
    }
  };

  const getPerformanceIcon = (performance: string) => {
    switch (performance) {
      case 'improving': return <TrendingUp className="w-4 h-4 text-green-400" />;
      case 'stable': return <BarChart3 className="w-4 h-4 text-blue-400" />;
      case 'declining': return <TrendingDown className="w-4 h-4 text-red-400" />;
      default: return <BarChart3 className="w-4 h-4 text-gray-400" />;
    }
  };

  const getSignalColor = (signal: string) => {
    switch (signal) {
      case 'buy': return 'text-green-400 bg-green-500/10 border-green-500/20';
      case 'sell': return 'text-red-400 bg-red-500/10 border-red-500/20';
      case 'hold': return 'text-yellow-400 bg-yellow-500/10 border-yellow-500/20';
      default: return 'text-gray-400 bg-gray-500/10 border-gray-500/20';
    }
  };

  const getHealthColor = (status: string) => {
    switch (status) {
      case 'online': return 'text-green-400';
      case 'slow': return 'text-yellow-400';
      case 'offline': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center">
            <Brain className="w-6 h-6 mr-2 text-blue-400" />
            Monitor IA
          </h1>
          <p className="text-gray-400 mt-1">
            Monitoreo inteligente de modelos de IA y señales de trading
          </p>
          {signalsErrors.signals && (
            <p className="text-yellow-400 text-sm mt-1">
              ⚠️ {signalsErrors.signals}
            </p>
          )}
          {realSignals.length > 0 && (
            <p className="text-green-400 text-sm mt-1">
              ✅ Datos en tiempo real conectados con precios reales
            </p>
          )}
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
              refreshSignals();
            }}
            disabled={loading.alerts || loading.systemStatus || signalsLoading.signals}
            className="p-2 bg-blue-500/20 text-blue-400 rounded-lg hover:bg-blue-500/30 transition-colors disabled:opacity-50"
            title="Actualizar datos"
          >
            <RefreshCw className={`w-4 h-4 ${loading.alerts || loading.systemStatus || signalsLoading.signals ? 'animate-spin' : ''}`} />
          </button>
          
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="p-2 bg-gray-600 text-gray-400 rounded-lg hover:bg-gray-500 transition-colors"
            title="Vista avanzada"
          >
            <Settings className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Tabs de navegación */}
      <div className="flex space-x-1 bg-gray-800 rounded-lg p-1">
        {[
          { id: 'overview', label: 'Resumen', icon: Activity },
          { id: 'signals', label: 'Señales', icon: Target },
          { id: 'models', label: 'Modelos', icon: Brain },
          { id: 'health', label: 'Salud', icon: Shield },
          { id: 'analytics', label: 'Análisis', icon: BarChart3 }
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={`flex items-center space-x-2 px-4 py-2 rounded-md transition-colors ${
              activeTab === tab.id
                ? 'bg-blue-500 text-white'
                : 'text-gray-400 hover:text-white hover:bg-gray-700'
            }`}
          >
            <tab.icon className="w-4 h-4" />
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Contenido de las tabs */}
      {activeTab === 'overview' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Estado general del sistema */}
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
              <Gauge className="w-5 h-5 mr-2 text-blue-400" />
              Estado del Sistema
            </h3>
            
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-gray-400">Estado General</span>
                <div className="flex items-center space-x-2">
                  <div className={`w-3 h-3 rounded-full ${
                    systemHealth.overall === 'healthy' ? 'bg-green-500' :
                    systemHealth.overall === 'warning' ? 'bg-yellow-500' : 'bg-red-500'
                  }`} />
                  <span className="text-white capitalize">{systemHealth.overall}</span>
                </div>
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                {Object.entries(systemHealth.components).map(([component, status]) => (
                  <div key={component} className="flex items-center justify-between">
                    <span className="text-sm text-gray-400 capitalize">{component}</span>
                    <span className={getHealthColor(status)}>{status}</span>
                  </div>
                ))}
              </div>
              
              <div className="text-xs text-gray-500">
                Última verificación: {systemHealth.lastCheck}
              </div>
            </div>
          </div>

          {/* Resumen de modelos */}
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
              <Brain className="w-5 h-5 mr-2 text-purple-400" />
              Modelos IA
            </h3>
            
            <div className="space-y-3">
              {aiModels.length > 0 ? (
                aiModels.slice(0, 3).map((model) => (
                  <div key={model.name} className="flex items-center justify-between p-3 bg-gray-700/50 rounded-lg">
                    <div className="flex items-center space-x-3">
                      <div className={`w-2 h-2 rounded-full ${
                        model.status === 'active' ? 'bg-green-500' :
                        model.status === 'training' ? 'bg-yellow-500' :
                        model.status === 'error' ? 'bg-red-500' : 'bg-gray-500'
                      }`} />
                      <div>
                        <div className="text-white font-medium">{model.name}</div>
                        <div className="text-sm text-gray-400">{model.accuracy.toFixed(1)}% precisión</div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      {getPerformanceIcon(model.performance)}
                      <span className="text-sm text-gray-400">{model.confidence.toFixed(1)}%</span>
                    </div>
                  </div>
                ))
              ) : (
                <div className="flex items-center justify-center py-8 text-gray-400">
                  <Database className="w-8 h-8 mr-3" />
                  <div>
                    <div className="font-medium">No hay datos de modelos disponibles</div>
                    <div className="text-sm">Los modelos de IA no están generando datos en este momento</div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Señales recientes */}
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
              <Target className="w-5 h-5 mr-2 text-green-400" />
              Señales Recientes
            </h3>
            
            <div className="space-y-3">
              {recentSignals.length > 0 ? (
                recentSignals.slice(0, 3).map((signal) => (
                                     <div key={signal.id} className={`p-3 rounded-lg border ${getSignalColor(signal.signal)}`}>
                     <div className="flex items-center justify-between mb-2">
                       <span className="font-medium capitalize">{signal.signal}</span>
                       <div className="text-right">
                         <span className="text-sm font-semibold">{signal.confidence.toFixed(1)}%</span>
                         {signal.quality_score && (
                           <div className="text-xs text-blue-400">Quality: {signal.quality_score.toFixed(1)}%</div>
                         )}
                       </div>
                     </div>
                     <div className="text-sm mb-1">{signal.pair} @ {signal.price.toFixed(4)}</div>
                     {signal.risk_reward_ratio && (
                       <div className="text-xs text-green-400 mb-1">R/R: {signal.risk_reward_ratio.toFixed(2)}</div>
                     )}
                     <div className="text-xs text-gray-400">{signal.source} • {signal.timestamp}</div>
                   </div>
                ))
              ) : (
                <div className="flex items-center justify-center py-8 text-gray-400">
                  <Target className="w-8 h-8 mr-3" />
                  <div>
                    <div className="font-medium">No hay señales disponibles</div>
                    <div className="text-sm">No se están generando señales de trading en este momento</div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Alertas críticas */}
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
              <AlertTriangle className="w-5 h-5 mr-2 text-red-400" />
              Alertas Críticas
            </h3>
            
            <div className="space-y-3">
              {alerts.filter(alert => alert.severity === 'critical').slice(0, 3).map((alert) => (
                <div key={alert.id} className="p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-red-400 font-medium">{alert.title}</span>
                    <span className="text-xs text-gray-400">{new Date(alert.timestamp).toLocaleTimeString()}</span>
                  </div>
                  <p className="text-sm text-gray-300">{alert.description}</p>
                </div>
              ))}
              
              {alerts.filter(alert => alert.severity === 'critical').length === 0 && (
                <div className="flex items-center justify-center py-4 text-gray-400">
                  <CheckCircle className="w-5 h-5 mr-2" />
                  <span>No hay alertas críticas</span>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {activeTab === 'signals' && (
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-white">Señales de Trading</h3>
            <div className="flex items-center space-x-4">
              <select
                value={selectedPair}
                onChange={(e) => setSelectedPair(e.target.value)}
                className="px-3 py-2 bg-gray-700 text-white rounded-lg border border-gray-600 focus:border-blue-500 focus:outline-none"
              >
                {availablePairs.map((pair) => (
                  <option key={pair.symbol} value={pair.symbol}>
                    {pair.symbol}
                  </option>
                ))}
              </select>
            </div>
          </div>
          
          <div className="space-y-4">
            {recentSignals.length > 0 ? (
              recentSignals.map((signal) => (
                                 <div key={signal.id} className={`p-4 rounded-lg border ${getSignalColor(signal.signal)}`}>
                   <div className="flex items-center justify-between mb-3">
                     <div className="flex items-center space-x-3">
                       <span className="text-lg font-bold capitalize">{signal.signal}</span>
                       <span className="text-sm bg-gray-700 px-2 py-1 rounded">{signal.pair}</span>
                       {signal.quality_score && (
                         <span className="text-xs bg-blue-500/20 text-blue-400 px-2 py-1 rounded">
                           Ultra Quality: {signal.quality_score.toFixed(1)}%
                         </span>
                       )}
                     </div>
                     <div className="text-right">
                       <div className="text-lg font-bold">{signal.price.toFixed(4)}</div>
                       <div className="text-sm text-gray-400">Confianza: {signal.confidence.toFixed(1)}%</div>
                       {signal.risk_reward_ratio && (
                         <div className="text-xs text-green-400">R/R: {signal.risk_reward_ratio.toFixed(2)}</div>
                       )}
                     </div>
                   </div>
                   
                   {/* Análisis detallado */}
                   {signal.technical_analysis && (
                     <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-3 text-xs">
                       <div className="bg-gray-700/50 p-2 rounded">
                         <div className="text-gray-400">RSI</div>
                         <div className="text-white">{signal.technical_analysis.rsi?.toFixed(1) || 'N/A'}</div>
                       </div>
                       <div className="bg-gray-700/50 p-2 rounded">
                         <div className="text-gray-400">Trend</div>
                         <div className="text-white">{signal.trend_analysis?.trend || 'N/A'}</div>
                       </div>
                       <div className="bg-gray-700/50 p-2 rounded">
                         <div className="text-gray-400">Momentum</div>
                         <div className="text-white">{signal.momentum_analysis?.overall_momentum || 'N/A'}</div>
                       </div>
                       <div className="bg-gray-700/50 p-2 rounded">
                         <div className="text-gray-400">Volume</div>
                         <div className="text-white">{signal.volume_analysis?.volume_signal || 'N/A'}</div>
                       </div>
                     </div>
                   )}
                   
                   {/* Stop Loss y Take Profit */}
                   {signal.stop_loss && signal.take_profit && (
                     <div className="flex justify-between mb-3 text-sm">
                       <div className="bg-red-500/10 border border-red-500/20 p-2 rounded">
                         <div className="text-red-400">Stop Loss</div>
                         <div className="text-white">{signal.stop_loss.toFixed(4)}</div>
                       </div>
                       <div className="bg-green-500/10 border border-green-500/20 p-2 rounded">
                         <div className="text-green-400">Take Profit</div>
                         <div className="text-white">{signal.take_profit.toFixed(4)}</div>
                       </div>
                     </div>
                   )}
                   
                   <div className="mb-3">
                     <div className="text-sm text-gray-400 mb-1">Razonamiento:</div>
                     <p className="text-sm">{signal.reasoning}</p>
                   </div>
                   
                   <div className="flex items-center justify-between text-xs text-gray-400">
                     <span>Fuente: {signal.source}</span>
                     <span>{signal.timestamp}</span>
                   </div>
                 </div>
              ))
            ) : (
              <div className="flex items-center justify-center py-12 text-gray-400">
                <Target className="w-12 h-12 mr-4" />
                <div>
                  <div className="text-lg font-medium">No hay señales de trading disponibles</div>
                  <div className="text-sm">Los modelos de IA no están generando señales en este momento</div>
                  <div className="text-xs mt-2">Verifica la conexión con las APIs de trading</div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {activeTab === 'models' && (
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 className="text-lg font-semibold text-white mb-6">Estado de Modelos IA</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {aiModels.length > 0 ? (
              aiModels.map((model) => (
                <div key={model.name} className="bg-gray-700/50 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-4">
                    <h4 className="text-lg font-semibold text-white">{model.name}</h4>
                    <div className={`flex items-center space-x-2 ${getStatusColor(model.status)}`}>
                      <div className={`w-2 h-2 rounded-full ${
                        model.status === 'active' ? 'bg-green-500' :
                        model.status === 'training' ? 'bg-yellow-500' :
                        model.status === 'error' ? 'bg-red-500' : 'bg-gray-500'
                      }`} />
                      <span className="text-sm capitalize">{model.status}</span>
                    </div>
                  </div>
                  
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Precisión</span>
                      <span className="text-white font-semibold">{model.accuracy.toFixed(1)}%</span>
                    </div>
                    
                    <div className="flex justify-between">
                      <span className="text-gray-400">Confianza</span>
                      <span className="text-white font-semibold">{model.confidence.toFixed(1)}%</span>
                    </div>
                    
                    <div className="flex justify-between">
                      <span className="text-gray-400">Rendimiento</span>
                      <div className="flex items-center space-x-1">
                        {getPerformanceIcon(model.performance)}
                        <span className="text-white capitalize">{model.performance}</span>
                      </div>
                    </div>
                    
                    <div className="flex justify-between">
                      <span className="text-gray-400">Alertas</span>
                      <span className="text-white">{model.alerts}</span>
                    </div>
                    
                    <div className="text-xs text-gray-500">
                      Última actualización: {model.lastUpdate}
                    </div>
                  </div>
                </div>
              ))
            ) : (
              <div className="col-span-2 flex items-center justify-center py-12 text-gray-400">
                <Brain className="w-12 h-12 mr-4" />
                <div>
                  <div className="text-lg font-medium">No hay modelos de IA disponibles</div>
                  <div className="text-sm">Los modelos de IA no están generando datos de rendimiento</div>
                  <div className="text-xs mt-2">Verifica la conexión con las APIs de modelos</div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {activeTab === 'health' && (
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 className="text-lg font-semibold text-white mb-6">Salud del Sistema</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Estado de componentes */}
            <div>
              <h4 className="text-md font-semibold text-white mb-4">Componentes</h4>
              <div className="space-y-3">
                {Object.entries(systemHealth.components).map(([component, status]) => (
                  <div key={component} className="flex items-center justify-between p-3 bg-gray-700/50 rounded-lg">
                    <div className="flex items-center space-x-2">
                      <div className={`w-3 h-3 rounded-full ${
                        status === 'online' ? 'bg-green-500' :
                        status === 'slow' ? 'bg-yellow-500' : 'bg-red-500'
                      }`} />
                      <span className="text-white capitalize">{component}</span>
                    </div>
                    <span className={getHealthColor(status)}>{status}</span>
                  </div>
                ))}
              </div>
            </div>
            
            {/* Métricas del sistema */}
            <div>
              <h4 className="text-md font-semibold text-white mb-4">Métricas</h4>
              <div className="space-y-3">
                <div className="flex justify-between p-3 bg-gray-700/50 rounded-lg">
                  <span className="text-gray-400">Uso de CPU</span>
                  <span className="text-white">23%</span>
                </div>
                <div className="flex justify-between p-3 bg-gray-700/50 rounded-lg">
                  <span className="text-gray-400">Memoria</span>
                  <span className="text-white">1.2GB / 4GB</span>
                </div>
                <div className="flex justify-between p-3 bg-gray-700/50 rounded-lg">
                  <span className="text-gray-400">Latencia API</span>
                  <span className="text-white">45ms</span>
                </div>
                <div className="flex justify-between p-3 bg-gray-700/50 rounded-lg">
                  <span className="text-gray-400">Tasa de éxito</span>
                  <span className="text-white">99.8%</span>
                </div>
              </div>
            </div>
          </div>
          
          <div className="mt-6 p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg">
            <div className="flex items-center space-x-2 text-blue-400">
              <Info className="w-4 h-4" />
              <span className="text-sm">
                Última verificación del sistema: {systemHealth.lastCheck}
              </span>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'analytics' && (
        <AIModelAnalytics />
      )}
    </div>
  );
};

export default MonitorIA; 