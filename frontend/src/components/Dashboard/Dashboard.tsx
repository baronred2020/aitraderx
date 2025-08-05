import React, { useState, useEffect } from 'react';
import { 
  Brain, 
  Cpu, 
  Zap, 
  Target, 
  Activity, 
  TrendingUp,
  ArrowUpRight,
  ArrowDownRight,
  Eye,
  EyeOff,
  RefreshCw,
  Play,
  Pause,
  Settings,
  CheckCircle,
  Loader2,
  AlertTriangle,
  DollarSign,
  BarChart3,
  Clock,
  Shield,
  Users,
  Database,
  Server
} from 'lucide-react';
import { useAuth } from '../../contexts/AuthContext';
import { 
  LineChart, 
  Line, 
  AreaChart, 
  Area, 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from 'recharts';


// Datos de los sistemas de IA disponibles
const aiSystems = [
  {
    name: 'Brain Max',
    description: 'Sistema de ensemble con múltiples modelos (RF, XGBoost, LightGBM, MLP)',
    precision: 85.7,
    status: 'active',
    pairs: ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD'],
    tradingStyles: ['Scalping', 'Day Trading', 'Swing Trading', 'Position Trading'],
    lastUpdate: '2025-07-24 00:59:04',
    icon: Brain,
    color: 'text-blue-400',
    bgColor: 'bg-blue-500/10',
    borderColor: 'border-blue-500/20',
    models: 6,
    winRate: 60.0,
    features: ['Análisis Técnico', 'Patrones de Mercado', 'Ensemble ML']
  },
  {
    name: 'Brain Ultra',
    description: 'Sistema multi-estrategia con adaptación dinámica (CatBoost, XGBoost, LightGBM)',
    precision: 88.7,
    status: 'active',
    pairs: ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD'],
    tradingStyles: ['Scalping', 'Day Trading', 'Swing Trading', 'Position Trading'],
    lastUpdate: '2025-07-30 21:32:31',
    icon: Zap,
    color: 'text-green-400',
    bgColor: 'bg-green-500/10',
    borderColor: 'border-green-500/20',
    models: 5,
    winRate: 72.0,
    features: ['Multi-Estrategia', 'Adaptación Dinámica', 'Optimización Continua']
  },
  {
    name: 'Brain Predictor',
    description: 'Sistema predictivo con forecasting y eventos económicos (GradientBoosting, RandomForest)',
    precision: 85.2,
    status: 'active',
    pairs: ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD'],
    tradingStyles: ['Day Trading', 'Swing Trading', 'Position Trading'],
    lastUpdate: '2025-07-25 11:36:41',
    icon: Target,
    color: 'text-orange-400',
    bgColor: 'bg-orange-500/10',
    borderColor: 'border-orange-500/20',
    models: 25,
    winRate: 68.0,
    features: ['Forecasting', 'Eventos Económicos', 'Análisis Fundamental']
  },
  {
    name: 'MegaMind',
    description: 'Ensemble colaborativo de los 3 cerebros con consenso inteligente',
    precision: 92.3,
    status: 'active',
    pairs: ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'EURGBP', 'GBPJPY'],
    tradingStyles: ['Scalping', 'Day Trading', 'Swing Trading', 'Position Trading'],
    lastUpdate: '2025-07-30 21:32:31',
    icon: Cpu,
    color: 'text-purple-400',
    bgColor: 'bg-purple-500/10',
    borderColor: 'border-purple-500/20',
    models: 36,
    winRate: 85.0,
    features: ['Consenso Inteligente', 'Fusión en Tiempo Real', 'Evolución Automática']
  }
];

// Datos de precisión por par de divisas
const precisionByPair = [
  { pair: 'EURUSD', precision: 82.3, volume: '2.4M', trend: 'up' },
  { pair: 'GBPUSD', precision: 79.8, volume: '1.8M', trend: 'up' },
  { pair: 'USDJPY', precision: 85.1, volume: '3.1M', trend: 'up' },
  { pair: 'AUDUSD', precision: 77.2, volume: '1.2M', trend: 'down' },
  { pair: 'USDCAD', precision: 80.5, volume: '0.9M', trend: 'up' },
];

// Datos de rendimiento por estilo de trading
const tradingStylesPerformance = [
  { style: 'Scalping', precision: 88.7, avgTime: '5-15 min', success: 72, models: 'Brain Ultra' },
  { style: 'Day Trading', precision: 85.7, avgTime: '1-4 horas', success: 60, models: 'Brain Max' },
  { style: 'Swing Trading', precision: 85.2, avgTime: '1-7 días', success: 68, models: 'Brain Predictor' },
  { style: 'Position Trading', precision: 92.3, avgTime: '1-4 semanas', success: 85, models: 'MegaMind' },
];

// Datos de actividad del sistema
const systemActivity = [
  { time: '09:00', predictions: 45, accuracy: 82.1 },
  { time: '10:00', predictions: 52, accuracy: 84.3 },
  { time: '11:00', predictions: 48, accuracy: 79.8 },
  { time: '12:00', predictions: 61, accuracy: 86.2 },
  { time: '13:00', predictions: 55, accuracy: 83.7 },
  { time: '14:00', predictions: 58, accuracy: 85.1 },
  { time: '15:00', predictions: 63, accuracy: 87.4 },
  { time: '16:00', predictions: 49, accuracy: 81.9 },
  { time: '17:00', predictions: 44, accuracy: 80.2 },
  { time: '18:00', predictions: 38, accuracy: 78.6 },
];

export const Dashboard: React.FC = () => {
  const { user } = useAuth();
  const [selectedTimeframe, setSelectedTimeframe] = useState('1H');
  const [systemStatus, setSystemStatus] = useState('operational');

  // Estado para RL
  const [rlStatus, setRlStatus] = useState<{ DQN?: any; PPO?: any }>({});
  const [loadingRL, setLoadingRL] = useState(true);
  const [errorRL, setErrorRL] = useState<string | null>(null);

  useEffect(() => {
    setLoadingRL(true);
    fetch('/api/rl/status')
      .then(res => res.json())
      .then(data => {
        setRlStatus(data);
        setLoadingRL(false);
      })
      .catch(err => {
        setErrorRL('No se pudo obtener el estado de los agentes RL');
        setLoadingRL(false);
      });
  }, []);

  const timeframes = ['1H', '4H', '1D', '1W', '1M'];

  const systemMetrics = [
    {
      title: 'Sistemas IA Activos',
      value: '4',
      change: '+1',
      changeValue: 'Nuevo: MegaMind',
      icon: Brain,
      color: 'text-blue-400',
      bgColor: 'bg-blue-500/10',
      borderColor: 'border-blue-500/20'
    },
    {
      title: 'Precisión Promedio',
      value: '88.0%',
      change: '+6.8%',
      changeValue: '+4.2%',
      icon: Target,
      color: 'text-green-400',
      bgColor: 'bg-green-500/10',
      borderColor: 'border-green-500/20'
    },
    {
      title: 'Modelos Totales',
      value: '72',
      change: '+31',
      changeValue: 'Ensemble + Individuales',
      icon: DollarSign,
      color: 'text-purple-400',
      bgColor: 'bg-purple-500/10',
      borderColor: 'border-purple-500/20'
    },
    {
      title: 'Win Rate Promedio',
      value: '71.3%',
      change: '+8.5%',
      changeValue: '+5.2%',
      icon: BarChart3,
      color: 'text-orange-400',
      bgColor: 'bg-orange-500/10',
      borderColor: 'border-orange-500/20'
    }
  ];

  return (
    <div className="p-6 space-y-6">
      {/* Header del Dashboard */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Dashboard de Sistemas IA</h1>
          <p className="text-gray-400">Monitoreo de sistemas de Inteligencia Artificial para trading</p>
        </div>
        <div className="flex items-center space-x-3">
          <div className={`flex items-center space-x-2 px-3 py-2 rounded-lg ${
            systemStatus === 'operational' 
              ? 'bg-green-500/20 text-green-400 border border-green-500/30' 
              : 'bg-red-500/20 text-red-400 border border-red-500/30'
          }`}>
            <div className={`w-2 h-2 rounded-full ${
              systemStatus === 'operational' ? 'bg-green-400' : 'bg-red-400'
            }`} />
            <span className="text-sm font-medium">
              {systemStatus === 'operational' ? 'Sistema Operativo' : 'Mantenimiento'}
            </span>
          </div>
          <button className="p-2 rounded-lg hover:bg-gray-700/50 transition-colors">
            <RefreshCw className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Métricas principales del sistema */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {systemMetrics.map((metric, index) => (
          <div key={index} className={`trading-card p-6 ${metric.bgColor} ${metric.borderColor}`}>
            <div className="flex items-center justify-between mb-4">
              <div className={`p-3 rounded-lg ${metric.bgColor}`}>
                <metric.icon className={`w-6 h-6 ${metric.color}`} />
              </div>
              <div className={`text-sm ${metric.color}`}>
                {metric.change.startsWith('+') ? (
                  <ArrowUpRight className="w-4 h-4" />
                ) : (
                  <ArrowDownRight className="w-4 h-4" />
                )}
              </div>
            </div>
            <div>
              <p className="text-sm text-gray-400 mb-1">{metric.title}</p>
              <p className="text-2xl font-bold text-white mb-1">{metric.value}</p>
              <div className="flex items-center space-x-2">
                <span className={`text-sm ${metric.color}`}>{metric.change}</span>
                <span className="text-xs text-gray-400">({metric.changeValue})</span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Sistemas de IA disponibles */}
      <div className="trading-card p-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="text-lg font-semibold text-white">Sistemas de IA Disponibles</h3>
            <p className="text-sm text-gray-400">Estado y rendimiento de los sistemas de trading con IA</p>
          </div>
          <button className="text-sm text-blue-400 hover:text-blue-300">Ver detalles</button>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-4 gap-6">
          {aiSystems.map((system, index) => (
            <div key={index} className={`p-6 rounded-lg border ${system.borderColor} ${system.bgColor}`}>
              <div className="flex items-center justify-between mb-4">
                <div className={`p-3 rounded-lg ${system.bgColor}`}>
                  <system.icon className={`w-6 h-6 ${system.color}`} />
                </div>
                <div className={`px-2 py-1 rounded text-xs font-medium ${
                  system.status === 'active' 
                    ? 'bg-green-500/20 text-green-400' 
                    : 'bg-red-500/20 text-red-400'
                }`}>
                  {system.status === 'active' ? 'Activo' : 'Inactivo'}
                </div>
              </div>
              <div className="mb-4">
                <h4 className="font-semibold text-white text-lg mb-1">{system.name}</h4>
                <p className="text-sm text-gray-400 mb-3">{system.description}</p>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-400">Precisión</span>
                  <span className={`text-lg font-bold ${system.color}`}>{system.precision}%</span>
                </div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-400">Win Rate</span>
                  <span className="text-sm font-semibold text-white">{system.winRate}%</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-400">Modelos</span>
                  <span className="text-sm font-semibold text-white">{system.models}</span>
                </div>
              </div>
              <div className="space-y-2">
                <div>
                  <p className="text-xs text-gray-400 mb-1">Características:</p>
                  <div className="flex flex-wrap gap-1">
                    {system.features.slice(0, 2).map((feature, idx) => (
                      <span key={idx} className="px-2 py-1 bg-gray-700 rounded text-xs text-white">
                        {feature}
                      </span>
                    ))}
                    {system.features.length > 2 && (
                      <span className="px-2 py-1 bg-gray-700 rounded text-xs text-white">
                        +{system.features.length - 2}
                      </span>
                    )}
                  </div>
                </div>
                <div>
                  <p className="text-xs text-gray-400 mb-1">Pares disponibles:</p>
                  <div className="flex flex-wrap gap-1">
                    {system.pairs.slice(0, 3).map((pair, idx) => (
                      <span key={idx} className="px-2 py-1 bg-gray-700 rounded text-xs text-white">
                        {pair}
                      </span>
                    ))}
                    {system.pairs.length > 3 && (
                      <span className="px-2 py-1 bg-gray-700 rounded text-xs text-white">
                        +{system.pairs.length - 3}
                      </span>
                    )}
                  </div>
                </div>
              </div>
              <div className="mt-4 pt-3 border-t border-gray-700">
                <p className="text-xs text-gray-400">
                  Última actualización: {system.lastUpdate}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Gráficos principales */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Actividad del sistema */}
        <div className="lg:col-span-2 trading-card p-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h3 className="text-lg font-semibold text-white">Actividad del Sistema</h3>
              <p className="text-sm text-gray-400">Predicciones y precisión en tiempo real</p>
            </div>
            <div className="flex items-center space-x-2">
              {timeframes.map((tf) => (
                <button
                  key={tf}
                  onClick={() => setSelectedTimeframe(tf)}
                  className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                    selectedTimeframe === tf
                      ? 'bg-blue-500 text-white'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  {tf}
                </button>
              ))}
            </div>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={systemActivity}>
              <defs>
                <linearGradient id="colorPredictions" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#38b2ac" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#38b2ac" stopOpacity={0}/>
                </linearGradient>
                <linearGradient id="colorAccuracy" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#f56565" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#f56565" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#2d3748" />
              <XAxis dataKey="time" stroke="#a0aec0" fontSize={12} />
              <YAxis stroke="#a0aec0" fontSize={12} />
              <Tooltip 
                contentStyle={{
                  backgroundColor: '#252b3d',
                  border: '1px solid #2d3748',
                  borderRadius: '8px',
                  color: '#ffffff'
                }}
              />
              <Area 
                type="monotone" 
                dataKey="predictions" 
                stroke="#38b2ac" 
                strokeWidth={2}
                fill="url(#colorPredictions)"
                name="Predicciones"
              />
              <Area 
                type="monotone" 
                dataKey="accuracy" 
                stroke="#f56565" 
                strokeWidth={2}
                fill="url(#colorAccuracy)"
                name="Precisión (%)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Rendimiento por estilo de trading */}
        <div className="trading-card p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-white">Rendimiento por Estilo</h3>
            <button className="p-2 rounded-lg hover:bg-gray-700/50 transition-colors">
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>
          <div className="space-y-4">
            {tradingStylesPerformance.map((style, index) => (
              <div key={index} className="bg-gray-800/50 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-semibold text-white">{style.style}</h4>
                  <span className="text-green-400 font-bold">{style.precision}%</span>
                </div>
                <div className="flex items-center justify-between text-sm text-gray-400">
                  <span>{style.avgTime}</span>
                  <span>{style.success}% éxito</span>
                </div>
                <div className="text-xs text-gray-400 mt-1">
                  Modelo: {style.models}
                </div>
                <div className="mt-2 w-full bg-gray-700 rounded-full h-2">
                  <div 
                    className="bg-gradient-to-r from-blue-500 to-green-500 h-2 rounded-full" 
                    style={{ width: `${style.success}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Precisión por par y información del sistema */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Precisión por par de divisas */}
        <div className="trading-card p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-white">Precisión por Par</h3>
            <button className="text-sm text-blue-400 hover:text-blue-300">Ver todos</button>
          </div>
          <div className="space-y-4">
            {precisionByPair.map((pair, index) => (
              <div key={index} className="flex items-center justify-between p-4 bg-gray-800/50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-teal-500 rounded-lg flex items-center justify-center">
                    <span className="text-xs font-bold text-white">{pair.pair.slice(0, 3)}</span>
                  </div>
                  <div>
                    <p className="font-semibold text-white">{pair.pair}</p>
                    <p className="text-xs text-gray-400">Vol: {pair.volume}</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="font-semibold text-white">{pair.precision}%</p>
                  <div className="flex items-center space-x-1">
                    {pair.trend === 'up' ? (
                      <ArrowUpRight className="w-3 h-3 text-green-400" />
                    ) : (
                      <ArrowDownRight className="w-3 h-3 text-red-400" />
                    )}
                    <span className={`text-xs ${pair.trend === 'up' ? 'text-green-400' : 'text-red-400'}`}>
                      {pair.trend === 'up' ? 'Mejorando' : 'Bajando'}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Información importante del sistema */}
        <div className="trading-card p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-white">Información del Sistema</h3>
            <button className="text-sm text-blue-400 hover:text-blue-300">Ver detalles</button>
          </div>
          <div className="space-y-4">
            <div className="flex items-center justify-between p-3 bg-gray-800/50 rounded-lg">
              <div className="flex items-center space-x-3">
                <Server className="w-5 h-5 text-blue-400" />
                <div>
                  <p className="font-semibold text-white">Estado del Servidor</p>
                  <p className="text-xs text-gray-400">Tiempo de respuesta</p>
                </div>
              </div>
              <div className="text-right">
                <p className="text-green-400 font-semibold">Operativo</p>
                <p className="text-xs text-gray-400">45ms</p>
              </div>
            </div>
            
            <div className="flex items-center justify-between p-3 bg-gray-800/50 rounded-lg">
              <div className="flex items-center space-x-3">
                <Database className="w-5 h-5 text-green-400" />
                <div>
                  <p className="font-semibold text-white">Base de Datos</p>
                  <p className="text-xs text-gray-400">Última actualización</p>
                </div>
              </div>
              <div className="text-right">
                <p className="text-green-400 font-semibold">Conectada</p>
                <p className="text-xs text-gray-400">Hace 2 min</p>
              </div>
            </div>
            
            <div className="flex items-center justify-between p-3 bg-gray-800/50 rounded-lg">
              <div className="flex items-center space-x-3">
                <Users className="w-5 h-5 text-purple-400" />
                <div>
                  <p className="font-semibold text-white">Usuarios Activos</p>
                  <p className="text-xs text-gray-400">Sesiones concurrentes</p>
                </div>
              </div>
              <div className="text-right">
                <p className="text-purple-400 font-semibold">1,247</p>
                <p className="text-xs text-gray-400">+12% hoy</p>
              </div>
            </div>
            
            <div className="flex items-center justify-between p-3 bg-gray-800/50 rounded-lg">
              <div className="flex items-center space-x-3">
                <Shield className="w-5 h-5 text-orange-400" />
                <div>
                  <p className="font-semibold text-white">Seguridad</p>
                  <p className="text-xs text-gray-400">Última verificación</p>
                </div>
              </div>
              <div className="text-right">
                <p className="text-green-400 font-semibold">Verificada</p>
                <p className="text-xs text-gray-400">Hace 5 min</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      
    </div>
  );
}; 