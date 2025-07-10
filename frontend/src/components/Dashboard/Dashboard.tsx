import React, { useState, useEffect } from 'react';
import { 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  Activity, 
  Target, 
  Shield,
  ArrowUpRight,
  ArrowDownRight,
  Eye,
  EyeOff,
  RefreshCw,
  Play,
  Pause,
  Settings,
  Cpu,
  Zap,
  CheckCircle,
  Loader2,
  AlertTriangle
} from 'lucide-react';
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
import { MonitorIA } from '../AIMonitor/MonitorIA';

// Datos de ejemplo para los gráficos
const performanceData = [
  { time: '09:00', value: 125000, pnl: 1200 },
  { time: '10:00', value: 126200, pnl: 1800 },
  { time: '11:00', value: 125800, pnl: 1600 },
  { time: '12:00', value: 127100, pnl: 2100 },
  { time: '13:00', value: 128300, pnl: 3300 },
  { time: '14:00', value: 127900, pnl: 2900 },
  { time: '15:00', value: 129200, pnl: 4200 },
  { time: '16:00', value: 130100, pnl: 5100 },
  { time: '17:00', value: 131200, pnl: 6200 },
  { time: '18:00', value: 132500, pnl: 7500 },
];

const aiPerformanceData = [
  { name: 'Precisión', value: 78.3, color: '#48bb78' },
  { name: 'Falsos Positivos', value: 12.1, color: '#ed8936' },
  { name: 'Falsos Negativos', value: 9.6, color: '#f56565' },
];

const symbols = [
  { pair: 'EURUSD', label: 'EUR/USD', price: '1.0854', change: '+0.12%', volume: '2.4M', trend: 'up' },
  { pair: 'GBPUSD', label: 'GBP/USD', price: '1.2654', change: '-0.08%', volume: '1.8M', trend: 'down' },
  { pair: 'USDJPY', label: 'USD/JPY', price: '148.23', change: '+0.25%', volume: '3.1M', trend: 'up' },
  { pair: 'AUDUSD', label: 'AUD/USD', price: '0.6543', change: '+0.18%', volume: '1.2M', trend: 'up' },
  { pair: 'USDCAD', label: 'USD/CAD', price: '1.3542', change: '-0.05%', volume: '0.9M', trend: 'down' },
];

const recentOrders = [
  { id: 1, pair: 'EURUSD', type: 'BUY', amount: '10,000', price: '1.0854', time: '14:32:15', status: 'filled' },
  { id: 2, pair: 'GBPUSD', type: 'SELL', amount: '5,000', price: '1.2654', time: '14:28:42', status: 'pending' },
  { id: 3, pair: 'USDJPY', type: 'BUY', amount: '15,000', price: '148.23', time: '14:25:18', status: 'filled' },
  { id: 4, pair: 'AUDUSD', type: 'SELL', amount: '8,000', price: '0.6543', time: '14:22:05', status: 'cancelled' },
];

export const Dashboard: React.FC = () => {
  const [balanceVisible, setBalanceVisible] = useState(true);
  const [isTradingActive, setIsTradingActive] = useState(true);
  const [selectedTimeframe, setSelectedTimeframe] = useState('1H');

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

  const metrics = [
    {
      title: 'Balance Total',
      value: balanceVisible ? '$125,430' : '****',
      change: '+2.3%',
      changeValue: '+$2,850',
      icon: DollarSign,
      color: 'text-green-400',
      bgColor: 'bg-green-500/10',
      borderColor: 'border-green-500/20'
    },
    {
      title: 'P&L Diario',
      value: '+$1,234',
      change: '+0.98%',
      changeValue: '+$12.34',
      icon: TrendingUp,
      color: 'text-green-400',
      bgColor: 'bg-green-500/10',
      borderColor: 'border-green-500/20'
    },
    {
      title: 'Precisión IA',
      value: '78.3%',
      change: '+1.2%',
      changeValue: '+0.8%',
      icon: Target,
      color: 'text-blue-400',
      bgColor: 'bg-blue-500/10',
      borderColor: 'border-blue-500/20'
    },
    {
      title: 'Sharpe Ratio',
      value: '1.45',
      change: '+0.12',
      changeValue: '+0.08',
      icon: Shield,
      color: 'text-purple-400',
      bgColor: 'bg-purple-500/10',
      borderColor: 'border-purple-500/20'
    }
  ];

  return (
    <div className="p-6 space-y-6">
      {/* Header del Dashboard */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Dashboard</h1>
          <p className="text-gray-400">Resumen de tu actividad de trading con IA</p>
        </div>
        <div className="flex items-center space-x-3">
          <button
            onClick={() => setBalanceVisible(!balanceVisible)}
            className="p-2 rounded-lg hover:bg-gray-700/50 transition-colors"
          >
            {balanceVisible ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
          </button>
          <button
            onClick={() => setIsTradingActive(!isTradingActive)}
            className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all ${
              isTradingActive 
                ? 'bg-red-500/20 text-red-400 border border-red-500/30' 
                : 'bg-green-500/20 text-green-400 border border-green-500/30'
            }`}
          >
            {isTradingActive ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            <span className="text-sm font-medium">
              {isTradingActive ? 'Pausar Trading' : 'Activar Trading'}
            </span>
          </button>
        </div>
      </div>

      {/* Métricas principales */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {metrics.map((metric, index) => (
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

      {/* Gráficos principales */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Gráfico de rendimiento */}
        <div className="lg:col-span-2 trading-card p-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h3 className="text-lg font-semibold text-white">Rendimiento del Portfolio</h3>
              <p className="text-sm text-gray-400">Evolución del balance en tiempo real</p>
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
            <AreaChart data={performanceData}>
              <defs>
                <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#38b2ac" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#38b2ac" stopOpacity={0}/>
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
                dataKey="value" 
                stroke="#38b2ac" 
                strokeWidth={2}
                fill="url(#colorValue)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Métricas de IA */}
        <div className="trading-card p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-white">Rendimiento IA</h3>
            <button className="p-2 rounded-lg hover:bg-gray-700/50 transition-colors">
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie
                data={aiPerformanceData}
                cx="50%"
                cy="50%"
                innerRadius={40}
                outerRadius={80}
                paddingAngle={5}
                dataKey="value"
              >
                {aiPerformanceData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip 
                contentStyle={{
                  backgroundColor: '#252b3d',
                  border: '1px solid #2d3748',
                  borderRadius: '8px',
                  color: '#ffffff'
                }}
              />
            </PieChart>
          </ResponsiveContainer>
          <div className="mt-4 space-y-2">
            {aiPerformanceData.map((item, index) => (
              <div key={index} className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <div 
                    className="w-3 h-3 rounded-full" 
                    style={{ backgroundColor: item.color }}
                  />
                  <span className="text-sm text-gray-300">{item.name}</span>
                </div>
                <span className="text-sm font-semibold text-white">{item.value}%</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Pares de trading y órdenes recientes */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Pares de trading */}
        <div className="trading-card p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-white">Pares Activos</h3>
            <button className="text-sm text-blue-400 hover:text-blue-300">Ver todos</button>
          </div>
          <div className="space-y-4">
            {symbols.map((symbol, index) => (
              <div key={index} className="flex items-center justify-between p-4 bg-gray-800/50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-teal-500 rounded-lg flex items-center justify-center">
                    <span className="text-xs font-bold text-white">{symbol.pair.split('/')[0]}</span>
                  </div>
                  <div>
                    <p className="font-semibold text-white">{symbol.label}</p>
                    <p className="text-xs text-gray-400">Vol: {symbol.volume}</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="font-semibold text-white">{symbol.price}</p>
                  <div className="flex items-center space-x-1">
                    {symbol.trend === 'up' ? (
                      <ArrowUpRight className="w-3 h-3 text-green-400" />
                    ) : (
                      <ArrowDownRight className="w-3 h-3 text-red-400" />
                    )}
                    <span className={`text-xs ${symbol.trend === 'up' ? 'text-green-400' : 'text-red-400'}`}>
                      {symbol.change}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Órdenes recientes */}
        <div className="trading-card p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-white">Órdenes Recientes</h3>
            <button className="text-sm text-blue-400 hover:text-blue-300">Ver todas</button>
          </div>
          <div className="space-y-3">
            {recentOrders.map((order) => (
              <div key={order.id} className="flex items-center justify-between p-3 bg-gray-800/50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className={`w-2 h-2 rounded-full ${
                    order.status === 'filled' ? 'bg-green-400' :
                    order.status === 'pending' ? 'bg-yellow-400' :
                    'bg-red-400'
                  }`} />
                  <div>
                    <p className="font-semibold text-white">{order.pair}</p>
                    <p className="text-xs text-gray-400">{order.time}</p>
                  </div>
                </div>
                <div className="text-right">
                  <div className="flex items-center space-x-2">
                    <span className={`text-sm font-semibold ${
                      order.type === 'BUY' ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {order.type}
                    </span>
                    <span className="text-sm text-white">{order.amount}</span>
                  </div>
                  <p className="text-xs text-gray-400">{order.price}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Estado del sistema IA */}
      <MonitorIA resumen />
    </div>
  );
}; 