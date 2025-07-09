import React, { useState } from 'react';
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
  ComposedChart,
  Scatter
} from 'recharts';
import { 
  TrendingUp, 
  TrendingDown, 
  Settings, 
  Maximize2,
  Minimize2,
  RefreshCw,
  Download
} from 'lucide-react';

// Datos de ejemplo para candlestick
const candlestickData = [
  { time: '09:00', open: 1.0850, high: 1.0860, low: 1.0845, close: 1.0855, volume: 1200 },
  { time: '10:00', open: 1.0855, high: 1.0870, low: 1.0850, close: 1.0865, volume: 1800 },
  { time: '11:00', open: 1.0865, high: 1.0875, low: 1.0855, close: 1.0860, volume: 1600 },
  { time: '12:00', open: 1.0860, high: 1.0880, low: 1.0850, close: 1.0875, volume: 2100 },
  { time: '13:00', open: 1.0875, high: 1.0890, low: 1.0865, close: 1.0885, volume: 3300 },
  { time: '14:00', open: 1.0885, high: 1.0895, low: 1.0875, close: 1.0880, volume: 2900 },
  { time: '15:00', open: 1.0880, high: 1.0900, low: 1.0870, close: 1.0895, volume: 4200 },
  { time: '16:00', open: 1.0895, high: 1.0910, low: 1.0885, close: 1.0905, volume: 5100 },
  { time: '17:00', open: 1.0905, high: 1.0920, low: 1.0895, close: 1.0915, volume: 6200 },
  { time: '18:00', open: 1.0915, high: 1.0930, low: 1.0905, close: 1.0925, volume: 7500 },
];

// Datos para indicadores técnicos
const technicalData = [
  { time: '09:00', price: 1.0855, sma20: 1.0840, sma50: 1.0830, rsi: 65, macd: 0.0012 },
  { time: '10:00', price: 1.0865, sma20: 1.0842, sma50: 1.0832, rsi: 68, macd: 0.0015 },
  { time: '11:00', price: 1.0860, sma20: 1.0845, sma50: 1.0835, rsi: 62, macd: 0.0010 },
  { time: '12:00', price: 1.0875, sma20: 1.0848, sma50: 1.0838, rsi: 72, macd: 0.0020 },
  { time: '13:00', price: 1.0885, sma20: 1.0850, sma50: 1.0840, rsi: 75, macd: 0.0025 },
  { time: '14:00', price: 1.0880, sma20: 1.0852, sma50: 1.0842, rsi: 70, macd: 0.0022 },
  { time: '15:00', price: 1.0895, sma20: 1.0855, sma50: 1.0845, rsi: 78, macd: 0.0030 },
  { time: '16:00', price: 1.0905, sma20: 1.0858, sma50: 1.0848, rsi: 82, macd: 0.0035 },
  { time: '17:00', price: 1.0915, sma20: 1.0860, sma50: 1.0850, rsi: 85, macd: 0.0040 },
  { time: '18:00', price: 1.0925, sma20: 1.0862, sma50: 1.0852, rsi: 88, macd: 0.0045 },
];

interface TradingChartProps {
  symbol: string;
  timeframe: string;
}

export const TradingChart: React.FC<TradingChartProps> = ({ symbol, timeframe }) => {
  const [chartType, setChartType] = useState<'candlestick' | 'line' | 'area'>('candlestick');
  const [showIndicators, setShowIndicators] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const chartTypes = [
    { id: 'candlestick', name: 'Candlestick', icon: '📊' },
    { id: 'line', name: 'Línea', icon: '📈' },
    { id: 'area', name: 'Área', icon: '📉' },
  ];

  const indicators = [
    { id: 'sma', name: 'SMA 20', color: '#38b2ac' },
    { id: 'ema', name: 'EMA 50', color: '#f56565' },
    { id: 'rsi', name: 'RSI', color: '#ed8936' },
    { id: 'macd', name: 'MACD', color: '#9f7aea' },
  ];

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-3 shadow-lg">
          <p className="text-white font-semibold">{label}</p>
          <div className="space-y-1 mt-2">
            <div className="flex justify-between">
              <span className="text-gray-400">Apertura:</span>
              <span className="text-white">{data.open}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Máximo:</span>
              <span className="text-green-400">{data.high}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Mínimo:</span>
              <span className="text-red-400">{data.low}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Cierre:</span>
              <span className="text-white">{data.close}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Volumen:</span>
              <span className="text-blue-400">{data.volume}</span>
            </div>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className={`trading-card ${isFullscreen ? 'fixed inset-0 z-50' : ''}`}>
      {/* Header del gráfico */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between p-3 sm:p-4 border-b border-gray-700/50">
        <div className="flex flex-col sm:flex-row sm:items-center space-y-2 sm:space-y-0 sm:space-x-4 mb-3 sm:mb-0">
          <div>
            <h3 className="text-base sm:text-lg font-semibold text-white">{symbol}</h3>
            <p className="text-xs sm:text-sm text-gray-400">{timeframe} • {new Date().toLocaleTimeString()}</p>
          </div>
          <div className="flex items-center space-x-2">
            <span className="text-lg sm:text-2xl font-bold text-green-400">1.0925</span>
            <div className="flex items-center text-green-400">
              <TrendingUp className="w-3 h-3 sm:w-4 sm:h-4" />
              <span className="text-xs sm:text-sm">+0.0010 (+0.09%)</span>
            </div>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          {/* Tipo de gráfico */}
          <div className="flex items-center space-x-1 bg-gray-800/50 rounded-lg p-1">
            {chartTypes.map((type) => (
              <button
                key={type.id}
                onClick={() => setChartType(type.id as any)}
                className={`px-2 sm:px-3 py-1 sm:py-2 rounded text-xs sm:text-sm font-medium transition-colors ${
                  chartType === type.id
                    ? 'bg-blue-500 text-white'
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                <span className="hidden sm:inline">{type.icon} {type.name}</span>
                <span className="sm:hidden">{type.icon}</span>
              </button>
            ))}
          </div>

          {/* Indicadores */}
          <button
            onClick={() => setShowIndicators(!showIndicators)}
            className={`p-1.5 sm:p-2 rounded-lg transition-colors ${
              showIndicators 
                ? 'bg-blue-500/20 text-blue-400' 
                : 'bg-gray-700/50 text-gray-400 hover:text-white'
            }`}
          >
            <Settings className="w-3 h-3 sm:w-4 sm:h-4" />
          </button>

          {/* Fullscreen */}
          <button
            onClick={() => setIsFullscreen(!isFullscreen)}
            className="p-1.5 sm:p-2 rounded-lg bg-gray-700/50 text-gray-400 hover:text-white transition-colors"
          >
            {isFullscreen ? <Minimize2 className="w-3 h-3 sm:w-4 sm:h-4" /> : <Maximize2 className="w-3 h-3 sm:w-4 sm:h-4" />}
          </button>

          {/* Refresh */}
          <button className="p-1.5 sm:p-2 rounded-lg bg-gray-700/50 text-gray-400 hover:text-white transition-colors">
            <RefreshCw className="w-3 h-3 sm:w-4 sm:h-4" />
          </button>

          {/* Download */}
          <button className="p-1.5 sm:p-2 rounded-lg bg-gray-700/50 text-gray-400 hover:text-white transition-colors">
            <Download className="w-3 h-3 sm:w-4 sm:h-4" />
          </button>
        </div>
      </div>

      {/* Gráfico principal */}
      <div className="p-3 sm:p-4">
        <ResponsiveContainer width="100%" height={300}>
          {chartType === 'candlestick' ? (
            <ComposedChart data={candlestickData}>
              <defs>
                <linearGradient id="volumeGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#38b2ac" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#38b2ac" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#2d3748" />
              <XAxis dataKey="time" stroke="#a0aec0" fontSize={12} />
              <YAxis stroke="#a0aec0" fontSize={12} />
              <Tooltip content={<CustomTooltip />} />
              
              {/* Candlestick bars */}
              <Bar dataKey="high" fill="transparent" stroke="#38b2ac" strokeWidth={2} />
              <Bar dataKey="low" fill="transparent" stroke="#38b2ac" strokeWidth={2} />
              <Bar dataKey="close" fill="#38b2ac" stroke="#38b2ac" />
              
              {/* Volume */}
              <Bar dataKey="volume" fill="url(#volumeGradient)" opacity={0.3} />
            </ComposedChart>
          ) : chartType === 'line' ? (
            <LineChart data={technicalData}>
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
              <Line 
                type="monotone" 
                dataKey="price" 
                stroke="#38b2ac" 
                strokeWidth={2}
                dot={{ fill: '#38b2ac', strokeWidth: 2, r: 4 }}
              />
              {showIndicators && (
                <>
                  <Line 
                    type="monotone" 
                    dataKey="sma20" 
                    stroke="#f56565" 
                    strokeWidth={1}
                    strokeDasharray="5 5"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="sma50" 
                    stroke="#ed8936" 
                    strokeWidth={1}
                    strokeDasharray="5 5"
                  />
                </>
              )}
            </LineChart>
          ) : (
            <AreaChart data={technicalData}>
              <defs>
                <linearGradient id="areaGradient" x1="0" y1="0" x2="0" y2="1">
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
                dataKey="price" 
                stroke="#38b2ac" 
                strokeWidth={2}
                fill="url(#areaGradient)"
              />
            </AreaChart>
          )}
        </ResponsiveContainer>

        {/* Indicadores técnicos */}
        {showIndicators && (
          <div className="mt-4 sm:mt-6">
            <h4 className="text-sm font-semibold text-white mb-3">Indicadores Técnicos</h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 sm:gap-4">
              {indicators.map((indicator) => (
                <div key={indicator.id} className="bg-gray-800/50 rounded-lg p-2 sm:p-3">
                  <div className="flex items-center justify-between">
                    <span className="text-xs sm:text-sm text-gray-400">{indicator.name}</span>
                    <div 
                      className="w-2 h-2 sm:w-3 sm:h-3 rounded-full" 
                      style={{ backgroundColor: indicator.color }}
                    />
                  </div>
                  <div className="text-sm sm:text-lg font-semibold text-white mt-1">
                    {indicator.id === 'rsi' ? '72.5' : 
                     indicator.id === 'macd' ? '0.0025' : '1.0862'}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}; 