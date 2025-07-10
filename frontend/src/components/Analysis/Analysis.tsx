import React, { useState } from 'react';
import { 
  TrendingUp, 
  TrendingDown, 
  AlertTriangle,
  CheckCircle,
  Clock,
  Activity
} from 'lucide-react';

export const Analysis: React.FC = () => {
  const [selectedTimeframe, setSelectedTimeframe] = useState('1H');
  const [selectedPair, setSelectedPair] = useState('EURUSD');
  const [analysisType, setAnalysisType] = useState<'technical' | 'fundamental'>('technical');

  const timeframes = ['1M', '5M', '15M', '1H', '4H', '1D', '1W'];
  const tradingPairs = [
    { pair: 'EURUSD', label: 'EUR/USD' },
    { pair: 'GBPUSD', label: 'GBP/USD' },
    { pair: 'USDJPY', label: 'USD/JPY' },
    { pair: 'AUDUSD', label: 'AUD/USD' },
    { pair: 'USDCAD', label: 'USD/CAD' },
  ];

  // Datos de ejemplo para análisis técnico
  const technicalIndicators = [
    { name: 'RSI', value: 65.4, status: 'neutral', trend: 'up' },
    { name: 'MACD', value: 0.0023, status: 'bullish', trend: 'up' },
    { name: 'Bollinger Bands', value: 'Upper', status: 'bearish', trend: 'down' },
    { name: 'Moving Average', value: '50 SMA', status: 'bullish', trend: 'up' }
  ];

  // Datos de ejemplo para análisis fundamental
  const fundamentalData = [
    { metric: 'GDP Growth', value: '2.1%', impact: 'positive', change: '+0.2%' },
    { metric: 'Inflation Rate', value: '3.2%', impact: 'negative', change: '+0.1%' },
    { metric: 'Interest Rate', value: '5.25%', impact: 'neutral', change: '0%' },
    { metric: 'Employment', value: '3.7%', impact: 'positive', change: '-0.1%' }
  ];

  // Señales de trading
  const tradingSignals = [
    { 
      type: 'BUY', 
      pair: 'EURUSD', 
      price: '1.0854', 
      strength: 'strong',
      reason: 'RSI oversold + Support level',
      time: '2 min ago'
    },
    { 
      type: 'SELL', 
      pair: 'GBPUSD', 
      price: '1.2654', 
      strength: 'medium',
      reason: 'Resistance level reached',
      time: '5 min ago'
    },
    { 
      type: 'HOLD', 
      pair: 'USDJPY', 
      price: '148.23', 
      strength: 'weak',
      reason: 'Mixed signals',
      time: '8 min ago'
    }
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'bullish': return 'text-green-400';
      case 'bearish': return 'text-red-400';
      case 'neutral': return 'text-yellow-400';
      default: return 'text-gray-400';
    }
  };

  const getSignalColor = (type: string) => {
    switch (type) {
      case 'BUY': return 'text-green-400 bg-green-500/20 border-green-500/30';
      case 'SELL': return 'text-red-400 bg-red-500/20 border-red-500/30';
      case 'HOLD': return 'text-yellow-400 bg-yellow-500/20 border-yellow-500/30';
      default: return 'text-gray-400 bg-gray-500/20 border-gray-500/30';
    }
  };

  const getStrengthColor = (strength: string) => {
    switch (strength) {
      case 'strong': return 'text-green-400';
      case 'medium': return 'text-yellow-400';
      case 'weak': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between space-y-4 sm:space-y-0">
        <div>
          <h1 className="text-2xl font-bold text-white">Análisis Avanzado</h1>
          <p className="text-gray-400">Herramientas de análisis técnico y fundamental</p>
        </div>
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2 bg-green-500/20 border border-green-500/30 rounded-lg px-3 py-2">
            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
            <span className="text-sm text-green-400 font-medium">Análisis Activo</span>
          </div>
        </div>
      </div>

      {/* Controles */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Par de Trading */}
        <div className="trading-card p-4">
          <label className="block text-sm text-gray-400 mb-2">Par de Trading</label>
          <select 
            value={selectedPair}
            onChange={(e) => setSelectedPair(e.target.value)}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white focus:border-blue-500 focus:outline-none"
          >
            {tradingPairs.map(pair => (
              <option key={pair.pair} value={pair.pair}>{pair.label}</option>
            ))}
          </select>
        </div>

        {/* Timeframe */}
        <div className="trading-card p-4">
          <label className="block text-sm text-gray-400 mb-2">Timeframe</label>
          <select 
            value={selectedTimeframe}
            onChange={(e) => setSelectedTimeframe(e.target.value)}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white focus:border-blue-500 focus:outline-none"
          >
            {timeframes.map(tf => (
              <option key={tf} value={tf}>{tf}</option>
            ))}
          </select>
        </div>

        {/* Tipo de Análisis */}
        <div className="trading-card p-4">
          <label className="block text-sm text-gray-400 mb-2">Tipo de Análisis</label>
          <div className="flex space-x-2">
            <button
              onClick={() => setAnalysisType('technical')}
              className={`flex-1 px-3 py-2 rounded text-sm font-medium transition-colors ${
                analysisType === 'technical'
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              Técnico
            </button>
            <button
              onClick={() => setAnalysisType('fundamental')}
              className={`flex-1 px-3 py-2 rounded text-sm font-medium transition-colors ${
                analysisType === 'fundamental'
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              Fundamental
            </button>
          </div>
        </div>
      </div>

      {/* Contenido Principal */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Análisis Principal */}
        <div className="lg:col-span-2">
          <div className="trading-card p-6">
            <h2 className="text-xl font-bold text-white mb-4">
              {analysisType === 'technical' ? 'Análisis Técnico' : 'Análisis Fundamental'}
            </h2>
            
            {analysisType === 'technical' ? (
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {technicalIndicators.map((indicator, index) => (
                    <div key={index} className="bg-gray-800/50 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-semibold text-white">{indicator.name}</span>
                        <span className={`text-sm ${getStatusColor(indicator.status)}`}>
                          {indicator.status}
                        </span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span className="text-lg font-bold text-white">{indicator.value}</span>
                        {indicator.trend === 'up' ? (
                          <TrendingUp className="w-4 h-4 text-green-400" />
                        ) : (
                          <TrendingDown className="w-4 h-4 text-red-400" />
                        )}
                      </div>
                    </div>
                  ))}
                </div>
                
                <div className="bg-gray-800/50 rounded-lg p-4">
                  <h3 className="text-lg font-semibold text-white mb-3">Resumen Técnico</h3>
                  <div className="space-y-2">
                    <div className="flex items-center space-x-2">
                      <CheckCircle className="w-4 h-4 text-green-400" />
                      <span className="text-sm text-gray-300">RSI indica sobreventa en EUR/USD</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <AlertTriangle className="w-4 h-4 text-yellow-400" />
                      <span className="text-sm text-gray-300">MACD muestra divergencia positiva</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Clock className="w-4 h-4 text-blue-400" />
                      <span className="text-sm text-gray-300">Soporte en 1.0840, resistencia en 1.0870</span>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {fundamentalData.map((data, index) => (
                    <div key={index} className="bg-gray-800/50 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-semibold text-white">{data.metric}</span>
                        <span className={`text-sm ${getStatusColor(data.impact)}`}>
                          {data.impact}
                        </span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span className="text-lg font-bold text-white">{data.value}</span>
                        <span className="text-sm text-gray-400">({data.change})</span>
                      </div>
                    </div>
                  ))}
                </div>
                
                <div className="bg-gray-800/50 rounded-lg p-4">
                  <h3 className="text-lg font-semibold text-white mb-3">Impacto en Mercado</h3>
                  <div className="space-y-2">
                    <div className="flex items-center space-x-2">
                      <TrendingUp className="w-4 h-4 text-green-400" />
                      <span className="text-sm text-gray-300">GDP positivo fortalece USD</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <TrendingDown className="w-4 h-4 text-red-400" />
                      <span className="text-sm text-gray-300">Inflación alta presiona EUR</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Activity className="w-4 h-4 text-blue-400" />
                      <span className="text-sm text-gray-300">Fed mantiene tasas estables</span>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Señales de Trading */}
        <div className="lg:col-span-1">
          <div className="trading-card p-6">
            <h2 className="text-xl font-bold text-white mb-4">Señales de Trading</h2>
            <div className="space-y-3">
              {tradingSignals.map((signal, index) => (
                <div key={index} className={`border rounded-lg p-3 ${getSignalColor(signal.type)}`}>
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-semibold">{signal.type}</span>
                    <span className={`text-xs ${getStrengthColor(signal.strength)}`}>
                      {signal.strength}
                    </span>
                  </div>
                  <div className="text-sm text-gray-300 mb-1">{signal.pair} @ {signal.price}</div>
                  <div className="text-xs text-gray-400 mb-2">{signal.reason}</div>
                  <div className="text-xs text-gray-500">{signal.time}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}; 