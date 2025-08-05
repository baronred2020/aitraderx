import React from 'react';

interface Strategy {
  id: string;
  name: string;
  brainType: string;
  pair: string;
  style: string;
  status: 'active' | 'paused' | 'stopped';
  currentPrice: number;
  totalTrades: number;
  winningTrades: number;
  totalPnL: number;
  openPositions: number;
  lastSignal: string;
  lastSignalTime: string;
  createdAt: string;
  lotSize: number;
  stopLossPips: number;
  takeProfitPips: number;
  minConfidence: number;
  maxPositions: number;
  riskPerTrade: number;
}

interface ActiveStrategiesProps {
  strategies: Strategy[];
  onStart: (id: string) => void;
  onStop: (id: string) => void;
  onDelete?: (id: string) => void;
  onDetails?: (strategy: Strategy) => void;
  isLoading: boolean;
}

const ActiveStrategies: React.FC<ActiveStrategiesProps> = ({ 
  strategies, 
  onStart, 
  onStop, 
  onDelete, 
  onDetails, 
  isLoading 
}) => {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'text-green-400';
      case 'paused': return 'text-yellow-400';
      case 'stopped': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active': return '🟢';
      case 'paused': return '🟡';
      case 'stopped': return '🔴';
      default: return '⚪';
    }
  };

  const getBrainIcon = (brainType: string) => {
    switch (brainType) {
      case 'Brain_Ultra': return '🧠';
      case 'Brain_Max': return '🧠';
      case 'Brain_Predictor': return '🔮';
      case 'Mega_Mind': return '🤖';
      default: return '🧠';
    }
  };

  const getStyleIcon = (style: string) => {
    switch (style) {
      case 'scalping': return '⚡';
      case 'day_trading': return '📈';
      case 'swing_trading': return '📊';
      default: return '📈';
    }
  };

  if (strategies.length === 0) {
    return (
      <div className="bg-gray-800 rounded-lg p-8 text-center">
        <div className="text-6xl mb-4">🤖</div>
        <h3 className="text-xl font-medium text-white mb-2">No hay estrategias configuradas</h3>
        <p className="text-gray-400 mb-6">Crea tu primera estrategia de trading automático para comenzar</p>
        <div className="text-sm text-gray-500">
          💡 Tip: Comienza con una estrategia de scalping en EURUSD usando Brain Ultra
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <h3 className="text-xl font-bold text-white">📊 Estrategias Activas</h3>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {strategies.map((strategy) => (
          <div key={strategy.id} className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-3">
                <div className="text-2xl">{getBrainIcon(strategy.brainType)}</div>
                <div>
                  <h4 className="text-lg font-semibold text-white">{strategy.name}</h4>
                  <div className="flex items-center space-x-2 text-sm text-gray-400">
                    <span>{strategy.pair}</span>
                    <span>•</span>
                    <span>{getStyleIcon(strategy.style)} {strategy.style}</span>
                  </div>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <span className={`text-sm font-medium ${getStatusColor(strategy.status)}`}>
                  {getStatusIcon(strategy.status)} {strategy.status}
                </span>
              </div>
            </div>

            {/* Current Price and P&L */}
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div className="bg-gray-700 rounded-lg p-3">
                <div className="text-sm text-gray-400 mb-1">Tipo</div>
                <div className="text-lg font-bold text-white">
                  {strategy.brainType}
                </div>
              </div>
              <div className="bg-gray-700 rounded-lg p-3">
                <div className="text-sm text-gray-400 mb-1">P&L Total</div>
                <div className={`text-lg font-bold ${(strategy.totalPnL || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  ${(strategy.totalPnL || 0).toFixed(2)}
                </div>
              </div>
            </div>

            {/* Trading Stats */}
            <div className="grid grid-cols-3 gap-3 mb-4">
              <div className="text-center">
                <div className="text-lg font-bold text-white">{strategy.totalTrades}</div>
                <div className="text-xs text-gray-400">Total Trades</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-bold text-green-400">{strategy.winningTrades}</div>
                <div className="text-xs text-gray-400">Ganadores</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-bold text-blue-400">{strategy.totalTrades}</div>
                <div className="text-xs text-gray-400">Total Trades</div>
              </div>
            </div>

            {/* Last Signal */}
            <div className="bg-gray-700 rounded-lg p-3 mb-4">
              <div className="text-sm text-gray-400 mb-1">Última Señal</div>
              <div className="flex items-center justify-between">
                <span className="text-white font-medium capitalize">
                  {strategy.lastSignal || 'N/A'}
                </span>
                <span className="text-xs text-gray-400">
                  {strategy.lastSignalTime || 'N/A'}
                </span>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex space-x-2">
              {strategy.status === 'active' ? (
                <button
                  onClick={() => onStop(strategy.id)}
                  disabled={isLoading}
                  className="flex-1 bg-red-600 hover:bg-red-700 disabled:bg-gray-600 text-white py-2 px-4 rounded-lg text-sm font-medium transition-colors"
                >
                  ⏸️ Pausar
                </button>
              ) : (
                <button
                  onClick={() => onStart(strategy.id)}
                  disabled={isLoading}
                  className="flex-1 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 text-white py-2 px-4 rounded-lg text-sm font-medium transition-colors"
                >
                  ▶️ Iniciar
                </button>
              )}
              
              <button 
                onClick={() => onDetails && onDetails(strategy)}
                disabled={isLoading}
                className="flex-1 bg-gray-600 hover:bg-gray-700 disabled:bg-gray-500 text-white py-2 px-4 rounded-lg text-sm font-medium transition-colors"
              >
                📊 Detalles
              </button>
              
              <button 
                onClick={() => onDelete && onDelete(strategy.id)}
                disabled={isLoading}
                className="bg-red-600 hover:bg-red-700 disabled:bg-gray-500 text-white py-2 px-3 rounded-lg text-sm font-medium transition-colors"
                title="Eliminar estrategia"
              >
                🗑️
              </button>
            </div>

            {/* Win Rate */}
            {strategy.totalTrades > 0 && (
              <div className="mt-3 text-center">
                <div className="text-sm text-gray-400">
                  Win Rate: <span className="text-white font-medium">
                    {strategy.totalTrades > 0 ? ((strategy.winningTrades / strategy.totalTrades) * 100).toFixed(1) : '0.0'}%
                  </span>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Summary Stats */}
      {strategies.length > 1 && (
        <div className="bg-gray-800 rounded-lg p-4 mt-6">
          <h4 className="text-lg font-medium text-white mb-3">📈 Resumen de Estrategias</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-green-400">
                {strategies.filter(s => s.status === 'active').length}
              </div>
              <div className="text-sm text-gray-400">Activas</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-400">
                {strategies.reduce((total, s) => total + s.totalTrades, 0)}
              </div>
              <div className="text-sm text-gray-400">Total Trades</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-400">
                ${(strategies.reduce((total, s) => total + (s.totalPnL || 0), 0)).toFixed(2)}
              </div>
              <div className="text-sm text-gray-400">P&L Total</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-yellow-400">
                {strategies.reduce((total, s) => total + s.totalTrades, 0)}
              </div>
              <div className="text-sm text-gray-400">Posiciones</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ActiveStrategies; 