import React, { useState } from 'react';

interface PortfolioStatsProps {}

const PortfolioStats: React.FC<PortfolioStatsProps> = () => {
  const [selectedPeriod, setSelectedPeriod] = useState<'1d' | '1w' | '1m' | '3m' | '1y'>('1m');

  // Mock data - esto vendría del backend
  const mockStats = {
    totalPnL: 1250.50,
    totalTrades: 156,
    winningTrades: 98,
    losingTrades: 58,
    winRate: 62.8,
    avgWin: 15.20,
    avgLoss: -8.50,
    maxDrawdown: -320.75,
    sharpeRatio: 1.85,
    profitFactor: 2.45,
    totalVolume: 15600.00,
    bestStrategy: 'Scalping EURUSD Brain Ultra',
    worstStrategy: 'Swing Trading GBPUSD Mega Mind',
    bestDay: '2024-01-15',
    worstDay: '2024-01-08'
  };

  const getPnLColor = (value: number) => {
    return value >= 0 ? 'text-green-400' : 'text-red-400';
  };

  const getPerformanceColor = (value: number) => {
    if (value >= 80) return 'text-green-400';
    if (value >= 60) return 'text-yellow-400';
    return 'text-red-400';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gray-800 rounded-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-2xl font-bold text-white mb-2">📊 Estadísticas del Portfolio</h2>
            <p className="text-gray-400">Análisis detallado del rendimiento de trading</p>
          </div>
          
          <select
            value={selectedPeriod}
            onChange={(e) => setSelectedPeriod(e.target.value as any)}
            className="bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
          >
            <option value="1d">Último día</option>
            <option value="1w">Última semana</option>
            <option value="1m">Último mes</option>
            <option value="3m">Últimos 3 meses</option>
            <option value="1y">Último año</option>
          </select>
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-gray-700 rounded-lg p-4">
            <div className={`text-3xl font-bold ${getPnLColor(mockStats.totalPnL)}`}>
              ${mockStats.totalPnL.toFixed(2)}
            </div>
            <div className="text-sm text-gray-400">P&L Total</div>
          </div>
          
          <div className="bg-gray-700 rounded-lg p-4">
            <div className="text-3xl font-bold text-white">
              {mockStats.totalTrades}
            </div>
            <div className="text-sm text-gray-400">Total Trades</div>
          </div>
          
          <div className="bg-gray-700 rounded-lg p-4">
            <div className={`text-3xl font-bold ${getPerformanceColor(mockStats.winRate)}`}>
              {mockStats.winRate}%
            </div>
            <div className="text-sm text-gray-400">Win Rate</div>
          </div>
          
          <div className="bg-gray-700 rounded-lg p-4">
            <div className="text-3xl font-bold text-blue-400">
              {mockStats.sharpeRatio.toFixed(2)}
            </div>
            <div className="text-sm text-gray-400">Sharpe Ratio</div>
          </div>
        </div>
      </div>

      {/* Detailed Stats */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Trading Performance */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-xl font-bold text-white mb-4">📈 Rendimiento de Trading</h3>
          
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Trades Ganadores</span>
              <span className="text-green-400 font-medium">{mockStats.winningTrades}</span>
            </div>
            
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Trades Perdedores</span>
              <span className="text-red-400 font-medium">{mockStats.losingTrades}</span>
            </div>
            
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Ganancia Promedio</span>
              <span className="text-green-400 font-medium">${mockStats.avgWin.toFixed(2)}</span>
            </div>
            
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Pérdida Promedio</span>
              <span className="text-red-400 font-medium">${mockStats.avgLoss.toFixed(2)}</span>
            </div>
            
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Profit Factor</span>
              <span className="text-blue-400 font-medium">{mockStats.profitFactor.toFixed(2)}</span>
            </div>
            
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Volumen Total</span>
              <span className="text-white font-medium">${mockStats.totalVolume.toFixed(2)}</span>
            </div>
          </div>
        </div>

        {/* Risk Metrics */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-xl font-bold text-white mb-4">🛡️ Métricas de Riesgo</h3>
          
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Máximo Drawdown</span>
              <span className="text-red-400 font-medium">${mockStats.maxDrawdown.toFixed(2)}</span>
            </div>
            
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Sharpe Ratio</span>
              <span className="text-blue-400 font-medium">{mockStats.sharpeRatio.toFixed(2)}</span>
            </div>
            
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Ratio Riesgo/Recompensa</span>
              <span className="text-white font-medium">1:1.8</span>
            </div>
            
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Volatilidad</span>
              <span className="text-yellow-400 font-medium">12.5%</span>
            </div>
            
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Beta</span>
              <span className="text-white font-medium">0.85</span>
            </div>
            
            <div className="flex justify-between items-center">
              <span className="text-gray-400">VaR (95%)</span>
              <span className="text-red-400 font-medium">-$150.00</span>
            </div>
          </div>
        </div>
      </div>

      {/* Strategy Performance */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-xl font-bold text-white mb-4">🎯 Rendimiento por Estrategia</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-gray-700 rounded-lg p-4">
            <h4 className="text-lg font-medium text-white mb-3">🏆 Mejor Estrategia</h4>
            <div className="space-y-2">
              <div className="text-green-400 font-medium">{mockStats.bestStrategy}</div>
              <div className="text-sm text-gray-400">P&L: +$450.25</div>
              <div className="text-sm text-gray-400">Win Rate: 75.2%</div>
              <div className="text-sm text-gray-400">Trades: 45</div>
            </div>
          </div>
          
          <div className="bg-gray-700 rounded-lg p-4">
            <h4 className="text-lg font-medium text-white mb-3">⚠️ Peor Estrategia</h4>
            <div className="space-y-2">
              <div className="text-red-400 font-medium">{mockStats.worstStrategy}</div>
              <div className="text-sm text-gray-400">P&L: -$125.80</div>
              <div className="text-sm text-gray-400">Win Rate: 45.8%</div>
              <div className="text-sm text-gray-400">Trades: 24</div>
            </div>
          </div>
        </div>
      </div>

      {/* Daily Performance */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-xl font-bold text-white mb-4">📅 Rendimiento Diario</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-gray-700 rounded-lg p-4">
            <h4 className="text-lg font-medium text-green-400 mb-3">📈 Mejor Día</h4>
            <div className="space-y-2">
              <div className="text-white font-medium">{mockStats.bestDay}</div>
              <div className="text-sm text-gray-400">P&L: +$85.50</div>
              <div className="text-sm text-gray-400">Trades: 8</div>
              <div className="text-sm text-gray-400">Win Rate: 87.5%</div>
            </div>
          </div>
          
          <div className="bg-gray-700 rounded-lg p-4">
            <h4 className="text-lg font-medium text-red-400 mb-3">📉 Peor Día</h4>
            <div className="space-y-2">
              <div className="text-white font-medium">{mockStats.worstDay}</div>
              <div className="text-sm text-gray-400">P&L: -$45.20</div>
              <div className="text-sm text-gray-400">Trades: 6</div>
              <div className="text-sm text-gray-400">Win Rate: 33.3%</div>
            </div>
          </div>
        </div>
      </div>

      {/* Performance Chart Placeholder */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-xl font-bold text-white mb-4">📊 Gráfico de Rendimiento</h3>
        <div className="bg-gray-700 rounded-lg p-8 text-center">
          <div className="text-4xl mb-2">📈</div>
          <p className="text-gray-400">Gráfico de rendimiento del portfolio</p>
          <p className="text-sm text-gray-500 mt-2">Aquí se mostrará el gráfico de P&L a lo largo del tiempo</p>
        </div>
      </div>
    </div>
  );
};

export default PortfolioStats; 