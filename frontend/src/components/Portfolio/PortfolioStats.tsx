import React, { useState } from 'react';
import { usePortfolio, PortfolioStats as PortfolioStatsType } from '../../hooks/usePortfolio';

interface PortfolioStatsProps {}

const PortfolioStats: React.FC<PortfolioStatsProps> = () => {
  const [selectedPeriod, setSelectedPeriod] = useState<'1d' | '1w' | '1m' | '3m' | '1y'>('1m');
  
  const { stats, isLoading, error, fetchStats } = usePortfolio();

  // Función para manejar cambio de período
  const handlePeriodChange = (period: '1d' | '1w' | '1m' | '3m' | '1y') => {
    setSelectedPeriod(period);
    fetchStats(period);
  };

  // Función para formatear números
  const formatNumber = (num: number, decimals: number = 2) => {
    return new Intl.NumberFormat('es-ES', {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals
    }).format(num);
  };

  // Función para formatear porcentajes
  const formatPercentage = (num: number) => {
    return `${formatNumber(num, 1)}%`;
  };

  // Función para formatear moneda
  const formatCurrency = (num: number) => {
    return new Intl.NumberFormat('es-ES', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2
    }).format(num);
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
            onChange={(e) => handlePeriodChange(e.target.value as any)}
            className="bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
            disabled={isLoading}
          >
            <option value="1d">Último día</option>
            <option value="1w">Última semana</option>
            <option value="1m">Último mes</option>
            <option value="3m">Últimos 3 meses</option>
            <option value="1y">Último año</option>
          </select>
        </div>

        {/* Loading State */}
        {isLoading && (
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
            <span className="ml-2 text-gray-400">Cargando estadísticas...</span>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="bg-red-900 border border-red-700 rounded-lg p-4 mb-6">
            <div className="flex items-center">
              <span className="text-red-400 text-xl mr-2">⚠️</span>
              <span className="text-red-300">Error: {error}</span>
            </div>
          </div>
        )}

        {/* Key Metrics */}
        {stats && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-gray-700 rounded-lg p-4">
              <div className={`text-3xl font-bold ${getPnLColor(stats.total_pnl)}`}>
                {formatCurrency(stats.total_pnl)}
              </div>
              <div className="text-sm text-gray-400">P&L Total</div>
            </div>
            
            <div className="bg-gray-700 rounded-lg p-4">
              <div className="text-3xl font-bold text-white">
                {stats.total_trades}
              </div>
              <div className="text-sm text-gray-400">Total Trades</div>
            </div>
            
            <div className="bg-gray-700 rounded-lg p-4">
              <div className={`text-3xl font-bold ${getPerformanceColor(stats.win_rate)}`}>
                {formatPercentage(stats.win_rate)}
              </div>
              <div className="text-sm text-gray-400">Win Rate</div>
            </div>
            
            <div className="bg-gray-700 rounded-lg p-4">
              <div className="text-3xl font-bold text-blue-400">
                {formatNumber(stats.sharpe_ratio)}
              </div>
              <div className="text-sm text-gray-400">Sharpe Ratio</div>
            </div>
          </div>
        )}
      </div>

      {/* Detailed Stats */}
      {stats && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Trading Performance */}
          <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-xl font-bold text-white mb-4">📈 Rendimiento de Trading</h3>
            
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Trades Ganadores</span>
                <span className="text-green-400 font-medium">{stats.winning_trades}</span>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Trades Perdedores</span>
                <span className="text-red-400 font-medium">{stats.losing_trades}</span>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Ganancia Promedio</span>
                <span className="text-green-400 font-medium">{formatCurrency(stats.avg_win)}</span>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Pérdida Promedio</span>
                <span className="text-red-400 font-medium">{formatCurrency(stats.avg_loss)}</span>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Profit Factor</span>
                <span className="text-blue-400 font-medium">{formatNumber(stats.profit_factor)}</span>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Predicciones Exitosas</span>
                <span className="text-white font-medium">{formatPercentage(stats.success_rate)}</span>
              </div>
            </div>
          </div>

          {/* Risk Metrics */}
          <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-xl font-bold text-white mb-4">🛡️ Métricas de Riesgo</h3>
            
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Máximo Drawdown</span>
                <span className="text-red-400 font-medium">{formatCurrency(stats.max_drawdown)}</span>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Sharpe Ratio</span>
                <span className="text-blue-400 font-medium">{formatNumber(stats.sharpe_ratio)}</span>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-gray-400">P&L Diario</span>
                <span className={`font-medium ${stats.daily_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {formatCurrency(stats.daily_pnl)}
                </span>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-gray-400">P&L Semanal</span>
                <span className={`font-medium ${stats.weekly_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {formatCurrency(stats.weekly_pnl)}
                </span>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-gray-400">P&L Mensual</span>
                <span className={`font-medium ${stats.monthly_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {formatCurrency(stats.monthly_pnl)}
                </span>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Señales Exitosas</span>
                <span className="text-white font-medium">{formatPercentage(stats.signal_success_rate)}</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Strategy Performance */}
      {stats && (
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-xl font-bold text-white mb-4">🎯 Rendimiento por Estrategia</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-gray-700 rounded-lg p-4">
              <h4 className="text-lg font-medium text-white mb-3">🏆 Mejor Par</h4>
              <div className="space-y-2">
                <div className="text-green-400 font-medium">{stats.best_pair || 'N/A'}</div>
                <div className="text-sm text-gray-400">Brain Type: {stats.best_brain_type || 'N/A'}</div>
                <div className="text-sm text-gray-400">Total P&L: {formatCurrency(stats.total_pnl)}</div>
                <div className="text-sm text-gray-400">Win Rate: {formatPercentage(stats.win_rate)}</div>
              </div>
            </div>
            
            <div className="bg-gray-700 rounded-lg p-4">
              <h4 className="text-lg font-medium text-white mb-3">⚠️ Peor Par</h4>
              <div className="space-y-2">
                <div className="text-red-400 font-medium">{stats.worst_pair || 'N/A'}</div>
                <div className="text-sm text-gray-400">Brain Type: {stats.worst_brain_type || 'N/A'}</div>
                <div className="text-sm text-gray-400">Total P&L: {formatCurrency(stats.total_pnl)}</div>
                <div className="text-sm text-gray-400">Win Rate: {formatPercentage(stats.win_rate)}</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Daily Performance */}
      {stats && (
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-xl font-bold text-white mb-4">📅 Rendimiento Diario</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-gray-700 rounded-lg p-4">
              <h4 className="text-lg font-medium text-green-400 mb-3">📈 P&L Diario</h4>
              <div className="space-y-2">
                <div className={`text-2xl font-bold ${stats.daily_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {formatCurrency(stats.daily_pnl)}
                </div>
                <div className="text-sm text-gray-400">Total Trades: {stats.total_trades}</div>
                <div className="text-sm text-gray-400">Win Rate: {formatPercentage(stats.win_rate)}</div>
              </div>
            </div>
            
            <div className="bg-gray-700 rounded-lg p-4">
              <h4 className="text-lg font-medium text-blue-400 mb-3">📊 Resumen</h4>
              <div className="space-y-2">
                <div className="text-sm text-gray-400">Predicciones: {stats.total_predictions}</div>
                <div className="text-sm text-gray-400">Señales: {stats.total_signals}</div>
                <div className="text-sm text-gray-400">Exitosas: {formatPercentage(stats.success_rate)}</div>
                <div className="text-sm text-gray-400">Profit Factor: {formatNumber(stats.profit_factor)}</div>
              </div>
            </div>
          </div>
        </div>
      )}

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