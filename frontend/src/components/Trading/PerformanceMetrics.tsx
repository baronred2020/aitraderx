import React from 'react';
import { TrendingUp, TrendingDown, DollarSign, Target, BarChart3, Activity } from 'lucide-react';

interface PerformanceStats {
  totalPositions: number;
  openPositions: number;
  closedPositions: number;
  totalPnL: number;
  totalPnLPercent: number;
  totalOrders: number;
  filledOrdersCount: number;
  fillRate: number;
  winRate: number;
}

interface PerformanceMetricsProps {
  stats: PerformanceStats;
  balance: number | null;
}

const PerformanceMetrics: React.FC<PerformanceMetricsProps> = ({ stats, balance }) => {
  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(amount);
  };

  const formatPercentage = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  const getPnLColor = (value: number) => {
    if (value > 0) return 'text-green-400';
    if (value < 0) return 'text-red-400';
    return 'text-gray-400';
  };

  const getPnLBgColor = (value: number) => {
    if (value > 0) return 'bg-green-500/20 border-green-500/30';
    if (value < 0) return 'bg-red-500/20 border-red-500/30';
    return 'bg-gray-500/20 border-gray-500/30';
  };

  // Calcular P&L promedio por posición
  const averagePnL = stats.closedPositions > 0 ? stats.totalPnL / stats.closedPositions : 0;

  return (
    <div className="trading-card p-4 sm:p-6">
      <h3 className="text-lg font-semibold text-white mb-4 sm:mb-6 flex items-center">
        <BarChart3 className="w-5 h-5 mr-2 text-blue-400" />
        P&L y Rendimiento
      </h3>

      {/* Cards principales - Mejorado para PC */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4 mb-6">
        {/* P&L Total */}
        <div className={`p-3 sm:p-4 rounded-lg border ${getPnLBgColor(stats.totalPnL)} min-w-0`}>
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs sm:text-sm text-gray-400 truncate">P&L Total</span>
            {stats.totalPnL > 0 ? (
              <TrendingUp className="w-4 h-4 text-green-400 flex-shrink-0" />
            ) : (
              <TrendingDown className="w-4 h-4 text-red-400 flex-shrink-0" />
            )}
          </div>
          <div className={`text-lg sm:text-xl font-bold ${getPnLColor(stats.totalPnL)} truncate`}>
            {formatCurrency(stats.totalPnL)}
          </div>
          <div className={`text-xs sm:text-sm ${getPnLColor(stats.totalPnLPercent)} truncate`}>
            {formatPercentage(stats.totalPnLPercent)}
          </div>
        </div>

        {/* Win Rate */}
        <div className="bg-blue-500/20 border border-blue-500/30 p-3 sm:p-4 rounded-lg min-w-0">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs sm:text-sm text-gray-400 truncate">Win Rate</span>
            <Target className="w-4 h-4 text-blue-400 flex-shrink-0" />
          </div>
          <div className="text-lg sm:text-xl font-bold text-blue-400">
            {stats.winRate.toFixed(1)}%
          </div>
          <div className="text-xs sm:text-sm text-gray-400 truncate">
            {stats.closedPositions} posiciones
          </div>
        </div>

        {/* Fill Rate */}
        <div className="bg-purple-500/20 border border-purple-500/30 p-3 sm:p-4 rounded-lg min-w-0">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs sm:text-sm text-gray-400 truncate">Fill Rate</span>
            <Activity className="w-4 h-4 text-purple-400 flex-shrink-0" />
          </div>
          <div className="text-lg sm:text-xl font-bold text-purple-400">
            {stats.fillRate.toFixed(1)}%
          </div>
          <div className="text-xs sm:text-sm text-gray-400 truncate">
            {stats.filledOrdersCount}/{stats.totalOrders} órdenes
          </div>
        </div>

        {/* Balance */}
        <div className="bg-green-500/20 border border-green-500/30 p-3 sm:p-4 rounded-lg min-w-0">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs sm:text-sm text-gray-400 truncate">Balance</span>
            <DollarSign className="w-4 h-4 text-green-400 flex-shrink-0" />
          </div>
          <div className="text-lg sm:text-xl font-bold text-green-400 truncate">
            {balance ? formatCurrency(balance) : 'N/A'}
          </div>
          <div className="text-xs sm:text-sm text-gray-400 truncate">
            Saldo disponible
          </div>
        </div>
      </div>

      {/* Estadísticas Detalladas - Mejorado para PC */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6">
        {/* Posiciones */}
        <div className="bg-gray-800/50 rounded-lg p-3 sm:p-4">
          <h4 className="text-sm font-semibold text-gray-400 mb-3">Posiciones</h4>
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-400">Total:</span>
              <span className="text-white font-medium">{stats.totalPositions}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-400">Abiertas:</span>
              <span className="text-green-400 font-medium">{stats.openPositions}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-400">Cerradas:</span>
              <span className="text-blue-400 font-medium">{stats.closedPositions}</span>
            </div>
          </div>
        </div>

        {/* Órdenes */}
        <div className="bg-gray-800/50 rounded-lg p-3 sm:p-4">
          <h4 className="text-sm font-semibold text-gray-400 mb-3">Órdenes</h4>
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-400">Total:</span>
              <span className="text-white font-medium">{stats.totalOrders}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-400">Ejecutadas:</span>
              <span className="text-green-400 font-medium">{stats.filledOrdersCount}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-400">Pendientes:</span>
              <span className="text-yellow-400 font-medium">{stats.totalOrders - stats.filledOrdersCount}</span>
            </div>
          </div>
        </div>

        {/* Rendimiento */}
        <div className="bg-gray-800/50 rounded-lg p-3 sm:p-4">
          <h4 className="text-sm font-semibold text-gray-400 mb-3">Rendimiento</h4>
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-400">P&L Promedio:</span>
              <span className={`font-medium ${getPnLColor(averagePnL)}`}>
                {formatCurrency(averagePnL)}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-400">Éxito:</span>
              <span className="text-green-400 font-medium">{stats.winRate.toFixed(1)}%</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-400">Eficiencia:</span>
              <span className="text-purple-400 font-medium">{stats.fillRate.toFixed(1)}%</span>
            </div>
          </div>
        </div>
      </div>

      {/* Indicadores de Estado - Mejorado para PC */}
      <div className="mt-4 sm:mt-6 flex flex-wrap gap-2">
        {stats.openPositions > 0 && (
          <span className="px-3 py-1 bg-green-500/20 text-green-400 text-xs rounded-full">
            {stats.openPositions} posiciones activas
          </span>
        )}
        {stats.totalPnL > 0 && (
          <span className="px-3 py-1 bg-green-500/20 text-green-400 text-xs rounded-full">
            En ganancia
          </span>
        )}
        {stats.totalPnL < 0 && (
          <span className="px-3 py-1 bg-red-500/20 text-red-400 text-xs rounded-full">
            En pérdida
          </span>
        )}
        {stats.winRate >= 60 && (
          <span className="px-3 py-1 bg-blue-500/20 text-blue-400 text-xs rounded-full">
            Alto win rate
          </span>
        )}
        {stats.fillRate >= 80 && (
          <span className="px-3 py-1 bg-purple-500/20 text-purple-400 text-xs rounded-full">
            Alta eficiencia
          </span>
        )}
      </div>
    </div>
  );
};

export default PerformanceMetrics; 