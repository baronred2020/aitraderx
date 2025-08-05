import React, { useState } from 'react';
import { usePortfolio, TradingHistoryItem } from '../../hooks/usePortfolio';

const TradingHistory: React.FC = () => {
  const [selectedPeriod, setSelectedPeriod] = useState<'1d' | '1w' | '1m' | '3m' | 'all'>('1w');
  const [selectedPair, setSelectedPair] = useState<string>('all');
  const [selectedBrainType, setSelectedBrainType] = useState<string>('all');

  const { history, isLoading, error, fetchHistory } = usePortfolio();

  // Función para manejar cambio de período
  const handlePeriodChange = (period: '1d' | '1w' | '1m' | '3m' | 'all') => {
    setSelectedPeriod(period);
    fetchHistory({ period });
  };

  // Función para manejar cambio de par
  const handlePairChange = (pair: string) => {
    setSelectedPair(pair);
    fetchHistory({ pair: pair === 'all' ? undefined : pair });
  };

  // Función para manejar cambio de brain type
  const handleBrainTypeChange = (brainType: string) => {
    setSelectedBrainType(brainType);
    fetchHistory({ brain_type: brainType === 'all' ? undefined : brainType });
  };

  // Obtener pares únicos
  const uniquePairs = ['all', ...Array.from(new Set(history.map(trade => trade.pair)))];
  
  // Obtener brain types únicos
  const uniqueBrainTypes = ['all', ...Array.from(new Set(history.map(trade => trade.brain_type)))];

  // Función para formatear fecha
  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString('es-ES', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  // Función para formatear moneda
  const formatCurrency = (num: number) => {
    return new Intl.NumberFormat('es-ES', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2
    }).format(num);
  };

  const getTypeColor = (type: string) => {
    return type === 'buy' ? 'text-green-400' : 'text-red-400';
  };

  const getTypeIcon = (type: string) => {
    return type === 'buy' ? '📈' : '📉';
  };

  const getPnLColor = (pnl: number) => {
    return pnl >= 0 ? 'text-green-400' : 'text-red-400';
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'closed': return 'text-green-400';
      case 'open': return 'text-yellow-400';
      case 'cancelled': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'closed': return '✅';
      case 'open': return '⏳';
      case 'cancelled': return '❌';
      default: return '❓';
    }
  };

  // Calcular métricas del historial filtrado
  const totalPnL = history.reduce((sum, trade) => sum + trade.pnl, 0);
  const winningTrades = history.filter(trade => trade.pnl > 0).length;
  const totalTrades = history.length;
  const winRate = totalTrades > 0 ? (winningTrades / totalTrades) * 100 : 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h2 className="text-2xl font-bold text-white mb-4">📈 Historial de Trading</h2>
        
        {/* Loading State */}
        {isLoading && (
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
            <span className="ml-2 text-gray-400">Cargando historial...</span>
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

        {/* Filters */}
        <div className="flex flex-wrap gap-4 mb-6">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Período</label>
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
              <option value="all">Todo</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Par</label>
            <select
              value={selectedPair}
              onChange={(e) => handlePairChange(e.target.value)}
              className="bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
              disabled={isLoading}
            >
              {uniquePairs.map(pair => (
                <option key={pair} value={pair}>
                  {pair === 'all' ? 'Todos los pares' : pair}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Brain Type</label>
            <select
              value={selectedBrainType}
              onChange={(e) => handleBrainTypeChange(e.target.value)}
              className="bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
              disabled={isLoading}
            >
              {uniqueBrainTypes.map(brainType => (
                <option key={brainType} value={brainType}>
                  {brainType === 'all' ? 'Todos los Brain Types' : brainType}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Summary Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-gray-700 rounded-lg p-4">
            <div className="text-2xl font-bold text-white">{totalTrades}</div>
            <div className="text-sm text-gray-400">Total Trades</div>
          </div>
          <div className="bg-gray-700 rounded-lg p-4">
            <div className="text-2xl font-bold text-green-400">{winningTrades}</div>
            <div className="text-sm text-gray-400">Trades Ganadores</div>
          </div>
          <div className="bg-gray-700 rounded-lg p-4">
            <div className="text-2xl font-bold text-blue-400">{winRate.toFixed(1)}%</div>
            <div className="text-sm text-gray-400">Win Rate</div>
          </div>
          <div className="bg-gray-700 rounded-lg p-4">
            <div className={`text-2xl font-bold ${getPnLColor(totalPnL)}`}>
              {formatCurrency(totalPnL)}
            </div>
            <div className="text-sm text-gray-400">P&L Total</div>
          </div>
        </div>
      </div>

      {/* Trades Table */}
      <div className="bg-gray-800 rounded-lg overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-700">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                  Trade
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                  Estrategia
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                  Par
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                  Tipo
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                  Entrada
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                  Salida
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                  Pips
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                  P&L
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                  Estado
                </th>
              </tr>
            </thead>
            <tbody className="bg-gray-800 divide-y divide-gray-700">
              {history.map((trade) => (
                <tr key={trade.id} className="hover:bg-gray-700">
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-white">
                    #{trade.id}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                    {trade.brain_type}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-white font-medium">
                    {trade.pair}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getTypeColor(trade.direction)}`}>
                      {getTypeIcon(trade.direction)} {trade.direction.toUpperCase()}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                    {trade.entry_price.toFixed(5)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                    {trade.exit_price ? trade.exit_price.toFixed(5) : '-'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-white">
                    {trade.pips || '-'}
                  </td>
                  <td className={`px-6 py-4 whitespace-nowrap text-sm font-medium ${getPnLColor(trade.pnl)}`}>
                    {formatCurrency(trade.pnl)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(trade.status)}`}>
                      {getStatusIcon(trade.status)} {trade.status.toUpperCase()}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {history.length === 0 && !isLoading && (
          <div className="text-center py-8">
            <div className="text-4xl mb-2">📊</div>
            <h3 className="text-lg font-medium text-white mb-2">No hay trades en este período</h3>
            <p className="text-gray-400">Cambia los filtros para ver más resultados</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default TradingHistory; 