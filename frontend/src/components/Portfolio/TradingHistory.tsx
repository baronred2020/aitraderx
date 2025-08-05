import React, { useState } from 'react';

interface Trade {
  id: string;
  strategyId: string;
  strategyName: string;
  pair: string;
  type: 'buy' | 'sell';
  entryPrice: number;
  exitPrice: number;
  lotSize: number;
  pnl: number;
  pips: number;
  entryTime: string;
  exitTime: string;
  status: 'closed' | 'open';
  exitReason: string;
}

const TradingHistory: React.FC = () => {
  const [selectedPeriod, setSelectedPeriod] = useState<'1d' | '1w' | '1m' | '3m' | 'all'>('1w');
  const [selectedStrategy, setSelectedStrategy] = useState<string>('all');

  // Mock data - esto vendría del backend
  const mockTrades: Trade[] = [
    {
      id: '1',
      strategyId: 'strategy1',
      strategyName: 'Scalping EURUSD Brain Ultra',
      pair: 'EURUSD',
      type: 'buy',
      entryPrice: 1.0856,
      exitPrice: 1.0862,
      lotSize: 0.1,
      pnl: 6.0,
      pips: 6,
      entryTime: '2024-01-15 10:30:00',
      exitTime: '2024-01-15 10:35:00',
      status: 'closed',
      exitReason: 'TAKE_PROFIT'
    },
    {
      id: '2',
      strategyId: 'strategy1',
      strategyName: 'Scalping EURUSD Brain Ultra',
      pair: 'EURUSD',
      type: 'sell',
      entryPrice: 1.0870,
      exitPrice: 1.0865,
      lotSize: 0.1,
      pnl: 5.0,
      pips: 5,
      entryTime: '2024-01-15 11:15:00',
      exitTime: '2024-01-15 11:20:00',
      status: 'closed',
      exitReason: 'TAKE_PROFIT'
    }
  ];

  const strategies = ['all', 'Scalping EURUSD Brain Ultra', 'Day Trading GBPUSD Brain Max'];

  const getTypeColor = (type: string) => {
    return type === 'buy' ? 'text-green-400' : 'text-red-400';
  };

  const getTypeIcon = (type: string) => {
    return type === 'buy' ? '📈' : '📉';
  };

  const getPnLColor = (pnl: number) => {
    return pnl >= 0 ? 'text-green-400' : 'text-red-400';
  };

  const getExitReasonColor = (reason: string) => {
    switch (reason) {
      case 'TAKE_PROFIT': return 'text-green-400';
      case 'STOP_LOSS': return 'text-red-400';
      case 'TIME_EXIT': return 'text-yellow-400';
      default: return 'text-gray-400';
    }
  };

  const filteredTrades = mockTrades.filter(trade => {
    if (selectedStrategy !== 'all' && trade.strategyName !== selectedStrategy) {
      return false;
    }
    // Aquí se agregaría filtro por período
    return true;
  });

  const totalPnL = filteredTrades.reduce((sum, trade) => sum + trade.pnl, 0);
  const winningTrades = filteredTrades.filter(trade => trade.pnl > 0).length;
  const totalTrades = filteredTrades.length;
  const winRate = totalTrades > 0 ? (winningTrades / totalTrades) * 100 : 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h2 className="text-2xl font-bold text-white mb-4">📈 Historial de Trading</h2>
        
        {/* Filters */}
        <div className="flex flex-wrap gap-4 mb-6">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Período</label>
            <select
              value={selectedPeriod}
              onChange={(e) => setSelectedPeriod(e.target.value as any)}
              className="bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
            >
              <option value="1d">Último día</option>
              <option value="1w">Última semana</option>
              <option value="1m">Último mes</option>
              <option value="3m">Últimos 3 meses</option>
              <option value="all">Todo</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Estrategia</label>
            <select
              value={selectedStrategy}
              onChange={(e) => setSelectedStrategy(e.target.value)}
              className="bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
            >
              {strategies.map(strategy => (
                <option key={strategy} value={strategy}>
                  {strategy === 'all' ? 'Todas las estrategias' : strategy}
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
              ${totalPnL.toFixed(2)}
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
              {filteredTrades.map((trade) => (
                <tr key={trade.id} className="hover:bg-gray-700">
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-white">
                    #{trade.id}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                    {trade.strategyName}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-white font-medium">
                    {trade.pair}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getTypeColor(trade.type)}`}>
                      {getTypeIcon(trade.type)} {trade.type.toUpperCase()}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                    ${trade.entryPrice.toFixed(5)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                    ${trade.exitPrice.toFixed(5)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-white">
                    {trade.pips}
                  </td>
                  <td className={`px-6 py-4 whitespace-nowrap text-sm font-medium ${getPnLColor(trade.pnl)}`}>
                    ${trade.pnl.toFixed(2)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getExitReasonColor(trade.exitReason)}`}>
                      {trade.exitReason.replace('_', ' ')}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {filteredTrades.length === 0 && (
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