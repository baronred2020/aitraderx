import React, { useState } from 'react';
import { 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  Target,
  ArrowUpRight,
  ArrowDownRight,
  Plus,
  Minus,
  Settings,
  Clock,
  AlertTriangle,
  CheckCircle,
  XCircle
} from 'lucide-react';
import { YahooTradingChart } from './YahooTradingChart';

export const TradingView: React.FC = () => {
  const [selectedSymbol, setSelectedSymbol] = useState('EURUSD');
  const [selectedTimeframe, setSelectedTimeframe] = useState('1H');
  const [orderType, setOrderType] = useState<'market' | 'limit' | 'stop'>('market');
  const [orderSide, setOrderSide] = useState<'buy' | 'sell'>('buy');
  const [orderAmount, setOrderAmount] = useState('10000');
  const [orderPrice, setOrderPrice] = useState('1.0925');

  const symbols = [
    { pair: 'EURUSD', label: 'EUR/USD', price: '1.0854', change: '+0.12%', volume: '2.4M', trend: 'up' },
    { pair: 'GBPUSD', label: 'GBP/USD', price: '1.2654', change: '-0.08%', volume: '1.8M', trend: 'down' },
    { pair: 'USDJPY', label: 'USD/JPY', price: '148.23', change: '+0.25%', volume: '3.1M', trend: 'up' },
    { pair: 'AUDUSD', label: 'AUD/USD', price: '0.6543', change: '+0.18%', volume: '1.2M', trend: 'up' },
    { pair: 'USDCAD', label: 'USD/CAD', price: '1.3542', change: '-0.05%', volume: '0.9M', trend: 'down' },
  ];

  const timeframes = ['1M', '5M', '15M', '1H', '4H', '1D', '1W'];

  const recentOrders = [
    { id: 1, pair: 'EURUSD', type: 'BUY', amount: '10,000', price: '1.0854', time: '14:32:15', status: 'filled' },
    { id: 2, pair: 'GBPUSD', type: 'SELL', amount: '5,000', price: '1.2654', time: '14:28:42', status: 'pending' },
    { id: 3, pair: 'USDJPY', type: 'BUY', amount: '15,000', price: '148.23', time: '14:25:18', status: 'filled' },
    { id: 4, pair: 'AUDUSD', type: 'SELL', amount: '8,000', price: '0.6543', time: '14:22:05', status: 'cancelled' },
  ];

  const openPositions = [
    { id: 1, pair: 'EURUSD', type: 'BUY', amount: '10,000', openPrice: '1.0854', currentPrice: '1.0925', pnl: '+$71', pnlPercent: '+0.65%' },
    { id: 2, pair: 'GBPUSD', type: 'SELL', amount: '5,000', openPrice: '1.2654', currentPrice: '1.2630', pnl: '+$12', pnlPercent: '+0.19%' },
    { id: 3, pair: 'USDJPY', type: 'BUY', amount: '15,000', openPrice: '148.23', currentPrice: '148.45', pnl: '+$22', pnlPercent: '+0.15%' },
  ];

  const handlePlaceOrder = () => {
    // Simular colocación de orden
    console.log('Orden colocada:', { orderType, orderSide, orderAmount, orderPrice });
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'filled':
        return <CheckCircle className="w-4 h-4 text-green-400" />;
      case 'pending':
        return <Clock className="w-4 h-4 text-yellow-400" />;
      case 'cancelled':
        return <XCircle className="w-4 h-4 text-red-400" />;
      default:
        return <AlertTriangle className="w-4 h-4 text-gray-400" />;
    }
  };

  return (
    <div className="p-4 sm:p-6 space-y-4 sm:space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between space-y-4 sm:space-y-0">
        <div>
          <h1 className="text-xl sm:text-2xl font-bold text-white">Trading en Vivo</h1>
          <p className="text-sm sm:text-base text-gray-400">Plataforma de trading profesional con IA</p>
        </div>
        <div className="flex flex-col sm:flex-row items-start sm:items-center space-y-2 sm:space-y-0 sm:space-x-3">
          <div className="flex items-center space-x-2 bg-green-500/20 border border-green-500/30 rounded-lg px-3 py-2">
            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
            <span className="text-sm text-green-400 font-medium">Trading Activo</span>
          </div>
          <button className="trading-button px-4 py-2">
            <Settings className="w-4 h-4 mr-2" />
            <span className="hidden sm:inline">Configuración</span>
          </button>
        </div>
      </div>

      {/* Layout principal - Responsive */}
      <div className="grid grid-cols-1 xl:grid-cols-4 gap-4 sm:gap-6">
        {/* Panel de símbolos */}
        <div className="xl:col-span-1 order-1">
          <div className="trading-card p-3 sm:p-4">
            <h3 className="text-base sm:text-lg font-semibold text-white mb-3 sm:mb-4">Símbolos</h3>
            <div className="space-y-2">
              {symbols.map((symbol, index) => (
                <button
                  key={index}
                  onClick={() => setSelectedSymbol(symbol.pair)}
                  className={`w-full flex items-center justify-between p-2 sm:p-3 rounded-lg transition-all ${
                    selectedSymbol === symbol.pair
                      ? 'bg-blue-500/20 border border-blue-500/30'
                      : 'bg-gray-800/50 hover:bg-gray-700/50'
                  }`}
                >
                  <div className="flex items-center space-x-2 sm:space-x-3">
                    <div className="w-6 h-6 sm:w-8 sm:h-8 bg-gradient-to-r from-blue-500 to-teal-500 rounded-lg flex items-center justify-center">
                      <span className="text-xs font-bold text-white">{symbol.pair.split('/')[0]}</span>
                    </div>
                    <div>
                      <p className="text-sm sm:text-base font-semibold text-white">{symbol.label}</p>
                      <p className="text-xs text-gray-400">Vol: {symbol.volume}</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-sm sm:text-base font-semibold text-white">{symbol.price}</p>
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
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Gráfico principal */}
        <div className="xl:col-span-2 order-3 xl:order-2">
          <YahooTradingChart symbol={selectedSymbol} timeframe={selectedTimeframe} />
        </div>

        {/* Panel de órdenes */}
        <div className="xl:col-span-1 order-2 xl:order-3 space-y-4 sm:space-y-6">
          {/* Nueva orden */}
          <div className="trading-card p-3 sm:p-4">
            <h3 className="text-base sm:text-lg font-semibold text-white mb-3 sm:mb-4">Nueva Orden</h3>
            
            {/* Tipo de orden */}
            <div className="mb-3 sm:mb-4">
              <label className="block text-sm text-gray-400 mb-2">Tipo de Orden</label>
              <div className="grid grid-cols-3 gap-2">
                {[
                  { id: 'market', name: 'Mercado', color: 'bg-blue-500' },
                  { id: 'limit', name: 'Límite', color: 'bg-green-500' },
                  { id: 'stop', name: 'Stop', color: 'bg-red-500' },
                ].map((type) => (
                  <button
                    key={type.id}
                    onClick={() => setOrderType(type.id as any)}
                    className={`px-2 sm:px-3 py-2 rounded text-xs font-medium transition-colors ${
                      orderType === type.id
                        ? `${type.color} text-white`
                        : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                    }`}
                  >
                    {type.name}
                  </button>
                ))}
              </div>
            </div>

            {/* Compra/Venta */}
            <div className="mb-3 sm:mb-4">
              <label className="block text-sm text-gray-400 mb-2">Lado</label>
              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={() => setOrderSide('buy')}
                  className={`px-3 sm:px-4 py-2 rounded font-medium transition-colors text-sm ${
                    orderSide === 'buy'
                      ? 'bg-green-500 text-white'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  <Plus className="w-4 h-4 inline mr-1 sm:mr-2" />
                  <span className="hidden sm:inline">Comprar</span>
                  <span className="sm:hidden">Comp</span>
                </button>
                <button
                  onClick={() => setOrderSide('sell')}
                  className={`px-3 sm:px-4 py-2 rounded font-medium transition-colors text-sm ${
                    orderSide === 'sell'
                      ? 'bg-red-500 text-white'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  <Minus className="w-4 h-4 inline mr-1 sm:mr-2" />
                  <span className="hidden sm:inline">Vender</span>
                  <span className="sm:hidden">Vend</span>
                </button>
              </div>
            </div>

            {/* Cantidad */}
            <div className="mb-3 sm:mb-4">
              <label className="block text-sm text-gray-400 mb-2">Cantidad</label>
              <input
                type="text"
                value={orderAmount}
                onChange={(e) => setOrderAmount(e.target.value)}
                className="w-full trading-input px-3 py-2 text-sm"
                placeholder="10000"
              />
            </div>

            {/* Precio */}
            {orderType !== 'market' && (
              <div className="mb-3 sm:mb-4">
                <label className="block text-sm text-gray-400 mb-2">Precio</label>
                <input
                  type="text"
                  value={orderPrice}
                  onChange={(e) => setOrderPrice(e.target.value)}
                  className="w-full trading-input px-3 py-2 text-sm"
                  placeholder="1.0925"
                />
              </div>
            )}

            {/* Botón de orden */}
            <button
              onClick={handlePlaceOrder}
              className={`w-full py-2 sm:py-3 rounded-lg font-semibold transition-all text-sm ${
                orderSide === 'buy'
                  ? 'bg-green-500 hover:bg-green-600 text-white'
                  : 'bg-red-500 hover:bg-red-600 text-white'
              }`}
            >
              {orderSide === 'buy' ? 'Comprar' : 'Vender'} {selectedSymbol}
            </button>
          </div>

          {/* Posiciones abiertas */}
          <div className="trading-card p-3 sm:p-4">
            <h3 className="text-base sm:text-lg font-semibold text-white mb-3 sm:mb-4">Posiciones Abiertas</h3>
            <div className="space-y-2 sm:space-y-3">
              {openPositions.map((position) => (
                <div key={position.id} className="bg-gray-800/50 rounded-lg p-2 sm:p-3">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <div className={`w-2 h-2 rounded-full ${
                        position.type === 'BUY' ? 'bg-green-400' : 'bg-red-400'
                      }`} />
                      <span className="text-sm sm:text-base font-semibold text-white">{position.pair}</span>
                      <span className={`text-xs px-2 py-1 rounded ${
                        position.type === 'BUY' 
                          ? 'bg-green-500/20 text-green-400' 
                          : 'bg-red-500/20 text-red-400'
                      }`}>
                        {position.type}
                      </span>
                    </div>
                    <span className={`text-sm font-semibold ${
                      position.pnl.startsWith('+') ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {position.pnl}
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-xs text-gray-400">
                    <span>{position.amount}</span>
                    <span>{position.currentPrice}</span>
                    <span>{position.pnlPercent}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Órdenes recientes - Full width en móvil */}
      <div className="trading-card p-4 sm:p-6">
        <h3 className="text-lg sm:text-xl font-semibold text-white mb-4 sm:mb-6">Órdenes Recientes</h3>
        <div className="overflow-x-auto">
          <table className="w-full trading-table">
            <thead>
              <tr>
                <th className="text-left p-2 sm:p-4">Par</th>
                <th className="text-left p-2 sm:p-4">Tipo</th>
                <th className="text-left p-2 sm:p-4 hidden sm:table-cell">Cantidad</th>
                <th className="text-left p-2 sm:p-4 hidden md:table-cell">Precio</th>
                <th className="text-left p-2 sm:p-4 hidden lg:table-cell">Hora</th>
                <th className="text-left p-2 sm:p-4">Estado</th>
              </tr>
            </thead>
            <tbody>
              {recentOrders.map((order) => (
                <tr key={order.id}>
                  <td className="font-semibold text-white p-2 sm:p-4">{order.pair}</td>
                  <td className="p-2 sm:p-4">
                    <span className={`text-xs px-2 py-1 rounded ${
                      order.type === 'BUY' 
                        ? 'bg-green-500/20 text-green-400' 
                        : 'bg-red-500/20 text-red-400'
                    }`}>
                      {order.type}
                    </span>
                  </td>
                  <td className="text-gray-300 p-2 sm:p-4 hidden sm:table-cell">{order.amount}</td>
                  <td className="text-gray-300 p-2 sm:p-4 hidden md:table-cell">{order.price}</td>
                  <td className="text-gray-400 p-2 sm:p-4 hidden lg:table-cell">{order.time}</td>
                  <td className="p-2 sm:p-4">
                    <div className="flex items-center space-x-2">
                      {getStatusIcon(order.status)}
                      <span className="text-sm capitalize">{order.status}</span>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}; 