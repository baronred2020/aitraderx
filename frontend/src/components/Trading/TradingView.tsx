import React, { useState, useEffect } from 'react';
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
import { useYahooMarketData } from '../../hooks/useYahooMarketData';
import Wallet from './Wallet';
import { useWallet } from '../../hooks/useWallet';
import { useAuth } from '../../contexts/AuthContext';

export const TradingView: React.FC = () => {
  const [selectedSymbol, setSelectedSymbol] = useState('EURUSD');
  const [selectedTimeframe, setSelectedTimeframe] = useState('1H');
  const [orderType, setOrderType] = useState<'market' | 'limit' | 'stop'>('market');
  const [orderSide, setOrderSide] = useState<'buy' | 'sell'>('buy');
  const [orderAmount, setOrderAmount] = useState('10000');
  const [orderPrice, setOrderPrice] = useState('1.0925');

  // Estado para SL y TP
  const [orderSL, setOrderSL] = useState('');
  const [orderTP, setOrderTP] = useState('');
  const [orderError, setOrderError] = useState('');

  const { isLoading: authLoading } = useAuth();
  const token = localStorage.getItem('auth_token') || '';
  const {
    balance,
    loading: walletLoading,
    error: walletError,
    trade,
    fetchWallet,
    refreshTransactions,
  } = useWallet(token);

  // Log de depuración
  console.log('TOKEN en TradingView:', token);
  console.log('BALANCE en TradingView:', balance);

  // Llamar a fetchWallet al montar el componente
  useEffect(() => {
    if (token) fetchWallet();
  }, [token, fetchWallet]);

  // Definir símbolos base sin precios hardcodeados
  const baseSymbols = [
    { pair: 'EURUSD', label: 'EUR/USD' },
    { pair: 'GBPUSD', label: 'GBP/USD' },
    { pair: 'USDJPY', label: 'USD/JPY' },
    { pair: 'AUDUSD', label: 'AUD/USD' },
    { pair: 'USDCAD', label: 'USD/CAD' },
  ];

  // Obtener datos de mercado reales
  const symbolList = baseSymbols.map(s => s.pair);
  const { data: marketData, loading: marketLoading, error: marketError, marketStatus } = useYahooMarketData(symbolList);

  // Combinar datos base con datos de mercado reales
  const symbols = baseSymbols.map(baseSymbol => {
    const marketInfo = marketData[baseSymbol.pair];
    const changePercent = marketInfo?.changePercent || '0.00';
    const isPositive = !changePercent.startsWith('-');
    
    return {
      ...baseSymbol,
      price: marketInfo?.price || 'Loading...',
      change: changePercent ? `${isPositive ? '+' : ''}${changePercent}%` : '0.00%',
      volume: marketInfo?.volume || '-',
      trend: isPositive ? 'up' : 'down'
    };
  });

  // Actualizar precio de orden cuando cambie el símbolo seleccionado
  useEffect(() => {
    const selectedMarketData = marketData[selectedSymbol];
    if (selectedMarketData?.price && selectedMarketData.price !== 'Loading...') {
      setOrderPrice(selectedMarketData.price);
    }
  }, [selectedSymbol, marketData]);

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

  // Validaciones y resumen
  const isBuy = orderSide === 'buy';
  const priceNum = parseFloat(orderPrice);
  const slNum = parseFloat(orderSL);
  const tpNum = parseFloat(orderTP);
  const amountNum = parseFloat(orderAmount);
  const validSL = !orderSL || (isBuy ? slNum < priceNum : slNum > priceNum);
  const validTP = !orderTP || (isBuy ? tpNum > priceNum : tpNum < priceNum);
  const validAmount = amountNum > 0 && !isNaN(amountNum);
  const validPrice = priceNum > 0 && !isNaN(priceNum);
  const canPlaceOrder = validSL && validTP && validAmount && validPrice;

  // Calcular costo estimado de la orden (para forex, usar la cantidad directamente)
  const estimatedCost = validAmount && validPrice ? amountNum : 0;
  const hasFunds = (balance ?? 0) >= estimatedCost;
  const canPlaceOrderWithFunds = canPlaceOrder && hasFunds;

  const orderSummary = `${orderSide === 'buy' ? 'Comprar' : 'Vender'} ${orderAmount} ${selectedSymbol} a ${orderPrice}` +
    (orderSL ? ` | SL: ${orderSL}` : '') + (orderTP ? ` | TP: ${orderTP}` : '');

  // Feedback visual
  const [orderSuccess, setOrderSuccess] = useState('');

  const handleAddFunds = (amount: number) => {
    // This function is now handled by useWallet, but keeping it for now
    // as it might be re-introduced or refactored later.
    console.log('Adding funds:', amount);
  };

  const handlePlaceOrder = async () => {
    if (!canPlaceOrderWithFunds) {
      setOrderError(!hasFunds ? 'Saldo insuficiente para operar.' : 'Verifica los datos de la orden (SL/TP, cantidad, precio).');
      return;
    }
    setOrderError('');
    setOrderSuccess('');
    const ok = await trade(estimatedCost, `Orden ${orderSide} ${orderAmount} ${selectedSymbol} a ${orderPrice}`);
    if (ok) {
      setOrderSuccess('¡Orden colocada exitosamente!');
      fetchWallet();
      refreshTransactions();
    } else {
      setOrderError('Error al colocar la orden.');
    }
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
          {/* Estado del mercado */}
          <div className={`flex items-center space-x-2 border rounded-lg px-3 py-2 ${
            marketStatus === 'open' 
              ? 'bg-green-500/20 border-green-500/30' 
              : 'bg-red-500/20 border-red-500/30'
          }`}>
            <div className={`w-2 h-2 rounded-full animate-pulse ${
              marketStatus === 'open' ? 'bg-green-400' : 'bg-red-400'
            }`}></div>
            <span className={`text-sm font-medium ${
              marketStatus === 'open' ? 'text-green-400' : 'text-red-400'
            }`}>
              {marketStatus === 'open' ? 'Mercado Abierto' : 'Mercado Cerrado'}
            </span>
          </div>
          
          <div className="flex items-center space-x-2 bg-green-500/20 border border-green-500/30 rounded-lg px-3 py-2">
            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
            <span className="text-sm text-green-400 font-medium">Trading Activo</span>
          </div>
          {marketLoading && (
            <div className="flex items-center space-x-2 bg-blue-500/20 border border-blue-500/30 rounded-lg px-3 py-2">
              <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
              <span className="text-sm text-blue-400 font-medium">Actualizando Precios</span>
            </div>
          )}
          {marketError && (
            <div className="flex items-center space-x-2 bg-red-500/20 border border-red-500/30 rounded-lg px-3 py-2">
              <AlertTriangle className="w-3 h-3 text-red-400" />
              <span className="text-sm text-red-400 font-medium">Error datos de mercado</span>
            </div>
          )}
          <button className="trading-button px-4 py-2">
            <Settings className="w-4 h-4 mr-2" />
            <span className="hidden sm:inline">Configuración</span>
          </button>
        </div>
      </div>

      {/* Layout principal - Responsive */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4 sm:gap-6">
        {/* Gráfico principal - Ocupa 2/3 partes */}
        <div className="xl:col-span-2 order-1">
          <YahooTradingChart symbol={selectedSymbol} timeframe={selectedTimeframe} />
        </div>

        {/* Panel lateral - Ocupa 1/3 parte */}
        <div className="xl:col-span-1 order-2 space-y-4 sm:space-y-6">
          {/* Wallet */}
          <Wallet />
          {/* Selector de símbolos desplegable */}
          <div className="trading-card p-3 sm:p-4">
            <h3 className="text-base sm:text-lg font-semibold text-white mb-3 sm:mb-4">Símbolo</h3>
            <div className="relative">
              <select
                value={selectedSymbol}
                onChange={(e) => setSelectedSymbol(e.target.value)}
                className="w-full trading-input px-3 py-2 text-sm appearance-none bg-gray-800 border border-gray-700 rounded-lg text-white focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
              >
                {symbols.map((symbol) => (
                  <option key={symbol.pair} value={symbol.pair} className="bg-gray-800 text-white">
                    {symbol.label} - {symbol.price} ({symbol.change})
                  </option>
                ))}
              </select>
              <div className="absolute inset-y-0 right-0 flex items-center px-2 pointer-events-none">
                <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </div>
            </div>
          </div>
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

            {/* Stop Loss (SL) */}
            <div className="mb-3 sm:mb-4">
              <label className="block text-sm text-gray-400 mb-2">Stop Loss (SL)</label>
              <input
                type="text"
                value={orderSL}
                onChange={(e) => setOrderSL(e.target.value)}
                className={`w-full trading-input px-3 py-2 text-sm ${orderSL && !validSL ? 'border-red-500' : ''}`}
                placeholder={isBuy ? 'Menor que el precio' : 'Mayor que el precio'}
              />
              {orderSL && !validSL && (
                <div className="text-xs text-red-400 mt-1">El SL debe ser {isBuy ? 'menor' : 'mayor'} que el precio.</div>
              )}
            </div>

            {/* Take Profit (TP) */}
            <div className="mb-3 sm:mb-4">
              <label className="block text-sm text-gray-400 mb-2">Take Profit (TP)</label>
              <input
                type="text"
                value={orderTP}
                onChange={(e) => setOrderTP(e.target.value)}
                className={`w-full trading-input px-3 py-2 text-sm ${orderTP && !validTP ? 'border-red-500' : ''}`}
                placeholder={isBuy ? 'Mayor que el precio' : 'Menor que el precio'}
              />
              {orderTP && !validTP && (
                <div className="text-xs text-red-400 mt-1">El TP debe ser {isBuy ? 'mayor' : 'menor'} que el precio.</div>
              )}
            </div>

            {/* Resumen de la orden */}
            <div className="mb-3 sm:mb-4 bg-gray-800/60 rounded p-2 text-xs text-gray-300">
              <div className="font-semibold text-white mb-1">Resumen:</div>
              <div>{orderSummary}</div>
              <div className="mt-1 text-blue-300">
                Costo estimado: ${estimatedCost.toLocaleString()}
              </div>
              <div className="mt-1 text-gray-400">
                Saldo disponible: {walletLoading || balance === null ? 'Cargando...' : `$${(balance ?? 0).toLocaleString()}`}
              </div>
            </div>

            {/* Botón de orden */}
            <button
              onClick={handlePlaceOrder}
              className={`w-full py-2 sm:py-3 rounded-lg font-semibold transition-all text-sm ${
                orderSide === 'buy'
                  ? 'bg-green-500 hover:bg-green-600 text-white'
                  : 'bg-red-500 hover:bg-red-600 text-white'
              } ${!canPlaceOrderWithFunds || walletLoading || authLoading || balance === null ? 'opacity-60 cursor-not-allowed' : ''}`}
              disabled={!canPlaceOrderWithFunds || walletLoading || authLoading || balance === null}
            >
              {orderSide === 'buy' ? 'Comprar' : 'Vender'} {selectedSymbol}
            </button>
            {orderError && (
              <div className="text-xs text-red-400 mt-2">{orderError}</div>
            )}
            {orderSuccess && (
              <div className="text-xs text-green-400 mt-2">{orderSuccess}</div>
            )}
            {walletError && (
              <div className="text-xs text-red-400 mt-2">{walletError}</div>
            )}
            {!hasFunds && (
              <div className="text-xs text-yellow-400 mt-2">Saldo insuficiente para esta operación.</div>
            )}
            
            {/* Información del estado del mercado */}
            {marketStatus === 'closed' && (
              <div className="mt-3 p-3 bg-yellow-500/10 border border-yellow-500/20 rounded-lg">
                <div className="flex items-center space-x-2 mb-2">
                  <Clock className="w-4 h-4 text-yellow-400" />
                  <span className="text-sm font-medium text-yellow-400">Mercado Cerrado</span>
                </div>
                <p className="text-xs text-yellow-300">
                  El mercado forex está cerrado durante los fines de semana. 
                  Los datos mostrados son del último cierre de mercado.
                </p>
                <p className="text-xs text-yellow-300 mt-1">
                  El trading se reanudará el lunes a las 9:00 AM.
                </p>
              </div>
            )}
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