import { useEffect, useCallback } from 'react';
import { useWallet } from './useWallet';
import { useYahooMarketData } from './useYahooMarketData';
import { useSimulatedPositions, SimulatedPosition } from './useSimulatedPositions';
import { useSimulatedOrders, SimulatedOrder } from './useSimulatedOrders';

interface MarketData {
  [symbol: string]: {
    price: string;
    change: string;
    changePercent: string;
    volume: string;
    high: string;
    low: string;
    open: string;
    previousClose: string;
  };
}

export const useSimulatedTrading = (token: string) => {
  // Hooks básicos
  const { 
    balance, 
    trade, 
    fetchWallet,
    canRecharge,
    validateRechargeAmount,
    RECHARGE_THRESHOLD,
    MIN_RECHARGE,
    MAX_RECHARGE
  } = useWallet(token);
  
  // Símbolos disponibles
  const symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD'];
  const { data: marketData, loading: marketLoading, error: marketError } = useYahooMarketData(symbols);
  
  // Hooks de trading simulado
  const {
    positions,
    openPositions,
    closedPositions,
    updatePositionsPnL,
    openPosition,
    closePosition,
    totalPnL,
    totalPnLPercent
  } = useSimulatedPositions();
  
  const {
    orders,
    pendingOrders,
    filledOrders,
    recentOrders,
    createOrder,
    executeOrders,
    cancelOrder,
    totalOrders,
    filledOrdersCount,
    fillRate
  } = useSimulatedOrders();

  // Actualizar P&L y ejecutar órdenes cuando cambien los precios
  useEffect(() => {
    if (Object.keys(marketData).length > 0) {
      updatePositionsPnL(marketData);
      executeOrders(marketData);
    }
  }, [marketData, updatePositionsPnL, executeOrders]);

  // Colocar orden y abrir posición
  const placeOrder = useCallback(async (
    symbol: string,
    type: 'market' | 'limit' | 'stop',
    side: 'buy' | 'sell',
    amount: number,
    price: number,
    stopLoss?: number,
    takeProfit?: number,
    description?: string
  ) => {
    try {
      // 1. Crear la orden
      const order = createOrder({
        symbol,
        type,
        side,
        amount,
        price,
        stopLoss,
        takeProfit,
        description
      });

      // 2. Si es orden de mercado, ejecutar inmediatamente
      if (type === 'market') {
        const currentPrice = parseFloat(marketData[symbol]?.price || price.toString());
        
        // Actualizar la orden como ejecutada
        executeOrders({
          [symbol]: { price: currentPrice.toString(), change: '0', changePercent: '0', volume: '0', high: '0', low: '0', open: '0', previousClose: '0' }
        });

        // 3. Descontar del balance virtual
        const success = await trade(amount, `Orden ${side.toUpperCase()} ${amount} ${symbol} a ${currentPrice}`);
        
        if (success) {
          // 4. Abrir posición
          openPosition({
            symbol,
            type: side === 'buy' ? 'BUY' : 'SELL',
            amount,
            openPrice: currentPrice,
            currentPrice: currentPrice,
            openTime: new Date(),
            stopLoss,
            takeProfit
          });

          return { success: true, order, message: 'Orden ejecutada exitosamente' };
        } else {
          return { success: false, message: 'Error al procesar el pago' };
        }
      } else {
        // Para órdenes limit/stop, solo crear la orden
        return { success: true, order, message: 'Orden creada exitosamente' };
      }
    } catch (error) {
      console.error('Error placing order:', error);
      return { success: false, message: 'Error al colocar la orden' };
    }
  }, [createOrder, executeOrders, trade, openPosition, marketData]);

  // Cerrar posición manualmente
  const closePositionManually = useCallback(async (positionId: string) => {
    const position = positions.find(pos => pos.id === positionId);
    if (!position) return { success: false, message: 'Posición no encontrada' };

    try {
      // Calcular P&L final
      const finalPnL = position.pnl;
      
      // Cerrar la posición
      closePosition(positionId);
      
      // Agregar P&L al balance (simulado)
      if (finalPnL > 0) {
        await trade(-finalPnL, `Cierre posición ${position.symbol} - Ganancia`);
      } else {
        await trade(Math.abs(finalPnL), `Cierre posición ${position.symbol} - Pérdida`);
      }

      return { success: true, message: 'Posición cerrada exitosamente' };
    } catch (error) {
      console.error('Error closing position:', error);
      return { success: false, message: 'Error al cerrar la posición' };
    }
  }, [positions, closePosition, trade]);

  // Calcular estadísticas de rendimiento
  const performanceStats = {
    totalPositions: positions.length,
    openPositions: openPositions.length,
    closedPositions: closedPositions.length,
    totalPnL,
    totalPnLPercent,
    totalOrders,
    filledOrdersCount,
    fillRate,
    winRate: closedPositions.length > 0 
      ? (closedPositions.filter(pos => pos.pnl > 0).length / closedPositions.length) * 100 
      : 0
  };

  return {
    // Datos de mercado
    marketData,
    marketLoading,
    marketError,
    
    // Wallet
    balance,
    fetchWallet,
    canRecharge,
    validateRechargeAmount,
    RECHARGE_THRESHOLD,
    MIN_RECHARGE,
    MAX_RECHARGE,
    
    // Posiciones
    positions,
    openPositions,
    closedPositions,
    totalPnL,
    totalPnLPercent,
    
    // Órdenes
    orders,
    pendingOrders,
    filledOrders,
    recentOrders,
    totalOrders,
    filledOrdersCount,
    fillRate,
    
    // Acciones
    placeOrder,
    closePosition: closePositionManually,
    cancelOrder,
    
    // Estadísticas
    performanceStats
  };
}; 