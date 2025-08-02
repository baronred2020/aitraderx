import { useState, useEffect, useCallback } from 'react';

export interface SimulatedOrder {
  id: string;
  symbol: string;
  type: 'market' | 'limit' | 'stop';
  side: 'buy' | 'sell';
  amount: number;
  price: number;
  stopLoss?: number;
  takeProfit?: number;
  status: 'pending' | 'filled' | 'cancelled' | 'rejected';
  createdAt: Date;
  filledAt?: Date;
  filledPrice?: number;
  description?: string;
}

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

export const useSimulatedOrders = () => {
  const [orders, setOrders] = useState<SimulatedOrder[]>([]);

  // Cargar órdenes desde localStorage al inicializar
  useEffect(() => {
    const savedOrders = localStorage.getItem('simulated_orders');
    if (savedOrders) {
      try {
        const parsed = JSON.parse(savedOrders);
        // Convertir fechas de string a Date
        const ordersWithDates = parsed.map((order: any) => ({
          ...order,
          createdAt: new Date(order.createdAt),
          filledAt: order.filledAt ? new Date(order.filledAt) : undefined
        }));
        setOrders(ordersWithDates);
      } catch (error) {
        console.error('Error loading orders:', error);
      }
    }
  }, []);

  // Guardar órdenes en localStorage cuando cambien
  useEffect(() => {
    localStorage.setItem('simulated_orders', JSON.stringify(orders));
  }, [orders]);

  // Crear nueva orden
  const createOrder = useCallback((orderData: Omit<SimulatedOrder, 'id' | 'status' | 'createdAt'>) => {
    const newOrder: SimulatedOrder = {
      ...orderData,
      id: Date.now().toString(),
      status: 'pending',
      createdAt: new Date()
    };
    
    setOrders(prev => [newOrder, ...prev]);
    return newOrder;
  }, []);

  // Ejecutar órdenes basado en precios de mercado
  const executeOrders = useCallback((marketData: MarketData) => {
    setOrders(prev => prev.map(order => {
      if (order.status !== 'pending') return order;
      
      const currentPrice = parseFloat(marketData[order.symbol]?.price || '0');
      if (currentPrice === 0) return order;
      
      let shouldExecute = false;
      let executionPrice = currentPrice;
      
      // Lógica de ejecución por tipo de orden
      switch (order.type) {
        case 'market':
          shouldExecute = true;
          executionPrice = currentPrice;
          break;
          
        case 'limit':
          if (order.side === 'buy') {
            // Orden de compra limit: ejecutar cuando el precio baje al nivel especificado
            shouldExecute = currentPrice <= order.price;
            executionPrice = order.price;
          } else {
            // Orden de venta limit: ejecutar cuando el precio suba al nivel especificado
            shouldExecute = currentPrice >= order.price;
            executionPrice = order.price;
          }
          break;
          
        case 'stop':
          if (order.side === 'buy') {
            // Stop de compra: ejecutar cuando el precio suba al nivel especificado
            shouldExecute = currentPrice >= order.price;
            executionPrice = currentPrice;
          } else {
            // Stop de venta: ejecutar cuando el precio baje al nivel especificado
            shouldExecute = currentPrice <= order.price;
            executionPrice = currentPrice;
          }
          break;
      }
      
      if (shouldExecute) {
        return {
          ...order,
          status: 'filled',
          filledAt: new Date(),
          filledPrice: executionPrice
        };
      }
      
      return order;
    }));
  }, []);

  // Cancelar orden
  const cancelOrder = useCallback((orderId: string) => {
    setOrders(prev => prev.map(order => 
      order.id === orderId ? { ...order, status: 'cancelled' } : order
    ));
  }, []);

  // Obtener órdenes por estado
  const pendingOrders = orders.filter(order => order.status === 'pending');
  const filledOrders = orders.filter(order => order.status === 'filled');
  const cancelledOrders = orders.filter(order => order.status === 'cancelled');

  // Obtener órdenes recientes (últimas 10)
  const recentOrders = orders.slice(0, 10);

  // Calcular estadísticas
  const totalOrders = orders.length;
  const filledOrdersCount = filledOrders.length;
  const pendingOrdersCount = pendingOrders.length;
  const cancelledOrdersCount = cancelledOrders.length;
  const fillRate = totalOrders > 0 ? (filledOrdersCount / totalOrders) * 100 : 0;

  return {
    orders,
    pendingOrders,
    filledOrders,
    cancelledOrders,
    recentOrders,
    createOrder,
    executeOrders,
    cancelOrder,
    totalOrders,
    filledOrdersCount,
    pendingOrdersCount,
    cancelledOrdersCount,
    fillRate
  };
}; 