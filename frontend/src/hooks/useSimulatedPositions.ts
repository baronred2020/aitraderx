import { useState, useEffect, useCallback } from 'react';

export interface SimulatedPosition {
  id: string;
  symbol: string;
  type: 'BUY' | 'SELL';
  amount: number;
  openPrice: number;
  currentPrice: number;
  openTime: Date;
  stopLoss?: number;
  takeProfit?: number;
  pnl: number;
  pnlPercent: number;
  status: 'open' | 'closed';
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

export const useSimulatedPositions = () => {
  const [positions, setPositions] = useState<SimulatedPosition[]>([]);

  // Cargar posiciones desde localStorage al inicializar
  useEffect(() => {
    const savedPositions = localStorage.getItem('simulated_positions');
    if (savedPositions) {
      try {
        const parsed = JSON.parse(savedPositions);
        // Convertir fechas de string a Date
        const positionsWithDates = parsed.map((pos: any) => ({
          ...pos,
          openTime: new Date(pos.openTime)
        }));
        setPositions(positionsWithDates);
      } catch (error) {
        console.error('Error loading positions:', error);
      }
    }
  }, []);

  // Guardar posiciones en localStorage cuando cambien
  useEffect(() => {
    localStorage.setItem('simulated_positions', JSON.stringify(positions));
  }, [positions]);

  // Calcular P&L dinámico basado en precios de mercado
  const updatePositionsPnL = useCallback((marketData: MarketData) => {
    setPositions(prev => prev.map(pos => {
      const currentPrice = parseFloat(marketData[pos.symbol]?.price || pos.currentPrice.toString());
      
      // Calcular P&L
      let pnl: number;
      if (pos.type === 'BUY') {
        pnl = (currentPrice - pos.openPrice) * pos.amount;
      } else {
        pnl = (pos.openPrice - currentPrice) * pos.amount;
      }
      
      const pnlPercent = (pnl / (pos.openPrice * pos.amount)) * 100;
      
      // Verificar stop loss y take profit
      let status = pos.status;
      if (status === 'open') {
        if (pos.stopLoss) {
          if (pos.type === 'BUY' && currentPrice <= pos.stopLoss) {
            status = 'closed';
          } else if (pos.type === 'SELL' && currentPrice >= pos.stopLoss) {
            status = 'closed';
          }
        }
        
        if (pos.takeProfit) {
          if (pos.type === 'BUY' && currentPrice >= pos.takeProfit) {
            status = 'closed';
          } else if (pos.type === 'SELL' && currentPrice <= pos.takeProfit) {
            status = 'closed';
          }
        }
      }
      
      return { 
        ...pos, 
        currentPrice, 
        pnl, 
        pnlPercent,
        status
      };
    }));
  }, []);

  // Abrir nueva posición
  const openPosition = useCallback((position: Omit<SimulatedPosition, 'id' | 'pnl' | 'pnlPercent' | 'status'>) => {
    const newPosition: SimulatedPosition = {
      ...position,
      id: Date.now().toString(),
      pnl: 0,
      pnlPercent: 0,
      status: 'open'
    };
    
    setPositions(prev => [...prev, newPosition]);
    return newPosition;
  }, []);

  // Cerrar posición
  const closePosition = useCallback((positionId: string) => {
    setPositions(prev => prev.map(pos => 
      pos.id === positionId ? { ...pos, status: 'closed' } : pos
    ));
  }, []);

  // Obtener posiciones abiertas
  const openPositions = positions.filter(pos => pos.status === 'open');
  
  // Obtener posiciones cerradas
  const closedPositions = positions.filter(pos => pos.status === 'closed');

  // Calcular P&L total
  const totalPnL = positions.reduce((sum, pos) => sum + pos.pnl, 0);
  const totalPnLPercent = positions.length > 0 
    ? positions.reduce((sum, pos) => sum + pos.pnlPercent, 0) / positions.length 
    : 0;

  return {
    positions,
    openPositions,
    closedPositions,
    updatePositionsPnL,
    openPosition,
    closePosition,
    totalPnL,
    totalPnLPercent
  };
}; 