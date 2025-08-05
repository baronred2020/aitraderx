import { useState, useEffect, useCallback } from 'react';

interface ScalpingStrategy {
  id: string;
  type: string;
  name: string;
  description: string;
  pair: string;
  timeframe: string;
  parameters: {
    position_size: number;
    stop_loss_pips: number;
    take_profit_pips: number;
    min_confidence: number;
    min_volatility: number;
    max_spread: number;
    risk_per_trade: number;
  };
  filters: {
    sessions: string[];
    min_volume: number;
    max_daily_trades: number;
  };
  status: 'created' | 'active' | 'stopped';
  created_at: string;
  started_at?: string;
  stopped_at?: string;
  last_signal?: any;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  total_pnl: number;
  current_price: number;
  balance: number;
  equity: number;
  win_rate: number;
  open_positions: number;
  closed_positions: number;
}

interface ScalpingStrategyConfig {
  parameters?: {
    position_size?: number;
    stop_loss_pips?: number;
    take_profit_pips?: number;
    min_confidence?: number;
    min_volatility?: number;
    max_spread?: number;
    risk_per_trade?: number;
  };
  filters?: {
    sessions?: string[];
    min_volume?: number;
    max_daily_trades?: number;
  };
  initial_balance?: number;
}

interface UseScalpingStrategiesReturn {
  strategies: ScalpingStrategy[];
  isLoading: boolean;
  error: string | null;
  createStrategy: (strategyType: string, config: ScalpingStrategyConfig) => Promise<any>;
  startStrategy: (strategyId: string) => Promise<any>;
  stopStrategy: (strategyId: string) => Promise<any>;
  deleteStrategy: (strategyId: string) => Promise<any>;
  getStrategyStatus: (strategyId: string) => Promise<any>;
  generateSignal: (strategyId: string, marketData: any) => Promise<any>;
  executeTrade: (strategyId: string, signal: any) => Promise<any>;
  getAvailableTypes: () => Promise<any>;
  refreshStrategies: () => Promise<void>;
}

export const useScalpingStrategies = (): UseScalpingStrategiesReturn => {
  const [strategies, setStrategies] = useState<ScalpingStrategy[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const API_BASE = 'http://localhost:8000/api/v1/trading/scalping';

  const fetchStrategies = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      const response = await fetch(`${API_BASE}/strategies`);
      const data = await response.json();
      
      if (data.success) {
        setStrategies(data.strategies || []);
      } else {
        setError(data.message || 'Error obteniendo estrategias');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error de conexión');
    } finally {
      setIsLoading(false);
    }
  }, []);

  const createStrategy = useCallback(async (strategyType: string, config: ScalpingStrategyConfig) => {
    try {
      setIsLoading(true);
      setError(null);
      
      const response = await fetch(`${API_BASE}/strategies`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          strategy_type: strategyType,
          config: config
        }),
      });
      
      const data = await response.json();
      
      if (data.success) {
        await fetchStrategies(); // Refrescar lista
        return data;
      } else {
        setError(data.message || 'Error creando estrategia');
        return data;
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Error de conexión';
      setError(errorMsg);
      return { success: false, error: errorMsg };
    } finally {
      setIsLoading(false);
    }
  }, [fetchStrategies]);

  const startStrategy = useCallback(async (strategyId: string) => {
    try {
      setIsLoading(true);
      setError(null);
      
      const response = await fetch(`${API_BASE}/strategies/${strategyId}/start`, {
        method: 'POST',
      });
      
      const data = await response.json();
      
      if (data.success) {
        await fetchStrategies(); // Refrescar lista
        return data;
      } else {
        setError(data.message || 'Error iniciando estrategia');
        return data;
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Error de conexión';
      setError(errorMsg);
      return { success: false, error: errorMsg };
    } finally {
      setIsLoading(false);
    }
  }, [fetchStrategies]);

  const stopStrategy = useCallback(async (strategyId: string) => {
    try {
      setIsLoading(true);
      setError(null);
      
      const response = await fetch(`${API_BASE}/strategies/${strategyId}/stop`, {
        method: 'POST',
      });
      
      const data = await response.json();
      
      if (data.success) {
        await fetchStrategies(); // Refrescar lista
        return data;
      } else {
        setError(data.message || 'Error deteniendo estrategia');
        return data;
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Error de conexión';
      setError(errorMsg);
      return { success: false, error: errorMsg };
    } finally {
      setIsLoading(false);
    }
  }, [fetchStrategies]);

  const deleteStrategy = useCallback(async (strategyId: string) => {
    try {
      setIsLoading(true);
      setError(null);
      
      const response = await fetch(`${API_BASE}/strategies/${strategyId}`, {
        method: 'DELETE',
      });
      
      const data = await response.json();
      
      if (data.success) {
        await fetchStrategies(); // Refrescar lista
        return data;
      } else {
        setError(data.message || 'Error eliminando estrategia');
        return data;
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Error de conexión';
      setError(errorMsg);
      return { success: false, error: errorMsg };
    } finally {
      setIsLoading(false);
    }
  }, [fetchStrategies]);

  const getStrategyStatus = useCallback(async (strategyId: string) => {
    try {
      const response = await fetch(`${API_BASE}/strategies/${strategyId}/status`);
      const data = await response.json();
      
      if (data.success) {
        return data;
      } else {
        return { success: false, error: data.message };
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Error de conexión';
      return { success: false, error: errorMsg };
    }
  }, []);

  const generateSignal = useCallback(async (strategyId: string, marketData: any) => {
    try {
      const response = await fetch(`${API_BASE}/strategies/${strategyId}/signal`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(marketData),
      });
      
      const data = await response.json();
      return data;
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Error de conexión';
      return { success: false, error: errorMsg };
    }
  }, []);

  const executeTrade = useCallback(async (strategyId: string, signal: any) => {
    try {
      const response = await fetch(`${API_BASE}/strategies/${strategyId}/execute`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(signal),
      });
      
      const data = await response.json();
      
      if (data.success) {
        await fetchStrategies(); // Refrescar lista
      }
      
      return data;
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Error de conexión';
      return { success: false, error: errorMsg };
    }
  }, [fetchStrategies]);

  const getAvailableTypes = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/available-types`);
      const data = await response.json();
      return data;
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Error de conexión';
      return { success: false, error: errorMsg };
    }
  }, []);

  const refreshStrategies = useCallback(async () => {
    await fetchStrategies();
  }, [fetchStrategies]);

  // Cargar estrategias al montar el componente
  useEffect(() => {
    fetchStrategies();
  }, [fetchStrategies]);

  return {
    strategies,
    isLoading,
    error,
    createStrategy,
    startStrategy,
    stopStrategy,
    deleteStrategy,
    getStrategyStatus,
    generateSignal,
    executeTrade,
    getAvailableTypes,
    refreshStrategies,
  };
}; 