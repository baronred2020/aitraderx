import { useState, useCallback, useEffect } from 'react';

interface Strategy {
  id: string;
  name: string;
  brainType: string;
  pair: string;
  style: string;
  status: 'active' | 'paused' | 'stopped';
  currentPrice: number;
  totalTrades: number;
  winningTrades: number;
  totalPnL: number;
  openPositions: number;
  lastSignal: string;
  lastSignalTime: string;
  createdAt: string;
  lotSize: number;
  stopLossPips: number;
  takeProfitPips: number;
  minConfidence: number;
  maxPositions: number;
  riskPerTrade: number;
}

interface StrategyConfig {
  name: string;
  brainType: string;
  pair: string;
  style: string;
  lotSize: number;
  stopLossPips: number;
  takeProfitPips: number;
  minConfidence: number;
  maxPositions: number;
  riskPerTrade: number;
}

interface UsePortfolioStrategiesReturn {
  strategies: Strategy[];
  isLoading: boolean;
  error: string | null;
  createStrategy: (config: StrategyConfig) => Promise<any>;
  startStrategy: (id: string) => Promise<any>;
  stopStrategy: (id: string) => Promise<any>;
  deleteStrategy: (id: string) => Promise<any>;
  refreshStrategies: () => Promise<void>;
}

export const usePortfolioStrategies = (): UsePortfolioStrategiesReturn => {
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const API_BASE = 'http://localhost:8000/api/v1/trading';

  const fetchStrategies = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      const response = await fetch(`${API_BASE}/strategies`);
      const data = await response.json();
      
      if (response.ok) {
        setStrategies(data.strategies || []);
      } else {
        setError(data.detail || 'Error obteniendo estrategias');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error de conexión');
    } finally {
      setIsLoading(false);
    }
  }, []);

  const createStrategy = useCallback(async (config: StrategyConfig) => {
    try {
      setIsLoading(true);
      setError(null);
      
      const response = await fetch(`${API_BASE}/strategies`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(config),
      });
      
      const data = await response.json();
      
      if (response.ok) {
        await fetchStrategies(); // Refrescar lista
        return { success: true, strategy: data.strategy };
      } else {
        const errorMsg = data.detail || 'Error creando estrategia';
        setError(errorMsg);
        return { success: false, error: errorMsg };
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
      
      if (response.ok) {
        await fetchStrategies(); // Refrescar lista
        return { success: true };
      } else {
        const errorMsg = data.detail || 'Error iniciando estrategia';
        setError(errorMsg);
        return { success: false, error: errorMsg };
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
      
      if (response.ok) {
        await fetchStrategies(); // Refrescar lista
        return { success: true };
      } else {
        const errorMsg = data.detail || 'Error deteniendo estrategia';
        setError(errorMsg);
        return { success: false, error: errorMsg };
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
      
      if (response.ok) {
        await fetchStrategies(); // Refrescar lista
        return { success: true };
      } else {
        const data = await response.json();
        const errorMsg = data.detail || 'Error eliminando estrategia';
        setError(errorMsg);
        return { success: false, error: errorMsg };
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Error de conexión';
      setError(errorMsg);
      return { success: false, error: errorMsg };
    } finally {
      setIsLoading(false);
    }
  }, [fetchStrategies]);

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
    refreshStrategies,
  };
}; 