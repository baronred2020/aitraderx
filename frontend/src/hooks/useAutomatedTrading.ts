import { useState, useCallback, useEffect } from 'react';

interface Strategy {
  id: string;
  name: string;
  type: string;
  pair: string;
  status: 'active' | 'inactive' | 'paused';
  totalPnL: number;
  totalTrades: number;
  winningTrades: number;
  lastSignal?: any;
  createdAt: string;
  updatedAt: string;
}

interface ConnectionInfo {
  isConnected: boolean;
  lastCheck: string;
  status: 'connected' | 'disconnected' | 'error';
  message?: string;
}

interface UseAutomatedTradingReturn {
  strategies: Strategy[];
  isLoading: boolean;
  error: string | null;
  connectionInfo: ConnectionInfo;
  createStrategy: (config: any) => Promise<any>;
  startStrategy: (id: string) => Promise<any>;
  stopStrategy: (id: string) => Promise<any>;
  deleteStrategy: (id: string) => Promise<any>;
  clearAllStrategies: () => void;
  downloadEA: () => Promise<void>;
  testConnection: () => Promise<any>;
}

export const useAutomatedTrading = (): UseAutomatedTradingReturn => {
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [connectionInfo, setConnectionInfo] = useState<ConnectionInfo>({
    isConnected: false,
    lastCheck: new Date().toISOString(),
    status: 'disconnected'
  });

  // Inicializar con estrategias vacías - solo datos reales
  useEffect(() => {
    // No crear estrategias mock, solo trabajar con datos reales
    setStrategies([]);
  }, []); // Solo se ejecuta una vez al montar el componente

  const createStrategy = useCallback(async (config: any) => {
    setIsLoading(true);
    setError(null);
    
    try {
      // Simular creación de estrategia
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const newStrategy: Strategy = {
        id: Date.now().toString(),
        name: config.name || 'Nueva Estrategia',
        type: config.type || 'scalping',
        pair: config.pair || 'EURUSD',
        status: 'inactive',
        totalPnL: 0,
        totalTrades: 0,
        winningTrades: 0,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      };
      
      setStrategies(prev => [...prev, newStrategy]);
      return { success: true, strategy: newStrategy };
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Error al crear estrategia';
      setError(errorMsg);
      return { success: false, error: errorMsg };
    } finally {
      setIsLoading(false);
    }
  }, []);

  const startStrategy = useCallback(async (id: string) => {
    setIsLoading(true);
    setError(null);
    
    try {
      await new Promise(resolve => setTimeout(resolve, 500));
      
      setStrategies(prev => prev.map(strategy => 
        strategy.id === id 
          ? { ...strategy, status: 'active' as const, updatedAt: new Date().toISOString() }
          : strategy
      ));
      
      return { success: true };
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Error al iniciar estrategia';
      setError(errorMsg);
      return { success: false, error: errorMsg };
    } finally {
      setIsLoading(false);
    }
  }, []);

  const stopStrategy = useCallback(async (id: string) => {
    setIsLoading(true);
    setError(null);
    
    try {
      await new Promise(resolve => setTimeout(resolve, 500));
      
      setStrategies(prev => prev.map(strategy => 
        strategy.id === id 
          ? { ...strategy, status: 'inactive' as const, updatedAt: new Date().toISOString() }
          : strategy
      ));
      
      return { success: true };
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Error al detener estrategia';
      setError(errorMsg);
      return { success: false, error: errorMsg };
    } finally {
      setIsLoading(false);
    }
  }, []);

  const deleteStrategy = useCallback(async (id: string) => {
    setIsLoading(true);
    setError(null);
    
    try {
      await new Promise(resolve => setTimeout(resolve, 500));
      
      setStrategies(prev => prev.filter(strategy => strategy.id !== id));
      
      return { success: true };
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Error al eliminar estrategia';
      setError(errorMsg);
      return { success: false, error: errorMsg };
    } finally {
      setIsLoading(false);
    }
  }, []);

  const clearAllStrategies = useCallback(() => {
    setStrategies([]);
  }, []);



  const downloadEA = useCallback(async () => {
    try {
      const response = await fetch('/api/v1/trading/mt4/download-ea');
      
      if (!response.ok) {
        throw new Error('Error al descargar EA');
      }
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'mt4_expert_advisor.mq4';
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Error al descargar EA';
      setError(errorMsg);
      throw err;
    }
  }, []);

  const testConnection = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Simular test de conexión
      const isConnected = Math.random() > 0.3; // 70% de probabilidad de éxito
      
      setConnectionInfo({
        isConnected,
        lastCheck: new Date().toISOString(),
        status: isConnected ? 'connected' : 'error',
        message: isConnected ? 'Conexión exitosa' : 'Error de conexión'
      });
      
      return { success: isConnected, message: isConnected ? 'Conexión exitosa' : 'Error de conexión' };
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Error al probar conexión';
      setError(errorMsg);
      setConnectionInfo({
        isConnected: false,
        lastCheck: new Date().toISOString(),
        status: 'error',
        message: errorMsg
      });
      return { success: false, error: errorMsg };
    } finally {
      setIsLoading(false);
    }
  }, []);

  return {
    strategies,
    isLoading,
    error,
    connectionInfo,
    createStrategy,
    startStrategy,
    stopStrategy,
    deleteStrategy,
    clearAllStrategies,
    downloadEA,
    testConnection
  };
}; 