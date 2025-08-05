import { useState, useEffect, useCallback } from 'react';
import { apiService } from '../services/api';

// Tipos de datos
export interface PortfolioStats {
  total_predictions: number;
  successful_predictions: number;
  success_rate: number;
  total_signals: number;
  successful_signals: number;
  signal_success_rate: number;
  total_pnl: number;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate: number;
  avg_win: number;
  avg_loss: number;
  max_drawdown: number;
  sharpe_ratio: number;
  profit_factor: number;
  best_pair?: string;
  best_brain_type?: string;
  worst_pair?: string;
  worst_brain_type?: string;
  best_day?: string;
  worst_day?: string;
  daily_pnl: number;
  weekly_pnl: number;
  monthly_pnl: number;
}

export interface TradingHistoryItem {
  id: number;
  pair: string;
  brain_type: string;
  type: string; // 'prediction' or 'signal'
  direction: string;
  entry_price: number;
  exit_price?: number;
  pnl: number;
  pips?: number;
  confidence: number;
  status: string; // 'open', 'closed', 'cancelled'
  entry_time: string;
  exit_time?: string;
  success?: boolean;
  success_percentage?: number;
}

export interface PortfolioPerformance {
  total_return: number;
  daily_return: number;
  weekly_return: number;
  monthly_return: number;
  risk_metrics: {
    sharpe_ratio: number;
    max_drawdown: number;
    win_rate: number;
    profit_factor: number;
    total_trades: number;
    winning_trades: number;
    losing_trades: number;
    average_win: number;
    average_loss: number;
    largest_win: number;
    largest_loss: number;
    volatility: number;
    beta: number;
    var_95: number;
  };
  performance_by_pair: Record<string, {
    total_trades: number;
    winning_trades: number;
    win_rate: number;
    avg_success: number;
    total_pnl: number;
  }>;
  performance_by_brain: Record<string, {
    total_trades: number;
    winning_trades: number;
    win_rate: number;
    avg_success: number;
    total_pnl: number;
  }>;
  recent_trades: TradingHistoryItem[];
}

export interface RiskMetrics {
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
  profit_factor: number;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  average_win: number;
  average_loss: number;
  largest_win: number;
  largest_loss: number;
  volatility: number;
  beta: number;
  var_95: number;
}

interface UsePortfolioReturn {
  // Estados
  stats: PortfolioStats | null;
  history: TradingHistoryItem[];
  performance: PortfolioPerformance | null;
  riskMetrics: RiskMetrics | null;
  isLoading: boolean;
  error: string | null;
  
  // Funciones
  fetchStats: (period?: string) => Promise<void>;
  fetchHistory: (params?: {
    limit?: number;
    period?: string;
    pair?: string;
    brain_type?: string;
  }) => Promise<void>;
  fetchPerformance: (period?: string) => Promise<void>;
  fetchRiskMetrics: (period?: string) => Promise<void>;
  refreshAll: () => Promise<void>;
}

export const usePortfolio = (): UsePortfolioReturn => {
  const [stats, setStats] = useState<PortfolioStats | null>(null);
  const [history, setHistory] = useState<TradingHistoryItem[]>([]);
  const [performance, setPerformance] = useState<PortfolioPerformance | null>(null);
  const [riskMetrics, setRiskMetrics] = useState<RiskMetrics | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Función para manejar errores
  const handleError = (err: any, operation: string) => {
    const errorMessage = err.response?.data?.detail || err.message || `Error en ${operation}`;
    setError(errorMessage);
    console.error(`Error en ${operation}:`, err);
  };

  // Obtener estadísticas del portfolio
  const fetchStats = useCallback(async (period: string = '1m') => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await apiService.getPortfolioStats(period);
      setStats(response);
    } catch (err) {
      handleError(err, 'fetching portfolio stats');
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Obtener historial de trading
  const fetchHistory = useCallback(async (params: {
    limit?: number;
    period?: string;
    pair?: string;
    brain_type?: string;
  } = {}) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await apiService.getPortfolioHistory(params);
      setHistory(response);
    } catch (err) {
      handleError(err, 'fetching trading history');
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Obtener rendimiento del portfolio
  const fetchPerformance = useCallback(async (period: string = '1m') => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await apiService.getPortfolioPerformance(period);
      setPerformance(response);
    } catch (err) {
      handleError(err, 'fetching portfolio performance');
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Obtener métricas de riesgo
  const fetchRiskMetrics = useCallback(async (period: string = '1m') => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await apiService.getPortfolioRiskMetrics(period);
      setRiskMetrics(response);
    } catch (err) {
      handleError(err, 'fetching risk metrics');
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Refrescar todos los datos
  const refreshAll = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      await Promise.all([
        fetchStats(),
        fetchHistory(),
        fetchPerformance(),
        fetchRiskMetrics()
      ]);
    } catch (err) {
      handleError(err, 'refrescar datos del portfolio');
    } finally {
      setIsLoading(false);
    }
  }, [fetchStats, fetchHistory, fetchPerformance, fetchRiskMetrics]);

  // Cargar datos iniciales
  useEffect(() => {
    refreshAll();
  }, [refreshAll]);

  return {
    // Estados
    stats,
    history,
    performance,
    riskMetrics,
    isLoading,
    error,
    
    // Funciones
    fetchStats,
    fetchHistory,
    fetchPerformance,
    fetchRiskMetrics,
    refreshAll
  };
}; 