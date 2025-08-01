import { useState, useEffect } from 'react';
import { apiService, RealMetrics } from '../services/api';

export const useRealMetrics = (
  brainType?: string,
  pair?: string,
  style?: string
) => {
  const [metrics, setMetrics] = useState<RealMetrics | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchMetrics = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await apiService.getRealMetrics(brainType, pair, style);
      setMetrics(response);
    } catch (err) {
      console.error('Error fetching real metrics:', err);
      setError('Error al obtener métricas reales');
    } finally {
      setLoading(false);
    }
  };

  const completeExpiredPredictions = async () => {
    try {
      setLoading(true);
      const response = await apiService.completeExpiredPredictionsWithRealResults();
      // Recargar métricas después de completar predicciones
      await fetchMetrics();
      return response;
    } catch (err) {
      console.error('Error completing expired predictions:', err);
      setError('Error al completar predicciones expiradas');
      throw err;
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
  }, [brainType, pair, style]);

  return {
    metrics,
    loading,
    error,
    refetch: fetchMetrics,
    completeExpiredPredictions
  };
}; 