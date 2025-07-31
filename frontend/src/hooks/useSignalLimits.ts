import { useState, useEffect, useCallback } from 'react';
import { apiService, SignalLimits } from '../services/api';
import { useAuth } from '../contexts/AuthContext';

export const useSignalLimits = (brainType: string, style: string = 'day_trading') => {
  const { user, subscription } = useAuth();
  const [limits, setLimits] = useState<SignalLimits | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const checkLimits = useCallback(async () => {
    if (!user?.id) {
      setError('Usuario no autenticado');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const signalLimits = await apiService.getSignalLimits(
        brainType,
        user.id,
        subscription?.planType || 'starter',
        style
      );
      setLimits(signalLimits);
    } catch (err) {
      console.error('Error checking signal limits:', err);
      setError('Error al verificar límites de señales');
    } finally {
      setLoading(false);
    }
  }, [brainType, user?.id, subscription?.planType, style]);

  const generateSignal = useCallback(async (
    pair: string = 'EURUSD'
  ) => {
    if (!user?.id) {
      throw new Error('Usuario no autenticado');
    }

    const response = await apiService.generateSignal(
      brainType,
      pair,
      style,
      user.id,
      subscription?.planType || 'starter'
    );

    // Actualizar límites después de generar señal
    if (response.success) {
      await checkLimits();
    }

    return response;
  }, [brainType, style, user?.id, subscription?.planType, checkLimits]);

  // Cargar límites al montar el componente
  useEffect(() => {
    if (user?.id) {
      checkLimits();
    }
  }, [user?.id, checkLimits]);

  return {
    limits,
    loading,
    error,
    checkLimits,
    generateSignal,
    canGenerate: limits?.can_generate ?? false,
    remainingSignals: limits?.remaining_signals ?? 0,
    maxSignals: limits?.max_signals_per_day ?? 0,
    hasUnlimited: limits?.has_unlimited ?? false
  };
}; 