import { useState, useCallback } from 'react';

export interface WalletTransaction {
  id: number;
  type: string;
  amount: number;
  description: string;
  created_at: string;
}

export function useWallet(token: string) {
  const [balance, setBalance] = useState<number | null>(null);
  const [transactions, setTransactions] = useState<WalletTransaction[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Helper para headers
  const authHeaders = {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json',
  };

  // Obtener balance y movimientos
  const fetchWallet = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    // Verificar si hay token
    if (!token || token === 'dev-token') {
      setError('No hay token de autenticación válido');
      setLoading(false);
      return;
    }
    
    try {
      const res = await fetch('http://localhost:8000/wallet', { headers: authHeaders });
      
      if (res.status === 401 || res.status === 403) {
        setError('Sesión expirada. Por favor, inicia sesión nuevamente.');
        setBalance(null);
        setTransactions([]);
        return;
      }
      
      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Error obteniendo wallet');
      }
      
      const data = await res.json();
      setBalance(data.balance);
      setTransactions(data.transactions || []);
      setError(null);
    } catch (e: any) {
      console.error('Error fetching wallet:', e);
      setError(e.message || 'Error obteniendo wallet');
      setBalance(null);
      setTransactions([]);
    } finally {
      setLoading(false);
    }
  }, [token]);

  // Recargar saldo
  const recharge = useCallback(async (amount: number) => {
    setLoading(true);
    setError(null);
    
    // Verificar si hay token
    if (!token || token === 'dev-token') {
      setError('No hay token de autenticación válido');
      setLoading(false);
      return false;
    }
    
    try {
      const res = await fetch(`http://localhost:8000/wallet/recharge?amount=${amount}`, {
        method: 'POST',
        headers: authHeaders,
      });
      
      if (res.status === 401 || res.status === 403) {
        setError('Sesión expirada. Por favor, inicia sesión nuevamente.');
        return false;
      }
      
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Error recargando saldo');
      }
      
      setBalance(data.balance);
      await fetchWallet();
      setError(null);
      return true;
    } catch (e: any) {
      console.error('Error recharging wallet:', e);
      setError(e.message || 'Error recargando saldo');
      return false;
    } finally {
      setLoading(false);
    }
  }, [token, fetchWallet]);

  // Operar (descontar saldo)
  const trade = useCallback(async (amount: number, description = '') => {
    setLoading(true);
    setError(null);
    
    // Verificar si hay token
    if (!token || token === 'dev-token') {
      setError('No hay token de autenticación válido');
      setLoading(false);
      return false;
    }
    
    try {
      const res = await fetch(`http://localhost:8000/wallet/trade?amount=${amount}&description=${encodeURIComponent(description)}`, {
        method: 'POST',
        headers: authHeaders,
      });
      
      if (res.status === 401 || res.status === 403) {
        setError('Sesión expirada. Por favor, inicia sesión nuevamente.');
        return false;
      }
      
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Error operando');
      }
      
      setBalance(data.balance);
      await fetchWallet();
      setError(null);
      return true;
    } catch (e: any) {
      console.error('Error trading wallet:', e);
      setError(e.message || 'Error operando');
      return false;
    } finally {
      setLoading(false);
    }
  }, [token, fetchWallet]);

  // Refrescar movimientos
  const refreshTransactions = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    // Verificar si hay token
    if (!token || token === 'dev-token') {
      setError('No hay token de autenticación válido');
      setLoading(false);
      return;
    }
    
    try {
      const res = await fetch('http://localhost:8000/wallet/transactions', { headers: authHeaders });
      
      if (res.status === 401 || res.status === 403) {
        setError('Sesión expirada. Por favor, inicia sesión nuevamente.');
        setTransactions([]);
        return;
      }
      
      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Error obteniendo movimientos');
      }
      
      const data = await res.json();
      setTransactions(data);
      setError(null);
    } catch (e: any) {
      console.error('Error refreshing transactions:', e);
      setError(e.message || 'Error obteniendo movimientos');
      setTransactions([]);
    } finally {
      setLoading(false);
    }
  }, [token]);

  return {
    balance,
    transactions,
    loading,
    error,
    fetchWallet,
    recharge,
    trade,
    refreshTransactions,
  };
} 