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
    try {
      const res = await fetch('http://localhost:8000/wallet', { headers: authHeaders });
      if (!res.ok) throw new Error('Error obteniendo wallet');
      const data = await res.json();
      setBalance(data.balance);
      setTransactions(data.transactions || []);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [token]);

  // Recargar saldo
  const recharge = useCallback(async (amount: number) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`http://localhost:8000/wallet/recharge?amount=${amount}`, {
        method: 'POST',
        headers: authHeaders,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Error recargando saldo');
      setBalance(data.balance);
      await fetchWallet();
      return true;
    } catch (e: any) {
      setError(e.message);
      return false;
    } finally {
      setLoading(false);
    }
  }, [token, fetchWallet]);

  // Operar (descontar saldo)
  const trade = useCallback(async (amount: number, description = '') => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`http://localhost:8000/wallet/trade?amount=${amount}&description=${encodeURIComponent(description)}`, {
        method: 'POST',
        headers: authHeaders,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Error operando');
      setBalance(data.balance);
      await fetchWallet();
      return true;
    } catch (e: any) {
      setError(e.message);
      return false;
    } finally {
      setLoading(false);
    }
  }, [token, fetchWallet]);

  // Refrescar movimientos
  const refreshTransactions = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('http://localhost:8000/wallet/transactions', { headers: authHeaders });
      if (!res.ok) throw new Error('Error obteniendo movimientos');
      const data = await res.json();
      setTransactions(data);
    } catch (e: any) {
      setError(e.message);
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