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

  // Constantes para las reglas del balance virtual
  const INITIAL_BALANCE = 10000.0; // $10,000 USD inicial
  const RECHARGE_THRESHOLD = 100.0; // Solo recargar si balance < $100
  const MIN_RECHARGE = 1.0; // Mínimo $1 USD
  const MAX_RECHARGE = 10000.0; // Máximo $10,000 USD

  // Helper para headers
  const authHeaders = {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json',
  };

  // Verificar si se puede recargar
  const canRecharge = (currentBalance: number | null) => {
    return (currentBalance ?? 0) < RECHARGE_THRESHOLD;
  };

  // Validar monto de recarga
  const validateRechargeAmount = (amount: number) => {
    if (amount < MIN_RECHARGE) {
      return { valid: false, error: `El monto mínimo de recarga es $${MIN_RECHARGE}` };
    }
    if (amount > MAX_RECHARGE) {
      return { valid: false, error: `El monto máximo de recarga es $${MAX_RECHARGE}` };
    }
    return { valid: true, error: null };
  };

  // Obtener balance y movimientos
  const fetchWallet = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    // Verificar si hay token
    if (!token) {
      setError('No hay token de autenticación válido');
      setLoading(false);
      return;
    }
    
    // Para modo desarrollo, usar datos simulados
    if (token === 'dev-token') {
      // Verificar si ya existe un balance en localStorage
      const savedBalance = localStorage.getItem('virtual_balance');
      const savedTransactions = localStorage.getItem('virtual_transactions');
      
      if (savedBalance && savedTransactions) {
        // Usar balance existente
        setBalance(parseFloat(savedBalance));
        setTransactions(JSON.parse(savedTransactions));
      } else {
        // Primer acceso: establecer balance inicial
        setBalance(INITIAL_BALANCE);
        const initialTransaction = {
          id: 1,
          type: 'deposit',
          amount: INITIAL_BALANCE,
          description: 'Depósito inicial - Usuario registrado',
          created_at: new Date().toISOString()
        };
        setTransactions([initialTransaction]);
        
        // Guardar en localStorage
        localStorage.setItem('virtual_balance', INITIAL_BALANCE.toString());
        localStorage.setItem('virtual_transactions', JSON.stringify([initialTransaction]));
      }
      setError(null);
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
    if (!token) {
      setError('No hay token de autenticación válido');
      setLoading(false);
      return false;
    }
    
    // Validar monto de recarga
    const validation = validateRechargeAmount(amount);
    if (!validation.valid) {
      setError(validation.error);
      setLoading(false);
      return false;
    }
    
    // Verificar si se puede recargar
    if (!canRecharge(balance)) {
      setError(`Solo puedes recargar cuando tu balance esté por debajo de $${RECHARGE_THRESHOLD}`);
      setLoading(false);
      return false;
    }
    
    // Para modo desarrollo, simular recarga
    if (token === 'dev-token') {
      const newBalance = (balance || 0) + amount;
      const newTransaction = {
        id: Date.now(),
        type: 'deposit',
        amount: amount,
        description: `Recarga de saldo - $${amount.toLocaleString()}`,
        created_at: new Date().toISOString()
      };
      
      setBalance(newBalance);
      setTransactions(prev => [newTransaction, ...prev]);
      
      // Guardar en localStorage
      localStorage.setItem('virtual_balance', newBalance.toString());
      localStorage.setItem('virtual_transactions', JSON.stringify([newTransaction, ...transactions]));
      
      setError(null);
      setLoading(false);
      return true;
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
  }, [token, balance, transactions, fetchWallet]);

  // Operar (descontar saldo)
  const trade = useCallback(async (amount: number, description: string = '') => {
    setLoading(true);
    setError(null);
    
    // Verificar si hay token
    if (!token) {
      setError('No hay token de autenticación válido');
      setLoading(false);
      return false;
    }
    
    // Para modo desarrollo, simular operación
    if (token === 'dev-token') {
      const newBalance = (balance || 0) - amount;
      if (newBalance < 0) {
        setError('Saldo insuficiente');
        setLoading(false);
        return false;
      }
      
      const newTransaction = {
        id: Date.now(),
        type: 'trade',
        amount: -amount,
        description: description || 'Operación de trading',
        created_at: new Date().toISOString()
      };
      
      setBalance(newBalance);
      setTransactions(prev => [newTransaction, ...prev]);
      
      // Guardar en localStorage
      localStorage.setItem('virtual_balance', newBalance.toString());
      localStorage.setItem('virtual_transactions', JSON.stringify([newTransaction, ...transactions]));
      
      setError(null);
      setLoading(false);
      return true;
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
  }, [token, balance, transactions, fetchWallet]);

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
    // Nuevas funciones para las reglas
    canRecharge: () => canRecharge(balance),
    validateRechargeAmount,
    RECHARGE_THRESHOLD,
    MIN_RECHARGE,
    MAX_RECHARGE,
  };
} 