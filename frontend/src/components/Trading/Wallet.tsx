import React, { useState, useEffect } from 'react';
import { useWallet } from '../../hooks/useWallet';
import { useAuth } from '../../contexts/AuthContext';

const Wallet: React.FC = () => {
  const { isLoading: authLoading } = useAuth();
  const token = localStorage.getItem('auth_token') || '';
  const {
    balance,
    transactions,
    loading,
    error,
    fetchWallet,
    recharge,
    refreshTransactions,
  } = useWallet(token);

  const [showModal, setShowModal] = useState(false);
  const [amount, setAmount] = useState('');
  const [localError, setLocalError] = useState('');
  const [success, setSuccess] = useState('');

  useEffect(() => {
    if (token && !authLoading) fetchWallet();
    // eslint-disable-next-line
  }, [token, authLoading]);

  const handleAddFunds = async () => {
    const value = parseFloat(amount);
    if (isNaN(value) || value <= 0) {
      setLocalError('Ingresa un monto válido.');
      return;
    }
    setLocalError('');
    const ok = await recharge(value);
    if (ok) {
      setSuccess('Recarga exitosa');
      setShowModal(false);
      setAmount('');
      refreshTransactions();
    }
  };

  return (
    <div className="trading-card p-3 sm:p-4 mb-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-base font-semibold text-white">💰 Balance virtual:</span>
        <span className="text-lg font-bold text-green-400">{loading || authLoading ? '...' : `$${balance?.toLocaleString()}`}</span>
      </div>
      <button
        onClick={() => { setShowModal(true); setSuccess(''); }}
        className="w-full py-2 mt-2 rounded-lg bg-blue-500 hover:bg-blue-600 text-white font-semibold transition-all text-sm"
        disabled={loading || authLoading}
      >
        Añadir saldo
      </button>
      {error && <div className="text-xs text-red-400 mt-2">{error}</div>}
      {success && <div className="text-xs text-green-400 mt-2">{success}</div>}
      {showModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-40">
          <div className="bg-gray-900 rounded-lg p-6 w-full max-w-xs border border-gray-700">
            <h3 className="text-lg font-semibold text-white mb-4">Añadir saldo virtual</h3>
            <input
              type="number"
              value={amount}
              onChange={e => setAmount(e.target.value)}
              className="w-full trading-input px-3 py-2 text-sm mb-2"
              placeholder="Monto a añadir"
              min="1"
            />
            {localError && <div className="text-xs text-red-400 mb-2">{localError}</div>}
            <div className="flex space-x-2 mt-2">
              <button
                onClick={handleAddFunds}
                className="flex-1 py-2 rounded-lg bg-green-500 hover:bg-green-600 text-white font-semibold text-sm"
                disabled={loading}
              >
                Confirmar
              </button>
              <button
                onClick={() => { setShowModal(false); setLocalError(''); }}
                className="flex-1 py-2 rounded-lg bg-gray-700 hover:bg-gray-600 text-gray-200 font-semibold text-sm"
              >
                Cancelar
              </button>
            </div>
          </div>
        </div>
      )}
      <div className="mt-4">
        <h4 className="text-sm font-semibold text-white mb-2">Movimientos recientes</h4>
        {loading ? (
          <div className="text-xs text-gray-400">Cargando...</div>
        ) : (
          <div className="max-h-40 overflow-y-auto text-xs">
            {transactions.length === 0 && <div className="text-gray-400">Sin movimientos</div>}
            {transactions.map(tx => (
              <div key={tx.id} className="flex justify-between border-b border-gray-700 py-1">
                <span className="font-semibold text-white">{tx.type}</span>
                <span className={tx.amount > 0 ? 'text-green-400' : 'text-red-400'}>
                  {tx.amount > 0 ? '+' : ''}{tx.amount}
                </span>
                <span className="text-gray-400">{tx.description}</span>
                <span className="text-gray-500">{new Date(tx.created_at).toLocaleString()}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default Wallet; 