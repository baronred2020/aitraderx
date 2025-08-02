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
    canRecharge,
    validateRechargeAmount,
    RECHARGE_THRESHOLD,
    MIN_RECHARGE,
    MAX_RECHARGE,
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

    // Validar monto de recarga
    const validation = validateRechargeAmount(value);
    if (!validation.valid) {
      setLocalError(validation.error || 'Error de validación desconocido');
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

  const handleAmountChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setAmount(value);
    
    // Limpiar errores al cambiar el valor
    if (localError) {
      setLocalError('');
    }
  };

  const getBalanceStatusColor = () => {
    if (balance === null) return 'text-gray-400';
    if (balance < RECHARGE_THRESHOLD) return 'text-red-400';
    if (balance < 1000) return 'text-yellow-400';
    return 'text-green-400';
  };

  const getBalanceStatusText = () => {
    if (balance === null) return '';
    if (balance < RECHARGE_THRESHOLD) return ' (Bajo - Puedes recargar)';
    if (balance < 1000) return ' (Medio)';
    return ' (Alto)';
  };

  return (
    <div className="trading-card p-3 sm:p-4 mb-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-base font-semibold text-white">💰 Balance virtual:</span>
        <div className="text-right">
          <span className={`text-lg font-bold ${getBalanceStatusColor()}`}>
            {loading || authLoading ? '...' : 
             balance === null ? 'No disponible' : 
             `$${balance?.toLocaleString()}`}
          </span>
          <div className="text-xs text-gray-400">
            {getBalanceStatusText()}
          </div>
        </div>
      </div>
      
      {/* Botón de recarga con validación */}
      <button
        onClick={() => { setShowModal(true); setSuccess(''); setLocalError(''); }}
        className={`w-full py-2 mt-2 rounded-lg font-semibold transition-all text-sm ${
          canRecharge() && !loading && !authLoading && balance !== null
            ? 'bg-blue-500 hover:bg-blue-600 text-white'
            : 'bg-gray-600 text-gray-400 cursor-not-allowed'
        }`}
        disabled={!canRecharge() || loading || authLoading || balance === null}
      >
        {canRecharge() ? 'Añadir saldo' : `Recarga disponible cuando balance menor a $${RECHARGE_THRESHOLD}`}
      </button>



      {error && (
        <div className="text-xs text-red-400 mt-2 p-2 bg-red-900/20 rounded border border-red-500/30">
          ⚠️ {error}
        </div>
      )}
      {success && (
        <div className="text-xs text-green-400 mt-2 p-2 bg-green-900/20 rounded border border-green-500/30">
          ✅ {success}
        </div>
      )}

      {/* Modal de recarga */}
      {showModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-40">
          <div className="bg-gray-900 rounded-lg p-6 w-full max-w-sm border border-gray-700">
            <h3 className="text-lg font-semibold text-white mb-4">Añadir saldo virtual</h3>
            
            {/* Información de reglas */}
            <div className="text-xs text-gray-400 mb-4 p-3 bg-gray-800/50 rounded border border-gray-700">
              <div className="font-semibold text-white mb-2">📋 Reglas de recarga:</div>
              <div>• Solo disponible cuando balance menor a ${RECHARGE_THRESHOLD}</div>
              <div>• Monto mínimo: ${MIN_RECHARGE}</div>
              <div>• Monto máximo: ${MAX_RECHARGE.toLocaleString()}</div>
            </div>

            <div className="mb-4">
              <label className="block text-sm text-gray-400 mb-2">Monto a añadir (USD)</label>
              <input
                type="number"
                value={amount}
                onChange={handleAmountChange}
                className="w-full trading-input px-3 py-2 text-sm"
                placeholder={`$${MIN_RECHARGE} - $${MAX_RECHARGE.toLocaleString()}`}
                min={MIN_RECHARGE}
                max={MAX_RECHARGE}
                step="0.01"
              />
            </div>

            {localError && (
              <div className="text-xs text-red-400 mb-3 p-2 bg-red-900/20 rounded border border-red-500/30">
                ⚠️ {localError}
              </div>
            )}

            <div className="flex space-x-2">
              <button
                onClick={handleAddFunds}
                className="flex-1 py-2 rounded-lg bg-green-500 hover:bg-green-600 text-white font-semibold text-sm"
                disabled={loading || !amount || parseFloat(amount) <= 0}
              >
                {loading ? 'Procesando...' : 'Confirmar'}
              </button>
              <button
                onClick={() => { 
                  setShowModal(false); 
                  setLocalError(''); 
                  setAmount('');
                }}
                className="flex-1 py-2 rounded-lg bg-gray-700 hover:bg-gray-600 text-gray-200 font-semibold text-sm"
              >
                Cancelar
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Movimientos recientes */}
      <div className="mt-4">
        <h4 className="text-sm font-semibold text-white mb-2">Movimientos recientes</h4>
        {loading ? (
          <div className="text-xs text-gray-400">Cargando...</div>
        ) : (
          <div className="max-h-40 overflow-y-auto text-xs">
            {transactions.length === 0 && <div className="text-gray-400">Sin movimientos</div>}
            {transactions.slice(0, 5).map(tx => (
              <div key={tx.id} className="flex justify-between items-center border-b border-gray-700 py-1">
                <div className="flex-1">
                  <div className="font-semibold text-white capitalize">{tx.type}</div>
                  <div className="text-gray-400 text-xs truncate">{tx.description}</div>
                </div>
                <div className="text-right ml-2">
                  <div className={`font-semibold ${tx.amount > 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {tx.amount > 0 ? '+' : ''}${tx.amount.toLocaleString()}
                  </div>
                  <div className="text-gray-500 text-xs">
                    {new Date(tx.created_at).toLocaleDateString()}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default Wallet; 