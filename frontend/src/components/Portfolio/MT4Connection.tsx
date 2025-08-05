import React, { useState, useEffect } from 'react';

interface MT4ConnectionProps {}

const MT4Connection: React.FC<MT4ConnectionProps> = () => {
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'connecting'>('disconnected');
  const [connectionInfo, setConnectionInfo] = useState({
    server: '',
    account: '',
    balance: 0,
    equity: 0,
    margin: 0,
    freeMargin: 0
  });

  useEffect(() => {
    // Simular verificación de conexión
    const checkConnection = async () => {
      setConnectionStatus('connecting');
      
      try {
        // Aquí se haría la llamada real al backend para verificar conexión MT4
        const response = await fetch('/api/v1/trading/mt4/status');
        const data = await response.json();
        
        if (data.connected) {
          setConnectionStatus('connected');
          setConnectionInfo({
            server: data.server || '',
            account: data.account || '',
            balance: data.balance || 0,
            equity: data.equity || 0,
            margin: data.margin || 0,
            freeMargin: data.freeMargin || 0
          });
        } else {
          setConnectionStatus('disconnected');
        }
      } catch (error) {
        setConnectionStatus('disconnected');
      }
    };

    checkConnection();
    
    // Verificar conexión cada 30 segundos
    const interval = setInterval(checkConnection, 30000);
    
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = () => {
    switch (connectionStatus) {
      case 'connected': return 'text-green-400';
      case 'connecting': return 'text-yellow-400';
      case 'disconnected': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const getStatusIcon = () => {
    switch (connectionStatus) {
      case 'connected': return '🟢';
      case 'connecting': return '🟡';
      case 'disconnected': return '🔴';
      default: return '⚪';
    }
  };

  const getStatusText = () => {
    switch (connectionStatus) {
      case 'connected': return 'Conectado';
      case 'connecting': return 'Conectando...';
      case 'disconnected': return 'Desconectado';
      default: return 'Desconocido';
    }
  };

  const handleConnect = async () => {
    setConnectionStatus('connecting');
    
    try {
      // Aquí se haría la llamada real para conectar con MT4
      const response = await fetch('/api/v1/trading/mt4/connect', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          server: 'Demo',
          login: '12345',
          password: 'password'
        })
      });
      
      const data = await response.json();
      
      if (data.status === 'connected') {
        setConnectionStatus('connected');
        setConnectionInfo({
          server: data.server || '',
          account: data.account || '',
          balance: data.balance || 0,
          equity: data.equity || 0,
          margin: data.margin || 0,
          freeMargin: data.freeMargin || 0
        });
      } else {
        setConnectionStatus('disconnected');
      }
    } catch (error) {
      setConnectionStatus('disconnected');
    }
  };

  const handleDisconnect = async () => {
    try {
      await fetch('/api/v1/trading/mt4/disconnect', { method: 'POST' });
      setConnectionStatus('disconnected');
      setConnectionInfo({
        server: '',
        account: '',
        balance: 0,
        equity: 0,
        margin: 0,
        freeMargin: 0
      });
    } catch (error) {
      console.error('Error disconnecting:', error);
    }
  };

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className="text-2xl">📊</div>
          <div>
            <h3 className="text-lg font-semibold text-white">Conexión MT4/MT5</h3>
            <p className="text-sm text-gray-400">Estado de conexión con MetaTrader</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-3">
          <div className={`flex items-center space-x-2 ${getStatusColor()}`}>
            <span className="text-lg">{getStatusIcon()}</span>
            <span className="font-medium">{getStatusText()}</span>
          </div>
          
          {connectionStatus === 'disconnected' ? (
            <button
              onClick={handleConnect}
              className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors"
            >
              🔗 Conectar
            </button>
          ) : (
            <button
              onClick={handleDisconnect}
              className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors"
            >
              🔌 Desconectar
            </button>
          )}
        </div>
      </div>

      {connectionStatus === 'connected' && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-gray-700 rounded-lg p-3">
            <div className="text-sm text-gray-400 mb-1">Servidor</div>
            <div className="text-white font-medium">{connectionInfo.server}</div>
          </div>
          
          <div className="bg-gray-700 rounded-lg p-3">
            <div className="text-sm text-gray-400 mb-1">Cuenta</div>
            <div className="text-white font-medium">{connectionInfo.account}</div>
          </div>
          
          <div className="bg-gray-700 rounded-lg p-3">
            <div className="text-sm text-gray-400 mb-1">Balance</div>
            <div className="text-white font-medium">${(connectionInfo.balance || 0).toFixed(2)}</div>
          </div>
          
          <div className="bg-gray-700 rounded-lg p-3">
            <div className="text-sm text-gray-400 mb-1">Equity</div>
            <div className="text-white font-medium">${(connectionInfo.equity || 0).toFixed(2)}</div>
          </div>
        </div>
      )}

      {connectionStatus === 'disconnected' && (
        <div className="bg-gray-700 rounded-lg p-4">
          <div className="text-center">
            <div className="text-4xl mb-2">🔌</div>
            <h4 className="text-white font-medium mb-2">MetaTrader Desconectado</h4>
            <p className="text-gray-400 text-sm mb-4">
              Conecta tu cuenta de MetaTrader para habilitar el trading automático
            </p>
            <div className="text-xs text-gray-500 space-y-1">
              <div>• Asegúrate de que MetaTrader esté abierto</div>
              <div>• Verifica que el Expert Advisor esté habilitado</div>
              <div>• Comprueba que el trading automático esté permitido</div>
            </div>
          </div>
        </div>
      )}

      {connectionStatus === 'connecting' && (
        <div className="bg-gray-700 rounded-lg p-4">
          <div className="text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-400 mx-auto mb-2"></div>
            <h4 className="text-white font-medium">Conectando con MetaTrader...</h4>
            <p className="text-gray-400 text-sm">Por favor espera</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default MT4Connection; 