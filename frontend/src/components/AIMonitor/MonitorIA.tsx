import React, { useEffect, useState } from 'react';
import { Cpu, Zap, CheckCircle, Loader2, AlertTriangle } from 'lucide-react';

interface MonitorIAProps {
  resumen?: boolean; // Si true, muestra solo tarjetas principales (para dashboard)
}

export const MonitorIA: React.FC<MonitorIAProps> = ({ resumen = false }) => {
  const [rlStatus, setRlStatus] = useState<{ DQN?: any; PPO?: any }>({});
  const [loadingRL, setLoadingRL] = useState(true);
  const [errorRL, setErrorRL] = useState<string | null>(null);

  useEffect(() => {
    setLoadingRL(true);
    fetch('/api/rl/status')
      .then(res => res.json())
      .then(data => {
        setRlStatus(data);
        setLoadingRL(false);
      })
      .catch(err => {
        setErrorRL('No se pudo obtener el estado de los agentes RL');
        setLoadingRL(false);
      });
  }, []);

  return (
    <div className={resumen ? '' : 'max-w-3xl mx-auto'}>
      <div className={`trading-card p-6 ${resumen ? '' : 'mb-8'}`}>
        <div className="flex flex-col items-center justify-center mb-6">
          <div className="bg-gradient-to-tr from-blue-500 to-purple-500 rounded-full p-4 mb-2">
            <span role="img" aria-label="robot" className="text-3xl">🤖</span>
          </div>
          <h2 className="text-2xl font-bold text-white mb-1">Monitor de IA</h2>
          <p className="text-gray-400 text-center">Monitoreo de Inteligencia Artificial<br />Seguimiento en tiempo real del rendimiento de los modelos IA</p>
        </div>
        <div className="space-y-4">
          {/* IA Tradicional */}
          <div className="flex items-center justify-between bg-gray-800/60 rounded-lg px-6 py-4">
            <div>
              <div className="font-bold text-white text-lg">IA Tradicional</div>
              <div className="text-gray-400 text-sm">Random Forest + LSTM</div>
            </div>
            <div className="flex flex-col items-end">
              <span className="text-green-400 font-bold text-xl">78.3%</span>
              <span className="text-xs text-gray-400">Precisión</span>
            </div>
          </div>
          {/* RL DQN */}
          <div className="flex items-center justify-between bg-gray-800/60 rounded-lg px-6 py-4">
            <div>
              <div className="font-bold text-white text-lg">Reinforcement Learning</div>
              <div className="text-gray-400 text-sm">DQN Agent</div>
            </div>
            <div className="flex flex-col items-end">
              {loadingRL ? (
                <span className="text-blue-400 flex items-center"><Loader2 className="w-4 h-4 mr-1 animate-spin" />Cargando</span>
              ) : errorRL ? (
                <span className="text-red-400 flex items-center"><AlertTriangle className="w-4 h-4 mr-1" />Error</span>
              ) : rlStatus.DQN && rlStatus.DQN.status === 'initialized' ? (
                <>
                  <span className="text-blue-400 font-bold">{rlStatus.DQN.trained ? 'Entrenado' : 'Entrenando'}</span>
                  <span className="text-xs text-gray-400">{rlStatus.DQN.performance_metrics?.precision ? `${rlStatus.DQN.performance_metrics.precision}% precisión` : 'Sin métrica'}</span>
                </>
              ) : (
                <span className="text-gray-400">No inicializado</span>
              )}
            </div>
          </div>
          {/* RL PPO */}
          <div className="flex items-center justify-between bg-gray-800/60 rounded-lg px-6 py-4">
            <div>
              <div className="font-bold text-white text-lg">Reinforcement Learning</div>
              <div className="text-gray-400 text-sm">PPO Agent</div>
            </div>
            <div className="flex flex-col items-end">
              {loadingRL ? (
                <span className="text-purple-400 flex items-center"><Loader2 className="w-4 h-4 mr-1 animate-spin" />Cargando</span>
              ) : errorRL ? (
                <span className="text-red-400 flex items-center"><AlertTriangle className="w-4 h-4 mr-1" />Error</span>
              ) : rlStatus.PPO && rlStatus.PPO.status === 'initialized' ? (
                <>
                  <span className="text-purple-400 font-bold">{rlStatus.PPO.trained ? 'Entrenado' : 'Entrenando'}</span>
                  <span className="text-xs text-gray-400">{rlStatus.PPO.performance_metrics?.precision ? `${rlStatus.PPO.performance_metrics.precision}% precisión` : 'Sin métrica'}</span>
                </>
              ) : (
                <span className="text-gray-400">No inicializado</span>
              )}
            </div>
          </div>
        </div>
        {!resumen && (
          <div className="mt-8 text-center text-gray-400 text-xs">
            Última actualización: {new Date().toLocaleString()}
          </div>
        )}
      </div>
    </div>
  );
}; 