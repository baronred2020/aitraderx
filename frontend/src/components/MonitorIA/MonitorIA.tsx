import React from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { useFeatureAccess } from '../../hooks/useFeatureAccess';
import MonitoringAgents from './MonitoringAgents';

const MonitorIA: React.FC = () => {
  const { subscription } = useAuth();
  const { checkFeature } = useFeatureAccess();

  // Verificar acceso a la funcionalidad de monitoreo
  if (!checkFeature('monitoring_agents')) {
    return (
      <div className="p-6">
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6 text-center">
          <h2 className="text-xl font-semibold text-yellow-800 mb-2">
            Monitor IA no disponible
          </h2>
          <p className="text-yellow-700 mb-4">
            Esta funcionalidad requiere un plan de suscripción superior.
          </p>
          <p className="text-sm text-yellow-600">
            Plan actual: {subscription?.planType || 'N/A'}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="border-b border-gray-200 pb-4">
        <h1 className="text-2xl font-bold text-gray-900">Monitor IA</h1>
        <p className="text-gray-600 mt-2">
          Sistema de monitoreo inteligente con agentes especializados para supervisar el rendimiento de los modelos de IA
        </p>
      </div>

      {/* Agentes de Monitoreo */}
      <MonitoringAgents />
    </div>
  );
};

export default MonitorIA; 