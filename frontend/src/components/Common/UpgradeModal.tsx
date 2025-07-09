import React from 'react';
import { X, Crown, Zap, Brain, Shield } from 'lucide-react';

interface UpgradeModalProps {
  isOpen: boolean;
  onClose: () => void;
  currentPlan: string;
  requiredPlan: string;
  feature: string;
}

const planInfo = {
  freemium: {
    name: 'Freemium',
    price: '$0',
    color: 'text-gray-400',
    icon: null
  },
  basic: {
    name: 'Básico',
    price: '$29/mes',
    color: 'text-blue-400',
    icon: Zap
  },
  pro: {
    name: 'Pro',
    price: '$99/mes',
    color: 'text-purple-400',
    icon: Brain
  },
  elite: {
    name: 'Elite',
    price: '$299/mes',
    color: 'text-yellow-400',
    icon: Crown
  }
};

const featureDescriptions = {
  'ai-monitor': 'Monitor de Inteligencia Artificial',
  'rl': 'Reinforcement Learning',
  'reports': 'Reportes Avanzados',
  'alerts': 'Sistema de Alertas',
  'mt4_integration': 'Integración MT4',
  'api_access': 'Acceso a API',
  'custom_models': 'Modelos Personalizados'
};

export const UpgradeModal: React.FC<UpgradeModalProps> = ({
  isOpen,
  onClose,
  currentPlan,
  requiredPlan,
  feature
}) => {
  if (!isOpen) return null;

  const currentPlanInfo = planInfo[currentPlan as keyof typeof planInfo];
  const requiredPlanInfo = planInfo[requiredPlan as keyof typeof planInfo];
  const featureName = featureDescriptions[feature as keyof typeof featureDescriptions] || feature;

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center p-4 z-50">
      <div className="glass-effect rounded-2xl p-8 max-w-md w-full border border-gray-700/50">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-r from-yellow-500 to-orange-500 rounded-lg flex items-center justify-center">
              <Crown className="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white">Upgrade Requerido</h2>
              <p className="text-sm text-gray-400">Desbloquea nuevas funcionalidades</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-gray-700/50 transition-colors"
          >
            <X className="w-5 h-5 text-gray-400" />
          </button>
        </div>

        {/* Contenido */}
        <div className="space-y-6">
          {/* Mensaje principal */}
          <div className="text-center">
            <p className="text-gray-300 mb-2">
              Para acceder a <span className="font-semibold text-blue-400">{featureName}</span>
            </p>
            <p className="text-sm text-gray-400">
              Necesitas actualizar tu plan de <span className={currentPlanInfo.color}>{currentPlanInfo.name}</span> a <span className={requiredPlanInfo.color}>{requiredPlanInfo.name}</span>
            </p>
          </div>

          {/* Comparación de planes */}
          <div className="bg-gray-800/50 rounded-lg p-4 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-gray-400">Plan Actual:</span>
              <span className={`font-semibold ${currentPlanInfo.color}`}>
                {currentPlanInfo.name} ({currentPlanInfo.price})
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-400">Plan Requerido:</span>
              <span className={`font-semibold ${requiredPlanInfo.color}`}>
                {requiredPlanInfo.name} ({requiredPlanInfo.price})
              </span>
            </div>
          </div>

          {/* Beneficios del upgrade */}
          <div>
            <h3 className="text-sm font-semibold text-white mb-3">Beneficios del Upgrade:</h3>
            <div className="space-y-2">
              {requiredPlan === 'basic' && (
                <>
                  <div className="flex items-center space-x-2 text-sm text-gray-300">
                    <Zap className="w-4 h-4 text-blue-400" />
                    <span>5 pares de trading</span>
                  </div>
                  <div className="flex items-center space-x-2 text-sm text-gray-300">
                    <Brain className="w-4 h-4 text-purple-400" />
                    <span>Análisis avanzado</span>
                  </div>
                  <div className="flex items-center space-x-2 text-sm text-gray-300">
                    <Shield className="w-4 h-4 text-green-400" />
                    <span>Sistema de alertas</span>
                  </div>
                </>
              )}
              {requiredPlan === 'pro' && (
                <>
                  <div className="flex items-center space-x-2 text-sm text-gray-300">
                    <Brain className="w-4 h-4 text-purple-400" />
                    <span>Monitor de IA</span>
                  </div>
                  <div className="flex items-center space-x-2 text-sm text-gray-300">
                    <Zap className="w-4 h-4 text-orange-400" />
                    <span>Reinforcement Learning</span>
                  </div>
                  <div className="flex items-center space-x-2 text-sm text-gray-300">
                    <Crown className="w-4 h-4 text-yellow-400" />
                    <span>Integración MT4</span>
                  </div>
                </>
              )}
              {requiredPlan === 'elite' && (
                <>
                  <div className="flex items-center space-x-2 text-sm text-gray-300">
                    <Crown className="w-4 h-4 text-yellow-400" />
                    <span>Acceso completo a API</span>
                  </div>
                  <div className="flex items-center space-x-2 text-sm text-gray-300">
                    <Brain className="w-4 h-4 text-purple-400" />
                    <span>Modelos personalizados</span>
                  </div>
                  <div className="flex items-center space-x-2 text-sm text-gray-300">
                    <Shield className="w-4 h-4 text-green-400" />
                    <span>Soporte prioritario 24/7</span>
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Botones */}
          <div className="flex space-x-3">
            <button
              onClick={onClose}
              className="flex-1 px-4 py-2 border border-gray-600 text-gray-300 rounded-lg hover:bg-gray-700/50 transition-colors"
            >
              Cancelar
            </button>
            <button
              onClick={() => {
                // Aquí iría la lógica para redirigir al upgrade
                window.open('/upgrade', '_blank');
                onClose();
              }}
              className="flex-1 px-4 py-2 bg-gradient-to-r from-blue-600 to-teal-600 hover:from-blue-700 hover:to-teal-700 text-white font-semibold rounded-lg transition-all"
            >
              Upgrade Ahora
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}; 