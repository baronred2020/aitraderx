import React from 'react';
import { 
  X, 
  CheckCircle, 
  XCircle, 
  Brain, 
  BarChart3, 
  Settings, 
  Users, 
  Crown,
  Zap,
  Shield,
  Clock,
  Star
} from 'lucide-react';

interface Plan {
  id: string;
  name: string;
  plan_type: 'starter' | 'trader' | 'expert' | 'premium' | 'institutional';
  price: number;
  currency: string;
  description: string;
  benefits: string[];
  ai_capabilities: {
    traditional_ai: boolean;
    reinforcement_learning: boolean;
    ensemble_ai: boolean;
    lstm_predictions: boolean;
    custom_models: boolean;
    auto_training: boolean;
  };
  api_limits: {
    daily_requests: number;
    prediction_days: number;
    backtest_days: number;
    trading_pairs: number;
    alerts_limit: number;
    portfolio_size: number;
  };
  ui_features: {
    advanced_charts: boolean;
    multiple_timeframes: boolean;
    rl_dashboard: boolean;
    ai_monitor: boolean;
    mt4_integration: boolean;
    api_access: boolean;
    custom_reports: boolean;
    priority_support: boolean;
  };
  max_indicators: number;
  max_predictions_per_day: number;
  max_backtests_per_month: number;
  max_portfolios: number;
  support_level: string;
  response_time_hours: number;
}

interface PlanDetailsModalProps {
  plan: Plan | null;
  isOpen: boolean;
  onClose: () => void;
  onUpgrade: (plan: Plan) => void;
}

const PlanDetailsModal: React.FC<PlanDetailsModalProps> = ({ 
  plan, 
  isOpen, 
  onClose, 
  onUpgrade 
}) => {
  if (!isOpen || !plan) return null;

  const getPlanIcon = (planType: string) => {
    switch (planType) {
      case 'starter':
        return <Shield className="w-8 h-8 text-green-400" />;
      case 'trader':
        return <Zap className="w-8 h-8 text-blue-400" />;
      case 'expert':
        return <Brain className="w-8 h-8 text-purple-400" />;
      case 'premium':
        return <Crown className="w-8 h-8 text-yellow-400" />;
      case 'institutional':
        return <Crown className="w-8 h-8 text-purple-600" />;
      default:
        return <Shield className="w-8 h-8 text-gray-400" />;
    }
  };

  const getSupportLevelColor = (level: string) => {
    switch (level) {
      case 'dedicated':
        return 'text-purple-400';
      case 'phone':
        return 'text-yellow-400';
      case 'email':
        return 'text-blue-400';
      case 'community':
        return 'text-green-400';
      default:
        return 'text-gray-400';
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-gray-800 rounded-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto border border-gray-700">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-700">
          <div className="flex items-center space-x-4">
            {getPlanIcon(plan.plan_type)}
            <div>
              <h2 className="text-2xl font-bold text-white">{plan.name}</h2>
              <p className="text-gray-400">{plan.description}</p>
            </div>
          </div>
          <div className="text-right">
            <div className="text-3xl font-bold text-white">
              ${plan.price}
              <span className="text-sm text-gray-400">/mes</span>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-gray-700 transition-colors"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        <div className="p-6 space-y-8">
          {/* Beneficios */}
          <div>
            <h3 className="text-xl font-semibold mb-4 flex items-center">
              <Star className="w-5 h-5 mr-2 text-yellow-400" />
              Beneficios Incluidos
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {plan.benefits.map((benefit, index) => (
                <div key={index} className="flex items-start space-x-3">
                  <CheckCircle className="w-5 h-5 text-green-400 flex-shrink-0 mt-0.5" />
                  <span className="text-gray-300">{benefit}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Capacidades de IA */}
          <div>
            <h3 className="text-xl font-semibold mb-4 flex items-center">
              <Brain className="w-5 h-5 mr-2 text-purple-400" />
              Capacidades de Inteligencia Artificial
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <div className="flex items-center space-x-3">
                {plan.ai_capabilities.traditional_ai ? (
                  <CheckCircle className="w-5 h-5 text-green-400" />
                ) : (
                  <XCircle className="w-5 h-5 text-red-400" />
                )}
                <span className="text-gray-300">AI Tradicional</span>
              </div>
              <div className="flex items-center space-x-3">
                {plan.ai_capabilities.reinforcement_learning ? (
                  <CheckCircle className="w-5 h-5 text-green-400" />
                ) : (
                  <XCircle className="w-5 h-5 text-red-400" />
                )}
                <span className="text-gray-300">Reinforcement Learning</span>
              </div>
              <div className="flex items-center space-x-3">
                {plan.ai_capabilities.ensemble_ai ? (
                  <CheckCircle className="w-5 h-5 text-green-400" />
                ) : (
                  <XCircle className="w-5 h-5 text-red-400" />
                )}
                <span className="text-gray-300">Ensemble AI</span>
              </div>
              <div className="flex items-center space-x-3">
                {plan.ai_capabilities.lstm_predictions ? (
                  <CheckCircle className="w-5 h-5 text-green-400" />
                ) : (
                  <XCircle className="w-5 h-5 text-red-400" />
                )}
                <span className="text-gray-300">Predicciones LSTM</span>
              </div>
              <div className="flex items-center space-x-3">
                {plan.ai_capabilities.custom_models ? (
                  <CheckCircle className="w-5 h-5 text-green-400" />
                ) : (
                  <XCircle className="w-5 h-5 text-red-400" />
                )}
                <span className="text-gray-300">Modelos Personalizados</span>
              </div>
              <div className="flex items-center space-x-3">
                {plan.ai_capabilities.auto_training ? (
                  <CheckCircle className="w-5 h-5 text-green-400" />
                ) : (
                  <XCircle className="w-5 h-5 text-red-400" />
                )}
                <span className="text-gray-300">Auto-Entrenamiento</span>
              </div>
            </div>
          </div>

          {/* Límites de API */}
          <div>
            <h3 className="text-xl font-semibold mb-4 flex items-center">
              <BarChart3 className="w-5 h-5 mr-2 text-blue-400" />
              Límites y Capacidades
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-gray-700/50 rounded-lg p-4">
                <div className="text-2xl font-bold text-white">{plan.api_limits.daily_requests.toLocaleString()}</div>
                <div className="text-sm text-gray-400">Requests/día</div>
              </div>
              <div className="bg-gray-700/50 rounded-lg p-4">
                <div className="text-2xl font-bold text-white">{plan.max_predictions_per_day.toLocaleString()}</div>
                <div className="text-sm text-gray-400">Predicciones/día</div>
              </div>
              <div className="bg-gray-700/50 rounded-lg p-4">
                <div className="text-2xl font-bold text-white">{plan.max_backtests_per_month.toLocaleString()}</div>
                <div className="text-sm text-gray-400">Backtests/mes</div>
              </div>
              <div className="bg-gray-700/50 rounded-lg p-4">
                <div className="text-2xl font-bold text-white">{plan.max_portfolios.toLocaleString()}</div>
                <div className="text-sm text-gray-400">Portfolios</div>
              </div>
            </div>
          </div>

          {/* Características de UI */}
          <div>
            <h3 className="text-xl font-semibold mb-4 flex items-center">
              <Settings className="w-5 h-5 mr-2 text-green-400" />
              Características de Interfaz
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="flex items-center space-x-3">
                {plan.ui_features.advanced_charts ? (
                  <CheckCircle className="w-5 h-5 text-green-400" />
                ) : (
                  <XCircle className="w-5 h-5 text-red-400" />
                )}
                <span className="text-gray-300">Gráficos Avanzados</span>
              </div>
              <div className="flex items-center space-x-3">
                {plan.ui_features.multiple_timeframes ? (
                  <CheckCircle className="w-5 h-5 text-green-400" />
                ) : (
                  <XCircle className="w-5 h-5 text-red-400" />
                )}
                <span className="text-gray-300">Múltiples Timeframes</span>
              </div>
              <div className="flex items-center space-x-3">
                {plan.ui_features.rl_dashboard ? (
                  <CheckCircle className="w-5 h-5 text-green-400" />
                ) : (
                  <XCircle className="w-5 h-5 text-red-400" />
                )}
                <span className="text-gray-300">Dashboard RL</span>
              </div>
              <div className="flex items-center space-x-3">
                {plan.ui_features.ai_monitor ? (
                  <CheckCircle className="w-5 h-5 text-green-400" />
                ) : (
                  <XCircle className="w-5 h-5 text-red-400" />
                )}
                <span className="text-gray-300">Monitor IA</span>
              </div>
              <div className="flex items-center space-x-3">
                {plan.ui_features.mt4_integration ? (
                  <CheckCircle className="w-5 h-5 text-green-400" />
                ) : (
                  <XCircle className="w-5 h-5 text-red-400" />
                )}
                <span className="text-gray-300">Integración MT4</span>
              </div>
              <div className="flex items-center space-x-3">
                {plan.ui_features.api_access ? (
                  <CheckCircle className="w-5 h-5 text-green-400" />
                ) : (
                  <XCircle className="w-5 h-5 text-red-400" />
                )}
                <span className="text-gray-300">Acceso API</span>
              </div>
              <div className="flex items-center space-x-3">
                {plan.ui_features.custom_reports ? (
                  <CheckCircle className="w-5 h-5 text-green-400" />
                ) : (
                  <XCircle className="w-5 h-5 text-red-400" />
                )}
                <span className="text-gray-300">Reportes Personalizados</span>
              </div>
              <div className="flex items-center space-x-3">
                {plan.ui_features.priority_support ? (
                  <CheckCircle className="w-5 h-5 text-green-400" />
                ) : (
                  <XCircle className="w-5 h-5 text-red-400" />
                )}
                <span className="text-gray-300">Soporte Prioritario</span>
              </div>
            </div>
          </div>

          {/* Soporte */}
          <div>
            <h3 className="text-xl font-semibold mb-4 flex items-center">
              <Users className="w-5 h-5 mr-2 text-blue-400" />
              Soporte y Respuesta
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-gray-700/50 rounded-lg p-4">
                <div className="flex items-center space-x-3 mb-2">
                  <Crown className="w-5 h-5 text-yellow-400" />
                  <span className="font-medium text-white">Nivel de Soporte</span>
                </div>
                <div className={`text-lg font-semibold ${getSupportLevelColor(plan.support_level)}`}>
                  {plan.support_level === 'dedicated' && 'Soporte Dedicado'}
                  {plan.support_level === 'phone' && 'Soporte Telefónico'}
                  {plan.support_level === 'email' && 'Soporte por Email'}
                  {plan.support_level === 'community' && 'Soporte Comunitario'}
                </div>
              </div>
              <div className="bg-gray-700/50 rounded-lg p-4">
                <div className="flex items-center space-x-3 mb-2">
                  <Clock className="w-5 h-5 text-green-400" />
                  <span className="font-medium text-white">Tiempo de Respuesta</span>
                </div>
                <div className="text-lg font-semibold text-white">
                  {plan.response_time_hours} horas
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Footer con botón de upgrade */}
        <div className="p-6 border-t border-gray-700">
          <button
            onClick={() => onUpgrade(plan)}
            className="w-full py-3 px-6 bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 rounded-lg font-medium transition-all duration-200 text-white"
          >
            <Zap className="w-5 h-5 inline mr-2" />
            Upgrade a {plan.name}
          </button>
        </div>
      </div>
    </div>
  );
};

export default PlanDetailsModal; 