import React, { useState, useEffect } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { 
  Crown, 
  Zap, 
  Brain, 
  Shield, 
  CreditCard, 
  Calendar, 
  CheckCircle, 
  XCircle,
  ArrowUpRight,
  Star,
  Clock,
  DollarSign,
  Users,
  BarChart3,
  Settings,
  Zap as ZapIcon,
  Info
} from 'lucide-react';
import PlanDetailsModal from './PlanDetailsModal';

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

interface UserSubscription {
  id: string;
  planType: 'starter' | 'trader' | 'expert' | 'premium' | 'institutional';
  status: 'active' | 'expired' | 'cancelled' | 'trial';
  startDate: string;
  endDate: string;
  isTrial: boolean;
}

const Subscriptions: React.FC = () => {
  const { user, subscription } = useAuth();
  const [plans, setPlans] = useState<Plan[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedPlan, setSelectedPlan] = useState<Plan | null>(null);
  const [showUpgradeModal, setShowUpgradeModal] = useState(false);
  const [showDetailsModal, setShowDetailsModal] = useState(false);
  const [selectedPlanForDetails, setSelectedPlanForDetails] = useState<Plan | null>(null);
  const [paymentMethod, setPaymentMethod] = useState('card');

  useEffect(() => {
    fetchPlans();
  }, []);

  const fetchPlans = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/subscriptions/plans');
      if (response.ok) {
        const plansData = await response.json();
        setPlans(plansData);
      }
    } catch (error) {
      console.error('Error fetching plans:', error);
    } finally {
      setLoading(false);
    }
  };

  const getPlanIcon = (planType: string) => {
    switch (planType) {
      case 'starter':
        return <Shield className="w-6 h-6 text-green-400" />;
      case 'trader':
        return <Zap className="w-6 h-6 text-blue-400" />;
      case 'expert':
        return <Brain className="w-6 h-6 text-purple-400" />;
      case 'premium':
        return <Crown className="w-6 h-6 text-yellow-400" />;
      case 'institutional':
        return <Crown className="w-6 h-6 text-purple-600" />;
      default:
        return <Shield className="w-6 h-6 text-gray-400" />;
    }
  };

  const getPlanColor = (planType: string) => {
    switch (planType) {
      case 'starter':
        return 'border-green-500/20 bg-green-500/5';
      case 'trader':
        return 'border-blue-500/20 bg-blue-500/5';
      case 'expert':
        return 'border-purple-500/20 bg-purple-500/5';
      case 'premium':
        return 'border-yellow-500/20 bg-yellow-500/5';
      case 'institutional':
        return 'border-purple-600/20 bg-purple-600/5';
      default:
        return 'border-gray-500/20 bg-gray-500/5';
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('es-ES', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });
  };

  const getDaysUntilExpiry = (endDate: string) => {
    const end = new Date(endDate);
    const now = new Date();
    const diffTime = end.getTime() - now.getTime();
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
    return diffDays;
  };

  const handleUpgrade = (plan: Plan) => {
    setSelectedPlan(plan);
    setShowUpgradeModal(true);
  };

  const handleShowDetails = (plan: Plan) => {
    setSelectedPlanForDetails(plan);
    setShowDetailsModal(true);
  };

  const handlePayment = async () => {
    if (!selectedPlan) return;

    try {
      const token = localStorage.getItem('token');
      if (!token) {
        alert('No hay token de autenticación');
        return;
      }

      const response = await fetch('http://localhost:8000/api/subscriptions/upgrade', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          plan_type: selectedPlan.plan_type,
          payment_method: paymentMethod
        })
      });

      if (response.ok) {
        const result = await response.json();
        alert(`¡Upgrade exitoso a ${selectedPlan.name}!`);
        setShowUpgradeModal(false);
        // Recargar la página para actualizar la información
        window.location.reload();
      } else {
        const error = await response.json();
        alert(`Error: ${error.detail || 'Error al procesar el upgrade'}`);
      }
    } catch (error) {
      console.error('Error processing payment:', error);
      alert('Error al procesar el pago. Inténtalo de nuevo.');
    }
  };

  const getCurrentPlan = () => {
    return plans.find(plan => plan.plan_type === subscription?.planType);
  };

  const getAvailableUpgrades = () => {
    const currentPlanIndex = plans.findIndex(plan => plan.plan_type === subscription?.planType);
    return plans.slice(currentPlanIndex + 1);
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
            Gestión de Suscripciones
          </h1>
          <p className="text-gray-400 mt-2">
            Administra tu suscripción actual y explora opciones de upgrade
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Suscripción Actual */}
          <div className="lg:col-span-1">
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <h2 className="text-2xl font-semibold mb-4 flex items-center">
                <Crown className="w-6 h-6 mr-2 text-yellow-400" />
                Tu Suscripción Actual
              </h2>

              {subscription ? (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      {getPlanIcon(subscription.planType)}
                      <div>
                        <h3 className="font-semibold text-lg capitalize">
                          {subscription.planType}
                        </h3>
                        <p className="text-gray-400 text-sm">
                          {getCurrentPlan()?.description}
                        </p>
                      </div>
                    </div>
                    <div className={`px-3 py-1 rounded-full text-xs font-medium ${
                      subscription.status === 'active' 
                        ? 'bg-green-500/20 text-green-400' 
                        : 'bg-red-500/20 text-red-400'
                    }`}>
                      {subscription.status === 'active' ? 'Activa' : 'Inactiva'}
                    </div>
                  </div>

                  <div className="space-y-3">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-400">Fecha de inicio:</span>
                      <span>{formatDate(subscription.startDate)}</span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-400">Fecha de expiración:</span>
                      <span>{formatDate(subscription.endDate)}</span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-400">Días restantes:</span>
                      <span className={`font-medium ${
                        getDaysUntilExpiry(subscription.endDate) <= 7 
                          ? 'text-red-400' 
                          : 'text-green-400'
                      }`}>
                        {getDaysUntilExpiry(subscription.endDate)} días
                      </span>
                    </div>
                  </div>

                  {subscription.isTrial && (
                    <div className="bg-blue-500/20 border border-blue-500/30 rounded-lg p-3">
                      <div className="flex items-center space-x-2">
                        <Clock className="w-4 h-4 text-blue-400" />
                        <span className="text-blue-400 text-sm font-medium">
                          Período de prueba activo
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center py-8">
                  <Shield className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-400">No tienes una suscripción activa</p>
                </div>
              )}
            </div>
          </div>

          {/* Planes Disponibles */}
          <div className="lg:col-span-2">
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <h2 className="text-2xl font-semibold mb-6 flex items-center">
                <ArrowUpRight className="w-6 h-6 mr-2 text-blue-400" />
                Planes Disponibles
              </h2>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {plans.map((plan) => {
                  const isCurrentPlan = plan.plan_type === subscription?.planType;
                  const isUpgrade = plans.findIndex(p => p.plan_type === subscription?.planType) < 
                                   plans.findIndex(p => p.plan_type === plan.plan_type);

                  return (
                    <div 
                      key={plan.id}
                      className={`relative rounded-xl p-6 border transition-all duration-200 hover:scale-105 ${
                        isCurrentPlan 
                          ? 'border-green-500/50 bg-green-500/10' 
                          : getPlanColor(plan.plan_type)
                      }`}
                    >
                      {isCurrentPlan && (
                        <div className="absolute -top-2 -right-2 bg-green-500 text-white px-2 py-1 rounded-full text-xs font-medium">
                          Actual
                        </div>
                      )}

                      <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center space-x-3">
                          {getPlanIcon(plan.plan_type)}
                          <div>
                            <h3 className="font-semibold text-lg">{plan.name}</h3>
                            <p className="text-gray-400 text-sm">{plan.description}</p>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="text-2xl font-bold">
                            ${plan.price}
                            <span className="text-sm text-gray-400">/mes</span>
                          </div>
                        </div>
                      </div>

                      <div className="space-y-3 mb-6">
                        <div className="grid grid-cols-2 gap-2 text-sm">
                          <div className="flex items-center space-x-2">
                            <BarChart3 className="w-4 h-4 text-blue-400" />
                            <span>{plan.api_limits.daily_requests} requests/día</span>
                          </div>
                          <div className="flex items-center space-x-2">
                            <Brain className="w-4 h-4 text-purple-400" />
                            <span>{plan.max_predictions_per_day} predicciones/día</span>
                          </div>
                          <div className="flex items-center space-x-2">
                            <Settings className="w-4 h-4 text-green-400" />
                            <span>{plan.max_backtests_per_month} backtests/mes</span>
                          </div>
                          <div className="flex items-center space-x-2">
                            <Users className="w-4 h-4 text-yellow-400" />
                            <span>{plan.max_portfolios} portfolios</span>
                          </div>
                        </div>
                      </div>

                                             <div className="space-y-2 mb-6">
                         <h4 className="font-medium text-sm text-gray-300">Beneficios principales:</h4>
                         <div className="space-y-1">
                           {plan.benefits.slice(0, 3).map((benefit, index) => (
                             <div key={index} className="flex items-center space-x-2 text-sm">
                               <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0" />
                               <span className="text-gray-300">{benefit}</span>
                             </div>
                           ))}
                         </div>
                         <button
                           onClick={() => handleShowDetails(plan)}
                           className="flex items-center space-x-2 text-blue-400 hover:text-blue-300 text-sm transition-colors"
                         >
                           <Info className="w-4 h-4" />
                           <span>Ver detalles completos</span>
                         </button>
                       </div>

                      <button
                        onClick={() => handleUpgrade(plan)}
                        disabled={isCurrentPlan || !isUpgrade}
                        className={`w-full py-3 px-4 rounded-lg font-medium transition-all duration-200 ${
                          isCurrentPlan
                            ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                            : isUpgrade
                            ? 'bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 text-white'
                            : 'bg-gray-600 text-gray-400 cursor-not-allowed'
                        }`}
                      >
                        {isCurrentPlan ? 'Plan Actual' : isUpgrade ? 'Upgrade' : 'No disponible'}
                      </button>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>

        {/* Modal de Upgrade */}
        {showUpgradeModal && selectedPlan && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
            <div className="bg-gray-800 rounded-xl p-6 max-w-md w-full mx-4 border border-gray-700">
              <h3 className="text-xl font-semibold mb-4">
                Upgrade a {selectedPlan.name}
              </h3>
              
              <div className="space-y-4">
                <div className="bg-gray-700 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-gray-300">Precio mensual:</span>
                    <span className="text-2xl font-bold">${selectedPlan.price}</span>
                  </div>
                  <p className="text-sm text-gray-400">
                    Renovación automática mensual
                  </p>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">
                    Método de pago
                  </label>
                  <select
                    value={paymentMethod}
                    onChange={(e) => setPaymentMethod(e.target.value)}
                    className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white"
                  >
                    <option value="card">Tarjeta de crédito/débito</option>
                    <option value="paypal">PayPal</option>
                    <option value="crypto">Criptomonedas</option>
                  </select>
                </div>

                <div className="flex space-x-3">
                  <button
                    onClick={() => setShowUpgradeModal(false)}
                    className="flex-1 py-2 px-4 border border-gray-600 rounded-lg hover:bg-gray-700 transition-colors"
                  >
                    Cancelar
                  </button>
                  <button
                    onClick={handlePayment}
                    className="flex-1 py-2 px-4 bg-gradient-to-r from-blue-500 to-purple-500 rounded-lg hover:from-blue-600 hover:to-purple-600 transition-all"
                  >
                    <CreditCard className="w-4 h-4 inline mr-2" />
                    Procesar Pago
                  </button>
                </div>
              </div>
            </div>
          </div>
                 )}

         {/* Modal de Detalles del Plan */}
         <PlanDetailsModal
           plan={selectedPlanForDetails}
           isOpen={showDetailsModal}
           onClose={() => setShowDetailsModal(false)}
           onUpgrade={handleUpgrade}
         />
       </div>
     </div>
   );
 };

export default Subscriptions; 