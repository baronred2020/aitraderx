import React, { useState, useEffect } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { Brain, Eye, EyeOff, Loader2, Check, CreditCard, Shield, Zap, Crown } from 'lucide-react';

interface Plan {
  id: string;
  name: string;
  plan_type: 'freemium' | 'basic' | 'pro' | 'elite';
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
}

interface PaymentMethod {
  id: string;
  name: string;
  icon: string;
  description: string;
}

export const Register: React.FC = () => {
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    password: '',
    confirmPassword: '',
    firstName: '',
    lastName: '',
    phone: ''
  });
  
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  
  // Estados para el plan y pago
  const [selectedPlan, setSelectedPlan] = useState<string>('freemium');
  const [showPlanSelection, setShowPlanSelection] = useState(false);
  const [plans, setPlans] = useState<Plan[]>([]);
  const [paymentMethod, setPaymentMethod] = useState<string>('');
  const [isProcessingPayment, setIsProcessingPayment] = useState(false);
  
  const { login } = useAuth();

  // Métodos de pago disponibles
  const paymentMethods: PaymentMethod[] = [
    {
      id: 'stripe',
      name: 'Tarjeta de Crédito/Débito',
      icon: '💳',
      description: 'Visa, Mastercard, American Express'
    },
    {
      id: 'paypal',
      name: 'PayPal',
      icon: '🔵',
      description: 'Pago seguro con PayPal'
    },
    {
      id: 'crypto',
      name: 'Criptomonedas',
      icon: '₿',
      description: 'Bitcoin, Ethereum, USDT'
    }
  ];

  // Cargar planes desde el backend
  useEffect(() => {
    const fetchPlans = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/subscriptions/plans');
        if (response.ok) {
          const plansData = await response.json();
          setPlans(plansData);
        }
      } catch (error) {
        console.error('Error cargando planes:', error);
      }
    };
    
    fetchPlans();
  }, []);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const validateForm = () => {
    if (!formData.username || !formData.email || !formData.password || !formData.confirmPassword) {
      setError('Todos los campos son obligatorios');
      return false;
    }
    
    if (formData.password !== formData.confirmPassword) {
      setError('Las contraseñas no coinciden');
      return false;
    }
    
    if (formData.password.length < 8) {
      setError('La contraseña debe tener al menos 8 caracteres');
      return false;
    }
    
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(formData.email)) {
      setError('Ingresa un email válido');
      return false;
    }
    
    return true;
  };

  const handleFreeRegistration = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!validateForm()) return;
    
    setIsLoading(true);
    setError('');
    
    try {
      // Registrar usuario con plan freemium
      const response = await fetch('http://localhost:8000/api/auth/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ...formData,
          plan_type: 'freemium'
        }),
      });

      if (response.ok) {
        const responseData = await response.json();
        setSuccess('¡Registro exitoso! Tu cuenta freemium ha sido creada.');
        
        // Guardar token en localStorage
        if (responseData.token) {
          localStorage.setItem('auth_token', responseData.token);
        }
        
        // Auto-login inmediato
        const loginSuccess = await login(formData.username, formData.password);
        if (!loginSuccess) {
          setError('Registro exitoso pero error al iniciar sesión automáticamente');
        }
      } else {
        try {
          const errorData = await response.json();
          let errorMessage = 'Error en el registro';
          
          if (errorData.detail) {
            if (typeof errorData.detail === 'string') {
              errorMessage = errorData.detail;
            } else if (Array.isArray(errorData.detail)) {
              errorMessage = errorData.detail.map((err: any) => err.msg || 'Error de validación').join(', ');
            }
          }
          
          setError(errorMessage);
        } catch (parseError) {
          setError('Error en el registro');
        }
      }
    } catch (error) {
      console.error('Registration error:', error);
      setError('Error de conexión. Verifica que el backend esté corriendo.');
    } finally {
      setIsLoading(false);
    }
  };

  const handlePaidRegistration = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!validateForm()) return;
    
    if (!paymentMethod) {
      setError('Selecciona un método de pago');
      return;
    }
    
    setIsProcessingPayment(true);
    setError('');
    
    try {
      // Registrar usuario con plan pagado
      const response = await fetch('http://localhost:8000/api/auth/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ...formData,
          plan_type: selectedPlan,
          payment_method: paymentMethod
        }),
      });

      if (response.ok) {
        const responseData = await response.json();
        setSuccess('¡Registro exitoso! Tu suscripción ha sido activada.');
        
        // Guardar token en localStorage
        if (responseData.token) {
          localStorage.setItem('auth_token', responseData.token);
        }
        
        // Auto-login inmediato
        const loginSuccess = await login(formData.username, formData.password);
        if (!loginSuccess) {
          setError('Registro exitoso pero error al iniciar sesión automáticamente');
        }
      } else {
        try {
          const errorData = await response.json();
          let errorMessage = 'Error en el registro';
          
          if (errorData.detail) {
            if (typeof errorData.detail === 'string') {
              errorMessage = errorData.detail;
            } else if (Array.isArray(errorData.detail)) {
              errorMessage = errorData.detail.map((err: any) => err.msg || 'Error de validación').join(', ');
            }
          }
          
          setError(errorMessage);
        } catch (parseError) {
          setError('Error en el registro');
        }
      }
    } catch (error) {
      console.error('Registration error:', error);
      setError('Error de conexión. Verifica que el backend esté corriendo.');
    } finally {
      setIsProcessingPayment(false);
    }
  };

  const getSelectedPlan = () => {
    return plans.find(plan => plan.plan_type === selectedPlan);
  };

  const selectedPlanData = getSelectedPlan();

  return (
    <div className="min-h-screen w-full bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 flex items-center justify-center p-4">
      <div className="w-full max-w-6xl">
        {/* Logo y título */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center mb-4">
            <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-teal-500 rounded-2xl flex items-center justify-center">
              <Brain className="w-8 h-8 text-white" />
            </div>
          </div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-teal-400 bg-clip-text text-transparent mb-2">
            AITRADERX
          </h1>
          <p className="text-gray-400">AI Trading Platform</p>
        </div>

        <div className={`grid gap-8 ${showPlanSelection ? 'grid-cols-1 lg:grid-cols-2' : 'grid-cols-1 max-w-md mx-auto'}`}>
          {/* Formulario de registro */}
          <div className="glass-effect rounded-2xl p-8 border border-gray-700/50">
            <h2 className="text-2xl font-bold text-white mb-6 text-center">Crear Cuenta</h2>
            
            <form onSubmit={showPlanSelection ? handlePaidRegistration : handleFreeRegistration} className="space-y-6">
              {/* Información personal */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label htmlFor="firstName" className="block text-sm font-medium text-gray-300 mb-2">
                    Nombre
                  </label>
                  <input
                    id="firstName"
                    name="firstName"
                    type="text"
                    value={formData.firstName}
                    onChange={handleInputChange}
                    className="w-full px-4 py-3 bg-gray-800/50 border border-gray-600/50 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                    placeholder="Tu nombre"
                    required
                  />
                </div>
                
                <div>
                  <label htmlFor="lastName" className="block text-sm font-medium text-gray-300 mb-2">
                    Apellido
                  </label>
                  <input
                    id="lastName"
                    name="lastName"
                    type="text"
                    value={formData.lastName}
                    onChange={handleInputChange}
                    className="w-full px-4 py-3 bg-gray-800/50 border border-gray-600/50 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                    placeholder="Tu apellido"
                    required
                  />
                </div>
              </div>

              <div>
                <label htmlFor="username" className="block text-sm font-medium text-gray-300 mb-2">
                  Usuario
                </label>
                <input
                  id="username"
                  name="username"
                  type="text"
                  value={formData.username}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 bg-gray-800/50 border border-gray-600/50 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                  placeholder="Elige un nombre de usuario"
                  required
                />
              </div>

              <div>
                <label htmlFor="email" className="block text-sm font-medium text-gray-300 mb-2">
                  Email
                </label>
                <input
                  id="email"
                  name="email"
                  type="email"
                  value={formData.email}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 bg-gray-800/50 border border-gray-600/50 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                  placeholder="tu@email.com"
                  required
                />
              </div>

              <div>
                <label htmlFor="phone" className="block text-sm font-medium text-gray-300 mb-2">
                  Teléfono (opcional)
                </label>
                <input
                  id="phone"
                  name="phone"
                  type="tel"
                  value={formData.phone}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 bg-gray-800/50 border border-gray-600/50 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                  placeholder="+1 (555) 123-4567"
                />
              </div>

              <div>
                <label htmlFor="password" className="block text-sm font-medium text-gray-300 mb-2">
                  Contraseña
                </label>
                <div className="relative">
                  <input
                    id="password"
                    name="password"
                    type={showPassword ? 'text' : 'password'}
                    value={formData.password}
                    onChange={handleInputChange}
                    className="w-full px-4 py-3 bg-gray-800/50 border border-gray-600/50 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all pr-12"
                    placeholder="Mínimo 8 caracteres"
                    required
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-white transition-colors"
                  >
                    {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                  </button>
                </div>
              </div>

              <div>
                <label htmlFor="confirmPassword" className="block text-sm font-medium text-gray-300 mb-2">
                  Confirmar Contraseña
                </label>
                <div className="relative">
                  <input
                    id="confirmPassword"
                    name="confirmPassword"
                    type={showConfirmPassword ? 'text' : 'password'}
                    value={formData.confirmPassword}
                    onChange={handleInputChange}
                    className="w-full px-4 py-3 bg-gray-800/50 border border-gray-600/50 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all pr-12"
                    placeholder="Confirma tu contraseña"
                    required
                  />
                  <button
                    type="button"
                    onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                    className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-white transition-colors"
                  >
                    {showConfirmPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                  </button>
                </div>
              </div>

              {/* Error */}
              {error && (
                <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-4">
                  <p className="text-red-400 text-sm">
                    {typeof error === 'string' ? error : 'Error en el registro'}
                  </p>
                </div>
              )}

              {/* Success */}
              {success && (
                <div className="bg-green-500/10 border border-green-500/20 rounded-lg p-4">
                  <p className="text-green-400 text-sm">{success}</p>
                </div>
              )}

              {/* Botón de registro */}
              <button
                type="submit"
                disabled={isLoading || isProcessingPayment}
                className="w-full bg-gradient-to-r from-blue-600 to-teal-600 hover:from-blue-700 hover:to-teal-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold py-3 px-6 rounded-lg transition-all duration-200 flex items-center justify-center space-x-2"
              >
                {isLoading || isProcessingPayment ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    <span>{isProcessingPayment ? 'Procesando pago...' : 'Creando cuenta...'}</span>
                  </>
                ) : (
                  <span>{showPlanSelection ? 'Completar Registro con Pago' : 'Registrarse Gratis'}</span>
                )}
              </button>
            </form>

            {/* Opción para cambiar a registro con pago */}
            {!showPlanSelection && (
              <div className="mt-6 text-center">
                <button
                  onClick={() => setShowPlanSelection(true)}
                  className="text-blue-400 hover:text-blue-300 transition-colors text-sm"
                >
                  ¿Quieres más funcionalidades? Ver planes premium
                </button>
              </div>
            )}
          </div>

          {/* Selección de plan y pago */}
          {showPlanSelection && (
            <div className="space-y-6">
              {/* Planes disponibles */}
              <div className="glass-effect rounded-2xl p-6 border border-gray-700/50">
                <h3 className="text-xl font-bold text-white mb-4">Elige tu Plan</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {plans.map((plan) => (
                    <div
                      key={plan.id}
                      className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                        selectedPlan === plan.plan_type
                          ? 'border-blue-500 bg-blue-500/10'
                          : 'border-gray-600 hover:border-gray-500'
                      }`}
                      onClick={() => setSelectedPlan(plan.plan_type)}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="font-semibold text-white">{plan.name}</h4>
                        <div className="flex items-center space-x-2">
                          {plan.plan_type === 'freemium' && <Shield className="w-4 h-4 text-green-400" />}
                          {plan.plan_type === 'basic' && <Zap className="w-4 h-4 text-blue-400" />}
                          {plan.plan_type === 'pro' && <Crown className="w-4 h-4 text-purple-400" />}
                          {plan.plan_type === 'elite' && <Crown className="w-4 h-4 text-yellow-400" />}
                        </div>
                      </div>
                      
                      <div className="text-2xl font-bold text-white mb-2">
                        {plan.price === 0 ? 'Gratis' : `$${plan.price}`}
                        {plan.price > 0 && <span className="text-sm text-gray-400">/mes</span>}
                      </div>
                      
                      <p className="text-gray-400 text-sm mb-3">{plan.description}</p>
                      
                      <div className="space-y-1">
                        {plan.benefits.slice(0, 3).map((benefit, index) => (
                          <div key={index} className="flex items-center space-x-2">
                            <Check className="w-3 h-3 text-green-400" />
                            <span className="text-xs text-gray-300">{benefit}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Métodos de pago */}
              {selectedPlan !== 'freemium' && (
                <div className="glass-effect rounded-2xl p-6 border border-gray-700/50">
                  <h3 className="text-xl font-bold text-white mb-4">Método de Pago</h3>
                  <div className="space-y-3">
                    {paymentMethods.map((method) => (
                      <div
                        key={method.id}
                        className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                          paymentMethod === method.id
                            ? 'border-blue-500 bg-blue-500/10'
                            : 'border-gray-600 hover:border-gray-500'
                        }`}
                        onClick={() => setPaymentMethod(method.id)}
                      >
                        <div className="flex items-center space-x-3">
                          <span className="text-2xl">{method.icon}</span>
                          <div>
                            <h4 className="font-semibold text-white">{method.name}</h4>
                            <p className="text-sm text-gray-400">{method.description}</p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Resumen del plan seleccionado */}
              {selectedPlanData && (
                <div className="glass-effect rounded-2xl p-6 border border-gray-700/50">
                  <h3 className="text-xl font-bold text-white mb-4">Resumen del Plan</h3>
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-gray-300">Plan:</span>
                      <span className="text-white font-semibold">{selectedPlanData.name}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">Precio:</span>
                      <span className="text-white font-semibold">
                        {selectedPlanData.price === 0 ? 'Gratis' : `$${selectedPlanData.price}/mes`}
                      </span>
                    </div>
                    {selectedPlanData.price > 0 && (
                      <div className="flex justify-between">
                        <span className="text-gray-300">Método de pago:</span>
                        <span className="text-white font-semibold">
                          {paymentMethods.find(m => m.id === paymentMethod)?.name || 'No seleccionado'}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="text-center mt-8">
          <p className="text-gray-400 text-sm">
            ¿Ya tienes cuenta?{' '}
            <button 
              onClick={() => window.location.href = '/login'}
              className="text-blue-400 hover:text-blue-300 transition-colors"
            >
              Inicia sesión aquí
            </button>
          </p>
        </div>
      </div>
    </div>
  );
}; 