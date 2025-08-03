// src/components/RL/RLDashboard.tsx
import React, { useState, useEffect } from 'react';
import { Brain, Zap, TrendingUp, Award, BarChart3, RefreshCw, Play, Square, Target, DollarSign, Activity, Shield, Settings, Eye, EyeOff, Crown, Cpu, Network, Rocket } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';
import { useFeatureAccess } from '../../hooks/useFeatureAccess';
import { UpgradeModal } from '../Common/UpgradeModal';
import { useAuth } from '../../contexts/AuthContext';

interface RLStatus {
  status: string;
  active_sessions: number;
  model_coordination: {
    brain_max_weight: number;
    brain_ultra_weight: number;
    brain_predictor_weight: number;
    megamind_weight: number;
  };
  current_strategy: string;
  market_regime: string;
  risk_level: string;
}

interface RLPerformance {
  total_trades: number;
  win_rate: number;
  profit_factor: number;
  sharpe_ratio: number;
  max_drawdown: number;
  total_return: number;
  model_performance: {
    brain_max: { accuracy: number; confidence: number };
    brain_ultra: { accuracy: number; confidence: number };
    brain_predictor: { accuracy: number; confidence: number };
    megamind: { accuracy: number; confidence: number };
  };
}

interface TradingSignal {
  signal_id: string;
  pair: string;
  signal: string;
  confidence: number;
  position_size: number;
  entry_price: number;  // Precio de entrada real
  stop_loss: number;
  take_profit: number;
  reasoning: string;
  models_used: string[];
  timestamp: string;
  current_market_price?: number;  // Precio actual de mercado
}

interface TrainingProgress {
  is_training: boolean;
  progress: number;
  current_episode: number;
  total_episodes: number;
  estimated_time_remaining?: number;
  status: string;
  error_message?: string;
}

interface TrainingPermission {
  can_train: boolean;
  reason: string;
  session_id?: string;
  days_until_next?: number;
}

interface TrainingValidation {
  valid: boolean;
  reason?: string;
  limits?: {
    min: number;
    max: number;
    recommended: number;
  };
  estimated_minutes?: number;
}

interface TrainingSession {
  success: boolean;
  session_id?: string;
  message?: string;
  error?: string;
}

export const RLDashboard: React.FC = () => {
  const { requireAccess, showUpgradeModal, upgradeInfo, closeUpgradeModal } = useFeatureAccess();
  const { subscription } = useAuth();
  
  // Verificar acceso al cargar el componente
  useEffect(() => {
    const hasAccess = requireAccess('rl');
    if (!hasAccess) {
      return; // No continuar si no tiene acceso
    }
    
    // Cargar datos iniciales
    loadRLData();
  }, [requireAccess]);

  const [rlStatus, setRlStatus] = useState<RLStatus | null>(null);
  const [rlPerformance, setRlPerformance] = useState<RLPerformance | null>(null);
  const [activeSignals, setActiveSignals] = useState<TradingSignal[]>([]);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState<TrainingProgress | null>(null);
  const [currentEpisode, setCurrentEpisode] = useState(0);
  const [estimatedTimeRemaining, setEstimatedTimeRemaining] = useState(0);
  const [episodes, setEpisodes] = useState(() => {
    // Establecer episodios por defecto según el plan
    const planType = subscription?.planType;
    switch (planType) {
      case 'premium':
        return 500; // Recomendado para Premium
      case 'institutional':
        return 1000; // Recomendado para Institutional
      default:
        return 500; // Default a Premium
    }
  });
  const [trainingPermission, setTrainingPermission] = useState<TrainingPermission | null>(null);
  const [trainingValidation, setTrainingValidation] = useState<TrainingValidation | null>(null);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [isStartingTraining, setIsStartingTraining] = useState(false);

  useEffect(() => {
    loadRLData();
    const interval = setInterval(loadRLData, 3000); // Actualización más frecuente
    return () => clearInterval(interval);
  }, []);

  // Verificar permisos de entrenamiento
  useEffect(() => {
    const checkTrainingPermission = async () => {
      try {
        // Obtener user_id del contexto de autenticación
        const userId = "4dabfd30-483d-4fa0-a8d0-bd151a46340f"; // TODO: Obtener del contexto de auth
        
        const response = await fetch(`/api/rl/can-train/${userId}`);
        if (response.ok) {
          const permission = await response.json();
          setTrainingPermission(permission);
        } else {
          console.error('Error checking training permission:', response.status);
          setTrainingPermission({
            can_train: false,
            reason: "Error verificando permisos"
          });
        }
      } catch (error) {
        console.error('Error checking training permission:', error);
        setTrainingPermission({
          can_train: false,
          reason: "Error de conexión"
        });
      }
    };

    // Verificar permisos al cargar y cuando cambie el estado de entrenamiento
    checkTrainingPermission();
  }, [trainingProgress?.is_training]); // Dependencia en el estado de entrenamiento

  const loadRLData = async () => {
    try {
      const [statusResponse, performanceResponse, signalsResponse] = await Promise.all([
        fetch('/api/rl/status'),
        fetch('/api/rl/performance'),
        fetch('/api/rl/active-signals')
      ]);

      if (statusResponse.ok) {
        const statusData = await statusResponse.json();
        setRlStatus(statusData);
      } else {
        console.error('Error loading RL status:', statusResponse.status);
      }

      if (performanceResponse.ok) {
        const perfData = await performanceResponse.json();
        setRlPerformance(perfData);
      } else {
        console.error('Error loading RL performance:', performanceResponse.status);
      }

      if (signalsResponse.ok) {
        const signalsData = await signalsResponse.json();
        setActiveSignals(signalsData.signals || signalsData || []);
      } else {
        console.error('Error loading active signals:', signalsResponse.status);
      }

    } catch (error) {
      console.error('Error loading RL data:', error);
    }
  };

  // Función para validar parámetros de entrenamiento
  const validateTrainingParams = async (episodes: number) => {
    try {
      const response = await fetch('/api/rl/validate-params', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          episodes: episodes,
          user_plan: subscription?.planType || "premium"
        })
      });

      if (response.ok) {
        const validation = await response.json();
        setTrainingValidation(validation);
        return validation.valid;
      } else {
        console.error('Error validating parameters:', response.status);
        setTrainingValidation({
          valid: false,
          reason: "Error validando parámetros"
        });
        return false;
      }
    } catch (error) {
      console.error('Error validating training parameters:', error);
      setTrainingValidation({
        valid: false,
        reason: "Error de conexión"
      });
      return false;
    }
  };

  // Función para iniciar entrenamiento
  const startTraining = async () => {
    if (!trainingPermission?.can_train) {
      alert(trainingPermission?.reason || 'No puedes iniciar entrenamiento');
      return;
    }

    const isValid = await validateTrainingParams(episodes);
    if (!isValid) {
      alert(trainingValidation?.reason || 'Parámetros de entrenamiento inválidos');
      return;
    }

    setIsStartingTraining(true);
    try {
      const userId = "4dabfd30-483d-4fa0-a8d0-bd151a46340f"; // TODO: Obtener del contexto de auth
      
      const response = await fetch('/api/rl/start-training', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: userId,
          episodes: episodes,
          algorithm: "dqn",
          trading_pair: "EURUSD",
          timeframe: "1h"
        })
      });

      if (response.ok) {
        const result = await response.json();
        
        if (result.success && result.session_id) {
          setCurrentSessionId(result.session_id);
          setTrainingProgress({
            is_training: true,
            progress: 0,
            current_episode: 0,
            total_episodes: episodes,
            status: 'running'
          });
          
          // Iniciar polling del progreso
          startProgressPolling(result.session_id);
        } else {
          alert(result.error || 'Error iniciando entrenamiento');
        }
      } else {
        console.error('Error starting training:', response.status);
        alert('Error iniciando entrenamiento');
      }
    } catch (error) {
      console.error('Error starting training:', error);
      alert('Error iniciando entrenamiento');
    } finally {
      setIsStartingTraining(false);
    }
  };

  // Función para hacer polling del progreso
  const startProgressPolling = (sessionId: string) => {
    const interval = setInterval(async () => {
      try {
        const response = await fetch(`/api/rl/training-progress/${sessionId}`);
        
        if (response.ok) {
          const progress = await response.json();
          
          setTrainingProgress(progress);
          setCurrentEpisode(progress.current_episode || 0);
          
          if (progress.estimated_time_remaining) {
            setEstimatedTimeRemaining(progress.estimated_time_remaining);
          }
          
          // Si el entrenamiento terminó, detener el polling
          if (!progress.is_training) {
            clearInterval(interval);
            if (progress.status === 'completed') {
              alert('¡Entrenamiento completado exitosamente!');
            } else if (progress.status === 'failed') {
              alert(`Error en entrenamiento: ${progress.error_message}`);
            }
          }
        } else {
          console.error('Error fetching training progress:', response.status);
        }
      } catch (error) {
        console.error('Error fetching training progress:', error);
      }
    }, 2000); // Poll cada 2 segundos
    
    // Limpiar intervalo después de 30 minutos (tiempo máximo de entrenamiento)
    setTimeout(() => {
      clearInterval(interval);
    }, 30 * 60 * 1000);
  };

  // Función para cancelar entrenamiento
  const cancelTraining = async () => {
    if (!currentSessionId) return;
    
    try {
      const userId = "4dabfd30-483d-4fa0-a8d0-bd151a46340f"; // TODO: Obtener del contexto de auth
      
      const response = await fetch(`/api/rl/cancel-training/${currentSessionId}?user_id=${userId}`, {
        method: 'POST'
      });
      
      if (response.ok) {
        const result = await response.json();
        
        if (result.success) {
          setTrainingProgress(null);
          setCurrentSessionId(null);
          alert('Entrenamiento cancelado');
        } else {
          alert(result.error || 'Error cancelando entrenamiento');
        }
      } else {
        console.error('Error canceling training:', response.status);
        alert('Error cancelando entrenamiento');
      }
    } catch (error) {
      console.error('Error canceling training:', error);
      alert('Error cancelando entrenamiento');
    }
  };

  const executeSignal = async (signal: TradingSignal) => {
    try {
      const response = await fetch('/api/rl/execute-signal', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(signal)
      });

      if (response.ok) {
        const result = await response.json();
        if (result.success) {
          alert(`Señal ejecutada: ${signal.signal} ${signal.pair}`);
          loadRLData(); // Recargar datos
        } else {
          alert(`Error ejecutando señal: ${result.error}`);
        }
      } else {
        console.error('Error executing signal:', response.status);
        alert('Error ejecutando señal');
      }
    } catch (error) {
      console.error('Error executing signal:', error);
      alert('Error ejecutando señal');
    }
  };

  if (rlStatus === null) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        <span className="ml-3 text-white">Cargando AITRADERX RL Director...</span>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900/20 to-purple-900/20 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header Mejorado */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="w-16 h-16 rounded-2xl flex items-center justify-center shadow-2xl" style={{ 
              background: 'linear-gradient(135deg, #3b82f6, #8b5cf6, #ec4899)',
              boxShadow: '0 20px 40px rgba(59, 130, 246, 0.3)'
            }}>
              <Rocket className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
                AITRADERX RL Director
              </h1>
              <p className="text-blue-300 font-medium text-lg">Sistema de Coordinación Inteligente de Modelos IA</p>
            </div>
          </div>
          <div className="flex items-center justify-center gap-6 text-sm text-gray-400">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span>Sistema Activo</span>
            </div>
            <div className="flex items-center gap-2">
              <Cpu className="w-4 h-4" />
              <span>4 Modelos Coordinados</span>
            </div>
            <div className="flex items-center gap-2">
              <Network className="w-4 h-4" />
              <span>IA Colaborativa</span>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Panel de Estado del RL Director */}
          <RLStatusPanel 
            rlStatus={rlStatus}
            showAdvanced={showAdvanced}
            trainingProgress={trainingProgress}
            currentEpisode={currentEpisode}
            estimatedTimeRemaining={estimatedTimeRemaining}
            episodes={episodes}
            setEpisodes={setEpisodes}
            trainingPermission={trainingPermission}
            trainingValidation={trainingValidation}
            isStartingTraining={isStartingTraining}
            onStartTraining={startTraining}
            onCancelTraining={cancelTraining}
            subscription={subscription}
          />

          {/* Panel de Coordinación de Modelos */}
          <ModelCoordinationPanel rlStatus={rlStatus} showAdvanced={showAdvanced} />
        </div>

        {/* Panel de Señales Activas */}
        <ActiveSignalsPanel activeSignals={activeSignals} onExecuteSignal={executeSignal} />

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
          {/* Panel de Rendimiento */}
          <RLPerformancePanel rlPerformance={rlPerformance} showAdvanced={showAdvanced} />

          {/* Panel de Comparación */}
          <AIComparisonPanel />
        </div>

        {/* Panel de Configuración Avanzada */}
        <AdvancedConfigurationPanel showAdvanced={showAdvanced} />

        {/* Toggle para Configuración Avanzada */}
        <div className="text-center mt-6">
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="glass-effect px-8 py-3 rounded-xl text-white font-medium transition-all duration-300 hover:scale-105 hover:shadow-2xl"
            style={{
              background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(139, 92, 246, 0.2))',
              border: '1px solid rgba(59, 130, 246, 0.3)'
            }}
          >
            {showAdvanced ? (
              <>
                <EyeOff className="w-5 h-5 inline mr-2" />
                Ocultar Configuración Avanzada
              </>
            ) : (
              <>
                <Settings className="w-5 h-5 inline mr-2" />
                Mostrar Configuración Avanzada
              </>
            )}
          </button>
        </div>
      </div>

      {/* Modal de upgrade para usuarios sin acceso */}
      {showUpgradeModal && upgradeInfo && (
        <UpgradeModal
          isOpen={showUpgradeModal}
          onClose={closeUpgradeModal}
          currentPlan={upgradeInfo.currentPlan}
          requiredPlan={upgradeInfo.requiredPlan}
          feature={upgradeInfo.feature}
        />
      )}
    </div>
  );
};

const RLStatusPanel: React.FC<{
  rlStatus: RLStatus | null;
  showAdvanced: boolean;
  trainingProgress: TrainingProgress | null;
  currentEpisode: number;
  estimatedTimeRemaining: number;
  episodes: number;
  setEpisodes: (episodes: number) => void;
  trainingPermission: TrainingPermission | null;
  trainingValidation: TrainingValidation | null;
  isStartingTraining: boolean;
  onStartTraining: () => void;
  onCancelTraining: () => void;
  subscription: { planType?: string } | null;
}> = ({ 
  rlStatus, 
  showAdvanced, 
  trainingProgress, 
  currentEpisode, 
  estimatedTimeRemaining,
  episodes,
  setEpisodes,
  trainingPermission,
  trainingValidation,
  isStartingTraining,
  onStartTraining,
  onCancelTraining,
  subscription
}) => {
  const isTraining = trainingProgress?.is_training;
  const canStartTraining = trainingPermission?.can_train && !isTraining;
  const isValidEpisodes = episodes >= 100 && episodes <= 5000;

  return (
    <div className="glass-effect p-6 rounded-2xl border" style={{
      background: 'linear-gradient(135deg, rgba(26, 31, 46, 0.8), rgba(45, 55, 72, 0.8))',
      borderColor: 'rgba(59, 130, 246, 0.2)'
    }}>
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{
          background: 'linear-gradient(135deg, #3b82f6, #1d4ed8)',
          boxShadow: '0 8px 20px rgba(59, 130, 246, 0.3)'
        }}>
          <Activity className="w-5 h-5 text-white" />
        </div>
        <div>
          <h3 className="text-xl font-bold text-white">Estado del RL Director</h3>
          <p className="text-blue-300 text-sm">Monitoreo en tiempo real</p>
        </div>
      </div>
      
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="rounded-xl p-4 text-white" style={{
          background: 'linear-gradient(135deg, #3b82f6, #1d4ed8)',
          boxShadow: '0 8px 20px rgba(59, 130, 246, 0.3)'
        }}>
          <div className="text-2xl font-bold flex items-center gap-2">
            {rlStatus?.status === 'active' ? '🟢 Activo' : '🔴 Inactivo'}
          </div>
          <div className="text-sm opacity-90">Estado del Sistema</div>
        </div>
        
        <div className="rounded-xl p-4 text-white" style={{
          background: 'linear-gradient(135deg, #10b981, #059669)',
          boxShadow: '0 8px 20px rgba(16, 185, 129, 0.3)'
        }}>
          <div className="text-2xl font-bold">{rlStatus?.active_sessions || 0}</div>
          <div className="text-sm opacity-90">Sesiones Activas</div>
        </div>
      </div>

      <div className="space-y-3 mb-6">
        <div className="flex justify-between items-center p-3 rounded-lg" style={{ backgroundColor: 'rgba(59, 130, 246, 0.1)' }}>
          <span className="text-gray-300">Estrategia Actual:</span>
          <span className="font-semibold text-white">{rlStatus?.current_strategy || 'N/A'}</span>
        </div>
        <div className="flex justify-between items-center p-3 rounded-lg" style={{ backgroundColor: 'rgba(139, 92, 246, 0.1)' }}>
          <span className="text-gray-300">Regimen de Mercado:</span>
          <span className="font-semibold text-white">{rlStatus?.market_regime || 'N/A'}</span>
        </div>
        <div className="flex justify-between items-center p-3 rounded-lg" style={{ backgroundColor: 'rgba(236, 72, 153, 0.1)' }}>
          <span className="text-gray-300">Nivel de Riesgo:</span>
          <span className="font-semibold text-white">{rlStatus?.risk_level || 'N/A'}</span>
        </div>
      </div>

      {showAdvanced && (
        <>
          <div className="border-t border-gray-700 pt-6 mb-6">
            <h4 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <Crown className="w-5 h-5 text-yellow-400" />
              Controles de Entrenamiento
            </h4>
            
            {/* Información de permisos */}
            {trainingPermission && (
              <div className={`p-4 rounded-xl mb-4 ${
                trainingPermission.can_train 
                  ? 'bg-green-900/30 border border-green-500/30' 
                  : 'bg-red-900/30 border border-red-500/30'
              }`}>
                <div className="text-sm font-medium text-white">
                  {trainingPermission.can_train ? '✅' : '❌'} {trainingPermission.reason}
                </div>
                {trainingPermission.days_until_next && (
                  <div className="text-xs text-gray-400 mt-1">
                    Próximo entrenamiento disponible en {trainingPermission.days_until_next} días
                  </div>
                )}
              </div>
            )}

            {/* Configuración de entrenamiento */}
            <div className="space-y-4 mb-4">
              {/* Información del Plan */}
              <div className="p-4 rounded-xl border" style={{ 
                backgroundColor: 'rgba(26, 31, 46, 0.8)',
                borderColor: 'rgba(139, 92, 246, 0.3)'
              }}>
                <div className="flex items-center gap-3 mb-3">
                  <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{
                    background: subscription?.planType === 'institutional' 
                      ? 'linear-gradient(135deg, #ec4899, #be185d)'
                      : 'linear-gradient(135deg, #8b5cf6, #7c3aed)',
                    boxShadow: '0 4px 12px rgba(139, 92, 246, 0.3)'
                  }}>
                    {subscription?.planType === 'institutional' ? (
                      <Crown className="w-4 h-4 text-white" />
                    ) : (
                      <Award className="w-4 h-4 text-white" />
                    )}
                  </div>
                  <div>
                    <h4 className="font-medium text-white">
                      Plan {subscription?.planType === 'institutional' ? 'Institutional' : 'Premium'}
                    </h4>
                    <p className="text-sm text-gray-400">
                      Límite: {subscription?.planType === 'institutional' ? '5,000' : '1,000'} episodios
                    </p>
                  </div>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Episodios de Entrenamiento
                </label>
                <input
                  type="number"
                  value={episodes}
                  onChange={(e) => setEpisodes(Number(e.target.value))}
                  min="100"
                  max={subscription?.planType === 'institutional' ? 5000 : 1000}
                  step="100"
                  className="w-full px-3 py-2 rounded-lg text-white bg-gray-800/50 border border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  disabled={isTraining}
                />
                <div className="text-xs text-gray-500 mt-1">
                  Min: 100, Max: {subscription?.planType === 'institutional' ? '5,000' : '1,000'}, 
                  Recomendado: {subscription?.planType === 'institutional' ? '1,000' : '500'}
                </div>
                {trainingValidation?.limits && (
                  <div className="text-xs text-blue-400 mt-1">
                    Límites para tu plan: {trainingValidation.limits.min} - {trainingValidation.limits.max} episodios
                  </div>
                )}
              </div>

              {trainingValidation?.estimated_minutes && (
                <div className="text-sm text-gray-400 flex items-center gap-2">
                  <span className="text-yellow-400">⏱️</span>
                  Tiempo estimado: {trainingValidation.estimated_minutes} minutos
                </div>
              )}
            </div>

            {/* Botones de control */}
            <div className="flex gap-3">
              <button
                onClick={onStartTraining}
                disabled={!canStartTraining || !isValidEpisodes || isStartingTraining}
                className={`flex-1 py-3 px-4 rounded-xl font-medium transition-all duration-300 ${
                  canStartTraining && isValidEpisodes && !isStartingTraining
                    ? 'bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white shadow-lg hover:shadow-xl hover:scale-105'
                    : 'bg-gray-700 text-gray-400 cursor-not-allowed'
                }`}
              >
                {isStartingTraining ? (
                  <>
                    <RefreshCw className="w-4 h-4 inline mr-2 animate-spin" />
                    Iniciando...
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 inline mr-2" />
                    Iniciar Entrenamiento
                  </>
                )}
              </button>
              
              {isTraining && (
                <button
                  onClick={onCancelTraining}
                  className="flex-1 py-3 px-4 bg-gradient-to-r from-red-600 to-pink-600 hover:from-red-700 hover:to-pink-700 text-white rounded-xl font-medium transition-all duration-300 shadow-lg hover:shadow-xl hover:scale-105"
                >
                  <Square className="w-4 h-4 inline mr-2" />
                  Cancelar
                </button>
              )}
            </div>

            {/* Barra de progreso */}
            {isTraining && trainingProgress && (
              <div className="mt-6 p-4 rounded-xl" style={{ backgroundColor: 'rgba(59, 130, 246, 0.1)' }}>
                <div className="flex justify-between text-sm text-gray-300 mb-3">
                  <span>Progreso: {Math.round(trainingProgress.progress * 100)}%</span>
                  <span>Episodio {currentEpisode} de {trainingProgress.total_episodes}</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-3 overflow-hidden">
                  <div
                    className="bg-gradient-to-r from-blue-500 to-purple-500 h-3 rounded-full transition-all duration-300 shadow-lg"
                    style={{ width: `${trainingProgress.progress * 100}%` }}
                  ></div>
                </div>
                {estimatedTimeRemaining > 0 && (
                  <div className="text-xs text-gray-400 mt-2 flex items-center gap-1">
                    <span className="text-yellow-400">⏱️</span>
                    Tiempo restante estimado: {estimatedTimeRemaining} minutos
                  </div>
                )}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
};

const ModelCoordinationPanel: React.FC<{
  rlStatus: RLStatus | null;
  showAdvanced: boolean;
}> = ({ rlStatus, showAdvanced }) => {
  if (!rlStatus) return null;

  const modelData = [
    {
      name: 'Brain Max',
      weight: rlStatus.model_coordination.brain_max_weight,
      status: 'Activo',
      confidence: 95,
      color: '#3b82f6',
      gradient: 'linear-gradient(135deg, #3b82f6, #1d4ed8)'
    },
    {
      name: 'Brain Ultra',
      weight: rlStatus.model_coordination.brain_ultra_weight,
      status: 'Activo',
      confidence: 92,
      color: '#10b981',
      gradient: 'linear-gradient(135deg, #10b981, #059669)'
    },
    {
      name: 'Brain Predictor',
      weight: rlStatus.model_coordination.brain_predictor_weight,
      status: 'Activo',
      confidence: 90,
      color: '#f59e0b',
      gradient: 'linear-gradient(135deg, #f59e0b, #d97706)'
    },
    {
      name: 'MegaMind',
      weight: rlStatus.model_coordination.megamind_weight,
      status: 'Activo',
      confidence: 98,
      color: '#8b5cf6',
      gradient: 'linear-gradient(135deg, #8b5cf6, #7c3aed)'
    }
  ];

  return (
    <div className="glass-effect p-6 rounded-2xl border" style={{
      background: 'linear-gradient(135deg, rgba(26, 31, 46, 0.8), rgba(45, 55, 72, 0.8))',
      borderColor: 'rgba(139, 92, 246, 0.2)'
    }}>
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{
          background: 'linear-gradient(135deg, #8b5cf6, #7c3aed)',
          boxShadow: '0 8px 20px rgba(139, 92, 246, 0.3)'
        }}>
          <Network className="w-5 h-5 text-white" />
        </div>
        <div>
          <h3 className="text-xl font-bold text-white">Coordinación de Modelos IA</h3>
          <p className="text-purple-300 text-sm">Sistema colaborativo inteligente</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Gráfico de Ponderación */}
        <div>
          <h4 className="font-medium mb-3 text-white">Ponderación Actual de Modelos</h4>
          <div className="bg-gray-800/30 rounded-xl p-4">
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie
                  data={modelData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={5}
                  dataKey="weight"
                >
                  {modelData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip 
                  formatter={(value) => [`${value}%`, 'Peso']}
                  contentStyle={{
                    backgroundColor: 'rgba(26, 31, 46, 0.95)',
                    border: '1px solid rgba(139, 92, 246, 0.3)',
                    borderRadius: '12px',
                    color: 'white'
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Estado de Modelos */}
        <div>
          <h4 className="font-medium mb-3 text-white">Estado de Modelos</h4>
          <div className="space-y-3">
            {modelData.map((model, index) => (
              <div key={index} className="p-4 rounded-xl border transition-all duration-300 hover:scale-105" style={{
                background: 'linear-gradient(135deg, rgba(26, 31, 46, 0.8), rgba(45, 55, 72, 0.8))',
                borderColor: 'rgba(139, 92, 246, 0.2)'
              }}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <div 
                      className="w-4 h-4 rounded-full mr-3 shadow-lg"
                      style={{ 
                        background: model.gradient,
                        boxShadow: `0 0 10px ${model.color}40`
                      }}
                    ></div>
                    <div>
                      <div className="font-semibold text-white">{model.name}</div>
                      <div className="text-sm text-gray-400 font-medium">{model.status}</div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="font-bold text-xl text-white">{model.weight}%</div>
                    <div className="text-sm text-gray-400 font-medium">{model.confidence}% conf</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

const ActiveSignalsPanel: React.FC<{
  activeSignals: TradingSignal[];
  onExecuteSignal: (signal: TradingSignal) => void;
}> = ({ activeSignals, onExecuteSignal }) => {
  return (
    <div className="glass-effect p-6 rounded-2xl border mt-6" style={{
      background: 'linear-gradient(135deg, rgba(26, 31, 46, 0.8), rgba(45, 55, 72, 0.8))',
      borderColor: 'rgba(236, 72, 153, 0.2)'
    }}>
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{
          background: 'linear-gradient(135deg, #ec4899, #be185d)',
          boxShadow: '0 8px 20px rgba(236, 72, 153, 0.3)'
        }}>
          <Target className="w-5 h-5 text-white" />
        </div>
        <div>
          <h3 className="text-xl font-bold text-white">Señales Activas ({activeSignals.length})</h3>
          <p className="text-pink-300 text-sm">Oportunidades de trading en tiempo real</p>
        </div>
      </div>

      {activeSignals.length === 0 ? (
        <div className="text-center py-12 text-gray-400">
          <Target className="w-16 h-16 mx-auto mb-4 text-gray-600" />
          <p className="text-lg font-medium">No hay señales activas en este momento</p>
          <p className="text-sm">El RL Director está analizando el mercado...</p>
        </div>
      ) : (
        <div className="space-y-4">
          {activeSignals.map((signal, index) => (
            <div key={index} className="rounded-xl p-4 border transition-all duration-300 hover:scale-105" style={{
              background: 'linear-gradient(135deg, rgba(26, 31, 46, 0.8), rgba(45, 55, 72, 0.8))',
              borderColor: signal.signal === 'BUY' ? 'rgba(34, 197, 94, 0.4)' : 'rgba(239, 68, 68, 0.4)'
            }}>
              <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center">
                <div className="flex-1">
                  <div className="flex items-center mb-3">
                    <div className={`px-3 py-1 rounded-full text-xs font-medium mr-3 ${
                      signal.signal === 'BUY' 
                        ? 'bg-green-600/80 text-white border border-green-400/50' 
                        : 'bg-red-600/80 text-white border border-red-400/50'
                    }`}>
                      {signal.signal}
                    </div>
                    <span className="font-bold text-xl text-white">{signal.pair}</span>
                    <span className="ml-3 text-sm text-gray-400">
                      {new Date(signal.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                  
                  <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 mb-4">
                    <div className="text-center p-2 rounded-lg min-h-[60px] flex flex-col justify-center" style={{ backgroundColor: 'rgba(59, 130, 246, 0.1)' }}>
                      <div className="text-sm font-medium text-gray-400">Precio Entrada</div>
                      <div className="font-bold text-base text-blue-400 leading-tight">{signal.entry_price?.toFixed(5) || 'N/A'}</div>
                    </div>
                    <div className="text-center p-2 rounded-lg min-h-[60px] flex flex-col justify-center" style={{ backgroundColor: 'rgba(139, 92, 246, 0.1)' }}>
                      <div className="text-sm font-medium text-gray-400">Confianza</div>
                      <div className="font-bold text-base text-purple-400 leading-tight">{signal.confidence}%</div>
                    </div>
                    <div className="text-center p-2 rounded-lg min-h-[60px] flex flex-col justify-center" style={{ backgroundColor: 'rgba(236, 72, 153, 0.1)' }}>
                      <div className="text-sm font-medium text-gray-400">Posición</div>
                      <div className="font-bold text-base text-pink-400 leading-tight">{signal.position_size}%</div>
                    </div>
                    <div className="text-center p-2 rounded-lg min-h-[60px] flex flex-col justify-center" style={{ backgroundColor: 'rgba(239, 68, 68, 0.1)' }}>
                      <div className="text-sm font-medium text-gray-400">Stop Loss</div>
                      <div className="font-bold text-base text-red-400 leading-tight">{signal.stop_loss?.toFixed(5) || 'N/A'}</div>
                    </div>
                    <div className="text-center p-2 rounded-lg min-h-[60px] flex flex-col justify-center" style={{ backgroundColor: 'rgba(16, 185, 129, 0.1)' }}>
                      <div className="text-sm font-medium text-gray-400">Take Profit</div>
                      <div className="font-bold text-base text-green-400 leading-tight">{signal.take_profit?.toFixed(5) || 'N/A'}</div>
                    </div>
                  </div>

                  <div className="text-sm text-gray-300 font-medium p-3 rounded-lg" style={{ backgroundColor: 'rgba(59, 130, 246, 0.05)' }}>
                    <strong className="text-white">Razón:</strong> {signal.reasoning}
                  </div>
                </div>

                <div className="mt-4 sm:mt-0 sm:ml-4">
                  <button
                    onClick={() => onExecuteSignal(signal)}
                    className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white px-6 py-3 rounded-xl font-medium transition-all duration-300 shadow-lg hover:shadow-xl hover:scale-105"
                  >
                    <Rocket className="w-4 h-4 inline mr-2" />
                    Ejecutar
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

const RLPerformancePanel: React.FC<{
  rlPerformance: RLPerformance | null;
  showAdvanced: boolean;
}> = ({ rlPerformance, showAdvanced }) => {
  return (
    <div className="glass-effect p-6 rounded-2xl border" style={{
      background: 'linear-gradient(135deg, rgba(26, 31, 46, 0.8), rgba(45, 55, 72, 0.8))',
      borderColor: 'rgba(139, 92, 246, 0.2)'
    }}>
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{
          background: 'linear-gradient(135deg, #8b5cf6, #7c3aed)',
          boxShadow: '0 8px 20px rgba(139, 92, 246, 0.3)'
        }}>
          <BarChart3 className="w-5 h-5 text-white" />
        </div>
        <div>
          <h3 className="text-xl font-bold text-white">Rendimiento del RL Director</h3>
          <p className="text-purple-300 text-sm">Análisis detallado de rendimiento</p>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 mb-6">
        <div className="rounded-xl p-3 text-white shadow-lg min-h-[80px] flex flex-col justify-center" style={{ background: 'linear-gradient(135deg, #10b981, #059669)' }}>
          <div className="text-lg font-bold text-white leading-tight">
            {((rlPerformance?.total_return || 0) * 100).toFixed(1)}%
          </div>
          <div className="text-xs text-gray-200 mt-1">Profit Total</div>
        </div>

        <div className="rounded-xl p-3 text-white shadow-lg min-h-[80px] flex flex-col justify-center" style={{ background: 'linear-gradient(135deg, #3b82f6, #1d4ed8)' }}>
          <div className="text-lg font-bold text-white leading-tight">
            {((rlPerformance?.win_rate || 0) * 100).toFixed(1)}%
          </div>
          <div className="text-xs text-gray-200 mt-1">Win Rate</div>
        </div>

        <div className="rounded-xl p-3 text-white shadow-lg min-h-[80px] flex flex-col justify-center" style={{ background: 'linear-gradient(135deg, #f59e0b, #d97706)' }}>
          <div className="text-lg font-bold text-white leading-tight">
            {rlPerformance?.profit_factor.toFixed(2)}
          </div>
          <div className="text-xs text-gray-200 mt-1">Profit Factor</div>
        </div>

        <div className="rounded-xl p-3 text-white shadow-lg min-h-[80px] flex flex-col justify-center" style={{ background: 'linear-gradient(135deg, #8b5cf6, #7c3aed)' }}>
          <div className="text-lg font-bold text-white leading-tight">
            {rlPerformance?.sharpe_ratio.toFixed(2)}
          </div>
          <div className="text-xs text-gray-200 mt-1">Sharpe Ratio</div>
        </div>

        <div className="rounded-xl p-3 text-white shadow-lg min-h-[80px] flex flex-col justify-center" style={{ background: 'linear-gradient(135deg, #ef4444, #991b1b)' }}>
          <div className="text-lg font-bold text-white leading-tight">
            {((rlPerformance?.max_drawdown || 0) * 100).toFixed(1)}%
          </div>
          <div className="text-xs text-gray-200 mt-1">Max Drawdown</div>
        </div>

        <div className="rounded-xl p-3 text-white shadow-lg min-h-[80px] flex flex-col justify-center" style={{ background: 'linear-gradient(135deg, #6366f1, #4f46e5)' }}>
          <div className="text-lg font-bold text-white leading-tight">
            {rlPerformance?.total_trades}
          </div>
          <div className="text-xs text-gray-200 mt-1">Trades</div>
        </div>
      </div>

      {showAdvanced && (
        <div className="border-t border-gray-700 pt-6">
          <h4 className="font-medium mb-3 text-white flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-yellow-400" />
            Análisis de Rendimiento
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="rounded-xl p-4" style={{ backgroundColor: 'rgba(26, 31, 46, 0.8)' }}>
              <h5 className="font-medium mb-3 text-white">Evaluación General</h5>
              <div className="space-y-3 text-sm">
                <div className="flex justify-between items-center">
                  <span className="font-medium text-gray-300">Rendimiento:</span>
                  <span className={`font-semibold ${
                    (rlPerformance?.total_return || 0) > 0.1 ? 'text-green-400' : 
                    (rlPerformance?.total_return || 0) > 0 ? 'text-yellow-400' : 'text-red-400'
                  }`}>
                    {(rlPerformance?.total_return || 0) > 0.1 ? 'Excelente' : 
                    (rlPerformance?.total_return || 0) > 0 ? 'Bueno' : 'Necesita Mejora'}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="font-medium text-gray-300">Consistencia:</span>
                  <span className={`font-semibold ${
                    (rlPerformance?.sharpe_ratio || 0) > 1.5 ? 'text-green-400' : 
                    (rlPerformance?.sharpe_ratio || 0) > 1 ? 'text-yellow-400' : 'text-red-400'
                  }`}>
                    {(rlPerformance?.sharpe_ratio || 0) > 1.5 ? 'Alta' : 
                    (rlPerformance?.sharpe_ratio || 0) > 1 ? 'Media' : 'Baja'}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="font-medium text-gray-300">Gestión de Riesgo:</span>
                  <span className={`font-semibold ${
                    (rlPerformance?.max_drawdown || 0) < 0.1 ? 'text-green-400' : 
                    (rlPerformance?.max_drawdown || 0) < 0.2 ? 'text-yellow-400' : 'text-red-400'
                  }`}>
                    {(rlPerformance?.max_drawdown || 0) < 0.1 ? 'Excelente' : 
                    (rlPerformance?.max_drawdown || 0) < 0.2 ? 'Buena' : 'Necesita Mejora'}
                  </span>
                </div>
              </div>
            </div>
            <div className="rounded-xl p-4" style={{ backgroundColor: 'rgba(26, 31, 46, 0.8)' }}>
              <h5 className="font-medium mb-3 text-white">Recomendaciones</h5>
              <div className="space-y-3 text-sm">
                {(rlPerformance?.total_return || 0) < 0.05 && (
                  <div className="flex items-center text-red-400 font-medium">
                    <span className="w-2 h-2 bg-red-500 rounded-full mr-2"></span>
                    Considerar reentrenamiento del agente
                  </div>
                )}
                {(rlPerformance?.sharpe_ratio || 0) < 1 && (
                  <div className="flex items-center text-yellow-400 font-medium">
                    <span className="w-2 h-2 bg-yellow-500 rounded-full mr-2"></span>
                    Optimizar gestión de riesgo
                  </div>
                )}
                {(rlPerformance?.max_drawdown || 0) > 0.15 && (
                  <div className="flex items-center text-red-400 font-medium">
                    <span className="w-2 h-2 bg-red-500 rounded-full mr-2"></span>
                    Reducir exposición al riesgo
                  </div>
                )}
                {(rlPerformance?.win_rate || 0) < 0.6 && (
                  <div className="flex items-center text-yellow-400 font-medium">
                    <span className="w-2 h-2 bg-yellow-500 rounded-full mr-2"></span>
                    Revisar criterios de entrada
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {showAdvanced && rlPerformance && (
        <div className="mt-6 border-t border-gray-700 pt-6">
          <h4 className="font-medium mb-4 text-white">Rendimiento por Modelo</h4>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <div className="text-center p-4 rounded-xl" style={{ backgroundColor: 'rgba(26, 31, 46, 0.8)' }}>
              <div className="text-xl font-bold text-white">
                {(rlPerformance.model_performance.brain_max.accuracy * 100).toFixed(1)}%
              </div>
              <div className="text-sm font-medium text-gray-400">Brain Max</div>
            </div>
            <div className="text-center p-4 rounded-xl" style={{ backgroundColor: 'rgba(26, 31, 46, 0.8)' }}>
              <div className="text-xl font-bold text-white">
                {(rlPerformance.model_performance.brain_ultra.accuracy * 100).toFixed(1)}%
              </div>
              <div className="text-sm font-medium text-gray-400">Brain Ultra</div>
            </div>
            <div className="text-center p-4 rounded-xl" style={{ backgroundColor: 'rgba(26, 31, 46, 0.8)' }}>
              <div className="text-xl font-bold text-white">
                {(rlPerformance.model_performance.brain_predictor.accuracy * 100).toFixed(1)}%
              </div>
              <div className="text-sm font-medium text-gray-400">Brain Predictor</div>
            </div>
            <div className="text-center p-4 rounded-xl" style={{ backgroundColor: 'rgba(26, 31, 46, 0.8)' }}>
              <div className="text-xl font-bold text-white">
                {(rlPerformance.model_performance.megamind.accuracy * 100).toFixed(1)}%
              </div>
              <div className="text-sm font-medium text-gray-400">MegaMind</div>
            </div>
            <div className="text-center p-4 rounded-xl" style={{ backgroundColor: 'rgba(26, 31, 46, 0.8)' }}>
              <div className="text-xl font-bold text-white">
                {(rlPerformance.model_performance.brain_max.confidence * 100).toFixed(1)}%
              </div>
              <div className="text-sm font-medium text-gray-400">Confianza</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

const AIComparisonPanel: React.FC = () => {
  const comparisonData = [
    {
      method: 'IA Tradicional',
      accuracy: 76.8,
      profit: 12.5,
      sharpe: 1.32,
      trades: 45,
      description: 'Brain Max + Ultra + Predictor',
      color: 'linear-gradient(135deg, #3b82f6, #1d4ed8)'
    },
    {
      method: 'MegaMind Ensemble',
      accuracy: 79.2,
      profit: 14.8,
      sharpe: 1.45,
      trades: 38,
      description: 'Ensemble de 3 modelos',
      color: 'linear-gradient(135deg, #8b5cf6, #7c3aed)'
    },
    {
      method: 'RL Director',
      accuracy: 82.1,
      profit: 18.7,
      sharpe: 1.68,
      trades: 32,
      description: 'Coordinación inteligente',
      color: 'linear-gradient(135deg, #ec4899, #be185d)'
    }
  ];

  return (
    <div className="glass-effect p-6 rounded-2xl border" style={{
      background: 'linear-gradient(135deg, rgba(26, 31, 46, 0.8), rgba(45, 55, 72, 0.8))',
      borderColor: 'rgba(139, 92, 246, 0.2)'
    }}>
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{
          background: 'linear-gradient(135deg, #8b5cf6, #7c3aed)',
          boxShadow: '0 8px 20px rgba(139, 92, 246, 0.3)'
        }}>
          <TrendingUp className="w-5 h-5 text-white" />
        </div>
        <div>
          <h3 className="text-xl font-bold text-white">Comparación de Métodos AITRADERX</h3>
          <p className="text-purple-300 text-sm">Análisis comparativo de eficiencia</p>
        </div>
      </div>

      <div className="space-y-4 mb-6">
        {comparisonData.map((method, index) => (
          <div key={index} className="p-4 rounded-xl border transition-all duration-300 hover:scale-105" style={{
            background: 'linear-gradient(135deg, rgba(26, 31, 46, 0.8), rgba(45, 55, 72, 0.8))',
            borderColor: 'rgba(139, 92, 246, 0.2)'
          }}>
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{
                  background: method.color,
                  boxShadow: '0 4px 12px rgba(139, 92, 246, 0.3)'
                }}>
                  <Brain className="w-4 h-4 text-white" />
                </div>
                <div>
                  <h4 className="font-semibold text-white text-lg">{method.method}</h4>
                  <p className="text-sm text-gray-400">{method.description}</p>
                </div>
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold text-white">{method.accuracy}%</div>
                <div className="text-sm text-gray-400">Precisión</div>
              </div>
            </div>
            
            <div className="grid grid-cols-3 gap-3">
              <div className="text-center p-2 rounded-lg min-h-[50px] flex flex-col justify-center" style={{ backgroundColor: 'rgba(16, 185, 129, 0.1)' }}>
                <div className="text-sm font-bold text-green-400 leading-tight">{method.profit}%</div>
                <div className="text-xs text-gray-400">Profit Anual</div>
              </div>
              <div className="text-center p-2 rounded-lg min-h-[50px] flex flex-col justify-center" style={{ backgroundColor: 'rgba(139, 92, 246, 0.1)' }}>
                <div className="text-sm font-bold text-purple-400 leading-tight">{method.sharpe}</div>
                <div className="text-xs text-gray-400">Sharpe Ratio</div>
              </div>
              <div className="text-center p-2 rounded-lg min-h-[50px] flex flex-col justify-center" style={{ backgroundColor: 'rgba(59, 130, 246, 0.1)' }}>
                <div className="text-sm font-bold text-blue-400 leading-tight">{method.trades}</div>
                <div className="text-xs text-gray-400">Trades/Mes</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="p-4 rounded-xl border" style={{ 
        backgroundColor: 'rgba(26, 31, 46, 0.8)',
        borderColor: 'rgba(236, 72, 153, 0.3)'
      }}>
        <div className="flex items-center gap-3 mb-3">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{
            background: 'linear-gradient(135deg, #ec4899, #be185d)',
            boxShadow: '0 4px 12px rgba(236, 72, 153, 0.3)'
          }}>
            <Award className="w-4 h-4 text-white" />
          </div>
          <h4 className="font-medium text-pink-400 text-lg">🏆 Resumen de Rendimiento</h4>
        </div>
        <p className="text-gray-300 text-sm leading-relaxed">
          El <strong className="text-white">RL Director</strong> muestra el mejor rendimiento general con <strong className="text-pink-400">82.1% de precisión</strong> 
          y <strong className="text-pink-400">18.7% de profit anual</strong>. Coordina inteligentemente los 4 modelos existentes para 
          maximizar el rendimiento y minimizar el riesgo.
        </p>
      </div>
    </div>
  );
};

// Interfaces para la configuración RL
interface RLConfigurationLimits {
  max_drawdown_percentage: {
    min: number;
    max: number;
    default: number;
    step: number;
  };
  max_position_size_percentage: {
    min: number;
    max: number;
    default: number;
    step: number;
  };
  min_confidence_threshold: {
    min: number;
    max: number;
    default: number;
    step: number;
  };
  retraining_frequency: {
    options: Array<{ value: string; label: string }>;
    default: string | null;
    available: boolean;
  };
  retraining_enabled: {
    default: boolean;
    available: boolean;
  };
  user_subscription: string;
}

interface RLConfiguration {
  max_drawdown_percentage: number;
  max_position_size_percentage: number;
  min_confidence_threshold: number;
  retraining_frequency: string;
  retraining_enabled: boolean;
}

const AdvancedConfigurationPanel: React.FC<{ showAdvanced: boolean }> = ({ showAdvanced }) => {
  const { user } = useAuth();
  const { subscription } = useAuth();
  const [configLimits, setConfigLimits] = useState<RLConfigurationLimits | null>(null);
  const [currentConfig, setCurrentConfig] = useState<RLConfiguration | null>(null);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);

  // Obtener límites de configuración
  useEffect(() => {
    const loadConfigurationLimits = async () => {
      try {
        setLoading(true);
        const userId = user?.id || "4dabfd30-483d-4fa0-a8d0-bd151a46340f"; // Obtener del contexto de auth
        
        const response = await fetch(`/api/rl/config-limits-new`);
        if (response.ok) {
          const data = await response.json();
          if (data.success) {
            setConfigLimits(data.limits);
          }
        }
      } catch (error) {
        console.error('Error loading configuration limits:', error);
      } finally {
        setLoading(false);
      }
    };

    // Cargar configuración actual del usuario
    const loadCurrentConfiguration = async () => {
      try {
        const userId = user?.id || "4dabfd30-483d-4fa0-a8d0-bd151a46340f"; // Obtener del contexto de auth
        
        const response = await fetch(`/api/rl/configuration/${userId}`);
        if (response.ok) {
          const data = await response.json();
          if (data.success) {
            setCurrentConfig(data.configuration);
          }
        }
      } catch (error) {
        console.error('Error loading current configuration:', error);
      }
    };

    if (showAdvanced) {
      loadConfigurationLimits();
      loadCurrentConfiguration();
    }
  }, [showAdvanced]);

  const handleSaveConfiguration = async () => {
    if (!currentConfig) return;
    
    try {
      setSaving(true);
      const userId = user?.id || "4dabfd30-483d-4fa0-a8d0-bd151a46340f"; // Obtener del contexto de auth
      
      const response = await fetch(`/api/rl/configuration/${userId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(currentConfig),
      });
      
      if (response.ok) {
        const data = await response.json();
        if (data.success) {
          // Mostrar mensaje de éxito
          console.log('Configuración guardada exitosamente');
        }
      }
    } catch (error) {
      console.error('Error saving configuration:', error);
    } finally {
      setSaving(false);
    }
  };

  const handleResetConfiguration = async () => {
    try {
      setSaving(true);
      const userId = user?.id || "4dabfd30-483d-4fa0-a8d0-bd151a46340f"; // Obtener del contexto de auth
      
      const response = await fetch(`/api/rl/configuration/${userId}/reset`, {
        method: 'POST',
      });
      
      if (response.ok) {
        const data = await response.json();
        if (data.success) {
          setCurrentConfig(data.configuration);
        }
      }
    } catch (error) {
      console.error('Error resetting configuration:', error);
    } finally {
      setSaving(false);
    }
  };

  if (!showAdvanced) return null;
  
  if (loading) {
    return (
      <div className="glass-effect p-6 rounded-2xl border" style={{
        background: 'linear-gradient(135deg, rgba(26, 31, 46, 0.8), rgba(45, 55, 72, 0.8))',
        borderColor: 'rgba(139, 92, 246, 0.2)'
      }}>
        <div className="flex items-center justify-center py-8">
          <div className="text-white">Cargando configuración...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="glass-effect p-6 rounded-2xl border" style={{
      background: 'linear-gradient(135deg, rgba(26, 31, 46, 0.8), rgba(45, 55, 72, 0.8))',
      borderColor: 'rgba(139, 92, 246, 0.2)'
    }}>
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{
          background: 'linear-gradient(135deg, #8b5cf6, #7c3aed)',
          boxShadow: '0 8px 20px rgba(139, 92, 246, 0.3)'
        }}>
          <Settings className="w-5 h-5 text-white" />
        </div>
        <div>
          <h3 className="text-xl font-bold text-white">Configuración Avanzada</h3>
          <p className="text-purple-300 text-sm">Ajustes de rendimiento y seguridad</p>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h4 className="font-medium mb-3 text-white">Parámetros de Riesgo</h4>
          <div className="space-y-3">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">
                Máximo Drawdown Permitido
              </label>
              <input 
                type="range" 
                min={configLimits?.max_drawdown_percentage.min || 5} 
                max={configLimits?.max_drawdown_percentage.max || 25} 
                step={configLimits?.max_drawdown_percentage.step || 1}
                value={currentConfig?.max_drawdown_percentage || configLimits?.max_drawdown_percentage.default || 15}
                onChange={(e) => setCurrentConfig(prev => prev ? {...prev, max_drawdown_percentage: parseFloat(e.target.value)} : null)}
                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider"
              />
              <div className="flex justify-between text-xs text-gray-500">
                <span>{configLimits?.max_drawdown_percentage.min || 5}%</span>
                <span>{currentConfig?.max_drawdown_percentage || configLimits?.max_drawdown_percentage.default || 15}%</span>
                <span>{configLimits?.max_drawdown_percentage.max || 25}%</span>
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">
                Tamaño Máximo de Posición
              </label>
              <input 
                type="range" 
                min={configLimits?.max_position_size_percentage.min || 1} 
                max={configLimits?.max_position_size_percentage.max || 10} 
                step={configLimits?.max_position_size_percentage.step || 0.5}
                value={currentConfig?.max_position_size_percentage || configLimits?.max_position_size_percentage.default || 5}
                onChange={(e) => setCurrentConfig(prev => prev ? {...prev, max_position_size_percentage: parseFloat(e.target.value)} : null)}
                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider"
              />
              <div className="flex justify-between text-xs text-gray-500">
                <span>{configLimits?.max_position_size_percentage.min || 1}%</span>
                <span>{currentConfig?.max_position_size_percentage || configLimits?.max_position_size_percentage.default || 5}%</span>
                <span>{configLimits?.max_position_size_percentage.max || 10}%</span>
              </div>
            </div>
          </div>
        </div>
        
        <div>
          <h4 className="font-medium mb-3 text-white">Configuración de Modelos</h4>
          <div className="space-y-3">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">
                Umbral de Confianza Mínima
              </label>
              <input 
                type="range" 
                min={configLimits?.min_confidence_threshold.min || 50} 
                max={configLimits?.min_confidence_threshold.max || 90} 
                step={configLimits?.min_confidence_threshold.step || 5}
                value={currentConfig?.min_confidence_threshold || configLimits?.min_confidence_threshold.default || 70}
                onChange={(e) => setCurrentConfig(prev => prev ? {...prev, min_confidence_threshold: parseFloat(e.target.value)} : null)}
                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider"
              />
              <div className="flex justify-between text-xs text-gray-500">
                <span>{configLimits?.min_confidence_threshold.min || 50}%</span>
                <span>{currentConfig?.min_confidence_threshold || configLimits?.min_confidence_threshold.default || 70}%</span>
                <span>{configLimits?.min_confidence_threshold.max || 90}%</span>
              </div>
            </div>
            
            {configLimits?.retraining_frequency.available && (
              <>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">
                    Frecuencia de Reentrenamiento
                  </label>
                  <select 
                    value={currentConfig?.retraining_frequency || configLimits?.retraining_frequency.default || ''}
                    onChange={(e) => setCurrentConfig(prev => prev ? {...prev, retraining_frequency: e.target.value} : null)}
                    className="w-full rounded-lg text-white bg-gray-800/50 border border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    {configLimits.retraining_frequency.options.map((option) => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </div>
                
                <div>
                  <label className="flex items-center space-x-2">
                    <input 
                      type="checkbox"
                      checked={currentConfig?.retraining_enabled || configLimits?.retraining_enabled.default || false}
                      onChange={(e) => setCurrentConfig(prev => prev ? {...prev, retraining_enabled: e.target.checked} : null)}
                      className="rounded border-gray-600 bg-gray-800/50 text-blue-500 focus:ring-blue-500"
                    />
                    <span className="text-sm font-medium text-gray-300">
                      Activar Reentrenamiento Automático
                    </span>
                  </label>
                </div>
              </>
            )}
            
            {!configLimits?.retraining_frequency.available && (
              <div className="p-3 bg-yellow-500/10 border border-yellow-500/20 rounded-lg">
                <p className="text-yellow-300 text-sm">
                  ⚠️ El reentrenamiento automático no está disponible en tu plan actual. 
                  {subscription?.planType === 'premium' ? ' Actualiza a Institutional para acceder a más opciones.' : ' Actualiza tu suscripción para acceder a esta función.'}
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
      
      <div className="mt-6 flex justify-end space-x-3">
        <button 
          onClick={handleResetConfiguration}
          disabled={saving}
          className="px-4 py-2 rounded-lg text-gray-400 hover:bg-gray-800/50 transition-colors disabled:opacity-50"
        >
          {saving ? 'Restaurando...' : 'Restaurar Valores'}
        </button>
        <button 
          onClick={handleSaveConfiguration}
          disabled={saving || !currentConfig}
          className="px-4 py-2 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white rounded-lg font-medium transition-colors shadow-lg hover:shadow-xl hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {saving ? 'Guardando...' : 'Guardar Configuración'}
        </button>
      </div>
    </div>
  );
};

// Servicios de API
export const rlService = {
  async getRLStatus() {
    const response = await fetch('/api/rl/status');
    return response.json();
  },

  async trainAgent(episodes: number) {
    const response = await fetch(`/api/rl/train?episodes=${episodes}`, {
      method: 'POST'
    });
    return response.json();
  },

  async getRLPrediction(marketData: any) {
    const response = await fetch('/api/rl/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(marketData)
    });
    return response.json();
  },

  async getRLPerformance() {
    const response = await fetch('/api/rl/performance');
    return response.json();
  },

  async getActiveSignals() {
    const response = await fetch('/api/rl/active-signals');
    return response.json();
  },

  async executeSignal(signal: TradingSignal) {
    const response = await fetch('/api/rl/execute-signal', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(signal)
    });
    return response.json();
  }
};

// Estilos CSS personalizados para los sliders
const sliderStyles = `
  .slider::-webkit-slider-thumb {
    appearance: none;
    height: 20px;
    width: 20px;
    border-radius: 50%;
    background: #3b82f6;
    cursor: pointer;
    border: 2px solid #ffffff;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
  }

  .slider::-moz-range-thumb {
    height: 20px;
    width: 20px;
    border-radius: 50%;
    background: #3b82f6;
    cursor: pointer;
    border: 2px solid #ffffff;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
  }

  .slider::-webkit-slider-track {
    background: #e5e7eb;
    border-radius: 8px;
    height: 8px;
  }

  .slider::-moz-range-track {
    background: #e5e7eb;
    border-radius: 8px;
    height: 8px;
  }
`;

// Agregar estilos al head del documento
if (typeof document !== 'undefined') {
  const style = document.createElement('style');
  style.textContent = sliderStyles;
  document.head.appendChild(style);
}
