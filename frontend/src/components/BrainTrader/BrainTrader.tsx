import React, { useState, useEffect } from 'react';
import { 
  Brain, 
  TrendingUp, 
  TrendingDown, 
  Target, 
  AlertTriangle,
  Info,
  BarChart3,
  Zap,
  Shield,
  Clock,
  DollarSign,
  Activity,
  CheckCircle,
  XCircle,
  Crown,
  Star,
  Settings,
  RefreshCw,
  Play,
  Pause,
  Eye,
  EyeOff,
  History,
  Calendar,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon
} from 'lucide-react';
import { useAuth } from '../../contexts/AuthContext';
import { useFeatureAccess } from '../../hooks/useFeatureAccess';
import { useBrainTraderApi } from '../../hooks/useBrainTraderApi';
import { useYahooMarketData } from '../../hooks/useYahooMarketData';
import { apiService } from '../../services/api';
import type { PredictionHistoryItem, PredictionLimits, UserStats, BrainTraderSignal } from '../../services/api';


interface BrainTraderProps {}

interface ModelInfo {
  brainType: 'brain_max' | 'brain_ultra' | 'brain_predictor' | 'mega_mind';
  pair: string;
  style: string;
  accuracy: number;
  lastUpdate: string;
  status: 'active' | 'training' | 'error';
}

interface Prediction {
  pair: string;
  direction: 'up' | 'down' | 'sideways';
  confidence: number;
  target_price: number;
  timeframe: string;
  reasoning: string;
  brain_type: string;
  timestamp: string;
  expires_at?: string;
}

interface Signal {
  pair: string;
  type: 'buy' | 'sell' | 'hold';
  strength: 'strong' | 'medium' | 'weak';
  confidence: number;
  entry_price: number;
  stop_loss: number;
  take_profit: number;
  brain_type: string;
  timestamp: string;
}

interface Trend {
  pair: string;
  direction: 'bullish' | 'bearish' | 'neutral';
  strength: number;
  timeframe: string;
  support: number;
  resistance: number;
  description: string;
  brain_type: string;
  timestamp: string;
}

export const BrainTrader: React.FC<BrainTraderProps> = () => {
  const { subscription } = useAuth();
  const { checkAccess, checkFeature } = useFeatureAccess();
  
  // Hook para obtener precios actuales de mercado
  const { data: marketData, loading: marketLoading, error: marketError } = useYahooMarketData(['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']);
  
  const {
    predictions,
    signals,
    trends,
    megaMindPredictions,
    megaMindCollaboration,
    megaMindArena,
    megaMindPerformance,
    availableBrains,
    defaultBrain,
    loading,
    errors,
    loadPredictions,
    loadPredictionsWithIntervals,
    loadSignals,
    loadTrends,
    loadMegaMindPredictions,
    loadMegaMindCollaboration,
    loadMegaMindArena,
    loadMegaMindPerformance,
    generateManualSignal,
    getSignalIntervals,
    getNextPredictionTime,
    addSignal,
    refreshAll,
  } = useBrainTraderApi(subscription?.planType || 'starter');
  
  // Estados principales
  const [selectedPair, setSelectedPair] = useState('EURUSD');
  const [selectedStyle, setSelectedStyle] = useState('day_trading');
  const [activeBrain, setActiveBrain] = useState<'brain_max' | 'brain_ultra' | 'brain_predictor' | 'mega_mind'>('brain_max');
  const [isAutoTrading, setIsAutoTrading] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Estados de datos
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);

  // Estados del sistema de predicciones
  const [currentPrediction, setCurrentPrediction] = useState<Prediction | null>(null);
  const [isGeneratingPrediction, setIsGeneratingPrediction] = useState(false);
  const [predictionLimits, setPredictionLimits] = useState<PredictionLimits | null>(null);
  const [predictionHistory, setPredictionHistory] = useState<PredictionHistoryItem[]>([]);
  const [userStats, setUserStats] = useState<UserStats | null>(null);
  const [activeTab, setActiveTab] = useState<'predictions' | 'signals' | 'trends' | 'history'>('predictions');
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [isLoadingStats, setIsLoadingStats] = useState(false);

  // Estados para el sistema de señales manuales
  const [isGeneratingSignal, setIsGeneratingSignal] = useState(false);
  const [signalQuality, setSignalQuality] = useState<number | null>(null);
  const [signalMessage, setSignalMessage] = useState<string>('');
  const [signalIntervals, setSignalIntervals] = useState<any>(null);
  const [isValidSignalTime, setIsValidSignalTime] = useState(false);

  // Estados para el sistema de intervalos
  const [intervalInfo, setIntervalInfo] = useState<any>(null);
  const [isValidPredictionTime, setIsValidPredictionTime] = useState(false);
  const [nextPredictionTime, setNextPredictionTime] = useState<string>('');
  const [timeUntilNext, setTimeUntilNext] = useState<number>(0);
  const [useIntervals, setUseIntervals] = useState(false);

  // Configuración según suscripción
  const getAvailablePairs = () => {
    if (!subscription || subscription.status !== 'active') {
      return ['EURUSD']; // Starter
    }
    
    switch (subscription.planType) {
      case 'starter':
        return ['EURUSD'];
      case 'trader':
        return ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD'];
      case 'expert':
      case 'premium':
      case 'institutional':
        return ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'EURGBP', 'GBPJPY', 'EURJPY'];
      default:
        return ['EURUSD'];
    }
  };

  const getAvailableStyles = () => {
    if (!subscription || subscription.status !== 'active') {
      return ['day_trading']; // Starter
    }
    
    switch (subscription.planType) {
      case 'starter':
        return ['day_trading'];
      case 'trader':
      case 'expert':
      case 'premium':
      case 'institutional':
        return ['scalping', 'day_trading', 'swing_trading', 'position_trading'];
      default:
        return ['day_trading'];
    }
  };

  const getAvailableBrains = () => {
    if (!subscription || subscription.status !== 'active') {
      return ['brain_max']; // Starter solo Brain Max
    }
    
    switch (subscription.planType) {
      case 'starter':
        return ['brain_max']; // Starter solo Brain Max
      case 'trader':
        return ['brain_max', 'mega_mind'];
      case 'expert':
        return ['brain_max', 'brain_ultra', 'mega_mind'];
      case 'premium':
        return ['brain_max', 'brain_ultra', 'brain_predictor', 'mega_mind'];
      case 'institutional':
        return ['brain_max', 'brain_ultra', 'brain_predictor', 'mega_mind'];
      default:
        return ['brain_max'];
    }
  };

  // Configuración especial para MEGA MIND
  const isMegaMindAvailable = () => {
    return true; // Disponible en todos los planes para testing
  };

  const getMegaMindFeatures = () => {
    return {
      brainCollaboration: true,
      brainFusion: true,
      brainArena: true,
      brainEvolution: true,
      brainOrchestration: true,
      multiTimeframeAnalysis: true,
      crossAssetCorrelation: true,
      institutionalRiskManagement: true,
      advancedPortfolioOptimization: true
    };
  };

  // Funciones según plan de suscripción
  const getAvailableFeatures = () => {
    if (!subscription || subscription.status !== 'active') {
      return {
        brainMax: true,
        brainUltra: false,
        brainPredictor: false,
        megaMind: false,
        multiTimeframe: false,
        crossAsset: false,
        economicCalendar: false,
        autoTraining: false,
        customModels: false,
        apiAccess: false
      };
    }

    switch (subscription.planType) {
      case 'starter':
        return {
          brainMax: true,
          brainUltra: false,
          brainPredictor: false,
          megaMind: false,
          multiTimeframe: false,
          crossAsset: false,
          economicCalendar: false,
          autoTraining: false,
          customModels: false,
          apiAccess: false
        };
      case 'trader':
        return {
          brainMax: true,
          brainUltra: false,
          brainPredictor: false,
          megaMind: false,
          multiTimeframe: false,
          crossAsset: false,
          economicCalendar: false,
          autoTraining: false,
          customModels: false,
          apiAccess: false
        };
      case 'expert':
        return {
          brainMax: true,
          brainUltra: true,
          brainPredictor: false,
          megaMind: false,
          multiTimeframe: true,
          crossAsset: true,
          economicCalendar: true,
          autoTraining: true,
          customModels: false,
          apiAccess: false
        };
      case 'premium':
        return {
          brainMax: true,
          brainUltra: true,
          brainPredictor: true,
          megaMind: false,
          multiTimeframe: true,
          crossAsset: true,
          economicCalendar: true,
          autoTraining: true,
          customModels: true,
          apiAccess: true
        };
      case 'institutional':
        return {
          brainMax: true,
          brainUltra: true,
          brainPredictor: true,
          megaMind: true,
          multiTimeframe: true,
          crossAsset: true,
          economicCalendar: true,
          autoTraining: true,
          customModels: true,
          apiAccess: true
        };
      default:
        return {
          brainMax: true,
          brainUltra: false,
          brainPredictor: false,
          megaMind: false,
          multiTimeframe: false,
          crossAsset: false,
          economicCalendar: false,
          autoTraining: false,
          customModels: false,
          apiAccess: false
        };
    }
  };

  const getPlanLimitations = () => {
    if (!subscription || subscription.status !== 'active') {
      return {
        maxPredictionsPerDay: 5,
        maxPairs: 1,
        maxTimeframes: 1,
        maxBacktests: 5,
        supportLevel: 'community'
      };
    }

    switch (subscription.planType) {
      case 'starter':
        return {
          maxPredictionsPerDay: 5,
          maxPairs: 1,
          maxTimeframes: 1,
          maxBacktests: 5,
          supportLevel: 'community'
        };
      case 'trader':
        return {
          maxPredictionsPerDay: 20,
          maxPairs: 5,
          maxTimeframes: 2,
          maxBacktests: 20,
          supportLevel: 'email'
        };
      case 'expert':
        return {
          maxPredictionsPerDay: 50,
          maxPairs: 50,
          maxTimeframes: 5,
          maxBacktests: 100,
          supportLevel: 'email'
        };
      case 'premium':
        return {
          maxPredictionsPerDay: 100,
          maxPairs: 1000,
          maxTimeframes: 10,
          maxBacktests: 500,
          supportLevel: 'phone'
        };
      case 'institutional':
        return {
          maxPredictionsPerDay: -1, // Sin límite
          maxPairs: 5000,
          maxTimeframes: 15,
          maxBacktests: 2000,
          supportLevel: 'dedicated'
        };
      default:
        return {
          maxPredictionsPerDay: 5,
          maxPairs: 1,
          maxTimeframes: 1,
          maxBacktests: 5,
          supportLevel: 'community'
        };
    }
  };

  // Función para obtener el precio actual del par seleccionado
  const getCurrentPrice = (pair: string) => {
    if (!marketData || !marketData[pair]) {
      return null;
    }
    return parseFloat(marketData[pair].price);
  };

  // Función para obtener el cambio de precio
  const getPriceChange = (pair: string) => {
    if (!marketData || !marketData[pair]) {
      return null;
    }
    return {
      change: parseFloat(marketData[pair].change),
      changePercent: parseFloat(marketData[pair].changePercent)
    };
  };

  // Cargar datos del modelo
  const loadModelData = async () => {
    try {
      // Actualizar información del modelo
      const currentModelInfo: ModelInfo = {
        brainType: activeBrain,
        pair: selectedPair,
        style: selectedStyle,
        accuracy: activeBrain === 'mega_mind' ? 95 : 85, // Valores fijos para simplicidad
        lastUpdate: new Date().toISOString(),
        status: 'active'
      };
      
      setModelInfo(currentModelInfo);
      
      // Cargar datos según el cerebro activo
      if (activeBrain === 'mega_mind') {
        await Promise.all([
          loadMegaMindPredictions(selectedPair, selectedStyle),
          loadMegaMindCollaboration(selectedPair),
          loadMegaMindArena(selectedPair),
          loadMegaMindPerformance(),
        ]);
      } else {
        await Promise.all([
          loadPredictions(activeBrain, selectedPair, selectedStyle),
          loadSignals(activeBrain, selectedPair),
          loadTrends(activeBrain, selectedPair),
        ]);
      }
      
    } catch (error) {
      console.error('Error cargando datos del modelo:', error);
    }
  };

  // Efectos
  useEffect(() => {
    loadModelData();
  }, [selectedPair, selectedStyle, activeBrain]);

  // Efecto para cargar datos del sistema de predicciones
  useEffect(() => {
    const loadPredictionData = async () => {
      await Promise.all([
        loadPredictionLimits(),
        loadActivePrediction(),
        loadPredictionHistory(),
        loadUserStats()
      ]);
    };
    
    loadPredictionData();
  }, [selectedPair, selectedStyle, activeBrain]);

  // Efecto para cargar intervalos de señales
  useEffect(() => {
    if (activeTab === 'signals') {
      console.log('Cargando intervalos de señales...');
      loadSignalIntervals();
    }
  }, [activeTab, selectedStyle, activeBrain]);

  // Efecto para cargar información de intervalos de predicciones
  useEffect(() => {
    if (useIntervals) {
      console.log('Cargando información de intervalos de predicciones...');
      loadIntervalInfo();
    }
  }, [selectedStyle, activeBrain, useIntervals]);

  // Debug: Log cuando cambian las señales
  useEffect(() => {
    console.log('🔍 Señales actualizadas:', signals.length, 'signals:', signals);
  }, [signals]);

  // Función para mostrar errores de API
  const hasApiErrors = () => {
    return Object.values(errors).some(error => error !== null);
  };

  const getApiErrorMessages = () => {
    return Object.entries(errors)
      .filter(([_, error]) => error !== null)
      .map(([key, error]) => `${key}: ${error}`)
      .join(', ');
  };

  // Funciones del sistema de predicciones
  const getPlanLimits = () => {
    if (!predictionLimits) {
      // Fallback: usar límites del plan actual
      const planLimits = getPlanLimitations();
      return { 
        used: 0, 
        limit: planLimits.maxPredictionsPerDay, 
        canGenerate: true, 
        hasUnlimited: planLimits.maxPredictionsPerDay === -1 
      };
    }
    
    const hasUnlimited = predictionLimits.has_unlimited || predictionLimits.max_predictions_per_day === -1;
    
    if (hasUnlimited) {
      return { 
        used: 0, 
        limit: -1, // Ilimitado
        canGenerate: true, 
        hasUnlimited: true 
      };
    }
    
    const used = predictionLimits.max_predictions_per_day - predictionLimits.remaining_predictions;
    const limit = predictionLimits.max_predictions_per_day;
    const canGenerate = predictionLimits.can_generate;
    
    return { used, limit, canGenerate, hasUnlimited: false };
  };

  const canGeneratePrediction = () => {
    const { canGenerate } = getPlanLimits();
    return canGenerate && !isGeneratingPrediction;
  };

  const generatePrediction = async () => {
    if (!canGeneratePrediction()) return;
    
    setIsGeneratingPrediction(true);
    try {
      const response = await apiService.generatePrediction(
        selectedPair,
        activeBrain,
        selectedStyle
      );
      
      if (response.success && response.prediction) {
        setCurrentPrediction(response.prediction);
      }
      
      // Actualizar límites e historial
      await Promise.all([
        loadPredictionLimits(),
        loadPredictionHistory(),
        loadUserStats()
      ]);
      
    } catch (error) {
      console.error('Error generando predicción:', error);
    } finally {
      setIsGeneratingPrediction(false);
    }
  };

  // Nueva función para generar predicciones con intervalos
  const generatePredictionWithIntervals = async () => {
    if (!canGeneratePrediction()) return;
    
    setIsGeneratingPrediction(true);
    try {
      // Usar el nuevo método con intervalos
      await loadPredictionsWithIntervals(
        activeBrain,
        selectedPair,
        selectedStyle,
        5, // limit
        subscription?.planType || 'starter'
      );
      
      // Actualizar límites e historial
      await Promise.all([
        loadPredictionLimits(),
        loadPredictionHistory(),
        loadUserStats()
      ]);
      
    } catch (error) {
      console.error('Error generando predicción con intervalos:', error);
    } finally {
      setIsGeneratingPrediction(false);
    }
  };

  // Nueva función para obtener información del próximo intervalo
  const getNextIntervalInfo = async () => {
    try {
      const intervalInfo = await getNextPredictionTime(activeBrain, selectedStyle);
      return intervalInfo;
    } catch (error) {
      console.error('Error obteniendo información del próximo intervalo:', error);
      return null;
    }
  };

  // Función para cargar información de intervalos
  const loadIntervalInfo = async () => {
    try {
      const info = await getNextIntervalInfo();
      if (info) {
        setIntervalInfo(info);
        setIsValidPredictionTime(info.is_valid_now);
        setNextPredictionTime(info.next_interval);
        setTimeUntilNext(info.time_until_next_minutes);
      }
    } catch (error) {
      console.error('Error cargando información de intervalos:', error);
    }
  };

  const loadPredictionLimits = async () => {
    try {
      const limits = await apiService.getPredictionLimits(selectedStyle);
      setPredictionLimits(limits);
    } catch (error) {
      console.error('Error cargando límites:', error);
    }
  };

  const loadActivePrediction = async () => {
    try {
      const active = await apiService.getActivePrediction(selectedStyle);
      if (active) {
        setCurrentPrediction(active);
      }
    } catch (error) {
      console.error('Error cargando predicción activa:', error);
    }
  };

  const loadPredictionHistory = async () => {
    setIsLoadingHistory(true);
    try {
      // ✅ Primero completar predicciones expiradas automáticamente (sin recargar)
      await _completeExpiredPredictionsInternal();
      
      // Luego cargar el historial actualizado
      const history = await apiService.getPredictionHistory();
      setPredictionHistory(history);
    } catch (error) {
      console.error('Error cargando historial:', error);
    } finally {
      setIsLoadingHistory(false);
    }
  };

  const loadUserStats = async () => {
    setIsLoadingStats(true);
    try {
      const stats = await apiService.getUserStats();
      setUserStats(stats);
    } catch (error) {
      console.error('Error cargando estadísticas:', error);
    } finally {
      setIsLoadingStats(false);
    }
  };

  // ✅ Función para completar predicciones expiradas (solo llamada manual)
  const completeExpiredPredictions = async () => {
    try {
      const result = await apiService.completeExpiredPredictions();
      if (result.success) {
        console.log('✅ Predicciones expiradas completadas:', result.message);
        // Recargar historial después de completar
        await loadPredictionHistory();
        // Recargar estadísticas
        await loadUserStats();
      }
    } catch (error) {
      console.error('Error completing expired predictions:', error);
    }
  };

  // ✅ Función interna para completar predicciones sin recargar (solo completar, no recargar)
  const _completeExpiredPredictionsInternal = async () => {
    try {
      const result = await apiService.completeExpiredPredictions();
      if (result.success && result.completed > 0) {
        console.log('✅ Predicciones expiradas completadas internamente:', result.message);
      }
    } catch (error) {
      console.error('Error completing expired predictions internally:', error);
    }
  };

  // Funciones para el sistema de señales manuales
  const generateSignal = async () => {
    setIsGeneratingSignal(true);
    setSignalQuality(null);
    setSignalMessage('');
    
    try {
      console.log('🔍 Generando señal...', { activeBrain, selectedPair, selectedStyle });
      
      const response = await apiService.generateSignal(
        activeBrain,
        selectedPair,
        selectedStyle
      );
      
      console.log('🔍 Respuesta del backend:', response);
      
      if (response.success) {
        setSignalQuality(response.signal_quality || 0);
        setSignalMessage(`Señal generada exitosamente (Calidad: ${response.signal_quality?.toFixed(1)}%)`);
        
        // Si la señal se generó exitosamente, agregarla al estado local
        console.log('🔍 Respuesta completa:', response);
        
        // Crear la señal desde los campos directos de la respuesta
        const newSignal: BrainTraderSignal = {
          pair: response.pair || selectedPair,
          type: (response.signal_type as 'buy' | 'sell' | 'hold') || 'hold',
          strength: (response.strength as 'strong' | 'medium' | 'weak') || 'medium',
          confidence: response.confidence || response.signal_quality || 0,
          entry_price: response.current_price || response.entry_price || 0,
          stop_loss: response.stop_loss || 0,
          take_profit: typeof response.take_profit === 'string' ? parseFloat(response.take_profit) : (response.take_profit || 0),
          brain_type: response.brain_type || activeBrain,
          timestamp: response.timestamp || response.generated_at || new Date().toISOString()
        };
        
        console.log('🔍 Nueva señal a agregar:', newSignal);
        console.log('🔍 Función addSignal disponible:', typeof addSignal);
        
        // Agregar la nueva señal al inicio de la lista
        addSignal(newSignal);
        
        console.log('🔍 Señal agregada, signals actual:', signals);
      } else {
        setSignalQuality(response.signal_quality || 0);
        setSignalMessage(response.message || 'Error generando señal');
        console.log('❌ Respuesta no exitosa:', response);
      }
    } catch (error) {
      setSignalQuality(0);
      setSignalMessage('Error generando señal');
      console.error('❌ Error generando señal:', error);
    } finally {
      setIsGeneratingSignal(false);
    }
  };

  const loadSignalIntervals = async () => {
    try {
      console.log('Llamando a getSignalIntervals...');
      const intervals = await apiService.getSignalIntervals(activeBrain, selectedStyle);
      console.log('Intervalos recibidos:', intervals);
      setSignalIntervals(intervals);
      setIsValidSignalTime(intervals.is_valid_time);
    } catch (error) {
      console.error('Error loading signal intervals:', error);
    }
  };

  const getNextInterval = () => {
    if (signalIntervals) {
      return signalIntervals.next_interval;
    }
    return '--:--';
  };

  const isSignalTime = () => {
    // Para el plan Starter, permitir siempre generar señales para testing
    if (!subscription || subscription.status !== 'active' || subscription.planType === 'starter') {
      return true;
    }
    return isValidSignalTime;
  };

  // Verificar acceso
  if (!checkAccess('brain-trader')) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <Brain className="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-gray-300 mb-2">Brain Trader</h3>
          <p className="text-gray-500 mb-4">Esta función requiere una suscripción activa</p>
          <button className="bg-gradient-to-r from-blue-500 to-teal-500 text-white px-6 py-2 rounded-lg hover:from-blue-600 hover:to-teal-600 transition-all">
            Actualizar Plan
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen" style={{ background: 'linear-gradient(to bottom right, var(--primary-bg), rgba(59, 130, 246, 0.2), var(--primary-bg))' }}>
      {/* Header Moderno */}
      <div className="sticky top-0 z-50 backdrop-blur-xl border-b" style={{ backgroundColor: 'rgba(15, 23, 42, 0.8)', borderColor: 'var(--border-color)' }}>
        <div className="px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            {/* Logo y Título */}
            <div className="flex items-center gap-3">
              <div className="relative">
                <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-2xl flex items-center justify-center shadow-lg" style={{ background: 'linear-gradient(to right, var(--accent-text), #06b6d4)', boxShadow: '0 10px 25px rgba(56, 178, 172, 0.25)' }}>
                  <Brain className="w-5 h-5 sm:w-6 sm:h-6 text-white" />
                </div>
                <div className="absolute -top-1 -right-1 w-3 h-3 rounded-full border-2 animate-pulse" style={{ backgroundColor: 'var(--success-color)', borderColor: 'var(--primary-bg)' }}></div>
              </div>
              
              <div className="flex-1 min-w-0">
                <h1 className="text-xl sm:text-2xl font-bold truncate" style={{ color: 'var(--primary-text)' }}>Brain Trader</h1>
                <p className="text-sm truncate" style={{ color: 'var(--secondary-text)' }}>Sistema de IA para trading automático</p>
              </div>
            </div>
            
            {/* Estado de API y Controles */}
            <div className="flex items-center gap-2 sm:gap-3">
              {/* Estado API */}
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-full border" style={{ backgroundColor: 'rgba(30, 41, 59, 0.5)', borderColor: 'var(--border-color)' }}>
                <div className={`w-2 h-2 rounded-full ${hasApiErrors() ? 'animate-pulse' : ''}`} style={{ backgroundColor: hasApiErrors() ? 'var(--danger-color)' : 'var(--success-color)' }}></div>
                <span className="text-xs sm:text-sm hidden sm:inline" style={{ color: 'var(--secondary-text)' }}>
                  {hasApiErrors() ? 'Error API' : 'API Online'}
                </span>
              </div>
              
              {/* Botón Auto Trading */}
              <button
                onClick={() => setIsAutoTrading(!isAutoTrading)}
                className="flex items-center gap-2 px-3 sm:px-4 py-2 rounded-xl transition-all duration-200 transform hover:scale-105 active:scale-95"
                style={{
                  backgroundColor: isAutoTrading ? 'rgba(239, 68, 68, 0.2)' : 'rgba(34, 197, 94, 0.2)',
                  color: isAutoTrading ? 'var(--danger-color)' : 'var(--success-color)',
                  border: `1px solid ${isAutoTrading ? 'rgba(239, 68, 68, 0.3)' : 'rgba(34, 197, 94, 0.3)'}`,
                  boxShadow: isAutoTrading ? '0 10px 25px rgba(239, 68, 68, 0.1)' : '0 10px 25px rgba(34, 197, 94, 0.1)'
                }}
              >
                {isAutoTrading ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                <span className="hidden sm:inline text-sm font-medium">
                  {isAutoTrading ? 'Detener' : 'Iniciar'} Auto Trading
                </span>
                <span className="sm:hidden text-sm font-medium">
                  {isAutoTrading ? 'Stop' : 'Start'}
                </span>
              </button>
              
              {/* Botón Avanzado */}
              <button
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="flex items-center gap-2 px-3 sm:px-4 py-2 rounded-xl transition-all duration-200 transform hover:scale-105 active:scale-95 border"
                style={{
                  backgroundColor: 'rgba(30, 41, 59, 0.5)',
                  color: 'var(--secondary-text)',
                  borderColor: 'var(--border-color)'
                }}
              >
                {showAdvanced ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                <span className="hidden sm:inline text-sm font-medium">Avanzado</span>
                <span className="sm:hidden text-sm font-medium">Adv</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Contenido Principal */}
      <div className="px-4 sm:px-6 lg:px-8 py-6 space-y-6">

      {/* Panel de Configuración Moderno */}
      <div className="backdrop-blur-sm rounded-2xl border p-4 sm:p-6" style={{ backgroundColor: 'rgba(30, 41, 59, 0.3)', borderColor: 'var(--border-color)' }}>
        <div className="flex items-center gap-3 mb-4">
          <Settings className="w-5 h-5" style={{ color: 'var(--accent-text)' }} />
          <h2 className="text-lg font-semibold" style={{ color: 'var(--primary-text)' }}>Configuración de Trading</h2>
        </div>
        
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Par de divisas */}
          <div className="space-y-2">
            <label className="text-sm font-medium flex items-center gap-2" style={{ color: 'var(--secondary-text)' }}>
              <Target className="w-4 h-4" />
              Par de Divisas
            </label>
            <select
              value={selectedPair}
              onChange={(e) => setSelectedPair(e.target.value)}
              className="w-full rounded-xl px-4 py-3 focus:outline-none focus:ring-2 transition-all duration-200"
              style={{
                backgroundColor: 'rgba(30, 41, 59, 0.5)',
                border: '1px solid var(--border-color)',
                color: 'var(--primary-text)'
              }}
            >
              {getAvailablePairs().map(pair => (
                <option key={pair} value={pair}>{pair}</option>
              ))}
            </select>
          </div>

          {/* Estilo de trading */}
          <div className="space-y-2">
            <label className="text-sm font-medium flex items-center gap-2" style={{ color: 'var(--secondary-text)' }}>
              <BarChart3 className="w-4 h-4" />
              Estilo de Trading
            </label>
            <select
              value={selectedStyle}
              onChange={(e) => setSelectedStyle(e.target.value)}
              className="w-full rounded-xl px-4 py-3 focus:outline-none focus:ring-2 transition-all duration-200"
              style={{
                backgroundColor: 'rgba(30, 41, 59, 0.5)',
                border: '1px solid var(--border-color)',
                color: 'var(--primary-text)'
              }}
            >
              {getAvailableStyles().map(style => (
                <option key={style} value={style}>
                  {style.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </option>
              ))}
            </select>
          </div>

          {/* Cerebro activo */}
          <div className="space-y-2">
            <label className="text-sm font-medium flex items-center gap-2" style={{ color: 'var(--secondary-text)' }}>
              <Brain className="w-4 h-4" />
              Cerebro IA
            </label>
            <select
              value={activeBrain}
              onChange={(e) => setActiveBrain(e.target.value as any)}
              className="w-full rounded-xl px-4 py-3 focus:outline-none focus:ring-2 transition-all duration-200"
              style={{
                backgroundColor: 'rgba(30, 41, 59, 0.5)',
                border: '1px solid var(--border-color)',
                color: 'var(--primary-text)'
              }}
            >
              {getAvailableBrains().map(brain => (
                <option key={brain} value={brain}>
                  {brain.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </option>
              ))}
            </select>
          </div>

          {/* Botón de actualizar */}
          <div className="space-y-2">
            <label className="text-sm font-medium" style={{ color: 'var(--secondary-text)' }}>&nbsp;</label>
            <button
              onClick={() => refreshAll(activeBrain, selectedPair, selectedStyle)}
              disabled={Object.values(loading).some(l => l)}
              className="w-full flex items-center justify-center gap-2 px-4 py-3 rounded-xl transition-all duration-200 transform hover:scale-105 active:scale-95 disabled:transform-none"
              style={{
                background: Object.values(loading).some(l => l) 
                  ? 'linear-gradient(to right, var(--tertiary-bg), var(--tertiary-bg))'
                  : 'linear-gradient(to right, var(--accent-text), #06b6d4)',
                color: 'var(--primary-text)',
                boxShadow: '0 10px 25px rgba(56, 178, 172, 0.25)'
              }}
            >
              {Object.values(loading).some(l => l) ? (
                <RefreshCw className="w-4 h-4 animate-spin" />
              ) : (
                <RefreshCw className="w-4 h-4" />
              )}
              <span className="text-sm font-medium">
                {Object.values(loading).some(l => l) ? 'Actualizando...' : 'Actualizar'}
              </span>
            </button>
          </div>
        </div>
      </div>

      {/* Información del Plan */}
      <div className="backdrop-blur-sm rounded-2xl border p-4 sm:p-6" style={{ backgroundColor: 'rgba(30, 41, 59, 0.3)', borderColor: 'var(--border-color)' }}>
        <div className="flex items-center gap-3 mb-4">
          <Info className="w-5 h-5" style={{ color: 'var(--accent-text)' }} />
          <h2 className="text-lg font-semibold" style={{ color: 'var(--primary-text)' }}>Información del Plan</h2>
        </div>
        
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-gray-600">Predicciones usadas hoy:</span>
            <span className="ml-2 font-semibold text-blue-600">
              {(() => {
                const { used, limit } = getPlanLimits();
                return `${used} / ${limit === -1 ? '∞' : limit}`;
              })()}
            </span>
          </div>
          <div>
            <span className="text-gray-600">Estado de predicción:</span>
            <span className="ml-2 font-semibold text-blue-600">
              {getPlanLimits().canGenerate ? 'Disponible' : 'No disponible'}
            </span>
          </div>
        </div>
      </div>

      {/* Navegación por Pestañas */}
      <div className="backdrop-blur-sm rounded-2xl border p-4 sm:p-6" style={{ backgroundColor: 'rgba(30, 41, 59, 0.3)', borderColor: 'var(--border-color)' }}>
        <div className="flex flex-wrap gap-2 mb-6">
          {[
            { id: 'predictions', label: 'Predicciones', icon: TrendingUpIcon },
            { id: 'signals', label: 'Señales', icon: Zap },
            { id: 'trends', label: 'Tendencias', icon: BarChart3 },
            { id: 'history', label: 'Historial', icon: History }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all duration-200 ${
                activeTab === tab.id
                  ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg'
                  : 'bg-gray-700/50 text-gray-300 hover:bg-gray-600/50'
              }`}
            >
              <tab.icon className="w-4 h-4" />
              <span className="text-sm font-medium">{tab.label}</span>
            </button>
          ))}
        </div>

        {/* Contenido de las Pestañas */}
        {activeTab === 'predictions' && (
          <div className="space-y-6">
            {/* Opción de Intervalos */}
            <div className="bg-white rounded-lg p-4 border border-gray-200">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-3">
                  <Clock className="w-5 h-5 text-blue-600" />
                  <div>
                    <h4 className="text-lg font-semibold text-gray-800">Sistema de Intervalos</h4>
                    <p className="text-sm text-gray-600">Generar predicciones respetando intervalos de tiempo</p>
                  </div>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={useIntervals}
                    onChange={(e) => setUseIntervals(e.target.checked)}
                    className="sr-only peer"
                  />
                  <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                </label>
              </div>
              
              {useIntervals && (
                <div className="space-y-3">
                  <div className="flex items-center justify-between p-3 bg-blue-50 rounded-lg">
                    <div className="flex items-center space-x-2">
                      <div className={`w-3 h-3 rounded-full ${isValidPredictionTime ? 'bg-green-500' : 'bg-yellow-500'}`}></div>
                      <span className="text-sm font-medium text-gray-700">
                        {isValidPredictionTime ? 'Momento válido para predicción' : 'Esperando próximo intervalo'}
                      </span>
                    </div>
                    <span className="text-sm text-gray-500">
                      {isValidPredictionTime ? '✅ Listo' : `⏳ ${timeUntilNext.toFixed(1)} min`}
                    </span>
                  </div>
                  
                  {nextPredictionTime && (
                    <div className="text-sm text-gray-600">
                      <span className="font-medium">Próximo intervalo:</span> {new Date(nextPredictionTime).toLocaleTimeString('es-ES', { hour: '2-digit', minute: '2-digit' })}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Botón Generar Predicción */}
            <div className="mb-6">
              <button
                onClick={useIntervals ? generatePredictionWithIntervals : generatePrediction}
                disabled={isGeneratingPrediction || !canGeneratePrediction() || (useIntervals && !isValidPredictionTime)}
                className={`w-full py-3 px-6 rounded-lg font-semibold transition-all duration-300 ${
                  canGeneratePrediction() && (!useIntervals || isValidPredictionTime)
                    ? 'bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white shadow-lg hover:shadow-xl transform hover:scale-105'
                    : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                }`}
              >
                {isGeneratingPrediction ? (
                  <div className="flex items-center justify-center">
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                    Generando predicción...
                  </div>
                ) : (
                  <div className="flex items-center justify-center">
                    <TrendingUpIcon className="w-5 h-5 mr-2" />
                    {useIntervals ? 'Generar Predicción con Intervalos' : 'Generar Predicción'}
                  </div>
                )}
              </button>
              
              {!canGeneratePrediction() && (
                <p className="text-sm text-gray-500 mt-2 text-center">
                  {predictionLimits && (predictionLimits.max_predictions_per_day - predictionLimits.remaining_predictions) >= predictionLimits.max_predictions_per_day
                    ? 'Has alcanzado el límite diario de predicciones'
                    : useIntervals && !isValidPredictionTime
                    ? `Espera hasta el próximo intervalo (${timeUntilNext.toFixed(1)} min)`
                    : 'No puedes generar una nueva predicción en este momento'
                  }
                </p>
              )}
            </div>

            {/* Precio Actual */}
            <div className="bg-gradient-to-r from-green-50 to-blue-50 border border-green-200 rounded-lg p-6 mb-4">
              <div className="flex items-center justify-between mb-4">
                <h4 className="text-lg font-semibold text-gray-800">Precio Actual</h4>
                <div className="flex items-center space-x-2">
                  {marketLoading ? (
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                  ) : (
                    <div className={`w-3 h-3 rounded-full ${
                      marketError 
                        ? marketError.includes('fin de semana') || marketError.includes('cerrado')
                          ? 'bg-yellow-500'
                          : 'bg-red-500'
                        : 'bg-green-500'
                    }`}></div>
                  )}
                  <span className="text-sm text-gray-500">
                    {marketLoading 
                      ? 'Actualizando...' 
                      : marketError 
                        ? marketError.includes('fin de semana') || marketError.includes('cerrado')
                          ? 'Mercado cerrado'
                          : 'Error'
                        : 'En vivo'
                    }
                  </span>
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-white rounded-lg p-4">
                  <h5 className="font-semibold text-gray-700 mb-2">Par Seleccionado</h5>
                  <div className="flex items-center">
                    <span className="text-lg font-bold text-gray-800 mr-2">{selectedPair}</span>
                    <span className="text-sm text-gray-500">({selectedStyle})</span>
                  </div>
                </div>
                
                <div className="bg-white rounded-lg p-4">
                  <h5 className="font-semibold text-gray-700 mb-2">Precio Actual</h5>
                  <div className="flex items-center">
                    <span className="text-2xl font-bold text-gray-800">
                      ${getCurrentPrice(selectedPair)?.toFixed(4) || '---'}
                    </span>
                    {getPriceChange(selectedPair) && (
                      <span className={`ml-2 text-sm font-semibold ${
                        getPriceChange(selectedPair)!.changePercent >= 0 
                          ? 'text-green-600' 
                          : 'text-red-600'
                      }`}>
                        {getPriceChange(selectedPair)!.changePercent >= 0 ? '+' : ''}
                        {getPriceChange(selectedPair)!.changePercent.toFixed(2)}%
                      </span>
                    )}
                  </div>
                </div>
                
                {currentPrediction && (
                  <>
                    <div className="bg-white rounded-lg p-4">
                      <h5 className="font-semibold text-gray-700 mb-2">Diferencia con Predicción</h5>
                      <div className="flex items-center">
                        {(() => {
                          const currentPrice = getCurrentPrice(selectedPair);
                          const targetPrice = currentPrediction.target_price;
                          if (!currentPrice) return <span className="text-gray-500">---</span>;
                          
                          const difference = targetPrice - currentPrice;
                          const differencePercent = (difference / currentPrice) * 100;
                          
                          return (
                            <span className={`text-lg font-bold ${
                              difference >= 0 ? 'text-green-600' : 'text-red-600'
                            }`}>
                              {difference >= 0 ? '+' : ''}{difference.toFixed(4)} 
                              ({differencePercent >= 0 ? '+' : ''}{differencePercent.toFixed(2)}%)
                            </span>
                          );
                        })()}
                      </div>
                    </div>
                    
                    <div className="bg-white rounded-lg p-4">
                      <h5 className="font-semibold text-gray-700 mb-2">Estado de Predicción</h5>
                      <div className="flex items-center">
                        {(() => {
                          const currentPrice = getCurrentPrice(selectedPair);
                          const targetPrice = currentPrediction.target_price;
                          if (!currentPrice) return <span className="text-gray-500">---</span>;
                          
                          const isExpired = new Date() > new Date(currentPrediction.expires_at || '');
                          if (isExpired) {
                            const isCorrect = (currentPrediction.direction === 'up' && currentPrice > targetPrice) ||
                                            (currentPrediction.direction === 'down' && currentPrice < targetPrice);
                            
                            return (
                              <span className={`text-sm font-semibold ${
                                isCorrect ? 'text-green-600' : 'text-red-600'
                              }`}>
                                {isCorrect ? '✅ Correcta' : '❌ Incorrecta'}
                              </span>
                            );
                          } else {
                            return <span className="text-blue-600 text-sm font-semibold">⏳ En progreso</span>;
                          }
                        })()}
                      </div>
                    </div>
                  </>
                )}
              </div>
              
              {marketError && (
                <div className="mt-4 bg-yellow-50 border border-yellow-200 rounded-lg p-3">
                  <div className="flex items-center">
                    <AlertTriangle className="w-4 h-4 text-yellow-600 mr-2" />
                    <div>
                      <p className="text-yellow-800 text-sm font-medium">
                        {marketError.includes('fin de semana') 
                          ? 'Mercado cerrado - datos de fin de semana'
                          : marketError.includes('cerrado')
                          ? 'Mercado cerrado - usando datos históricos'
                          : `Error obteniendo datos de mercado: ${marketError}`
                        }
                      </p>
                      <p className="text-yellow-700 text-xs mt-1">
                        Los precios mostrados pueden no estar actualizados en tiempo real
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Predicción Actual */}
            {currentPrediction && (
              <div className="bg-gradient-to-r from-blue-50 to-purple-50 border border-blue-200 rounded-lg p-6 mb-4">
                <div className="flex items-center justify-between mb-4">
                  <h4 className="text-lg font-semibold text-gray-800">Predicción Actual</h4>
                  <div className="flex items-center space-x-2">
                    <span className="text-sm text-gray-500">
                      Expira: {new Date(currentPrediction.expires_at || '').toLocaleString()}
                    </span>
                    <div className={`w-3 h-3 rounded-full ${
                      new Date() < new Date(currentPrediction.expires_at || '') 
                        ? 'bg-green-500' 
                        : 'bg-red-500'
                    }`}></div>
                  </div>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="bg-white rounded-lg p-4">
                    <h5 className="font-semibold text-gray-700 mb-2">Dirección</h5>
                    <div className="flex items-center">
                      {currentPrediction.direction === 'up' ? (
                        <TrendingUpIcon className="w-6 h-6 text-green-600 mr-2" />
                      ) : (
                        <TrendingDownIcon className="w-6 h-6 text-red-600 mr-2" />
                      )}
                      <span className={`text-lg font-bold ${
                        currentPrediction.direction === 'up' ? 'text-green-600' : 'text-red-600'
                      }`}>
                        {currentPrediction.direction.toUpperCase()}
                      </span>
                    </div>
                  </div>
                  
                  <div className="bg-white rounded-lg p-4">
                    <h5 className="font-semibold text-gray-700 mb-2">Confianza</h5>
                    <div className="flex items-center">
                      <div className="w-full bg-gray-200 rounded-full h-2 mr-2">
                        <div 
                          className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${currentPrediction.confidence}%` }}
                        ></div>
                      </div>
                      <span className="text-sm font-semibold text-gray-700">
                        {currentPrediction.confidence}%
                      </span>
                    </div>
                  </div>
                  
                  <div className="bg-white rounded-lg p-4">
                    <h5 className="font-semibold text-gray-700 mb-2">Precio Objetivo</h5>
                    <span className="text-lg font-bold text-gray-800">
                      ${currentPrediction.target_price.toFixed(4)}
                    </span>
                  </div>
                  
                  <div className="bg-white rounded-lg p-4">
                    <h5 className="font-semibold text-gray-700 mb-2">Timeframe</h5>
                    <span className="text-lg font-bold text-gray-800">
                      {currentPrediction.timeframe}
                    </span>
                  </div>
                </div>
                
                <div className="mt-4 bg-white rounded-lg p-4">
                  <h5 className="font-semibold text-gray-700 mb-2">Análisis Técnico</h5>
                  <p className="text-gray-600 text-sm">
                    {currentPrediction.reasoning}
                  </p>
                </div>
              </div>
            )}

            {/* Mensaje cuando no hay predicción */}
            {!currentPrediction && !isGeneratingPrediction && (
              <div className="text-center py-8">
                <TrendingUpIcon className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                <h4 className="text-lg font-semibold text-gray-600 mb-2">
                  No hay predicción activa
                </h4>
                <p className="text-gray-500">
                  Haz clic en "Generar Predicción" para obtener una nueva predicción
                </p>
              </div>
            )}
          </div>
        )}

        {activeTab === 'signals' && (
          <div className="space-y-6">
            {/* Botón de Generación Manual */}
            <div className="backdrop-blur-sm rounded-2xl border p-4 sm:p-6" style={{ backgroundColor: 'rgba(30, 41, 59, 0.3)', borderColor: 'var(--border-color)' }}>
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h3 className="text-lg font-semibold" style={{ color: 'var(--primary-text)' }}>
                    Generar Señal Manual
                  </h3>
                  <p className="text-sm" style={{ color: 'var(--secondary-text)' }}>
                    {selectedStyle.replace('_', ' ').toUpperCase()} - {signalIntervals?.timeframe || '15M'} intervalos
                  </p>
                </div>
                
                <div className="text-right">
                  <p className="text-sm" style={{ color: 'var(--secondary-text)' }}>
                    Próximo intervalo: {getNextInterval()}
                  </p>
                  <p className={`text-xs ${isSignalTime() ? 'text-green-400' : 'text-yellow-400'}`}>
                    {isSignalTime() ? '✅ Momento válido' : '⏰ Esperando intervalo'}
                  </p>
                </div>
              </div>
              
              <button
                onClick={generateSignal}
                disabled={isGeneratingSignal || !isSignalTime()}
                className={`w-full py-3 px-4 rounded-xl font-medium transition-all duration-200 ${
                  isSignalTime() && !isGeneratingSignal
                    ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white hover:from-blue-700 hover:to-purple-700'
                    : 'bg-gray-600 text-gray-400 cursor-not-allowed'
                }`}
              >
                {isGeneratingSignal ? (
                  <div className="flex items-center justify-center gap-2">
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    Generando señal...
                  </div>
                ) : (
                  <div className="flex items-center justify-center gap-2">
                    <Zap className="w-4 h-4" />
                    Generar Señal
                  </div>
                )}
              </button>
              
              {/* Mensaje de Calidad */}
              {signalQuality !== null && (
                <div className={`mt-3 p-3 rounded-lg ${
                  signalQuality >= 70 
                    ? 'bg-green-500/20 border border-green-500/30' 
                    : 'bg-red-500/20 border border-red-500/30'
                }`}>
                  <p className={`text-sm font-medium ${
                    signalQuality >= 70 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {signalMessage}
                  </p>
                  {signalQuality > 0 && signalQuality < 70 && (
                    <p className="text-xs text-gray-400 mt-1">
                      Se requiere calidad mínima del 70% para mostrar la señal
                    </p>
                  )}
                </div>
              )}

              {/* Información de Intervalos */}
              {signalIntervals && (
                <div className="mt-4 p-3 rounded-lg" style={{ backgroundColor: 'rgba(30, 41, 59, 0.3)' }}>
                  <p className="text-xs text-gray-400 mb-2">Próximos intervalos:</p>
                  <div className="flex flex-wrap gap-2">
                    {signalIntervals.upcoming_intervals?.slice(0, 4).map((interval: string, index: number) => (
                      <span key={index} className="px-2 py-1 rounded text-xs" style={{ backgroundColor: 'rgba(100, 116, 139, 0.2)' }}>
                        {interval}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Contenido de Señales */}
            {signals.length > 0 ? (
              <div className="space-y-3">
                {signals.map((signal, index) => (
                  <div key={index} className="rounded-xl p-4 hover:bg-slate-700/50 transition-all duration-200 border" style={{ backgroundColor: 'rgba(30, 41, 59, 0.3)', borderColor: 'var(--border-color)' }}>
                    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                      <div className="flex items-center gap-3">
                        <div className={`w-12 h-12 rounded-xl flex items-center justify-center shadow-lg`} style={{
                          backgroundColor: signal.type === 'buy' ? 'rgba(34, 197, 94, 0.2)' :
                          signal.type === 'sell' ? 'rgba(239, 68, 68, 0.2)' : 'rgba(100, 116, 139, 0.2)',
                          color: signal.type === 'buy' ? 'var(--success-color)' :
                          signal.type === 'sell' ? 'var(--danger-color)' : 'var(--secondary-text)',
                          boxShadow: signal.type === 'buy' ? '0 10px 25px rgba(34, 197, 94, 0.25)' :
                          signal.type === 'sell' ? '0 10px 25px rgba(239, 68, 68, 0.25)' : '0 10px 25px rgba(100, 116, 139, 0.25)'
                        }}>
                          {signal.type === 'buy' ? <CheckCircle className="w-6 h-6" /> :
                           signal.type === 'sell' ? <XCircle className="w-6 h-6" /> :
                           <Pause className="w-6 h-6" />}
                        </div>
                        
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-1">
                            <p className="font-semibold text-lg" style={{ color: 'var(--primary-text)' }}>{signal.pair}</p>
                            <span className="px-2 py-1 rounded-full text-xs font-medium" style={{
                              backgroundColor: signal.type === 'buy' ? 'rgba(34, 197, 94, 0.2)' :
                              signal.type === 'sell' ? 'rgba(239, 68, 68, 0.2)' : 'rgba(100, 116, 139, 0.2)',
                              color: signal.type === 'buy' ? 'var(--success-color)' :
                              signal.type === 'sell' ? 'var(--danger-color)' : 'var(--secondary-text)'
                            }}>
                              {signal.type.toUpperCase()}
                            </span>
                          </div>
                          <p className="text-sm" style={{ color: 'var(--secondary-text)' }}>Fuerza: {signal.strength}</p>
                        </div>
                      </div>
                      
                      <div className="text-right">
                        <div className="rounded-lg p-3" style={{ backgroundColor: 'rgba(30, 41, 59, 0.3)' }}>
                          <p className="font-bold text-lg" style={{ color: 'var(--primary-text)' }}>${signal.entry_price.toFixed(4)}</p>
                          <p className="text-sm" style={{ color: 'var(--secondary-text)' }}>{signal.confidence.toFixed(1)}% confianza</p>
                          <div className="flex gap-2 mt-1 text-xs">
                            <span style={{ color: 'var(--danger-color)' }}>SL: ${signal.stop_loss.toFixed(4)}</span>
                            <span style={{ color: 'var(--success-color)' }}>TP: ${signal.take_profit.toFixed(4)}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8">
                <Zap className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                <h4 className="text-lg font-semibold text-gray-600 mb-2">
                  No hay señales disponibles
                </h4>
                <p className="text-gray-500">
                  Use el botón "Generar Señal" en los intervalos de {signalIntervals?.timeframe || '15'} minutos
                </p>
              </div>
            )}
          </div>
        )}

        {activeTab === 'trends' && (
          <div className="space-y-6">
            {/* Contenido de Tendencias */}
            {trends.length > 0 ? (
              <div className="space-y-3">
                {trends.map((trend, index) => (
                  <div key={index} className="rounded-xl p-4 hover:bg-slate-700/50 transition-all duration-200 border" style={{ backgroundColor: 'rgba(30, 41, 59, 0.3)', borderColor: 'var(--border-color)' }}>
                    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-3">
                      <div className="flex items-center gap-3">
                        <div className={`w-10 h-10 rounded-xl flex items-center justify-center shadow-lg`} style={{
                          backgroundColor: trend.direction === 'bullish' ? 'rgba(34, 197, 94, 0.2)' :
                          trend.direction === 'bearish' ? 'rgba(239, 68, 68, 0.2)' : 'rgba(100, 116, 139, 0.2)',
                          color: trend.direction === 'bullish' ? 'var(--success-color)' :
                          trend.direction === 'bearish' ? 'var(--danger-color)' : 'var(--secondary-text)',
                          boxShadow: trend.direction === 'bullish' ? '0 10px 25px rgba(34, 197, 94, 0.25)' :
                          trend.direction === 'bearish' ? '0 10px 25px rgba(239, 68, 68, 0.25)' : '0 10px 25px rgba(100, 116, 139, 0.25)'
                        }}>
                          {trend.direction === 'bullish' ? <TrendingUp className="w-5 h-5" /> :
                           trend.direction === 'bearish' ? <TrendingDown className="w-5 h-5" /> :
                           <Activity className="w-5 h-5" />}
                        </div>
                        
                        <div>
                          <p className="font-semibold text-lg" style={{ color: 'var(--primary-text)' }}>{trend.pair}</p>
                          <div className="flex items-center gap-2">
                            <span className="px-2 py-1 rounded-full text-xs font-medium capitalize" style={{
                              backgroundColor: trend.direction === 'bullish' ? 'rgba(34, 197, 94, 0.2)' :
                              trend.direction === 'bearish' ? 'rgba(239, 68, 68, 0.2)' : 'rgba(100, 116, 139, 0.2)',
                              color: trend.direction === 'bullish' ? 'var(--success-color)' :
                              trend.direction === 'bearish' ? 'var(--danger-color)' : 'var(--secondary-text)'
                            }}>
                              {trend.direction}
                            </span>
                            <span className="text-sm" style={{ color: 'var(--secondary-text)' }}>{trend.timeframe}</span>
                          </div>
                        </div>
                      </div>
                      
                      <div className="text-right">
                        <div className="rounded-lg p-3" style={{ backgroundColor: 'rgba(30, 41, 59, 0.3)' }}>
                          <p className="font-bold text-lg" style={{ color: 'var(--primary-text)' }}>{trend.strength.toFixed(0)}%</p>
                          <p className="text-sm" style={{ color: 'var(--secondary-text)' }}>Fuerza</p>
                        </div>
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-3 text-sm">
                      <div className="rounded-lg p-3" style={{ backgroundColor: 'rgba(30, 41, 59, 0.3)' }}>
                        <p className="font-medium" style={{ color: 'var(--secondary-text)' }}>Soporte</p>
                        <p className="font-semibold" style={{ color: 'var(--primary-text)' }}>${trend.support.toFixed(4)}</p>
                      </div>
                      <div className="rounded-lg p-3" style={{ backgroundColor: 'rgba(30, 41, 59, 0.3)' }}>
                        <p className="font-medium" style={{ color: 'var(--secondary-text)' }}>Resistencia</p>
                        <p className="font-semibold" style={{ color: 'var(--primary-text)' }}>${trend.resistance.toFixed(4)}</p>
                      </div>
                    </div>
                    
                    <div className="mt-3 p-3 rounded-lg" style={{ backgroundColor: 'rgba(30, 41, 59, 0.2)' }}>
                      <p className="text-sm" style={{ color: 'var(--secondary-text)' }}>{trend.description}</p>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8">
                <BarChart3 className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                <h4 className="text-lg font-semibold text-gray-600 mb-2">
                  No hay tendencias disponibles
                </h4>
                <p className="text-gray-500">
                  Las tendencias aparecerán aquí cuando estén disponibles
                </p>
              </div>
            )}
          </div>
        )}

        {activeTab === 'history' && (
          <div className="space-y-6">
            {/* Estadísticas del Usuario */}
            {userStats && (
              <div className="bg-gradient-to-r from-blue-50 to-purple-50 border border-blue-200 rounded-lg p-6 mb-6">
                <h4 className="text-lg font-semibold text-gray-800 mb-4">Estadísticas del Usuario</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-white rounded-lg p-4">
                    <h5 className="font-semibold text-gray-700 mb-2">Total Predicciones</h5>
                    <span className="text-2xl font-bold text-blue-600">{userStats.total_predictions}</span>
                  </div>
                  <div className="bg-white rounded-lg p-4">
                    <h5 className="font-semibold text-gray-700 mb-2">Predicciones Exitosas</h5>
                    <span className="text-2xl font-bold text-green-600">{userStats.successful_predictions}</span>
                  </div>
                  <div className="bg-white rounded-lg p-4">
                    <h5 className="font-semibold text-gray-700 mb-2">Tasa de Éxito</h5>
                    <span className="text-2xl font-bold text-purple-600">
                      {((userStats.successful_predictions / userStats.total_predictions) * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="bg-white rounded-lg p-4">
                    <h5 className="font-semibold text-gray-700 mb-2">Promedio de Éxito</h5>
                    <span className="text-2xl font-bold text-orange-600">
                      {userStats.average_success_percentage?.toFixed(2) || '0.00'}%
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* Historial de Predicciones */}
            <div>
              <div className="flex items-center justify-between mb-4">
                <h4 className="text-lg font-semibold text-gray-800">Historial de Predicciones</h4>
                <button
                  onClick={completeExpiredPredictions}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center gap-2"
                >
                  <Clock className="w-4 h-4" />
                  Completar Expiradas
                </button>
              </div>
              {isLoadingHistory ? (
                <div className="text-center py-8">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
                  <p className="text-gray-500">Cargando historial...</p>
                </div>
              ) : predictionHistory.length > 0 ? (
                <div className="space-y-4">
                  {predictionHistory.map((item, index) => (
                    <div key={index} className="bg-white rounded-lg p-6 border border-gray-200 shadow-sm hover:shadow-md transition-shadow">
                      {/* Header con estado de la predicción */}
                                              <div className="flex items-center justify-between mb-4">
                          <div className="flex items-center gap-3">
                            <div className={`w-12 h-12 rounded-full flex items-center justify-center ${
                              item.prediction_success === true ? 'bg-green-100' :
                              item.prediction_success === false ? 'bg-red-100' :
                              'bg-gray-100'
                            }`}>
                              {item.prediction_success === true ? (
                                <CheckCircle className="w-6 h-6 text-green-600" />
                              ) : item.prediction_success === false ? (
                                <XCircle className="w-6 h-6 text-red-600" />
                              ) : (
                                <Clock className="w-6 h-6 text-gray-600" />
                              )}
                            </div>
                          <div>
                            <div className="flex items-center gap-2">
                              <p className="font-bold text-lg text-gray-800">{item.pair}</p>
                              <span className={`px-2 py-1 rounded-full text-xs font-semibold ${
                                item.direction === 'up' ? 'bg-green-100 text-green-700' :
                                item.direction === 'down' ? 'bg-red-100 text-red-700' :
                                'bg-gray-100 text-gray-700'
                              }`}>
                                {item.direction.toUpperCase()}
                              </span>
                            </div>
                            <p className="text-sm text-gray-500">
                              {new Date(item.created_at).toLocaleDateString('es-ES', {
                                year: 'numeric',
                                month: 'short',
                                day: 'numeric',
                                hour: '2-digit',
                                minute: '2-digit'
                              })}
                            </p>
                          </div>
                        </div>
                        <div className="text-right">
                          <p className={`font-bold text-2xl ${
                            item.prediction_success === true ? 'text-green-600' :
                            item.prediction_success === false ? 'text-red-600' :
                            'text-gray-600'
                          }`}>
                            {item.prediction_success === true ? '✓' :
                             item.prediction_success === false ? '✗' :
                             '⏳'}
                          </p>
                          <p className="text-sm text-gray-500">
                            {item.confidence}% confianza
                          </p>
                        </div>
                      </div>
                      
                      {/* Comparación de precios */}
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                        <div className="bg-gray-50 rounded-lg p-4">
                          <p className="font-medium text-gray-700 mb-1">Precio al Generar</p>
                          <p className="font-bold text-lg text-gray-800">${item.current_price?.toFixed(5) || 'N/A'}</p>
                          <p className="text-xs text-gray-500">Precio capturado en el momento</p>
                        </div>
                        <div className="bg-blue-50 rounded-lg p-4">
                          <p className="font-medium text-gray-700 mb-1">Precio Objetivo</p>
                          <p className="font-bold text-lg text-blue-600">${item.target_price.toFixed(5)}</p>
                          <p className="text-xs text-gray-500">Precio predicho</p>
                        </div>
                        <div className="bg-green-50 rounded-lg p-4">
                          <p className="font-medium text-gray-700 mb-1">Precio Real</p>
                          <p className="font-bold text-lg text-green-600">
                            ${item.actual_price_at_expiry?.toFixed(5) || 'Pendiente'}
                          </p>
                          <p className="text-xs text-gray-500">Precio al expirar</p>
                        </div>
                      </div>
                      
                      {/* Métricas de rendimiento */}
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                        <div className="text-center">
                          <p className="text-sm font-medium text-gray-600">Diferencia</p>
                          <p className={`font-bold text-lg ${
                            item.actual_price_at_expiry ? 
                              (item.actual_price_at_expiry > item.target_price ? 'text-green-600' : 'text-red-600') : 
                              'text-gray-500'
                          }`}>
                            {item.actual_price_at_expiry ? 
                              `${((item.actual_price_at_expiry - item.target_price) / item.target_price * 100).toFixed(3)}%` : 
                              'N/A'
                            }
                          </p>
                        </div>
                        <div className="text-center">
                          <p className="text-sm font-medium text-gray-600">Movimiento Real</p>
                          <p className={`font-bold text-lg ${
                            item.actual_price_at_expiry ? 
                              (item.actual_price_at_expiry > item.current_price ? 'text-green-600' : 'text-red-600') : 
                              'text-gray-500'
                          }`}>
                            {item.actual_price_at_expiry ? 
                              `${((item.actual_price_at_expiry - item.current_price) / item.current_price * 100).toFixed(3)}%` : 
                              'N/A'
                            }
                          </p>
                        </div>
                        <div className="text-center">
                          <p className="text-sm font-medium text-gray-600">Estado</p>
                          <p className={`font-bold text-lg ${
                            item.prediction_success === true ? 'text-green-600' :
                            item.prediction_success === false ? 'text-red-600' :
                            'text-gray-600'
                          }`}>
                            {item.prediction_success === true ? 'Correcta' :
                             item.prediction_success === false ? 'Incorrecta' :
                             'Pendiente'}
                          </p>
                        </div>
                        <div className="text-center">
                          <p className="text-sm font-medium text-gray-600">Porcentaje Éxito</p>
                          <p className={`font-bold text-lg ${
                            item.success_percentage !== null && item.success_percentage !== undefined ? 'text-purple-600' : 'text-gray-500'
                          }`}>
                            {item.success_percentage !== null && item.success_percentage !== undefined ? 
                              `${item.success_percentage.toFixed(1)}%` : 'Pendiente'}
                          </p>
                        </div>
                      </div>
                      
                      {/* Información adicional */}
                      <div className="border-t pt-4">
                        <div className="flex items-center justify-between text-sm text-gray-600">
                          <div className="flex items-center gap-2">
                            <span className="font-medium">Brain:</span>
                            <span className="bg-purple-100 text-purple-700 px-2 py-1 rounded">
                              {item.brain_type || 'brain_max'}
                            </span>
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="font-medium">Timeframe:</span>
                            <span className="bg-blue-100 text-blue-700 px-2 py-1 rounded">
                              {item.timeframe}
                            </span>
                          </div>
                        </div>
                        {item.reasoning && (
                          <div className="mt-3 p-3 bg-gray-50 rounded-lg">
                            <p className="text-sm text-gray-700">
                              <span className="font-medium">Razonamiento:</span> {item.reasoning}
                            </p>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-12">
                  <History className="w-20 h-20 text-gray-300 mx-auto mb-4" />
                  <h4 className="text-xl font-semibold text-gray-600 mb-2">
                    No hay historial disponible
                  </h4>
                  <p className="text-gray-500 mb-6">
                    El historial aparecerá aquí después de generar predicciones con el botón "Generar Predicción"
                  </p>
                  <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 max-w-md mx-auto">
                    <p className="text-sm text-blue-700">
                      💡 <strong>Consejo:</strong> Cada predicción que generes se guardará automáticamente 
                      y se comparará con el precio real cuando expire el timeframe.
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        </div>

        {/* Errores de API */}
      {hasApiErrors() && (
        <div className="backdrop-blur-sm border rounded-2xl p-4 sm:p-6 animate-fade-in" style={{ backgroundColor: 'rgba(239, 68, 68, 0.1)', borderColor: 'rgba(239, 68, 68, 0.3)' }}>
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{ backgroundColor: 'rgba(239, 68, 68, 0.2)' }}>
              <AlertTriangle className="w-5 h-5" style={{ color: 'var(--danger-color)' }} />
            </div>
            <div>
              <h3 className="text-lg font-semibold" style={{ color: 'var(--danger-color)' }}>Errores de Conexión</h3>
              <p className="text-sm" style={{ color: 'rgba(239, 68, 68, 0.8)' }}>Problemas con la API de trading</p>
            </div>
          </div>
          <div className="rounded-xl p-3 mb-4" style={{ backgroundColor: 'rgba(239, 68, 68, 0.1)' }}>
            <p className="text-sm font-mono" style={{ color: 'rgba(239, 68, 68, 0.8)' }}>{getApiErrorMessages()}</p>
          </div>
          <button
            onClick={() => refreshAll(activeBrain, selectedPair, selectedStyle)}
            className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200 transform hover:scale-105 active:scale-95"
            style={{
              backgroundColor: 'var(--danger-color)',
              color: 'var(--primary-text)'
            }}
          >
            <RefreshCw className="w-4 h-4" />
            Reintentar Conexión
          </button>
        </div>
      )}

      {/* Información del Modelo */}
      {modelInfo && (
        <div className="backdrop-blur-sm rounded-2xl border p-4 sm:p-6" style={{ backgroundColor: 'rgba(30, 41, 59, 0.3)', borderColor: 'var(--border-color)' }}>
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{ backgroundColor: 'rgba(59, 130, 246, 0.2)' }}>
                <Info className="w-5 h-5" style={{ color: 'var(--accent-text)' }} />
              </div>
              <div>
                <h3 className="text-lg font-semibold" style={{ color: 'var(--primary-text)' }}>Información del Modelo</h3>
                <p className="text-sm" style={{ color: 'var(--secondary-text)' }}>Estado y métricas del cerebro IA</p>
              </div>
            </div>
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full border" style={{ backgroundColor: 'rgba(30, 41, 59, 0.5)', borderColor: 'var(--border-color)' }}>
              <div className={`w-2 h-2 rounded-full ${
                modelInfo.status === 'active' ? 'animate-pulse' :
                modelInfo.status === 'training' ? 'animate-pulse' : ''
              }`} style={{ 
                backgroundColor: modelInfo.status === 'active' ? 'var(--success-color)' :
                modelInfo.status === 'training' ? 'var(--warning-color)' : 'var(--danger-color)'
              }}></div>
              <span className="text-sm capitalize font-medium" style={{ color: 'var(--secondary-text)' }}>{modelInfo.status}</span>
            </div>
          </div>
          
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="rounded-xl p-4" style={{ backgroundColor: 'rgba(30, 41, 59, 0.3)' }}>
              <div className="flex items-center gap-2 mb-2">
                <Brain className="w-4 h-4" style={{ color: 'var(--accent-text)' }} />
                <p className="text-sm font-medium" style={{ color: 'var(--secondary-text)' }}>Cerebro</p>
              </div>
              <p className="font-semibold text-lg" style={{ color: 'var(--primary-text)' }}>{modelInfo.brainType.replace('_', ' ').toUpperCase()}</p>
            </div>
            <div className="rounded-xl p-4" style={{ backgroundColor: 'rgba(30, 41, 59, 0.3)' }}>
              <div className="flex items-center gap-2 mb-2">
                <Target className="w-4 h-4" style={{ color: 'var(--success-color)' }} />
                <p className="text-sm font-medium" style={{ color: 'var(--secondary-text)' }}>Precisión</p>
              </div>
              <p className="font-semibold text-lg" style={{ color: 'var(--primary-text)' }}>{modelInfo.accuracy.toFixed(1)}%</p>
            </div>
            <div className="rounded-xl p-4" style={{ backgroundColor: 'rgba(30, 41, 59, 0.3)' }}>
              <div className="flex items-center gap-2 mb-2">
                <Clock className="w-4 h-4" style={{ color: '#a855f7' }} />
                <p className="text-sm font-medium" style={{ color: 'var(--secondary-text)' }}>Última Actualización</p>
              </div>
              <p className="font-semibold text-sm" style={{ color: 'var(--primary-text)' }}>{new Date(modelInfo.lastUpdate).toLocaleString()}</p>
            </div>
          </div>
        </div>
      )}



      {/* Mega Mind - Sección Especial */}
      {isMegaMindAvailable() && activeBrain === 'mega_mind' && (
        <div className="backdrop-blur-sm rounded-2xl border p-4 sm:p-6" style={{ 
          background: 'linear-gradient(to bottom right, rgba(147, 51, 234, 0.3), rgba(59, 130, 246, 0.3), rgba(99, 102, 241, 0.3))',
          borderColor: 'rgba(147, 51, 234, 0.3)'
        }}>
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-xl flex items-center justify-center shadow-lg" style={{ 
              background: 'linear-gradient(to right, #a855f7, #eab308)',
              boxShadow: '0 10px 25px rgba(147, 51, 234, 0.25)'
            }}>
              <Crown className="w-5 h-5 text-white" />
            </div>
            <div>
              <h3 className="text-lg font-semibold" style={{ color: 'var(--primary-text)' }}>MEGA MIND</h3>
              <p className="text-sm" style={{ color: 'rgba(147, 51, 234, 0.8)' }}>Fusión de Cerebros IA - Máxima Precisión</p>
            </div>
          </div>
          
          <div className="space-y-3">
            {megaMindPredictions.map((prediction, index) => (
              <div key={index} className="rounded-xl p-4 hover:bg-purple-700/30 transition-all duration-200 border" style={{ backgroundColor: 'rgba(147, 51, 234, 0.2)', borderColor: 'rgba(147, 51, 234, 0.3)' }}>
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                  <div className="flex items-center gap-3">
                    <div className={`w-12 h-12 rounded-xl flex items-center justify-center shadow-lg`} style={{
                      backgroundColor: prediction.direction === 'up' ? 'rgba(34, 197, 94, 0.2)' :
                      prediction.direction === 'down' ? 'rgba(239, 68, 68, 0.2)' : 'rgba(100, 116, 139, 0.2)',
                      color: prediction.direction === 'up' ? 'var(--success-color)' :
                      prediction.direction === 'down' ? 'var(--danger-color)' : 'var(--secondary-text)',
                      boxShadow: prediction.direction === 'up' ? '0 10px 25px rgba(34, 197, 94, 0.25)' :
                      prediction.direction === 'down' ? '0 10px 25px rgba(239, 68, 68, 0.25)' : '0 10px 25px rgba(100, 116, 139, 0.25)'
                    }}>
                      {prediction.direction === 'up' ? <TrendingUp className="w-6 h-6" /> :
                       prediction.direction === 'down' ? <TrendingDown className="w-6 h-6" /> :
                       <Activity className="w-6 h-6" />}
                    </div>
                    
                    <div className="flex-1 min-w-0">
                      <p className="font-semibold text-lg" style={{ color: 'var(--primary-text)' }}>{prediction.pair}</p>
                      <p className="text-sm line-clamp-2" style={{ color: 'var(--secondary-text)' }}>{prediction.reasoning}</p>
                      <div className="flex items-center gap-2 mt-1">
                        <div className="w-2 h-2 rounded-full" style={{ backgroundColor: '#a855f7' }}></div>
                        <p className="text-xs" style={{ color: 'rgba(147, 51, 234, 0.8)' }}>Colaboración: {(prediction.collaboration_score * 100).toFixed(1)}%</p>
                      </div>
                    </div>
                  </div>
                  
                  <div className="text-right">
                    <div className="rounded-lg p-3" style={{ backgroundColor: 'rgba(147, 51, 234, 0.3)' }}>
                      <p className="font-bold text-lg" style={{ color: 'var(--primary-text)' }}>${prediction.target_price.toFixed(4)}</p>
                      <p className="text-sm" style={{ color: 'var(--secondary-text)' }}>{prediction.confidence.toFixed(1)}% confianza</p>
                      <p className="text-xs" style={{ color: 'rgba(147, 51, 234, 0.8)' }}>Método: {prediction.fusion_method}</p>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
          
          {/* Información adicional de Mega Mind */}
          {megaMindCollaboration && (
            <div className="mt-4 rounded-xl p-4 border" style={{ backgroundColor: 'rgba(147, 51, 234, 0.2)', borderColor: 'rgba(147, 51, 234, 0.2)' }}>
              <h4 className="text-sm font-semibold mb-3 flex items-center gap-2" style={{ color: 'rgba(147, 51, 234, 0.8)' }}>
                <Brain className="w-4 h-4" />
                Estado de Colaboración
              </h4>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                <div className="rounded-lg p-2" style={{ backgroundColor: 'rgba(147, 51, 234, 0.3)' }}>
                  <p className="font-medium" style={{ color: 'rgba(147, 51, 234, 0.8)' }}>Score</p>
                  <p className="font-semibold" style={{ color: 'var(--primary-text)' }}>{(megaMindCollaboration.collaboration_score * 100).toFixed(1)}%</p>
                </div>
                <div className="rounded-lg p-2" style={{ backgroundColor: 'rgba(147, 51, 234, 0.3)' }}>
                  <p className="font-medium" style={{ color: 'rgba(147, 51, 234, 0.8)' }}>Consenso</p>
                  <p className="font-semibold" style={{ color: 'var(--primary-text)' }}>{(megaMindCollaboration.consensus_level * 100).toFixed(1)}%</p>
                </div>
                <div className="rounded-lg p-2" style={{ backgroundColor: 'rgba(147, 51, 234, 0.3)' }}>
                  <p className="font-medium" style={{ color: 'rgba(147, 51, 234, 0.8)' }}>Estado</p>
                  <p className="font-semibold capitalize" style={{ color: 'var(--primary-text)' }}>{megaMindCollaboration.collaboration_status}</p>
                </div>
                <div className="rounded-lg p-2" style={{ backgroundColor: 'rgba(147, 51, 234, 0.3)' }}>
                  <p className="font-medium" style={{ color: 'rgba(147, 51, 234, 0.8)' }}>Cerebros</p>
                  <p className="font-semibold" style={{ color: 'var(--primary-text)' }}>3 Activos</p>
                </div>
              </div>
            </div>
          )}

          {/* Información general de Mega Mind cuando no hay predicciones */}
          {megaMindPredictions.length === 0 && (
            <div className="mt-4 rounded-xl p-4 border" style={{ backgroundColor: 'rgba(147, 51, 234, 0.1)', borderColor: 'rgba(147, 51, 234, 0.2)' }}>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-4">
                <div className="rounded-xl p-3" style={{ backgroundColor: 'rgba(147, 51, 234, 0.3)' }}>
                  <p className="text-sm font-medium" style={{ color: 'rgba(147, 51, 234, 0.8)' }}>Colaboración de Cerebros</p>
                  <p className="font-semibold" style={{ color: 'var(--primary-text)' }}>✓ Activada</p>
                </div>
                <div className="rounded-xl p-3" style={{ backgroundColor: 'rgba(147, 51, 234, 0.3)' }}>
                  <p className="text-sm font-medium" style={{ color: 'rgba(147, 51, 234, 0.8)' }}>Fusión de Estrategias</p>
                  <p className="font-semibold" style={{ color: 'var(--primary-text)' }}>✓ Optimizada</p>
                </div>
                <div className="rounded-xl p-3" style={{ backgroundColor: 'rgba(147, 51, 234, 0.3)' }}>
                  <p className="text-sm font-medium" style={{ color: 'rgba(147, 51, 234, 0.8)' }}>Análisis Multi-Timeframe</p>
                  <p className="font-semibold" style={{ color: 'var(--primary-text)' }}>✓ Avanzado</p>
                </div>
                <div className="rounded-xl p-3" style={{ backgroundColor: 'rgba(147, 51, 234, 0.3)' }}>
                  <p className="text-sm font-medium" style={{ color: 'rgba(147, 51, 234, 0.8)' }}>Correlación Cross-Asset</p>
                  <p className="font-semibold" style={{ color: 'var(--primary-text)' }}>✓ Institucional</p>
                </div>
              </div>
              
              <div className="rounded-xl p-4 border" style={{ backgroundColor: 'rgba(147, 51, 234, 0.1)', borderColor: 'rgba(147, 51, 234, 0.2)' }}>
                <p className="text-sm leading-relaxed" style={{ color: 'rgba(147, 51, 234, 0.8)' }}>
                  <strong>MEGA MIND</strong> combina la potencia de Brain Max, Brain Ultra y Brain Predictor 
                  para crear estrategias de trading institucionales con precisión superior al 95%.
                </p>
              </div>
            </div>
          )}
        </div>
      )}



      {/* Información de suscripción */}
      <div className="backdrop-blur-sm rounded-2xl border p-4 sm:p-6" style={{ 
        background: 'linear-gradient(to bottom right, rgba(59, 130, 246, 0.1), rgba(20, 184, 166, 0.1), rgba(6, 182, 212, 0.1))',
        borderColor: 'rgba(59, 130, 246, 0.2)'
      }}>
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{ background: 'linear-gradient(to right, var(--accent-text), #06b6d4)' }}>
            {subscription?.planType === 'institutional' && <Crown className="w-5 h-5 text-white" />}
            {subscription?.planType === 'premium' && <Crown className="w-5 h-5 text-white" />}
            {subscription?.planType === 'expert' && <Star className="w-5 h-5 text-white" />}
            {!subscription?.planType && <Shield className="w-5 h-5 text-white" />}
          </div>
          <div>
            <h3 className="text-lg font-semibold" style={{ color: 'var(--primary-text)' }}>Plan Actual</h3>
            <p className="text-sm" style={{ color: 'var(--secondary-text)' }}>{subscription?.planType?.toUpperCase() || 'STARTER'}</p>
          </div>
        </div>
        
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-4">
          <div className="rounded-xl p-3" style={{ backgroundColor: 'rgba(30, 41, 59, 0.3)' }}>
            <p className="text-sm font-medium" style={{ color: 'var(--secondary-text)' }}>Pares Disponibles</p>
            <p className="font-semibold text-lg" style={{ color: 'var(--primary-text)' }}>{getAvailablePairs().length} pares</p>
          </div>
          <div className="rounded-xl p-3" style={{ backgroundColor: 'rgba(30, 41, 59, 0.3)' }}>
            <p className="text-sm font-medium" style={{ color: 'var(--secondary-text)' }}>Estilos de Trading</p>
            <p className="font-semibold text-lg" style={{ color: 'var(--primary-text)' }}>{getAvailableStyles().length} estilos</p>
          </div>
          <div className="rounded-xl p-3" style={{ backgroundColor: 'rgba(30, 41, 59, 0.3)' }}>
            <p className="text-sm font-medium" style={{ color: 'var(--secondary-text)' }}>Cerebros IA</p>
            <p className="font-semibold text-lg" style={{ color: 'var(--primary-text)' }}>{getAvailableBrains().length} cerebros</p>
          </div>
        </div>

        {/* Características del Plan */}
        <div className="mb-4">
          <h4 className="text-sm font-medium text-slate-300 mb-3">Características Disponibles</h4>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
            {Object.entries(getAvailableFeatures()).map(([feature, available]) => (
              <div key={feature} className="flex items-center gap-2 p-2 bg-slate-700/30 rounded-lg">
                <div className={`w-2 h-2 rounded-full ${available ? 'bg-green-400 animate-pulse' : 'bg-slate-500'}`}></div>
                <span className={`${available ? 'text-white' : 'text-slate-500'}`}>
                  {feature.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Límites del Plan */}
        <div>
          <h4 className="text-sm font-medium mb-3" style={{ color: 'var(--secondary-text)' }}>Límites del Plan</h4>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
            <div className="rounded-lg p-2" style={{ backgroundColor: 'rgba(30, 41, 59, 0.3)' }}>
              <p className="font-medium" style={{ color: 'var(--secondary-text)' }}>Predicciones/día</p>
              <p className="font-semibold" style={{ color: 'var(--primary-text)' }}>{getPlanLimitations().maxPredictionsPerDay}</p>
            </div>
            <div className="rounded-lg p-2" style={{ backgroundColor: 'rgba(30, 41, 59, 0.3)' }}>
              <p className="font-medium" style={{ color: 'var(--secondary-text)' }}>Timeframes</p>
              <p className="font-semibold" style={{ color: 'var(--primary-text)' }}>{getPlanLimitations().maxTimeframes}</p>
            </div>
            <div className="rounded-lg p-2" style={{ backgroundColor: 'rgba(30, 41, 59, 0.3)' }}>
              <p className="font-medium" style={{ color: 'var(--secondary-text)' }}>Backtests/mes</p>
              <p className="font-semibold" style={{ color: 'var(--primary-text)' }}>{getPlanLimitations().maxBacktests}</p>
            </div>
            <div className="rounded-lg p-2" style={{ backgroundColor: 'rgba(30, 41, 59, 0.3)' }}>
              <p className="font-medium" style={{ color: 'var(--secondary-text)' }}>Soporte</p>
              <p className="font-semibold capitalize" style={{ color: 'var(--primary-text)' }}>{getPlanLimitations().supportLevel}</p>
            </div>
          </div>
        </div>
      </div>



      {/* Funciones Avanzadas según Plan */}
      {getAvailableFeatures().multiTimeframe && (
        <div className="backdrop-blur-sm rounded-2xl border p-4 sm:p-6" style={{ 
          background: 'linear-gradient(to bottom right, rgba(34, 197, 94, 0.1), rgba(59, 130, 246, 0.1), rgba(20, 184, 166, 0.1))',
          borderColor: 'rgba(34, 197, 94, 0.2)'
        }}>
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{ background: 'linear-gradient(to right, var(--success-color), var(--accent-text))' }}>
              <BarChart3 className="w-5 h-5 text-white" />
            </div>
            <div>
              <h3 className="text-lg font-semibold" style={{ color: 'var(--primary-text)' }}>Análisis Multi-Timeframe</h3>
              <p className="text-sm" style={{ color: 'rgba(34, 197, 94, 0.8)' }}>Análisis de confluencia temporal</p>
            </div>
          </div>
          
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div className="rounded-xl p-3" style={{ backgroundColor: 'rgba(34, 197, 94, 0.3)' }}>
              <p className="text-sm font-medium" style={{ color: 'rgba(34, 197, 94, 0.8)' }}>Timeframes Disponibles</p>
              <p className="font-semibold" style={{ color: 'var(--primary-text)' }}>1m, 5m, 15m, 1H, 4H, 1D</p>
            </div>
            <div className="rounded-xl p-3" style={{ backgroundColor: 'rgba(34, 197, 94, 0.3)' }}>
              <p className="text-sm font-medium" style={{ color: 'rgba(34, 197, 94, 0.8)' }}>Análisis Confluencia</p>
              <p className="font-semibold" style={{ color: 'var(--primary-text)' }}>✓ Activado</p>
            </div>
            <div className="rounded-xl p-3" style={{ backgroundColor: 'rgba(34, 197, 94, 0.3)' }}>
              <p className="text-sm font-medium" style={{ color: 'rgba(34, 197, 94, 0.8)' }}>Señales Multi-TF</p>
              <p className="font-semibold" style={{ color: 'var(--primary-text)' }}>✓ Generadas</p>
            </div>
          </div>
        </div>
      )}

      {getAvailableFeatures().crossAsset && (
        <div className="backdrop-blur-sm rounded-2xl border p-4 sm:p-6" style={{ 
          background: 'linear-gradient(to bottom right, rgba(249, 115, 22, 0.1), rgba(239, 68, 68, 0.1), rgba(236, 72, 153, 0.1))',
          borderColor: 'rgba(249, 115, 22, 0.2)'
        }}>
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{ background: 'linear-gradient(to right, #f97316, #ef4444)' }}>
              <Activity className="w-5 h-5 text-white" />
            </div>
            <div>
              <h3 className="text-lg font-semibold" style={{ color: 'var(--primary-text)' }}>Análisis Cross-Asset</h3>
              <p className="text-sm" style={{ color: 'rgba(249, 115, 22, 0.8)' }}>Correlaciones entre activos</p>
            </div>
          </div>
          
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div className="rounded-xl p-3" style={{ backgroundColor: 'rgba(249, 115, 22, 0.3)' }}>
              <p className="text-sm font-medium" style={{ color: 'rgba(249, 115, 22, 0.8)' }}>DXY Correlation</p>
              <p className="font-semibold" style={{ color: 'var(--primary-text)' }}>✓ Monitoreada</p>
            </div>
            <div className="rounded-xl p-3" style={{ backgroundColor: 'rgba(249, 115, 22, 0.3)' }}>
              <p className="text-sm font-medium" style={{ color: 'rgba(249, 115, 22, 0.8)' }}>Gold Correlation</p>
              <p className="font-semibold" style={{ color: 'var(--primary-text)' }}>✓ Analizada</p>
            </div>
            <div className="rounded-xl p-3" style={{ backgroundColor: 'rgba(249, 115, 22, 0.3)' }}>
              <p className="text-sm font-medium" style={{ color: 'rgba(249, 115, 22, 0.8)' }}>S&P 500 Correlation</p>
              <p className="font-semibold" style={{ color: 'var(--primary-text)' }}>✓ Calculada</p>
            </div>
            <div className="rounded-xl p-3" style={{ backgroundColor: 'rgba(249, 115, 22, 0.3)' }}>
              <p className="text-sm font-medium" style={{ color: 'rgba(249, 115, 22, 0.8)' }}>Oil Correlation</p>
              <p className="font-semibold" style={{ color: 'var(--primary-text)' }}>✓ Integrada</p>
            </div>
          </div>
        </div>
      )}

      {getAvailableFeatures().economicCalendar && (
        <div className="backdrop-blur-sm rounded-2xl border p-4 sm:p-6" style={{ 
          background: 'linear-gradient(to bottom right, rgba(234, 179, 8, 0.1), rgba(249, 115, 22, 0.1), rgba(239, 68, 68, 0.1))',
          borderColor: 'rgba(234, 179, 8, 0.2)'
        }}>
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{ background: 'linear-gradient(to right, #eab308, #f97316)' }}>
              <Clock className="w-5 h-5 text-white" />
            </div>
            <div>
              <h3 className="text-lg font-semibold" style={{ color: 'var(--primary-text)' }}>Calendario Económico</h3>
              <p className="text-sm" style={{ color: 'rgba(234, 179, 8, 0.8)' }}>Eventos económicos importantes</p>
            </div>
          </div>
          
          <div className="space-y-3">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between p-3 rounded-xl border" style={{ backgroundColor: 'rgba(234, 179, 8, 0.1)', borderColor: 'rgba(234, 179, 8, 0.2)' }}>
              <div>
                <p className="font-semibold" style={{ color: 'var(--primary-text)' }}>FOMC Interest Rate Decision</p>
                <p className="text-sm" style={{ color: 'rgba(234, 179, 8, 0.8)' }}>En 3 días - Alto Impacto</p>
              </div>
              <div className="text-right mt-2 sm:mt-0">
                <p className="font-semibold" style={{ color: 'var(--primary-text)' }}>USD</p>
                <p className="text-sm" style={{ color: 'rgba(234, 179, 8, 0.8)' }}>Bullish</p>
              </div>
            </div>
            
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between p-3 rounded-xl border" style={{ backgroundColor: 'rgba(234, 179, 8, 0.1)', borderColor: 'rgba(234, 179, 8, 0.2)' }}>
              <div>
                <p className="font-semibold" style={{ color: 'var(--primary-text)' }}>Non-Farm Payrolls</p>
                <p className="text-sm" style={{ color: 'rgba(234, 179, 8, 0.8)' }}>En 7 días - Alto Impacto</p>
              </div>
              <div className="text-right mt-2 sm:mt-0">
                <p className="font-semibold" style={{ color: 'var(--primary-text)' }}>USD</p>
                <p className="text-sm" style={{ color: 'rgba(234, 179, 8, 0.8)' }}>Neutral</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {getAvailableFeatures().autoTraining && (
        <div className="backdrop-blur-sm rounded-2xl border p-4 sm:p-6" style={{ 
          background: 'linear-gradient(to bottom right, rgba(99, 102, 241, 0.1), rgba(147, 51, 234, 0.1), rgba(236, 72, 153, 0.1))',
          borderColor: 'rgba(99, 102, 241, 0.2)'
        }}>
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{ background: 'linear-gradient(to right, #6366f1, #a855f7)' }}>
              <RefreshCw className="w-5 h-5 text-white" />
            </div>
            <div>
              <h3 className="text-lg font-semibold" style={{ color: 'var(--primary-text)' }}>Auto-Training Inteligente</h3>
              <p className="text-sm" style={{ color: 'rgba(99, 102, 241, 0.8)' }}>Entrenamiento automático de modelos</p>
            </div>
          </div>
          
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-4">
            <div className="rounded-xl p-3" style={{ backgroundColor: 'rgba(99, 102, 241, 0.3)' }}>
              <p className="text-sm font-medium" style={{ color: 'rgba(99, 102, 241, 0.8)' }}>Estado</p>
              <p className="font-semibold" style={{ color: 'var(--primary-text)' }}>✓ Activo</p>
            </div>
            <div className="rounded-xl p-3" style={{ backgroundColor: 'rgba(99, 102, 241, 0.3)' }}>
              <p className="text-sm font-medium" style={{ color: 'rgba(99, 102, 241, 0.8)' }}>Última Actualización</p>
              <p className="font-semibold" style={{ color: 'var(--primary-text)' }}>Hace 2 horas</p>
            </div>
            <div className="rounded-xl p-3" style={{ backgroundColor: 'rgba(99, 102, 241, 0.3)' }}>
              <p className="text-sm font-medium" style={{ color: 'rgba(99, 102, 241, 0.8)' }}>Próximo Entrenamiento</p>
              <p className="font-semibold" style={{ color: 'var(--primary-text)' }}>En 6 horas</p>
            </div>
          </div>
          
          <div className="rounded-xl p-4 border" style={{ backgroundColor: 'rgba(99, 102, 241, 0.1)', borderColor: 'rgba(99, 102, 241, 0.2)' }}>
            <p className="text-sm leading-relaxed" style={{ color: 'rgba(99, 102, 241, 0.8)' }}>
              El sistema se entrena automáticamente con nuevos datos de mercado para mantener 
              la precisión óptima de los modelos.
            </p>
          </div>
        </div>
      )}

      {/* Espaciado final para móvil */}
      <div className="h-4 sm:h-6"></div>
      </div>
    </div>
  );
}; 