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
  EyeOff
} from 'lucide-react';
import { useAuth } from '../../contexts/AuthContext';
import { useFeatureAccess } from '../../hooks/useFeatureAccess';

interface BrainTraderProps {}

interface ModelInfo {
  brainType: 'brain_max' | 'brain_ultra' | 'brain_predictor';
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
  targetPrice: number;
  timeframe: string;
  reasoning: string;
}

interface Signal {
  pair: string;
  type: 'buy' | 'sell' | 'hold';
  strength: 'strong' | 'medium' | 'weak';
  confidence: number;
  entryPrice: number;
  stopLoss: number;
  takeProfit: number;
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
}

export const BrainTrader: React.FC<BrainTraderProps> = () => {
  const { subscription } = useAuth();
  const { checkAccess, checkFeature } = useFeatureAccess();
  
  // Estados principales
  const [selectedPair, setSelectedPair] = useState('EURUSD');
  const [selectedStyle, setSelectedStyle] = useState('day_trading');
  const [activeBrain, setActiveBrain] = useState<'brain_max' | 'brain_ultra' | 'brain_predictor'>('brain_max');
  const [isAutoTrading, setIsAutoTrading] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Estados de datos
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [signals, setSignals] = useState<Signal[]>([]);
  const [trends, setTrends] = useState<Trend[]>([]);
  const [loading, setLoading] = useState(false);

  // Configuración según suscripción
  const getAvailablePairs = () => {
    if (!subscription || subscription.status !== 'active') {
      return ['EURUSD']; // Freemium
    }
    
    switch (subscription.planType) {
      case 'freemium':
        return ['EURUSD'];
      case 'basic':
        return ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD'];
      case 'pro':
      case 'elite':
        return ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'EURGBP', 'GBPJPY', 'EURJPY'];
      default:
        return ['EURUSD'];
    }
  };

  const getAvailableStyles = () => {
    if (!subscription || subscription.status !== 'active') {
      return ['day_trading']; // Freemium
    }
    
    switch (subscription.planType) {
      case 'freemium':
        return ['day_trading'];
      case 'basic':
      case 'pro':
      case 'elite':
        return ['scalping', 'day_trading', 'swing_trading', 'position_trading'];
      default:
        return ['day_trading'];
    }
  };

  const getAvailableBrains = () => {
    if (!subscription || subscription.status !== 'active') {
      return ['brain_max']; // Freemium
    }
    
    switch (subscription.planType) {
      case 'freemium':
        return ['brain_max'];
      case 'basic':
        return ['brain_max'];
      case 'pro':
        return ['brain_max', 'brain_ultra'];
      case 'elite':
        return ['brain_max', 'brain_ultra', 'brain_predictor'];
      case 'institutional':
        return ['brain_max', 'brain_ultra', 'brain_predictor', 'mega_mind'];
      default:
        return ['brain_max'];
    }
  };

  // Configuración especial para MEGA MIND
  const isMegaMindAvailable = () => {
    return subscription?.planType === 'institutional';
  };

  const getMegaMindFeatures = () => {
    return {
      brainCollaboration: true,
      brainFusion: true,
      brainArena: true,
      brainEvolution: true,
      brainOrchestration: true
    };
  };

  // Cargar datos del modelo
  const loadModelData = async () => {
    setLoading(true);
    try {
      // Simular llamada a API
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Datos simulados según el cerebro activo
      const mockModelInfo: ModelInfo = {
        brainType: activeBrain,
        pair: selectedPair,
        style: selectedStyle,
        accuracy: Math.random() * 20 + 80, // 80-100%
        lastUpdate: new Date().toISOString(),
        status: 'active'
      };
      
      const mockPredictions: Prediction[] = [
        {
          pair: selectedPair,
          direction: Math.random() > 0.5 ? 'up' : 'down',
          confidence: Math.random() * 30 + 70,
          targetPrice: 1.0925 + (Math.random() - 0.5) * 0.01,
          timeframe: '1H',
          reasoning: 'Análisis técnico basado en RSI y MACD'
        }
      ];
      
      const mockSignals: Signal[] = [
        {
          pair: selectedPair,
          type: Math.random() > 0.5 ? 'buy' : 'sell',
          strength: Math.random() > 0.7 ? 'strong' : 'medium',
          confidence: Math.random() * 40 + 60,
          entryPrice: 1.0925,
          stopLoss: 1.0925 - 0.005,
          takeProfit: 1.0925 + 0.015,
          timestamp: new Date().toISOString()
        }
      ];
      
      const mockTrends: Trend[] = [
        {
          pair: selectedPair,
          direction: Math.random() > 0.5 ? 'bullish' : 'bearish',
          strength: Math.random() * 50 + 50,
          timeframe: '4H',
          support: 1.0900,
          resistance: 1.0950,
          description: 'Tendencia alcista con soporte en 1.0900'
        }
      ];
      
      setModelInfo(mockModelInfo);
      setPredictions(mockPredictions);
      setSignals(mockSignals);
      setTrends(mockTrends);
      
    } catch (error) {
      console.error('Error cargando datos del modelo:', error);
    } finally {
      setLoading(false);
    }
  };

  // Efectos
  useEffect(() => {
    loadModelData();
  }, [selectedPair, selectedStyle, activeBrain]);

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
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-teal-500 rounded-xl flex items-center justify-center">
            <Brain className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white">Brain Trader</h1>
            <p className="text-gray-400">Sistema de IA para trading automático</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-3">
          <button
            onClick={() => setIsAutoTrading(!isAutoTrading)}
            className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all ${
              isAutoTrading 
                ? 'bg-red-500/20 text-red-400 border border-red-500/30' 
                : 'bg-green-500/20 text-green-400 border border-green-500/30'
            }`}
          >
            {isAutoTrading ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            <span>{isAutoTrading ? 'Detener' : 'Iniciar'} Auto Trading</span>
          </button>
          
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center space-x-2 px-4 py-2 bg-gray-700/50 text-gray-300 rounded-lg hover:bg-gray-600/50 transition-all"
          >
            {showAdvanced ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
            <span>Avanzado</span>
          </button>
        </div>
      </div>

      {/* Configuración */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {/* Par de divisas */}
        <div className="space-y-2">
          <label className="text-sm font-medium text-gray-300">Par de Divisas</label>
          <select
            value={selectedPair}
            onChange={(e) => setSelectedPair(e.target.value)}
            className="w-full bg-gray-800/50 border border-gray-600/50 rounded-lg px-3 py-2 text-white focus:border-blue-500 focus:outline-none"
          >
            {getAvailablePairs().map(pair => (
              <option key={pair} value={pair}>{pair}</option>
            ))}
          </select>
        </div>

        {/* Estilo de trading */}
        <div className="space-y-2">
          <label className="text-sm font-medium text-gray-300">Estilo de Trading</label>
          <select
            value={selectedStyle}
            onChange={(e) => setSelectedStyle(e.target.value)}
            className="w-full bg-gray-800/50 border border-gray-600/50 rounded-lg px-3 py-2 text-white focus:border-blue-500 focus:outline-none"
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
          <label className="text-sm font-medium text-gray-300">Cerebro IA</label>
          <select
            value={activeBrain}
            onChange={(e) => setActiveBrain(e.target.value as any)}
            className="w-full bg-gray-800/50 border border-gray-600/50 rounded-lg px-3 py-2 text-white focus:border-blue-500 focus:outline-none"
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
          <label className="text-sm font-medium text-gray-300">&nbsp;</label>
          <button
            onClick={loadModelData}
            disabled={loading}
            className="w-full flex items-center justify-center space-x-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white px-4 py-2 rounded-lg transition-all"
          >
            {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <RefreshCw className="w-4 h-4" />}
            <span>{loading ? 'Actualizando...' : 'Actualizar'}</span>
          </button>
        </div>
      </div>

      {/* Información del Modelo */}
      {modelInfo && (
        <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700/50">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-white flex items-center space-x-2">
              <Info className="w-5 h-5 text-blue-400" />
              <span>Información del Modelo</span>
            </h3>
            <div className="flex items-center space-x-2">
              <div className={`w-3 h-3 rounded-full ${
                modelInfo.status === 'active' ? 'bg-green-400' :
                modelInfo.status === 'training' ? 'bg-yellow-400' : 'bg-red-400'
              }`}></div>
              <span className="text-sm text-gray-400 capitalize">{modelInfo.status}</span>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="space-y-2">
              <p className="text-sm text-gray-400">Cerebro</p>
              <p className="text-white font-medium">{modelInfo.brainType.replace('_', ' ').toUpperCase()}</p>
            </div>
            <div className="space-y-2">
              <p className="text-sm text-gray-400">Precisión</p>
              <p className="text-white font-medium">{modelInfo.accuracy.toFixed(1)}%</p>
            </div>
            <div className="space-y-2">
              <p className="text-sm text-gray-400">Última Actualización</p>
              <p className="text-white font-medium">{new Date(modelInfo.lastUpdate).toLocaleString()}</p>
            </div>
          </div>
        </div>
      )}

      {/* Predicciones */}
      {predictions.length > 0 && (
        <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700/50">
          <h3 className="text-lg font-semibold text-white flex items-center space-x-2 mb-4">
            <Target className="w-5 h-5 text-green-400" />
            <span>Predicciones</span>
          </h3>
          
          <div className="space-y-4">
            {predictions.map((prediction, index) => (
              <div key={index} className="flex items-center justify-between p-4 bg-gray-700/30 rounded-lg">
                <div className="flex items-center space-x-4">
                  <div className={`w-12 h-12 rounded-lg flex items-center justify-center ${
                    prediction.direction === 'up' ? 'bg-green-500/20 text-green-400' :
                    prediction.direction === 'down' ? 'bg-red-500/20 text-red-400' :
                    'bg-gray-500/20 text-gray-400'
                  }`}>
                    {prediction.direction === 'up' ? <TrendingUp className="w-6 h-6" /> :
                     prediction.direction === 'down' ? <TrendingDown className="w-6 h-6" /> :
                     <Activity className="w-6 h-6" />}
                  </div>
                  
                  <div>
                    <p className="text-white font-medium">{prediction.pair}</p>
                    <p className="text-sm text-gray-400">{prediction.reasoning}</p>
                  </div>
                </div>
                
                <div className="text-right">
                  <p className="text-white font-medium">${prediction.targetPrice.toFixed(4)}</p>
                  <p className="text-sm text-gray-400">{prediction.confidence.toFixed(1)}% confianza</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Señales */}
      {signals.length > 0 && (
        <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700/50">
          <h3 className="text-lg font-semibold text-white flex items-center space-x-2 mb-4">
            <Zap className="w-5 h-5 text-yellow-400" />
            <span>Señales de Trading</span>
          </h3>
          
          <div className="space-y-4">
            {signals.map((signal, index) => (
              <div key={index} className="flex items-center justify-between p-4 bg-gray-700/30 rounded-lg">
                <div className="flex items-center space-x-4">
                  <div className={`w-12 h-12 rounded-lg flex items-center justify-center ${
                    signal.type === 'buy' ? 'bg-green-500/20 text-green-400' :
                    signal.type === 'sell' ? 'bg-red-500/20 text-red-400' :
                    'bg-gray-500/20 text-gray-400'
                  }`}>
                    {signal.type === 'buy' ? <CheckCircle className="w-6 h-6" /> :
                     signal.type === 'sell' ? <XCircle className="w-6 h-6" /> :
                     <Pause className="w-6 h-6" />}
                  </div>
                  
                  <div>
                    <p className="text-white font-medium">{signal.pair} - {signal.type.toUpperCase()}</p>
                    <p className="text-sm text-gray-400">Fuerza: {signal.strength}</p>
                  </div>
                </div>
                
                <div className="text-right">
                  <p className="text-white font-medium">${signal.entryPrice.toFixed(4)}</p>
                  <p className="text-sm text-gray-400">{signal.confidence.toFixed(1)}% confianza</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Tendencias */}
      {trends.length > 0 && (
        <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700/50">
          <h3 className="text-lg font-semibold text-white flex items-center space-x-2 mb-4">
            <BarChart3 className="w-5 h-5 text-purple-400" />
            <span>Tendencias</span>
          </h3>
          
          <div className="space-y-4">
            {trends.map((trend, index) => (
              <div key={index} className="p-4 bg-gray-700/30 rounded-lg">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-3">
                    <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                      trend.direction === 'bullish' ? 'bg-green-500/20 text-green-400' :
                      trend.direction === 'bearish' ? 'bg-red-500/20 text-red-400' :
                      'bg-gray-500/20 text-gray-400'
                    }`}>
                      {trend.direction === 'bullish' ? <TrendingUp className="w-5 h-5" /> :
                       trend.direction === 'bearish' ? <TrendingDown className="w-5 h-5" /> :
                       <Activity className="w-5 h-5" />}
                    </div>
                    
                    <div>
                      <p className="text-white font-medium">{trend.pair}</p>
                      <p className="text-sm text-gray-400 capitalize">{trend.direction}</p>
                    </div>
                  </div>
                  
                  <div className="text-right">
                    <p className="text-white font-medium">{trend.strength.toFixed(0)}%</p>
                    <p className="text-sm text-gray-400">Fuerza</p>
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-gray-400">Soporte</p>
                    <p className="text-white">${trend.support.toFixed(4)}</p>
                  </div>
                  <div>
                    <p className="text-gray-400">Resistencia</p>
                    <p className="text-white">${trend.resistance.toFixed(4)}</p>
                  </div>
                </div>
                
                <p className="text-sm text-gray-400 mt-3">{trend.description}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Información de suscripción */}
      <div className="bg-gradient-to-r from-blue-500/10 to-teal-500/10 rounded-xl p-6 border border-blue-500/20">
        <div className="flex items-center space-x-3 mb-4">
          {subscription?.planType === 'elite' && <Crown className="w-6 h-6 text-yellow-400" />}
          {subscription?.planType === 'pro' && <Star className="w-6 h-6 text-purple-400" />}
          <h3 className="text-lg font-semibold text-white">Plan Actual: {subscription?.planType?.toUpperCase() || 'FREEMIUM'}</h3>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          <div>
            <p className="text-gray-400">Pares Disponibles</p>
            <p className="text-white">{getAvailablePairs().length} pares</p>
          </div>
          <div>
            <p className="text-gray-400">Estilos de Trading</p>
            <p className="text-white">{getAvailableStyles().length} estilos</p>
          </div>
          <div>
            <p className="text-gray-400">Cerebros IA</p>
            <p className="text-white">{getAvailableBrains().length} cerebros</p>
          </div>
        </div>
      </div>
    </div>
  );
}; 