import React, { useState, useEffect } from 'react';
import { 
  CheckCircle,
  Target,
  Brain,
  Lock
} from 'lucide-react';
import { useIntelligentAnalysis, TradingType, AnalysisResult } from '../../hooks/useIntelligentAnalysis';
import { useAuth } from '../../contexts/AuthContext';
import { useFeatureAccess } from '../../hooks/useFeatureAccess';

export const Analysis: React.FC = () => {
  const { subscription, user } = useAuth();
  const { requireAccess, showUpgradeModal, upgradeInfo, closeUpgradeModal } = useFeatureAccess();
  
  // Verificar acceso a la sección de análisis
  const hasAccess = requireAccess('analysis');
  
  const [selectedPair, setSelectedPair] = useState('EURUSD');
  const [selectedTradingType, setSelectedTradingType] = useState<TradingType>({
    id: 'day_trading',
    name: 'Day Trading',
    description: '15-30 minutos Balance óptimo',
    timeframe: '15-30m',
    reason: 'Operaciones intradía con balance riesgo/beneficio',
    color: 'text-blue-400'
  });

  // Hook de análisis inteligente
  const { executeAnalysis, lastAnalysis, isAnalyzing } = useIntelligentAnalysis();

  // Tipos de trading disponibles
  const tradingTypes: TradingType[] = [
    {
      id: 'scalping',
      name: 'Scalping',
      description: '1-5 minutos Máxima precisión',
      timeframe: '1-5m',
      reason: 'Operaciones rápidas de alta precisión',
      color: 'text-pink-400'
    },
    {
      id: 'day_trading',
      name: 'Day Trading',
      description: '15-30 minutos Balance óptimo',
      timeframe: '15-30m',
      reason: 'Operaciones intradía con balance riesgo/beneficio',
      color: 'text-blue-400'
    },
    {
      id: 'swing_trading',
      name: 'Swing Trading',
      description: '1-4 horas Tendencias medias',
      timeframe: '1-4h',
      reason: 'Captura de tendencias de mediano plazo',
      color: 'text-green-400'
    },
    {
      id: 'position_trading',
      name: 'Position Trading',
      description: '1 día Tendencias largas',
      timeframe: '1d',
      reason: 'Posiciones de largo plazo',
      color: 'text-yellow-400'
    }
  ];

  const tradingPairs = [
    { pair: 'EURUSD', label: 'EUR/USD' },
    { pair: 'GBPUSD', label: 'GBP/USD' },
    { pair: 'USDJPY', label: 'USD/JPY' },
    { pair: 'AUDUSD', label: 'AUD/USD' },
    { pair: 'USDCAD', label: 'USD/CAD' },
  ];

  // Filtrar opciones según el plan de suscripción
  const getAvailableTradingTypes = (): TradingType[] => {
    // El admin tiene acceso completo a todos los tipos de trading
    if (user?.role === 'admin') {
      return tradingTypes;
    }
    
    if (!subscription) return [tradingTypes[1]]; // Solo day_trading por defecto
    
    switch (subscription.planType) {
      case 'starter':
        return [tradingTypes[1]]; // Solo day_trading
      case 'trader':
        return [tradingTypes[1], tradingTypes[2]]; // day_trading y swing_trading
      case 'expert':
      case 'premium':
      case 'institutional':
        return tradingTypes; // Todos los tipos
      default:
        return [tradingTypes[1]];
    }
  };

  const getAvailableTradingPairs = () => {
    // El admin tiene acceso completo a todos los pares de trading
    if (user?.role === 'admin') {
      return tradingPairs;
    }
    
    if (!subscription) return tradingPairs.slice(0, 1); // Solo EURUSD por defecto
    
    switch (subscription.planType) {
      case 'starter':
        return tradingPairs.slice(0, 1); // Solo EURUSD
      case 'trader':
        return tradingPairs.slice(0, 3); // EURUSD, GBPUSD, USDJPY
      case 'expert':
      case 'premium':
      case 'institutional':
        return tradingPairs; // Todos los pares
      default:
        return tradingPairs.slice(0, 1);
    }
  };

  // Asegurar que las selecciones actuales sean válidas para el plan
  useEffect(() => {
    const availableTypes = getAvailableTradingTypes();
    const availablePairs = getAvailableTradingPairs();
    
    // Verificar si el tipo de trading seleccionado está disponible
    if (!availableTypes.find(type => type.id === selectedTradingType.id)) {
      setSelectedTradingType(availableTypes[0]);
    }
    
    // Verificar si el par seleccionado está disponible
    if (!availablePairs.find(pair => pair.pair === selectedPair)) {
      setSelectedPair(availablePairs[0].pair);
    }
  }, [subscription]);

  // Función para ejecutar análisis inteligente
  const handleIntelligentAnalysis = async () => {
    try {
      console.log(`Ejecutando análisis inteligente para ${selectedPair} con tipo: ${selectedTradingType.name}`);
      await executeAnalysis(selectedTradingType, selectedPair);
    } catch (error) {
      console.error('Error ejecutando análisis inteligente:', error);
    }
  };

  // Ejecutar análisis automáticamente cuando cambie el par o tipo de trading
  useEffect(() => {
    if (hasAccess) {
      handleIntelligentAnalysis();
    }
  }, [selectedPair, selectedTradingType, hasAccess]);

  const getRiskLevelColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'low': return 'text-green-400 bg-green-500/20';
      case 'medium': return 'text-yellow-400 bg-yellow-500/20';
      case 'high': return 'text-red-400 bg-red-500/20';
      default: return 'text-gray-400 bg-gray-500/20';
    }
  };

  // Si no tiene acceso, mostrar mensaje de restricción
  if (!hasAccess) {
    return (
      <div className="p-6">
        <div className="trading-card p-8 text-center">
          <Lock className="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <h2 className="text-xl font-bold text-white mb-2">Análisis Inteligente</h2>
          <p className="text-gray-400 mb-4">
            Esta función requiere un plan de suscripción activo.
          </p>
          <button 
            onClick={() => requireAccess('analysis')}
            className="bg-purple-600 hover:bg-purple-700 text-white px-6 py-2 rounded-lg transition-colors"
          >
            Ver Planes Disponibles
          </button>
        </div>
      </div>
    );
  }

  const availableTradingTypes = getAvailableTradingTypes();
  const availableTradingPairs = getAvailableTradingPairs();

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between space-y-4 sm:space-y-0">
        <div>
          <h1 className="text-2xl font-bold text-white">Análisis Inteligente</h1>
          <p className="text-gray-400">Análisis avanzado con IA para optimizar tus decisiones de trading</p>
          {subscription && (
            <div className="flex items-center space-x-2 mt-2">
              <span className="text-xs text-gray-500">Plan:</span>
              <span className="text-xs font-medium text-purple-400 capitalize">{subscription.planType}</span>
              {user?.role === 'admin' && (
                <span className="text-xs text-green-400 bg-green-500/20 px-2 py-1 rounded">
                  Admin - Acceso Completo
                </span>
              )}
              {subscription.planType === 'starter' && user?.role !== 'admin' && (
                <span className="text-xs text-yellow-400 bg-yellow-500/20 px-2 py-1 rounded">
                  Limitado a EUR/USD y Day Trading
                </span>
              )}
            </div>
          )}
        </div>
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2 bg-purple-500/20 border border-purple-500/30 rounded-lg px-3 py-2">
            <div className="w-2 h-2 bg-purple-400 rounded-full animate-pulse"></div>
            <span className="text-sm text-purple-400 font-medium">IA Activa</span>
          </div>
        </div>
      </div>

      {/* Controles Simplificados */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Par de Trading */}
        <div className="trading-card p-4">
          <label className="block text-sm text-gray-400 mb-2">
            Par de Trading
            {subscription?.planType === 'starter' && user?.role !== 'admin' && (
              <span className="text-xs text-yellow-400 ml-2">(Limitado)</span>
            )}
          </label>
          <select 
            value={selectedPair}
            onChange={(e) => setSelectedPair(e.target.value)}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white focus:border-purple-500 focus:outline-none"
            disabled={availableTradingPairs.length === 1 && user?.role !== 'admin'}
          >
            {availableTradingPairs.map(pair => (
              <option key={pair.pair} value={pair.pair}>{pair.label}</option>
            ))}
          </select>
          {subscription?.planType === 'starter' && user?.role !== 'admin' && (
            <p className="text-xs text-gray-500 mt-1">
              Plan Starter: Solo EUR/USD disponible. Actualiza para más pares.
            </p>
          )}
        </div>

        {/* Tipo de Trading */}
        <div className="trading-card p-4">
          <label className="block text-sm text-gray-400 mb-2">
            Tipo de Trading
            {subscription?.planType === 'starter' && user?.role !== 'admin' && (
              <span className="text-xs text-yellow-400 ml-2">(Limitado)</span>
            )}
          </label>
          <select 
            value={selectedTradingType.id}
            onChange={(e) => {
              const type = availableTradingTypes.find(t => t.id === e.target.value);
              if (type) setSelectedTradingType(type);
            }}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white focus:border-purple-500 focus:outline-none"
            disabled={availableTradingTypes.length === 1 && user?.role !== 'admin'}
          >
            {availableTradingTypes.map(type => (
              <option key={type.id} value={type.id}>{type.name}</option>
            ))}
          </select>
          {subscription?.planType === 'starter' && user?.role !== 'admin' && (
            <p className="text-xs text-gray-500 mt-1">
              Plan Starter: Solo Day Trading disponible. Actualiza para más estrategias.
            </p>
          )}
        </div>
      </div>

      {/* Contenido Principal */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Análisis Inteligente Principal */}
        <div className="lg:col-span-2">
          <div className="trading-card p-6">
            <h2 className="text-xl font-bold text-white mb-4 flex items-center space-x-2">
              <Brain className="w-5 h-5 text-purple-400" />
              <span>Análisis Inteligente</span>
              {isAnalyzing && (
                <div className="flex items-center space-x-2 text-sm text-purple-400">
                  <div className="w-4 h-4 border border-purple-400 border-t-transparent rounded-full animate-spin"></div>
                  <span>Analizando...</span>
                </div>
              )}
            </h2>
            
            <div className="space-y-6">
              {/* Información del Tipo de Trading */}
              <div className="bg-gradient-to-r from-purple-900/20 to-blue-900/20 rounded-lg p-4 border border-purple-500/30">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-lg font-semibold text-purple-300 flex items-center space-x-2">
                    <Target className="w-5 h-5" />
                    <span>{selectedTradingType.name}</span>
                  </h3>
                  <div className={`px-3 py-1 rounded-full text-sm font-medium ${selectedTradingType.color} bg-gray-800/50`}>
                    {selectedTradingType.timeframe}
                  </div>
                </div>
                <p className="text-gray-300 text-sm mb-2">{selectedTradingType.description}</p>
                <p className="text-gray-400 text-xs">{selectedTradingType.reason}</p>
              </div>

              {/* Resultados del Análisis Inteligente */}
              {lastAnalysis ? (
                <div className="space-y-4">
                  {/* Métricas Principales */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="bg-gray-800/50 rounded-lg p-4">
                      <div className="text-sm text-gray-400 mb-1">Confianza</div>
                      <div className="text-2xl font-bold text-white">{(lastAnalysis.confidence * 100).toFixed(1)}%</div>
                      <div className="text-xs text-gray-500">Nivel de confianza del análisis</div>
                    </div>
                    <div className="bg-gray-800/50 rounded-lg p-4">
                      <div className="text-sm text-gray-400 mb-1">Nivel de Riesgo</div>
                      <div className={`text-2xl font-bold ${getRiskLevelColor(lastAnalysis.riskLevel)}`}>
                        {lastAnalysis.riskLevel.toUpperCase()}
                      </div>
                      <div className="text-xs text-gray-500">Evaluación de riesgo</div>
                    </div>
                    <div className="bg-gray-800/50 rounded-lg p-4">
                      <div className="text-sm text-gray-400 mb-1">Timestamp</div>
                      <div className="text-sm font-semibold text-white">
                        {lastAnalysis.timestamp.toLocaleTimeString()}
                      </div>
                      <div className="text-xs text-gray-500">Última actualización</div>
                    </div>
                  </div>

                  {/* Análisis Técnico Detallado */}
                  {lastAnalysis.technicalAnalysis && (
                    <div className="bg-gray-800/50 rounded-lg p-4">
                      <h4 className="text-lg font-semibold text-white mb-3">Análisis Técnico Detallado</h4>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div>
                          <div className="text-sm text-gray-400">RSI</div>
                          <div className="text-lg font-bold text-white">
                            {lastAnalysis.technicalAnalysis.rsi?.toFixed(1) || 'N/A'}
                          </div>
                        </div>
                        <div>
                          <div className="text-sm text-gray-400">MACD</div>
                          <div className="text-lg font-bold text-white">
                            {lastAnalysis.technicalAnalysis.macd?.toFixed(4) || 'N/A'}
                          </div>
                        </div>
                        <div>
                          <div className="text-sm text-gray-400">Tendencia</div>
                          <div className={`text-lg font-bold ${
                            lastAnalysis.technicalAnalysis.trend === 'bullish' ? 'text-green-400' :
                            lastAnalysis.technicalAnalysis.trend === 'bearish' ? 'text-red-400' : 'text-yellow-400'
                          }`}>
                            {lastAnalysis.technicalAnalysis.trend.toUpperCase()}
                          </div>
                        </div>
                        <div>
                          <div className="text-sm text-gray-400">Volatilidad</div>
                          <div className="text-lg font-bold text-white">
                            {(lastAnalysis.technicalAnalysis.volatility * 100).toFixed(1)}%
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Recomendaciones */}
                  <div className="bg-gray-800/50 rounded-lg p-4">
                    <h4 className="text-lg font-semibold text-white mb-3">Recomendaciones IA</h4>
                    <div className="space-y-2">
                      {lastAnalysis.recommendations.map((recommendation, index) => (
                        <div key={index} className="flex items-start space-x-2">
                          <CheckCircle className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
                          <span className="text-sm text-gray-300">{recommendation}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-8">
                  <Brain className="w-12 h-12 text-gray-500 mx-auto mb-4" />
                  <p className="text-gray-400">Selecciona un par y tipo de trading para comenzar el análisis inteligente</p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Panel Lateral */}
        <div className="lg:col-span-1 space-y-6">
          {/* Análisis Actual */}
          <div className="trading-card p-6">
            <h2 className="text-xl font-bold text-white mb-4">Análisis Actual</h2>
            <div className="space-y-3">
              {lastAnalysis ? (
                <div className="border border-purple-500/30 rounded-lg p-3 bg-purple-500/10">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-semibold text-purple-400">ANÁLISIS IA</span>
                    <span className={`text-xs ${getRiskLevelColor(lastAnalysis.riskLevel)}`}>
                      {lastAnalysis.riskLevel.toUpperCase()}
                    </span>
                  </div>
                  <div className="text-sm text-gray-300 mb-1">{selectedPair}</div>
                  <div className="text-xs text-gray-400 mb-2">
                    Confianza: {(lastAnalysis.confidence * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-gray-500">
                    {lastAnalysis.timestamp.toLocaleString()}
                  </div>
                </div>
              ) : (
                <div className="text-center py-4">
                  <p className="text-gray-500 text-sm">Ejecuta un análisis para ver resultados</p>
                </div>
              )}
            </div>
          </div>



          {/* Información del Tipo de Trading */}
          <div className="trading-card p-6">
            <h2 className="text-xl font-bold text-white mb-4">Configuración Actual</h2>
            <div className="space-y-3">
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="text-sm text-gray-400 mb-1">Par Seleccionado</div>
                <div className="text-lg font-semibold text-white">{selectedPair}</div>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="text-sm text-gray-400 mb-1">Tipo de Trading</div>
                <div className="text-lg font-semibold text-white">{selectedTradingType.name}</div>
                <div className="text-xs text-gray-500">{selectedTradingType.timeframe}</div>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="text-sm text-gray-400 mb-1">Descripción</div>
                <div className="text-sm text-gray-300">{selectedTradingType.description}</div>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Modal de Upgrade */}
      {showUpgradeModal && upgradeInfo && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-gray-900 border border-gray-700 rounded-lg p-6 max-w-md w-full mx-4">
            <div className="text-center">
              <Lock className="w-12 h-12 text-yellow-400 mx-auto mb-4" />
              <h3 className="text-xl font-bold text-white mb-2">Función Premium</h3>
              <p className="text-gray-400 mb-4">
                Esta función requiere el plan <span className="text-purple-400 font-semibold">{upgradeInfo.requiredPlan}</span> o superior.
              </p>
              <div className="bg-gray-800 rounded-lg p-3 mb-4">
                <p className="text-sm text-gray-300">
                  Plan actual: <span className="text-yellow-400">{upgradeInfo.currentPlan}</span>
                </p>
                <p className="text-sm text-gray-300">
                  Plan requerido: <span className="text-green-400">{upgradeInfo.requiredPlan}</span>
                </p>
              </div>
              <div className="flex space-x-3">
                <button
                  onClick={closeUpgradeModal}
                  className="flex-1 bg-gray-700 hover:bg-gray-600 text-white px-4 py-2 rounded-lg transition-colors"
                >
                  Cancelar
                </button>
                <button
                  onClick={() => {
                    closeUpgradeModal();
                    // Redirigir a la página de suscripciones
                    window.location.href = '/subscriptions';
                  }}
                  className="flex-1 bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded-lg transition-colors"
                >
                  Ver Planes
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}; 