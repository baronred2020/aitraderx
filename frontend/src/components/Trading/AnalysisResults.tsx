import React, { useState } from 'react';
import { TrendingUp, TrendingDown, AlertTriangle, CheckCircle, Clock, Target, Activity, BarChart3, Bell, Settings, Save } from 'lucide-react';
import { AnalysisResult } from '../../hooks/useIntelligentAnalysis';

interface AnalysisResultsProps {
  result: AnalysisResult;
  onClose: () => void;
}

export const AnalysisResults: React.FC<AnalysisResultsProps> = ({ result, onClose }) => {
  const [isApplying, setIsApplying] = useState(false);
  const [appliedConfig, setAppliedConfig] = useState<any>(null);

  const getRiskLevelColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'low':
        return 'text-green-400 bg-green-400/10';
      case 'medium':
        return 'text-yellow-400 bg-yellow-400/10';
      case 'high':
        return 'text-red-400 bg-red-400/10';
      default:
        return 'text-gray-400 bg-gray-400/10';
    }
  };

  const getRiskLevelIcon = (riskLevel: string) => {
    switch (riskLevel) {
      case 'low':
        return <CheckCircle className="w-4 h-4" />;
      case 'medium':
        return <AlertTriangle className="w-4 h-4" />;
      case 'high':
        return <TrendingDown className="w-4 h-4" />;
      default:
        return <AlertTriangle className="w-4 h-4" />;
    }
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'bullish':
        return <TrendingUp className="w-4 h-4 text-green-400" />;
      case 'bearish':
        return <TrendingDown className="w-4 h-4 text-red-400" />;
      default:
        return <Activity className="w-4 h-4 text-yellow-400" />;
    }
  };

  const getTrendColor = (trend: string) => {
    switch (trend) {
      case 'bullish':
        return 'text-green-400';
      case 'bearish':
        return 'text-red-400';
      default:
        return 'text-yellow-400';
    }
  };

  const applyRecommendations = async () => {
    setIsApplying(true);
    
    try {
      // Simular proceso de aplicación
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Configurar parámetros de trading basados en el análisis
      const config = {
        symbol: 'EURUSD', // Obtener del contexto
        tradingType: result.tradingType,
        stopLoss: result.technicalAnalysis?.currentPrice ? 
          result.technicalAnalysis.currentPrice * (result.riskLevel === 'high' ? 0.98 : 0.99) : null,
        takeProfit: result.technicalAnalysis?.currentPrice ? 
          result.technicalAnalysis.currentPrice * (result.riskLevel === 'high' ? 1.02 : 1.01) : null,
        positionSize: result.riskLevel === 'high' ? 0.5 : result.riskLevel === 'medium' ? 1 : 2,
        alerts: generateAlerts(result),
        timestamp: new Date(),
        confidence: result.confidence,
        riskLevel: result.riskLevel
      };
      
      // Guardar configuración en localStorage
      const savedConfigs = JSON.parse(localStorage.getItem('tradingConfigs') || '[]');
      savedConfigs.push(config);
      localStorage.setItem('tradingConfigs', JSON.stringify(savedConfigs));
      
      setAppliedConfig(config);
      
      // Mostrar notificación de éxito
      alert(`✅ Configuración aplicada exitosamente!\n\n• Stop Loss: ${config.stopLoss?.toFixed(4)}\n• Take Profit: ${config.takeProfit?.toFixed(4)}\n• Tamaño de posición: ${config.positionSize}x\n• Alertas configuradas: ${config.alerts.length}`);
      
    } catch (error) {
      console.error('Error aplicando recomendaciones:', error);
      alert('❌ Error aplicando recomendaciones. Inténtalo de nuevo.');
    } finally {
      setIsApplying(false);
    }
  };

  const generateAlerts = (analysis: AnalysisResult) => {
    const alerts = [];
    
    if (analysis.technicalAnalysis) {
      const { currentPrice, rsi, macd, trend, volatility } = analysis.technicalAnalysis;
      
      // Alerta de RSI
      if (rsi && rsi < 30) {
        alerts.push({
          type: 'RSI_OVERSOLD',
          condition: 'RSI < 30',
          message: 'RSI en sobreventa - Oportunidad de compra',
          price: currentPrice * 0.995
        });
      } else if (rsi && rsi > 70) {
        alerts.push({
          type: 'RSI_OVERBOUGHT',
          condition: 'RSI > 70',
          message: 'RSI en sobrecompra - Considerar venta',
          price: currentPrice * 1.005
        });
      }
      
      // Alerta de tendencia
      if (trend === 'bullish') {
        alerts.push({
          type: 'TREND_BULLISH',
          condition: 'Tendencia alcista',
          message: 'Tendencia alcista confirmada',
          price: currentPrice * 1.01
        });
      } else if (trend === 'bearish') {
        alerts.push({
          type: 'TREND_BEARISH',
          condition: 'Tendencia bajista',
          message: 'Tendencia bajista confirmada',
          price: currentPrice * 0.99
        });
      }
      
      // Alerta de volatilidad
      if (volatility > 3) {
        alerts.push({
          type: 'HIGH_VOLATILITY',
          condition: `Volatilidad alta (${volatility.toFixed(1)}%)`,
          message: 'Alta volatilidad - Usar stop-loss más amplios',
          price: currentPrice * 0.98
        });
      }
      
      // Alerta de MACD
      if (macd && macd > 0) {
        alerts.push({
          type: 'MACD_BULLISH',
          condition: 'MACD alcista',
          message: 'MACD positivo - Momentum alcista',
          price: currentPrice * 1.005
        });
      } else if (macd && macd < 0) {
        alerts.push({
          type: 'MACD_BEARISH',
          condition: 'MACD bajista',
          message: 'MACD negativo - Momentum bajista',
          price: currentPrice * 0.995
        });
      }
    }
    
    return alerts;
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-gray-800 rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto">
        <div className="p-6">
          {/* Header */}
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-3">
              <div 
                className="p-2 rounded-lg"
                style={{ backgroundColor: `${result.tradingType.color}20` }}
              >
                <Target className="w-6 h-6" style={{ color: result.tradingType.color }} />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-white">
                  Análisis Inteligente - {result.tradingType.name}
                </h3>
                <p className="text-sm text-gray-400">
                  {result.timestamp.toLocaleString()}
                </p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-white transition-colors"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Información del trading type */}
          <div className="bg-gray-700/50 rounded-lg p-4 mb-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="flex items-center space-x-2">
                <Clock className="w-4 h-4 text-gray-400" />
                <div>
                  <p className="text-xs text-gray-400">Timeframe</p>
                  <p className="text-sm font-medium text-white">{result.tradingType.timeframe}</p>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <TrendingUp className="w-4 h-4 text-gray-400" />
                <div>
                  <p className="text-xs text-gray-400">Razón</p>
                  <p className="text-sm font-medium text-white">{result.tradingType.reason}</p>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <div 
                  className="w-4 h-4 rounded-full"
                  style={{ backgroundColor: result.tradingType.color }}
                />
                <div>
                  <p className="text-xs text-gray-400">Tipo</p>
                  <p className="text-sm font-medium text-white">{result.tradingType.name}</p>
                </div>
              </div>
            </div>
          </div>

          {/* Análisis Técnico Real */}
          {result.technicalAnalysis && (
            <div className="bg-gray-700/50 rounded-lg p-4 mb-6">
              <h4 className="text-sm font-semibold text-white mb-4 flex items-center space-x-2">
                <BarChart3 className="w-4 h-4" />
                <span>Análisis Técnico en Tiempo Real</span>
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="bg-gray-800/50 rounded p-3">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs text-gray-400">Precio Actual</span>
                  </div>
                  <div className="text-lg font-semibold text-white">
                    {result.technicalAnalysis.currentPrice.toFixed(4)}
                  </div>
                </div>
                
                <div className="bg-gray-800/50 rounded p-3">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs text-gray-400">RSI</span>
                    <span className={`text-xs font-medium ${
                      result.technicalAnalysis.rsi && result.technicalAnalysis.rsi < 30 ? 'text-green-400' :
                      result.technicalAnalysis.rsi && result.technicalAnalysis.rsi > 70 ? 'text-red-400' : 'text-yellow-400'
                    }`}>
                      {result.technicalAnalysis.rsi ? result.technicalAnalysis.rsi.toFixed(1) : 'N/A'}
                    </span>
                  </div>
                  <div className="text-sm text-gray-300">
                    {result.technicalAnalysis.rsi && result.technicalAnalysis.rsi < 30 ? 'Sobreventa' :
                     result.technicalAnalysis.rsi && result.technicalAnalysis.rsi > 70 ? 'Sobrecompra' : 'Neutral'}
                  </div>
                </div>
                
                <div className="bg-gray-800/50 rounded p-3">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs text-gray-400">MACD</span>
                    <span className={`text-xs font-medium ${
                      result.technicalAnalysis.macd && result.technicalAnalysis.macd > 0 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {result.technicalAnalysis.macd ? result.technicalAnalysis.macd.toFixed(4) : 'N/A'}
                    </span>
                  </div>
                  <div className="text-sm text-gray-300">
                    {result.technicalAnalysis.macd && result.technicalAnalysis.macd > 0 ? 'Alcista' : 'Bajista'}
                  </div>
                </div>
                
                <div className="bg-gray-800/50 rounded p-3">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs text-gray-400">Tendencia</span>
                    {getTrendIcon(result.technicalAnalysis.trend)}
                  </div>
                  <div className={`text-sm font-medium capitalize ${getTrendColor(result.technicalAnalysis.trend)}`}>
                    {result.technicalAnalysis.trend}
                  </div>
                </div>
                
                <div className="bg-gray-800/50 rounded p-3">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs text-gray-400">ADX</span>
                    <span className={`text-xs font-medium ${
                      result.technicalAnalysis.adx && result.technicalAnalysis.adx > 25 ? 'text-green-400' : 'text-yellow-400'
                    }`}>
                      {result.technicalAnalysis.adx ? result.technicalAnalysis.adx.toFixed(1) : 'N/A'}
                    </span>
                  </div>
                  <div className="text-sm text-gray-300">
                    {result.technicalAnalysis.adx && result.technicalAnalysis.adx > 25 ? 'Tendencia Fuerte' : 'Tendencia Débil'}
                  </div>
                </div>
                
                <div className="bg-gray-800/50 rounded p-3">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs text-gray-400">Volatilidad</span>
                    <span className={`text-xs font-medium ${
                      result.technicalAnalysis.volatility > 3 ? 'text-red-400' :
                      result.technicalAnalysis.volatility < 1 ? 'text-green-400' : 'text-yellow-400'
                    }`}>
                      {result.technicalAnalysis.volatility.toFixed(1)}%
                    </span>
                  </div>
                  <div className="text-sm text-gray-300">
                    {result.technicalAnalysis.volatility > 3 ? 'Alta' :
                     result.technicalAnalysis.volatility < 1 ? 'Baja' : 'Normal'}
                  </div>
                </div>
                
                <div className="bg-gray-800/50 rounded p-3">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs text-gray-400">Volumen</span>
                    <span className={`text-xs font-medium ${
                      result.technicalAnalysis.volumeRatio > 1.5 ? 'text-green-400' :
                      result.technicalAnalysis.volumeRatio < 0.7 ? 'text-red-400' : 'text-yellow-400'
                    }`}>
                      {(result.technicalAnalysis.volumeRatio * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="text-sm text-gray-300">
                    {result.technicalAnalysis.volumeRatio > 1.5 ? 'Alto' :
                     result.technicalAnalysis.volumeRatio < 0.7 ? 'Bajo' : 'Normal'}
                  </div>
                </div>
                
                <div className="bg-gray-800/50 rounded p-3">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs text-gray-400">SMA {result.tradingType.id === 'scalping' ? '10' : 
                                                              result.tradingType.id === 'day_trading' ? '20' :
                                                              result.tradingType.id === 'swing_trading' ? '50' : '100'}</span>
                    <span className="text-xs text-gray-400">
                      {result.technicalAnalysis.sma20 ? result.technicalAnalysis.sma20.toFixed(4) : 'N/A'}
                    </span>
                  </div>
                  <div className="text-sm text-gray-300">
                    {result.technicalAnalysis.sma20 && result.technicalAnalysis.currentPrice > result.technicalAnalysis.sma20 ? 'Soporte' : 'Resistencia'}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Métricas principales */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            <div className="bg-gray-700/50 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-400">Nivel de Confianza</span>
                <span className="text-lg font-semibold text-blue-400">
                  {(result.confidence * 100).toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-gray-600 rounded-full h-2">
                <div 
                  className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${result.confidence * 100}%` }}
                />
              </div>
              <p className="text-xs text-gray-400 mt-2">
                Basado en análisis técnico real con datos de Yahoo Finance
              </p>
            </div>

            <div className="bg-gray-700/50 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-400">Nivel de Riesgo</span>
                <div className={`flex items-center space-x-1 px-2 py-1 rounded-full ${getRiskLevelColor(result.riskLevel)}`}>
                  {getRiskLevelIcon(result.riskLevel)}
                  <span className="text-xs font-medium capitalize">{result.riskLevel}</span>
                </div>
              </div>
              <p className="text-xs text-gray-400 mt-2">
                {result.riskLevel === 'low' && 'Riesgo bajo, operaciones conservadoras'}
                {result.riskLevel === 'medium' && 'Riesgo moderado, equilibrio entre riesgo y retorno'}
                {result.riskLevel === 'high' && 'Riesgo alto, requiere atención constante'}
              </p>
            </div>
          </div>

          {/* Recomendaciones */}
          <div className="bg-gray-700/50 rounded-lg p-4">
            <h4 className="text-sm font-semibold text-white mb-3">Recomendaciones Basadas en Análisis Real</h4>
            <div className="space-y-2">
              {result.recommendations.map((recommendation, index) => (
                <div key={index} className="flex items-start space-x-2">
                  <div className="w-1.5 h-1.5 rounded-full bg-blue-400 mt-2 flex-shrink-0" />
                  <p className="text-sm text-gray-300">{recommendation}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Configuración que se aplicará */}
          {result.technicalAnalysis && (
            <div className="bg-gray-700/50 rounded-lg p-4 mb-6">
              <h4 className="text-sm font-semibold text-white mb-3 flex items-center space-x-2">
                <Settings className="w-4 h-4" />
                <span>Configuración que se Aplicará</span>
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                <div className="bg-gray-800/50 rounded p-3">
                  <div className="text-gray-400 mb-1">Stop Loss Sugerido</div>
                  <div className="text-red-400 font-semibold">
                    {(result.technicalAnalysis.currentPrice * (result.riskLevel === 'high' ? 0.98 : 0.99)).toFixed(4)}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    -{(((result.riskLevel === 'high' ? 0.98 : 0.99) - 1) * 100).toFixed(1)}% riesgo
                  </div>
                </div>
                <div className="bg-gray-800/50 rounded p-3">
                  <div className="text-gray-400 mb-1">Take Profit Sugerido</div>
                  <div className="text-green-400 font-semibold">
                    {(result.technicalAnalysis.currentPrice * (result.riskLevel === 'high' ? 1.02 : 1.01)).toFixed(4)}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    +{(((result.riskLevel === 'high' ? 1.02 : 1.01) - 1) * 100).toFixed(1)}% beneficio
                  </div>
                </div>
                <div className="bg-gray-800/50 rounded p-3">
                  <div className="text-gray-400 mb-1">Tamaño de Posición</div>
                  <div className="text-blue-400 font-semibold">
                    {result.riskLevel === 'high' ? '0.5x' : result.riskLevel === 'medium' ? '1x' : '2x'}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    Basado en nivel de riesgo
                  </div>
                </div>
              </div>
              <div className="mt-3 p-3 bg-blue-900/20 rounded border border-blue-500/30">
                <div className="flex items-center space-x-2 text-blue-300">
                  <Bell className="w-4 h-4" />
                  <span className="text-sm font-medium">Alertas Automáticas</span>
                </div>
                <div className="text-xs text-gray-400 mt-1">
                  Se configurarán {generateAlerts(result).length} alertas basadas en el análisis técnico
                </div>
              </div>
            </div>
          )}

          {/* Footer */}
          <div className="mt-6 pt-4 border-t border-gray-700">
            <div className="flex justify-end space-x-3">
              <button
                onClick={onClose}
                className="px-4 py-2 text-sm font-medium text-gray-400 hover:text-white transition-colors"
              >
                Cerrar
              </button>
              <button
                onClick={applyRecommendations}
                disabled={isApplying}
                className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors flex items-center space-x-2 ${
                  isApplying 
                    ? 'bg-gray-600 text-gray-400 cursor-not-allowed' 
                    : 'bg-blue-500 hover:bg-blue-600 text-white'
                }`}
              >
                {isApplying ? (
                  <>
                    <div className="w-4 h-4 border border-white border-t-transparent rounded-full animate-spin"></div>
                    <span>Aplicando...</span>
                  </>
                ) : (
                  <>
                    <Save className="w-4 h-4" />
                    <span>Aplicar Recomendaciones</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}; 