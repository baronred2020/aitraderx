import React from 'react';
import { TrendingUp, TrendingDown, AlertTriangle, CheckCircle, Clock, Target } from 'lucide-react';
import { AnalysisResult } from '../../hooks/useIntelligentAnalysis';

interface AnalysisResultsProps {
  result: AnalysisResult;
  onClose: () => void;
}

export const AnalysisResults: React.FC<AnalysisResultsProps> = ({ result, onClose }) => {
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

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-gray-800 rounded-lg shadow-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
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
            <h4 className="text-sm font-semibold text-white mb-3">Recomendaciones</h4>
            <div className="space-y-2">
              {result.recommendations.map((recommendation, index) => (
                <div key={index} className="flex items-start space-x-2">
                  <div className="w-1.5 h-1.5 rounded-full bg-blue-400 mt-2 flex-shrink-0" />
                  <p className="text-sm text-gray-300">{recommendation}</p>
                </div>
              ))}
            </div>
          </div>

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
                onClick={() => {
                  // Aquí se podría implementar la funcionalidad para aplicar las recomendaciones
                  alert('Funcionalidad de aplicación de recomendaciones próximamente');
                }}
                className="px-4 py-2 text-sm font-medium bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-colors"
              >
                Aplicar Recomendaciones
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}; 