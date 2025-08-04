import React, { useState, useEffect } from 'react';
import { 
  Brain, 
  TrendingUp, 
  TrendingDown, 
  BarChart3, 
  Target, 
  Clock, 
  AlertTriangle,
  CheckCircle,
  XCircle,
  Activity,
  Zap,
  DollarSign,
  Calendar,
  Filter,
  Download,
  RefreshCw,
  Info
} from 'lucide-react';
import { useRealSignals } from '../../hooks/useRealSignals';

interface ModelPerformance {
  modelName: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  profitFactor: number;
  winRate: number;
  totalTrades: number;
  avgReturn: number;
  maxDrawdown: number;
  sharpeRatio: number;
  lastUpdate: string;
  status: 'improving' | 'stable' | 'declining';
  alerts: number;
}

interface PerformanceMetric {
  date: string;
  accuracy: number;
  profitFactor: number;
  winRate: number;
}

const AIModelAnalytics: React.FC = () => {
  const [selectedModel, setSelectedModel] = useState<string>('all');
  const [timeRange, setTimeRange] = useState<'1d' | '7d' | '30d' | '90d'>('7d');
  const [showDetails, setShowDetails] = useState(false);

  // Usar datos reales del hook - SIN fallbacks
  const { 
    modelPerformance: realModelPerformance,
    loading: performanceLoading,
    errors: performanceErrors,
    loadModelPerformance,
    refreshAll
  } = useRealSignals();

  // Solo usar datos reales, sin fallbacks
  const availableModels = realModelPerformance;
  
  const filteredModels = selectedModel === 'all' 
    ? availableModels 
    : availableModels.filter(model => model.modelName === selectedModel);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'improving': return 'text-green-400';
      case 'stable': return 'text-blue-400';
      case 'declining': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'improving': return <TrendingUp className="w-4 h-4 text-green-400" />;
      case 'stable': return <BarChart3 className="w-4 h-4 text-blue-400" />;
      case 'declining': return <TrendingDown className="w-4 h-4 text-red-400" />;
      default: return <BarChart3 className="w-4 h-4 text-gray-400" />;
    }
  };

  const getMetricColor = (value: number, threshold: number, reverse: boolean = false) => {
    const isGood = reverse ? value < threshold : value > threshold;
    return isGood ? 'text-green-400' : 'text-red-400';
  };



  const exportData = () => {
    const csvContent = "data:text/csv;charset=utf-8," 
      + "Model,Accuracy,Precision,Recall,F1-Score,Profit Factor,Win Rate,Total Trades,Avg Return,Max Drawdown,Sharpe Ratio\n"
      + filteredModels.map(model => 
          `${model.modelName},${model.accuracy},${model.precision},${model.recall},${model.f1Score},${model.profitFactor},${model.winRate},${model.totalTrades},${model.avgReturn},${model.maxDrawdown},${model.sharpeRatio}`
        ).join("\n");
    
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `ai_model_performance_${new Date().toISOString().split('T')[0]}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="space-y-6">
      {/* Header con controles */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center">
            <Brain className="w-5 h-5 mr-2 text-purple-400" />
            Análisis de Rendimiento IA
          </h2>
                     <p className="text-gray-400 text-sm">
             Métricas detalladas y análisis de rendimiento de modelos de IA
           </p>
           {performanceErrors.performance && (
             <p className="text-yellow-400 text-sm mt-1">
               ⚠️ {performanceErrors.performance}
             </p>
           )}
           {realModelPerformance.length > 0 && (
             <p className="text-green-400 text-sm mt-1">
               ✅ Datos de rendimiento en tiempo real
             </p>
           )}
        </div>
        
        <div className="flex items-center space-x-3">
                     <select
             value={selectedModel}
             onChange={(e) => setSelectedModel(e.target.value)}
             className="px-3 py-2 bg-gray-700 text-white rounded-lg border border-gray-600 focus:border-blue-500 focus:outline-none"
           >
             <option value="all">Todos los modelos</option>
             {availableModels.map(model => (
               <option key={model.modelName} value={model.modelName}>
                 {model.modelName}
               </option>
             ))}
           </select>
          
          <select
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value as any)}
            className="px-3 py-2 bg-gray-700 text-white rounded-lg border border-gray-600 focus:border-blue-500 focus:outline-none"
          >
            <option value="1d">Último día</option>
            <option value="7d">Última semana</option>
            <option value="30d">Último mes</option>
            <option value="90d">Último trimestre</option>
          </select>
          
                     <button
             onClick={() => refreshAll()}
             disabled={performanceLoading.performance}
             className="flex items-center space-x-2 px-3 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors disabled:opacity-50"
           >
             <RefreshCw className={`w-4 h-4 ${performanceLoading.performance ? 'animate-spin' : ''}`} />
             <span>Actualizar</span>
           </button>
           
           <button
             onClick={exportData}
             className="flex items-center space-x-2 px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
           >
             <Download className="w-4 h-4" />
             <span>Exportar</span>
           </button>
        </div>
      </div>

      {/* Resumen de métricas principales */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div className="flex items-center justify-between mb-2">
            <span className="text-gray-400 text-sm">Precisión Promedio</span>
            <Target className="w-4 h-4 text-blue-400" />
          </div>
          <div className="text-2xl font-bold text-white">
            {(filteredModels.reduce((sum, model) => sum + model.accuracy, 0) / filteredModels.length).toFixed(1)}%
          </div>
          <div className="text-xs text-gray-400 mt-1">
            {filteredModels.length} modelos activos
          </div>
        </div>
        
        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div className="flex items-center justify-between mb-2">
            <span className="text-gray-400 text-sm">Profit Factor Promedio</span>
            <DollarSign className="w-4 h-4 text-green-400" />
          </div>
          <div className="text-2xl font-bold text-white">
            {(filteredModels.reduce((sum, model) => sum + model.profitFactor, 0) / filteredModels.length).toFixed(2)}
          </div>
          <div className="text-xs text-gray-400 mt-1">
            Rentabilidad promedio
          </div>
        </div>
        
        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div className="flex items-center justify-between mb-2">
            <span className="text-gray-400 text-sm">Win Rate Promedio</span>
            <Activity className="w-4 h-4 text-yellow-400" />
          </div>
          <div className="text-2xl font-bold text-white">
            {(filteredModels.reduce((sum, model) => sum + model.winRate, 0) / filteredModels.length).toFixed(1)}%
          </div>
          <div className="text-xs text-gray-400 mt-1">
            Tasa de éxito
          </div>
        </div>
        
        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div className="flex items-center justify-between mb-2">
            <span className="text-gray-400 text-sm">Total Trades</span>
            <Zap className="w-4 h-4 text-purple-400" />
          </div>
          <div className="text-2xl font-bold text-white">
            {filteredModels.reduce((sum, model) => sum + model.totalTrades, 0)}
          </div>
          <div className="text-xs text-gray-400 mt-1">
            Operaciones totales
          </div>
        </div>
      </div>

      {/* Tabla detallada de rendimiento */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
        <div className="p-4 border-b border-gray-700">
          <h3 className="text-lg font-semibold text-white">Rendimiento Detallado por Modelo</h3>
        </div>
        
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-700/50">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                  Modelo
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                  Precisión
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                  Profit Factor
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                  Win Rate
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                  Sharpe Ratio
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                  Max Drawdown
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                  Estado
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                  Acciones
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-700">
              {filteredModels.length > 0 ? (
                filteredModels.map((model) => (
                  <tr key={model.modelName} className="hover:bg-gray-700/30">
                    <td className="px-4 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <div className="flex-shrink-0 h-8 w-8 bg-purple-500/20 rounded-lg flex items-center justify-center">
                          <Brain className="w-4 h-4 text-purple-400" />
                        </div>
                        <div className="ml-3">
                          <div className="text-sm font-medium text-white">{model.modelName}</div>
                          <div className="text-xs text-gray-400">{model.totalTrades} trades</div>
                        </div>
                      </div>
                    </td>
                    <td className="px-4 py-4 whitespace-nowrap">
                      <div className={`text-sm font-medium ${getMetricColor(model.accuracy, 75)}`}>
                        {model.accuracy.toFixed(1)}%
                      </div>
                    </td>
                    <td className="px-4 py-4 whitespace-nowrap">
                      <div className={`text-sm font-medium ${getMetricColor(model.profitFactor, 1.2)}`}>
                        {model.profitFactor.toFixed(2)}
                      </div>
                    </td>
                    <td className="px-4 py-4 whitespace-nowrap">
                      <div className={`text-sm font-medium ${getMetricColor(model.winRate, 65)}`}>
                        {model.winRate.toFixed(1)}%
                      </div>
                    </td>
                    <td className="px-4 py-4 whitespace-nowrap">
                      <div className={`text-sm font-medium ${getMetricColor(model.sharpeRatio, 1.0)}`}>
                        {model.sharpeRatio.toFixed(2)}
                      </div>
                    </td>
                    <td className="px-4 py-4 whitespace-nowrap">
                      <div className={`text-sm font-medium ${getMetricColor(model.maxDrawdown, -0.1, true)}`}>
                        {(model.maxDrawdown * 100).toFixed(1)}%
                      </div>
                    </td>
                    <td className="px-4 py-4 whitespace-nowrap">
                      <div className="flex items-center space-x-2">
                        {getStatusIcon(model.status)}
                        <span className={`text-sm ${getStatusColor(model.status)} capitalize`}>
                          {model.status}
                        </span>
                      </div>
                    </td>
                    <td className="px-4 py-4 whitespace-nowrap">
                      <button
                        onClick={() => setShowDetails(!showDetails)}
                        className="text-blue-400 hover:text-blue-300 text-sm"
                      >
                        {showDetails ? 'Ocultar' : 'Detalles'}
                      </button>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={8} className="px-4 py-12 text-center">
                    <div className="flex flex-col items-center justify-center text-gray-400">
                      <Brain className="w-12 h-12 mb-4" />
                      <div className="text-lg font-medium">No hay datos de rendimiento disponibles</div>
                      <div className="text-sm">Los modelos de IA no están generando datos de rendimiento</div>
                      <div className="text-xs mt-2">Verifica la conexión con las APIs de modelos</div>
                    </div>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Detalles expandidos */}
      {showDetails && (
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 className="text-lg font-semibold text-white mb-4">Métricas Detalladas</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredModels.length > 0 ? (
              filteredModels.map((model) => (
                <div key={model.modelName} className="bg-gray-700/50 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-4">
                    <h4 className="text-lg font-semibold text-white">{model.modelName}</h4>
                    <div className="flex items-center space-x-2">
                      {getStatusIcon(model.status)}
                      <span className={`text-sm ${getStatusColor(model.status)} capitalize`}>
                        {model.status}
                      </span>
                    </div>
                  </div>
                  
                  <div className="space-y-3">
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div>
                        <span className="text-gray-400">Precision:</span>
                        <div className="text-white font-medium">{(model.precision * 100).toFixed(1)}%</div>
                      </div>
                      <div>
                        <span className="text-gray-400">Recall:</span>
                        <div className="text-white font-medium">{(model.recall * 100).toFixed(1)}%</div>
                      </div>
                      <div>
                        <span className="text-gray-400">F1-Score:</span>
                        <div className="text-white font-medium">{(model.f1Score * 100).toFixed(1)}%</div>
                      </div>
                      <div>
                        <span className="text-gray-400">Avg Return:</span>
                        <div className="text-white font-medium">{(model.avgReturn * 100).toFixed(2)}%</div>
                      </div>
                    </div>
                    
                    <div className="pt-3 border-t border-gray-600">
                      <div className="flex items-center justify-between text-xs text-gray-400">
                        <span>Última actualización:</span>
                        <span>{model.lastUpdate}</span>
                      </div>
                      <div className="flex items-center justify-between text-xs text-gray-400 mt-1">
                        <span>Alertas:</span>
                        <span className={model.alerts > 0 ? 'text-red-400' : 'text-green-400'}>
                          {model.alerts}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              ))
            ) : (
              <div className="col-span-full flex items-center justify-center py-12 text-gray-400">
                <Brain className="w-12 h-12 mr-4" />
                <div>
                  <div className="text-lg font-medium">No hay métricas detalladas disponibles</div>
                  <div className="text-sm">Los modelos de IA no están generando datos de rendimiento</div>
                  <div className="text-xs mt-2">Verifica la conexión con las APIs de modelos</div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Información adicional */}
      <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-4">
        <div className="flex items-start space-x-3">
          <Info className="w-5 h-5 text-blue-400 mt-0.5" />
          <div>
            <h4 className="text-sm font-medium text-blue-400 mb-1">Interpretación de Métricas</h4>
                         <div className="text-sm text-gray-300 space-y-1">
               <p><strong>Profit Factor:</strong> &gt;1.2 = Rentable, &gt;1.5 = Excelente</p>
               <p><strong>Win Rate:</strong> &gt;65% = Bueno, &gt;70% = Excelente</p>
               <p><strong>Sharpe Ratio:</strong> &gt;1.0 = Bueno, &gt;1.5 = Excelente</p>
               <p><strong>Max Drawdown:</strong> &lt;10% = Aceptable, &lt;5% = Excelente</p>
             </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AIModelAnalytics; 