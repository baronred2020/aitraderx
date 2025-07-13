import React, { useState } from 'react';
import { TradingChart } from './TradingChart';
import { YahooTradingChart } from './YahooTradingChart';
import { Database, Zap, TrendingUp, Clock } from 'lucide-react';

interface TradingChartDemoProps {
  symbol: string;
  timeframe: string;
}

export const TradingChartDemo: React.FC<TradingChartDemoProps> = ({ symbol, timeframe }) => {
  const [dataSource, setDataSource] = useState<'twelvedata' | 'yahoo'>('yahoo');

  const dataSources = [
    {
      id: 'yahoo',
      name: 'Yahoo Finance',
      description: 'Datos gratuitos y confiables',
      icon: <Zap className="w-4 h-4" />,
      features: [
        '✅ Sin límites de API',
        '✅ Datos en tiempo real',
        '✅ Cobertura global',
        '✅ Histórico completo'
      ]
    },
    {
      id: 'twelvedata',
      name: 'Twelve Data',
      description: 'API profesional con límites',
      icon: <Database className="w-4 h-4" />,
      features: [
        '⚠️ 800 llamadas/día',
        '✅ Datos profesionales',
        '✅ Múltiples intervalos',
        '⚠️ Requiere API key'
      ]
    }
  ];

  return (
    <div className="space-y-4">
      {/* Selector de fuente de datos */}
      <div className="bg-gray-800/50 rounded-lg p-4">
        <h3 className="text-lg font-semibold text-white mb-4">Fuente de Datos</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {dataSources.map((source) => (
            <div
              key={source.id}
              onClick={() => setDataSource(source.id as 'twelvedata' | 'yahoo')}
              className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                dataSource === source.id
                  ? 'border-blue-500 bg-blue-500/10'
                  : 'border-gray-600 bg-gray-700/50 hover:border-gray-500'
              }`}
            >
              <div className="flex items-center space-x-3 mb-2">
                <div className={`p-2 rounded-lg ${
                  dataSource === source.id ? 'bg-blue-500' : 'bg-gray-600'
                }`}>
                  {source.icon}
                </div>
                <div>
                  <h4 className="font-semibold text-white">{source.name}</h4>
                  <p className="text-sm text-gray-400">{source.description}</p>
                </div>
              </div>
              
              <ul className="space-y-1">
                {source.features.map((feature, index) => (
                  <li key={index} className="text-xs text-gray-300 flex items-center space-x-2">
                    <span>{feature}</span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        {/* Comparación de rendimiento */}
        <div className="mt-6 p-4 bg-gray-700/50 rounded-lg">
          <h4 className="font-semibold text-white mb-3 flex items-center space-x-2">
            <TrendingUp className="w-4 h-4" />
            <span>Comparación de Rendimiento</span>
          </h4>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div className="text-center">
              <div className="text-2xl font-bold text-green-400">
                {dataSource === 'yahoo' ? '∞' : '800'}
              </div>
              <div className="text-gray-400">Llamadas/día</div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-400">
                {dataSource === 'yahoo' ? '2min' : '15min'}
              </div>
              <div className="text-gray-400">Cache</div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-400">
                {dataSource === 'yahoo' ? 'Gratis' : '$99/mes'}
              </div>
              <div className="text-gray-400">Costo</div>
            </div>
          </div>
        </div>
      </div>

      {/* Gráfico seleccionado */}
      <div className="bg-gray-800/50 rounded-lg overflow-hidden">
        <div className="p-4 border-b border-gray-700/50">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-white">
                {symbol} - {timeframe}
              </h3>
              <p className="text-sm text-gray-400">
                Fuente: {dataSource === 'yahoo' ? 'Yahoo Finance' : 'Twelve Data'}
              </p>
            </div>
            
            <div className="flex items-center space-x-2">
              <div className={`px-3 py-1 rounded-full text-xs font-medium ${
                dataSource === 'yahoo' 
                  ? 'bg-green-500/20 text-green-400' 
                  : 'bg-yellow-500/20 text-yellow-400'
              }`}>
                {dataSource === 'yahoo' ? 'Gratis' : 'Limitado'}
              </div>
              
              <div className="flex items-center space-x-1 text-gray-400">
                <Clock className="w-3 h-3" />
                <span className="text-xs">
                  {dataSource === 'yahoo' ? 'Actualizado cada 2min' : 'Actualizado cada 15min'}
                </span>
              </div>
            </div>
          </div>
        </div>
        
        <div className="p-4">
          {dataSource === 'yahoo' ? (
            <YahooTradingChart symbol={symbol} />
          ) : (
            <TradingChart symbol={symbol} timeframe={timeframe} />
          )}
        </div>
      </div>

      {/* Información adicional */}
      <div className="bg-gray-800/50 rounded-lg p-4">
        <h4 className="font-semibold text-white mb-3">Recomendación</h4>
        
        {dataSource === 'yahoo' ? (
          <div className="space-y-2">
            <p className="text-green-400 text-sm">
              ✅ Yahoo Finance es la mejor opción para desarrollo y uso gratuito
            </p>
            <ul className="text-sm text-gray-300 space-y-1 ml-4">
              <li>• Sin límites de API</li>
              <li>• Datos confiables y actualizados</li>
              <li>• Cobertura global de mercados</li>
              <li>• Ideal para prototipos y aplicaciones pequeñas</li>
            </ul>
          </div>
        ) : (
          <div className="space-y-2">
            <p className="text-yellow-400 text-sm">
              ⚠️ Twelve Data es mejor para aplicaciones comerciales con alto volumen
            </p>
            <ul className="text-sm text-gray-300 space-y-1 ml-4">
              <li>• Límite de 800 llamadas/día (gratuito)</li>
              <li>• Datos profesionales de alta calidad</li>
              <li>• Soporte técnico disponible</li>
              <li>• Planes de pago para mayor volumen</li>
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}; 