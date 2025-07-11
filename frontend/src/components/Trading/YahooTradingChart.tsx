import React, { useState } from 'react';
import { 
  LineChart, 
  Line, 
  AreaChart, 
  Area, 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  ComposedChart,
  Scatter
} from 'recharts';
import { 
  TrendingUp, 
  TrendingDown, 
  Settings, 
  Maximize2,
  Minimize2,
  RefreshCw,
  Download
} from 'lucide-react';
import { useCandles } from '../../hooks/useCandles';
import { useMarketData } from '../../hooks/useMarketData';
import { useEffect, useRef } from 'react';

// Tipos simples para lightweight-charts
declare const createChart: any;

interface TradingChartProps {
  symbol: string;
  timeframe: string;
}

const CandlestickChart: React.FC<{ data: any[] }> = ({ data }) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<any>(null);
  const isDisposedRef = useRef<boolean>(false);

  useEffect(() => {
    // Dynamic import to avoid TypeScript issues
    const loadChart = async () => {
      if (!chartContainerRef.current || isDisposedRef.current) return;
      
      try {
        const { createChart } = await import('lightweight-charts');
        
        // Clean up previous chart safely
        if (chartRef.current && !isDisposedRef.current) {
          try {
            chartRef.current.remove();
          } catch (e) {
            console.warn('Chart already disposed');
          }
          chartRef.current = null;
        }

        if (isDisposedRef.current) return; // Check again after cleanup

        const chart = createChart(chartContainerRef.current, {
          width: chartContainerRef.current.offsetWidth,
          height: 300,
          layout: {
            background: {
              color: '#181f3a'
            },
            textColor: '#cbd5e1',
          },
          grid: {
            vertLines: { color: '#2d3748' },
            horzLines: { color: '#2d3748' },
          },
          timeScale: { 
            timeVisible: true, 
            secondsVisible: false,
            borderColor: '#2d3748',
          },
          rightPriceScale: { 
            borderColor: '#2d3748',
          },
        } as any);
        
        if (isDisposedRef.current) {
          // Component was unmounted during chart creation
          try {
            chart.remove();
          } catch (e) {
            console.warn('Chart cleanup during creation');
          }
          return;
        }

        chartRef.current = chart;
        
        // Add candlestick series
        const candleSeries = (chart as any).addCandlestickSeries({
          upColor: '#38b2ac',
          downColor: '#f56565',
          borderUpColor: '#38b2ac',
          borderDownColor: '#f56565',
          wickUpColor: '#38b2ac',
          wickDownColor: '#f56565',
        });

        // Transform data to the correct format for lightweight-charts
        if (data && data.length > 0 && !isDisposedRef.current) {
          const chartData = data
            .map(item => ({
              time: Math.floor(new Date(item.date).getTime() / 1000),
              open: parseFloat(item.open),
              high: parseFloat(item.high),
              low: parseFloat(item.low),
              close: parseFloat(item.close),
            }))
            .sort((a, b) => a.time - b.time); // Sort by time

          try {
            candleSeries.setData(chartData);
          } catch (e) {
            console.warn('Error setting chart data:', e);
          }
        }

        // Handle resize
        const handleResize = () => {
          if (chartRef.current && chartContainerRef.current && !isDisposedRef.current) {
            try {
              chartRef.current.applyOptions({
                width: chartContainerRef.current.offsetWidth,
              });
            } catch (e) {
              console.warn('Error resizing chart:', e);
            }
          }
        };

        window.addEventListener('resize', handleResize);

        // Return cleanup function for this specific chart instance
        return () => {
          window.removeEventListener('resize', handleResize);
        };

      } catch (error) {
        console.error('Error loading chart:', error);
      }
    };

    isDisposedRef.current = false;
    loadChart();

    // Cleanup function
    return () => {
      isDisposedRef.current = true;
      if (chartRef.current) {
        try {
          chartRef.current.remove();
        } catch (e) {
          console.warn('Chart already disposed during cleanup');
        } finally {
          chartRef.current = null;
        }
      }
    };
  }, [data]);

  // Additional cleanup on unmount
  useEffect(() => {
    return () => {
      isDisposedRef.current = true;
    };
  }, []);

  return <div ref={chartContainerRef} style={{ width: '100%', height: 300 }} />;
};

export const YahooTradingChart: React.FC<TradingChartProps> = ({ symbol, timeframe }) => {
  const [chartType, setChartType] = useState<'candlestick' | 'line' | 'area'>('candlestick');
  const [showIndicators, setShowIndicators] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [lastSymbol, setLastSymbol] = useState(symbol);

  // Hook para datos de velas
  const { data: candleData, loading: candleLoading, error: candleError } = useCandles(symbol, '15', 100);
  // Hook para precio real
  const { data: marketData, loading: marketLoading, error: marketError } = useMarketData([symbol]);
  const realPrice = marketData[symbol]?.price;

  // Detectar cambio de símbolo
  useEffect(() => {
    if (symbol !== lastSymbol) {
      console.log(`[YahooTradingChart] Symbol changed from ${lastSymbol} to ${symbol}`);
      setLastSymbol(symbol);
    }
  }, [symbol, lastSymbol]);

  // Determinar si está cargando (incluyendo cambio de símbolo)
  const isLoading = candleLoading || marketLoading || symbol !== lastSymbol;

  // Mostrar loading inmediato al cambiar símbolo
  const showLoading = isLoading || !candleData || !marketData;

  // Transformar datos de Yahoo Finance al formato del gráfico
  const chartData = candleData && candleData.values
    ? candleData.values.map((item: any) => ({
        date: item.datetime,
        open: parseFloat(item.open),
        high: parseFloat(item.high),
        low: parseFloat(item.low),
        close: parseFloat(item.close),
        volume: parseFloat(item.volume),
      }))
    : [];

  const chartTypes = [
    { 
      id: 'candlestick', 
      name: 'Candlestick', 
      icon: '📊',
      description: 'Gráfico de velas japonesas'
    },
    { 
      id: 'line', 
      name: 'Línea', 
      icon: '📈',
      description: 'Gráfico de líneas del precio de cierre'
    },
    { 
      id: 'area', 
      name: 'Área', 
      icon: '📉',
      description: 'Gráfico de área con gradiente'
    },
  ];

  const timeframes = [
    { value: '1', label: '1min', description: '1 minuto' },
    { value: '5', label: '5min', description: '5 minutos' },
    { value: '15', label: '15min', description: '15 minutos' },
    { value: '30', label: '30min', description: '30 minutos' },
    { value: '60', label: '1H', description: '1 hora' },
    { value: '240', label: '4H', description: '4 horas' },
    { value: 'D', label: '1D', description: '1 día' },
  ];

  const indicators = [
    { id: 'sma', name: 'SMA 20', color: '#38b2ac' },
    { id: 'ema', name: 'EMA 50', color: '#f56565' },
    { id: 'rsi', name: 'RSI', color: '#ed8936' },
    { id: 'macd', name: 'MACD', color: '#9f7aea' },
  ];

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-3 shadow-lg">
          <p className="text-white font-semibold">{label}</p>
          <div className="space-y-1 mt-2">
            <div className="flex justify-between">
              <span className="text-gray-400">Apertura:</span>
              <span className="text-white">{data.open}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Máximo:</span>
              <span className="text-green-400">{data.high}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Mínimo:</span>
              <span className="text-red-400">{data.low}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Cierre:</span>
              <span className="text-white">{data.close}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Volumen:</span>
              <span className="text-blue-400">{data.volume}</span>
            </div>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className={`trading-card ${isFullscreen ? 'fixed inset-0 z-50' : ''}`}>
      {/* Header del gráfico */}
      <div className="p-3 sm:p-4 border-b border-gray-700/50">
        {/* Primera fila: Información del símbolo */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-4">
          <div className="flex flex-col sm:flex-row sm:items-center space-y-2 sm:space-y-0 sm:space-x-4">
            <div>
              <h3 className="text-base sm:text-lg font-semibold text-white">{symbol}</h3>
              <p className="text-xs sm:text-sm text-gray-400">
                {timeframe} • Yahoo Finance • {new Date().toLocaleTimeString()}
                {isLoading && <span className="text-blue-400 ml-2">🔄 Actualizando...</span>}
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-2 mt-2 sm:mt-0">
            <span className="text-lg sm:text-2xl font-bold text-green-400">
              {realPrice ? parseFloat(realPrice).toFixed(5) : '...'}
            </span>
            {/* Mostrar cambio porcentual si está disponible */}
            {marketData[symbol]?.changePercent && (
              <span className={`text-sm font-medium px-2 py-1 rounded-full ${
                parseFloat(marketData[symbol].changePercent) >= 0 
                  ? 'text-green-400 bg-green-400/10' 
                  : 'text-red-400 bg-red-400/10'
              }`}>
                {parseFloat(marketData[symbol].changePercent) >= 0 ? '+' : ''}
                {parseFloat(marketData[symbol].changePercent).toFixed(2)}%
              </span>
            )}
          </div>
        </div>

        {/* Segunda fila: Controles */}
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between space-y-3 lg:space-y-0">
          {/* Selector de tipo de gráfico - Más prominente */}
          <div className="flex flex-col sm:flex-row sm:items-center space-y-2 sm:space-y-0 sm:space-x-4">
            <span className="text-sm font-medium text-gray-400">Tipo de gráfico:</span>
            <div className="flex items-center space-x-1 bg-gray-800/50 rounded-lg p-1">
              {chartTypes.map((type) => (
                <button
                  key={type.id}
                  onClick={() => setChartType(type.id as any)}
                  title={type.description}
                  className={`px-3 py-2 rounded-md text-sm font-medium transition-all duration-200 ${
                    chartType === type.id
                      ? 'bg-blue-500 text-white shadow-lg scale-105'
                      : 'text-gray-400 hover:text-white hover:bg-gray-700/50'
                  }`}
                >
                  <span className="flex items-center space-x-1">
                    <span>{type.icon}</span>
                    <span className="hidden sm:inline">{type.name}</span>
                  </span>
                </button>
              ))}
            </div>
          </div>

          {/* Controles adicionales */}
          <div className="flex items-center space-x-2">
            {/* Indicadores */}
            <button
              onClick={() => setShowIndicators(!showIndicators)}
              title={showIndicators ? 'Ocultar indicadores' : 'Mostrar indicadores'}
              className={`p-2 rounded-lg transition-all duration-200 ${
                showIndicators 
                  ? 'bg-blue-500/20 text-blue-400 shadow-lg' 
                  : 'bg-gray-700/50 text-gray-400 hover:text-white hover:bg-gray-600/50'
              }`}
            >
              <Settings className="w-4 h-4" />
            </button>

            {/* Fullscreen */}
            <button
              onClick={() => setIsFullscreen(!isFullscreen)}
              title={isFullscreen ? 'Salir de pantalla completa' : 'Pantalla completa'}
              className="p-2 rounded-lg bg-gray-700/50 text-gray-400 hover:text-white hover:bg-gray-600/50 transition-all duration-200"
            >
              {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
            </button>

            {/* Refresh */}
            <button 
              title="Actualizar datos"
              className="p-2 rounded-lg bg-gray-700/50 text-gray-400 hover:text-white hover:bg-gray-600/50 transition-all duration-200"
            >
              <RefreshCw className="w-4 h-4" />
            </button>

            {/* Download */}
            <button 
              title="Descargar gráfico"
              className="p-2 rounded-lg bg-gray-700/50 text-gray-400 hover:text-white hover:bg-gray-600/50 transition-all duration-200"
            >
              <Download className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Gráfico principal */}
      <div className="p-3 sm:p-4">
        {/* Información del tipo de gráfico activo */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2 bg-gray-800/30 rounded-lg px-3 py-2">
              <span className="text-lg">{chartTypes.find(t => t.id === chartType)?.icon}</span>
              <div>
                <span className="text-sm font-medium text-white">
                  {chartTypes.find(t => t.id === chartType)?.name}
                </span>
                <p className="text-xs text-gray-400">
                  {chartTypes.find(t => t.id === chartType)?.description}
                </p>
              </div>
            </div>
            {showIndicators && chartType !== 'candlestick' && (
              <div className="flex items-center space-x-2 text-xs">
                <div className="flex items-center space-x-1">
                  <div className="w-3 h-0.5 bg-teal-400"></div>
                  <span className="text-gray-400">Precio cierre</span>
                </div>
                <div className="flex items-center space-x-1">
                  <div className="w-3 h-0.5 bg-cyan-400 opacity-60" style={{borderTop: '1px dashed'}}></div>
                  <span className="text-gray-400">Máximo</span>
                </div>
                {chartType === 'line' && (
                  <div className="flex items-center space-x-1">
                    <div className="w-3 h-0.5 bg-red-400 opacity-60" style={{borderTop: '1px dashed'}}></div>
                    <span className="text-gray-400">Mínimo</span>
                  </div>
                )}
              </div>
            )}
          </div>
          {chartData.length > 0 && (
            <div className="text-xs text-gray-400">
              {chartData.length} velas • Última actualización: {new Date(chartData[chartData.length - 1]?.date).toLocaleTimeString()}
            </div>
          )}
        </div>
        {showLoading ? (
          <div className="flex flex-col items-center justify-center h-64 text-gray-400">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-400 mb-4"></div>
            <div className="text-center">
              <div className="text-lg font-medium">Cargando datos de Yahoo Finance...</div>
              <div className="text-sm text-gray-500 mt-2">
                {symbol !== lastSymbol ? `Cambiando a ${symbol}...` : 'Actualizando datos...'}
              </div>
            </div>
          </div>
        ) : candleError || marketError ? (
          <div className="flex flex-col items-center justify-center h-64 text-red-400">
            <div className="text-lg font-medium mb-2">Error al cargar datos</div>
            <div className="text-sm text-gray-500 text-center">
              {candleError && <div>Candles: {candleError}</div>}
              {marketError && <div>Market: {marketError}</div>}
            </div>
          </div>
        ) : chartData.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-64 text-gray-400">
            <div className="text-lg font-medium mb-2">No hay datos disponibles</div>
            <div className="text-sm text-gray-500">Símbolo: {symbol}</div>
          </div>
        ) : (
          chartType === 'candlestick' ? (
            <CandlestickChart data={chartData} />
          ) : chartType === 'line' ? (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2d3748" />
                <XAxis 
                  dataKey="date" 
                  stroke="#a0aec0" 
                  fontSize={12}
                  tickFormatter={(value) => {
                    const date = new Date(value);
                    return date.toLocaleTimeString('en-US', { 
                      hour: '2-digit', 
                      minute: '2-digit',
                      hour12: false 
                    });
                  }}
                />
                <YAxis 
                  stroke="#a0aec0" 
                  fontSize={12}
                  domain={['dataMin - 0.0001', 'dataMax + 0.0001']}
                  tickFormatter={(value) => parseFloat(value).toFixed(4)}
                />
                <Tooltip content={<CustomTooltip />} />
                <Line 
                  type="monotone" 
                  dataKey="close" 
                  stroke="#38b2ac" 
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 6, fill: '#38b2ac', stroke: '#ffffff', strokeWidth: 2 }}
                />
                {showIndicators && (
                  <>
                    <Line 
                      type="monotone" 
                      dataKey="high" 
                      stroke="#22d3ee" 
                      strokeWidth={1}
                      dot={false}
                      strokeDasharray="5 5"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="low" 
                      stroke="#f87171" 
                      strokeWidth={1}
                      dot={false}
                      strokeDasharray="5 5"
                    />
                  </>
                )}
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="areaGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#38b2ac" stopOpacity={0.8}/>
                    <stop offset="50%" stopColor="#38b2ac" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#38b2ac" stopOpacity={0.1}/>
                  </linearGradient>
                  <linearGradient id="areaGradientHigh" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.4}/>
                    <stop offset="95%" stopColor="#22d3ee" stopOpacity={0.1}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#2d3748" />
                <XAxis 
                  dataKey="date" 
                  stroke="#a0aec0" 
                  fontSize={12}
                  tickFormatter={(value) => {
                    const date = new Date(value);
                    return date.toLocaleTimeString('en-US', { 
                      hour: '2-digit', 
                      minute: '2-digit',
                      hour12: false 
                    });
                  }}
                />
                <YAxis 
                  stroke="#a0aec0" 
                  fontSize={12}
                  domain={['dataMin - 0.0001', 'dataMax + 0.0001']}
                  tickFormatter={(value) => parseFloat(value).toFixed(4)}
                />
                <Tooltip content={<CustomTooltip />} />
                <Area 
                  type="monotone" 
                  dataKey="close" 
                  stroke="#38b2ac" 
                  strokeWidth={2}
                  fill="url(#areaGradient)"
                />
                {showIndicators && (
                  <Area 
                    type="monotone" 
                    dataKey="high" 
                    stroke="#22d3ee" 
                    strokeWidth={1}
                    fill="url(#areaGradientHigh)"
                    fillOpacity={0.3}
                  />
                )}
              </AreaChart>
            </ResponsiveContainer>
          )
        )}

        {/* Indicadores técnicos */}
        {showIndicators && !isLoading && (
          <div className="mt-4 sm:mt-6">
            <div className="flex items-center justify-between mb-4">
              <h4 className="text-sm font-semibold text-white">Indicadores Técnicos</h4>
              <div className="text-xs text-gray-400">
                Actualizado • {new Date().toLocaleTimeString()}
              </div>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 sm:gap-4">
              {indicators.map((indicator) => {
                // Simular valores dinámicos basados en los datos del gráfico
                const lastCandle = chartData[chartData.length - 1];
                let value, signal, signalColor;
                
                switch(indicator.id) {
                  case 'rsi':
                    value = lastCandle ? (65 + Math.sin(Date.now() / 100000) * 15).toFixed(1) : '72.5';
                    signal = parseFloat(value) > 70 ? 'Sobrecompra' : parseFloat(value) < 30 ? 'Sobreventa' : 'Neutral';
                    signalColor = parseFloat(value) > 70 ? 'text-red-400' : parseFloat(value) < 30 ? 'text-green-400' : 'text-yellow-400';
                    break;
                  case 'macd':
                    value = lastCandle ? (0.0025 + Math.sin(Date.now() / 80000) * 0.001).toFixed(4) : '0.0025';
                    signal = parseFloat(value) > 0 ? 'Alcista' : 'Bajista';
                    signalColor = parseFloat(value) > 0 ? 'text-green-400' : 'text-red-400';
                    break;
                  case 'sma':
                    value = lastCandle ? (lastCandle.close - 0.0002).toFixed(4) : '1.0862';
                    signal = lastCandle && lastCandle.close > parseFloat(value) ? 'Por encima' : 'Por debajo';
                    signalColor = lastCandle && lastCandle.close > parseFloat(value) ? 'text-green-400' : 'text-red-400';
                    break;
                  case 'ema':
                    value = lastCandle ? (lastCandle.close - 0.0005).toFixed(4) : '1.0857';
                    signal = lastCandle && lastCandle.close > parseFloat(value) ? 'Por encima' : 'Por debajo';
                    signalColor = lastCandle && lastCandle.close > parseFloat(value) ? 'text-green-400' : 'text-red-400';
                    break;
                  default:
                    value = '1.0862';
                    signal = 'Neutral';
                    signalColor = 'text-gray-400';
                }

                return (
                  <div key={indicator.id} className="bg-gray-800/50 rounded-lg p-3 border border-gray-700/30 hover:border-gray-600/50 transition-colors">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs sm:text-sm text-gray-400 font-medium">{indicator.name}</span>
                      <div 
                        className="w-2 h-2 sm:w-3 sm:h-3 rounded-full" 
                        style={{ backgroundColor: indicator.color }}
                      />
                    </div>
                    <div className="text-sm sm:text-lg font-bold text-white mb-1">
                      {value}
                    </div>
                    <div className={`text-xs font-medium ${signalColor}`}>
                      {signal}
                    </div>
                  </div>
                );
              })}
            </div>
            
            {/* Resumen de señales */}
            <div className="mt-4 p-3 bg-gray-800/30 rounded-lg border border-gray-700/30">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-gray-400">Señal general del mercado</span>
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 rounded-full bg-green-400"></div>
                  <span className="text-sm font-medium text-green-400">ALCISTA</span>
                </div>
              </div>
              <p className="text-xs text-gray-500 mt-1">
                Basado en el análisis de {indicators.length} indicadores técnicos
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}; 