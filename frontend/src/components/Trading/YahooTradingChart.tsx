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
    { id: 'candlestick', name: 'Candlestick', icon: '📊' },
    { id: 'line', name: 'Línea', icon: '📈' },
    { id: 'area', name: 'Área', icon: '📉' },
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
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between p-3 sm:p-4 border-b border-gray-700/50">
        <div className="flex flex-col sm:flex-row sm:items-center space-y-2 sm:space-y-0 sm:space-x-4 mb-3 sm:mb-0">
          <div>
            <h3 className="text-base sm:text-lg font-semibold text-white">{symbol}</h3>
            <p className="text-xs sm:text-sm text-gray-400">
              {timeframe} • Yahoo Finance • {new Date().toLocaleTimeString()}
              {isLoading && <span className="text-blue-400 ml-2">🔄 Actualizando...</span>}
            </p>
          </div>
          <div className="flex items-center space-x-2">
            <span className="text-lg sm:text-2xl font-bold text-green-400">
              {realPrice ? parseFloat(realPrice).toFixed(5) : '...'}
            </span>
            {/* Mostrar cambio porcentual si está disponible */}
            {marketData[symbol]?.changePercent && (
              <span className={`text-sm font-medium ${
                parseFloat(marketData[symbol].changePercent) >= 0 
                  ? 'text-green-400' 
                  : 'text-red-400'
              }`}>
                {parseFloat(marketData[symbol].changePercent) >= 0 ? '+' : ''}
                {parseFloat(marketData[symbol].changePercent).toFixed(2)}%
              </span>
            )}
          </div>
        </div>

        <div className="flex items-center space-x-2">
          {/* Tipo de gráfico */}
          <div className="flex items-center space-x-1 bg-gray-800/50 rounded-lg p-1">
            {chartTypes.map((type) => (
              <button
                key={type.id}
                onClick={() => setChartType(type.id as any)}
                className={`px-2 sm:px-3 py-1 sm:py-2 rounded text-xs sm:text-sm font-medium transition-colors ${
                  chartType === type.id
                    ? 'bg-blue-500 text-white'
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                <span className="hidden sm:inline">{type.icon} {type.name}</span>
                <span className="sm:hidden">{type.icon}</span>
              </button>
            ))}
          </div>

          {/* Indicadores */}
          <button
            onClick={() => setShowIndicators(!showIndicators)}
            className={`p-1.5 sm:p-2 rounded-lg transition-colors ${
              showIndicators 
                ? 'bg-blue-500/20 text-blue-400' 
                : 'bg-gray-700/50 text-gray-400 hover:text-white'
            }`}
          >
            <Settings className="w-3 h-3 sm:w-4 sm:h-4" />
          </button>

          {/* Fullscreen */}
          <button
            onClick={() => setIsFullscreen(!isFullscreen)}
            className="p-1.5 sm:p-2 rounded-lg bg-gray-700/50 text-gray-400 hover:text-white transition-colors"
          >
            {isFullscreen ? <Minimize2 className="w-3 h-3 sm:w-4 sm:h-4" /> : <Maximize2 className="w-3 h-3 sm:w-4 sm:h-4" />}
          </button>

          {/* Refresh */}
          <button className="p-1.5 sm:p-2 rounded-lg bg-gray-700/50 text-gray-400 hover:text-white transition-colors">
            <RefreshCw className="w-3 h-3 sm:w-4 sm:h-4" />
          </button>

          {/* Download */}
          <button className="p-1.5 sm:p-2 rounded-lg bg-gray-700/50 text-gray-400 hover:text-white transition-colors">
            <Download className="w-3 h-3 sm:w-4 sm:h-4" />
          </button>
        </div>
      </div>

      {/* Gráfico principal */}
      <div className="p-3 sm:p-4">
        {isLoading ? (
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
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2d3748" />
              <XAxis dataKey="date" stroke="#a0aec0" fontSize={12} />
              <YAxis stroke="#a0aec0" fontSize={12} />
              <Tooltip 
                contentStyle={{
                  backgroundColor: '#252b3d',
                  border: '1px solid #2d3748',
                  borderRadius: '8px',
                  color: '#ffffff'
                }}
              />
              <Line 
                type="monotone" 
                dataKey="close" 
                stroke="#38b2ac" 
                strokeWidth={2}
                dot={{ fill: '#38b2ac', strokeWidth: 2, r: 4 }}
              />
            </LineChart>
          ) : (
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="areaGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#38b2ac" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#38b2ac" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#2d3748" />
              <XAxis dataKey="date" stroke="#a0aec0" fontSize={12} />
              <YAxis stroke="#a0aec0" fontSize={12} />
              <Tooltip 
                contentStyle={{
                  backgroundColor: '#252b3d',
                  border: '1px solid #2d3748',
                  borderRadius: '8px',
                  color: '#ffffff'
                }}
              />
              <Area 
                type="monotone" 
                dataKey="close" 
                stroke="#38b2ac" 
                strokeWidth={2}
                fill="url(#areaGradient)"
              />
            </AreaChart>
          )
        )}

        {/* Indicadores técnicos */}
        {showIndicators && !isLoading && (
          <div className="mt-4 sm:mt-6">
            <h4 className="text-sm font-semibold text-white mb-3">Indicadores Técnicos</h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 sm:gap-4">
              {indicators.map((indicator) => (
                <div key={indicator.id} className="bg-gray-800/50 rounded-lg p-2 sm:p-3">
                  <div className="flex items-center justify-between">
                    <span className="text-xs sm:text-sm text-gray-400">{indicator.name}</span>
                    <div 
                      className="w-2 h-2 sm:w-3 sm:h-3 rounded-full" 
                      style={{ backgroundColor: indicator.color }}
                    />
                  </div>
                  <div className="text-sm sm:text-lg font-semibold text-white mt-1">
                    {indicator.id === 'rsi' ? '72.5' : 
                     indicator.id === 'macd' ? '0.0025' : '1.0862'}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}; 