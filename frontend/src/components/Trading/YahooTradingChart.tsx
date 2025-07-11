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

const CandlestickChart: React.FC<{ 
  data: any[]; 
  zoomLevel: number;
  onTouchStart?: (e: React.TouchEvent) => void; 
  onTouchMove?: (e: React.TouchEvent) => void; 
  onTouchEnd?: (e: React.TouchEvent) => void; 
}> = ({ data, zoomLevel, onTouchStart, onTouchMove, onTouchEnd }) => {
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
            rightOffset: 12,
            barSpacing: Math.max(4, 8 * zoomLevel), // Espaciado dinámico para zoom visual
            minBarSpacing: 2,
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
  }, [data, zoomLevel]);

  // Additional cleanup on unmount
  useEffect(() => {
    return () => {
      isDisposedRef.current = true;
    };
  }, []);

  return (
    <div 
      ref={chartContainerRef} 
      style={{ width: '100%', height: 300 }}
      onTouchStart={onTouchStart}
      onTouchMove={onTouchMove}
      onTouchEnd={onTouchEnd}
      className="touch-none select-none"
    />
  );
};

export const YahooTradingChart: React.FC<TradingChartProps> = ({ symbol, timeframe }) => {
  const [chartType, setChartType] = useState<'candlestick' | 'line' | 'area'>('candlestick');
  const [showIndicators, setShowIndicators] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [lastSymbol, setLastSymbol] = useState(symbol);
  const [initialLoadingCandles, setInitialLoadingCandles] = useState(true);
  const [initialLoadingMarket, setInitialLoadingMarket] = useState(true);
  const [lastUpdateTime, setLastUpdateTime] = useState<Date>(new Date());
  const [zoomLevel, setZoomLevel] = useState(1);
  const [dataRange, setDataRange] = useState({ start: 0, end: 100 });
  const [touchStart, setTouchStart] = useState<{ x: number; y: number; distance?: number } | null>(null);
  const [isTouching, setIsTouching] = useState(false);

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
      // Reset initial loading flags when symbol changes
      setInitialLoadingCandles(true);
      setInitialLoadingMarket(true);
    }
  }, [symbol, lastSymbol]);

  // Manejar el estado de loading inicial para candles
  useEffect(() => {
    if (candleData && initialLoadingCandles) {
      setInitialLoadingCandles(false);
      setLastUpdateTime(new Date());
    }
  }, [candleData, initialLoadingCandles]);

  // Manejar el estado de loading inicial para market data
  useEffect(() => {
    if (marketData && Object.keys(marketData).length > 0 && initialLoadingMarket) {
      setInitialLoadingMarket(false);
      setLastUpdateTime(new Date());
    }
  }, [marketData, initialLoadingMarket]);

  // Detectar actualizaciones en background
  useEffect(() => {
    if (!initialLoadingCandles && !initialLoadingMarket && (candleData || marketData)) {
      setLastUpdateTime(new Date());
    }
  }, [candleData, marketData, initialLoadingCandles, initialLoadingMarket]);

  // Determinar si está cargando - solo mostrar loading en carga inicial o cambio de símbolo
  const isInitialLoading = (initialLoadingCandles && candleLoading) || 
                          (initialLoadingMarket && marketLoading) || 
                          symbol !== lastSymbol;

  // Solo mostrar loading para carga inicial, no para actualizaciones en background
  const showLoading = isInitialLoading || (!candleData && !candleError) || (!marketData && !marketError);

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

  // Aplicar zoom a los datos - nuevo enfoque para zoom visual real
  const getZoomedData = () => {
    if (chartData.length === 0) return { data: [], visibleRange: { start: 0, end: chartData.length } };
    
    const totalItems = chartData.length;
    // Para zoom real: menos items = velas más anchas visualmente
    const itemsToShow = Math.max(5, Math.floor(totalItems / zoomLevel)); // Mínimo 5 items para zoom máximo
    
    // Calcular rango basado en posición de pan
    const centerPoint = (dataRange.start + dataRange.end) / 2 / 100; // Punto central como fracción
    const halfRange = itemsToShow / 2;
    
    let startIndex = Math.floor(totalItems * centerPoint - halfRange);
    let endIndex = Math.floor(totalItems * centerPoint + halfRange);
    
    // Ajustar límites
    if (startIndex < 0) {
      startIndex = 0;
      endIndex = Math.min(itemsToShow, totalItems);
    }
    if (endIndex > totalItems) {
      endIndex = totalItems;
      startIndex = Math.max(0, totalItems - itemsToShow);
    }
    
    return {
      data: chartData.slice(startIndex, endIndex),
      visibleRange: { start: startIndex, end: endIndex },
      totalItems
    };
  };

  const zoomedResult = getZoomedData();
  const zoomedChartData = zoomedResult.data;

  // Funciones de zoom
  const handleZoomIn = () => {
    setZoomLevel(prev => Math.min(prev * 1.5, 10)); // Máximo zoom 10x
  };

  const handleZoomOut = () => {
    setZoomLevel(prev => Math.max(prev / 1.5, 0.5)); // Mínimo zoom 0.5x
  };

  const handleResetZoom = () => {
    setZoomLevel(1);
    setDataRange({ start: 0, end: 100 });
  };

  // Resetear zoom cuando cambia el símbolo
  useEffect(() => {
    if (symbol !== lastSymbol) {
      setZoomLevel(1);
      setDataRange({ start: 0, end: 100 });
    }
  }, [symbol, lastSymbol]);

  const handlePanLeft = () => {
    setDataRange(prev => {
      const itemsToShow = Math.max(5, Math.floor(chartData.length / zoomLevel));
      const step = (itemsToShow / chartData.length) * 100 * 0.2; // Mover 20% del rango visible
      const range = prev.end - prev.start;
      const newStart = Math.max(0, prev.start - step);
      const newEnd = Math.min(100, newStart + range);
      return { start: newStart, end: newEnd };
    });
  };

  const handlePanRight = () => {
    setDataRange(prev => {
      const itemsToShow = Math.max(5, Math.floor(chartData.length / zoomLevel));
      const step = (itemsToShow / chartData.length) * 100 * 0.2; // Mover 20% del rango visible
      const range = prev.end - prev.start;
      const newEnd = Math.min(100, prev.end + step);
      const newStart = Math.max(0, newEnd - range);
      return { start: newStart, end: newEnd };
    });
  };

  // Funciones de gestos táctiles
  const getTouchDistance = (touches: React.TouchList) => {
    if (touches.length < 2) return 0;
    const touch1 = touches[0];
    const touch2 = touches[1];
    return Math.sqrt(
      Math.pow(touch2.clientX - touch1.clientX, 2) + 
      Math.pow(touch2.clientY - touch1.clientY, 2)
    );
  };

  const handleTouchStart = (e: React.TouchEvent) => {
    e.preventDefault();
    setIsTouching(true);
    
    if (e.touches.length === 1) {
      // Single touch - preparar para pan
      setTouchStart({
        x: e.touches[0].clientX,
        y: e.touches[0].clientY
      });
    } else if (e.touches.length === 2) {
      // Multi-touch - preparar para zoom
      const distance = getTouchDistance(e.touches);
      setTouchStart({
        x: (e.touches[0].clientX + e.touches[1].clientX) / 2,
        y: (e.touches[0].clientY + e.touches[1].clientY) / 2,
        distance
      });
    }
  };

  const handleTouchMove = (e: React.TouchEvent) => {
    if (!touchStart || !isTouching) return;
    e.preventDefault();

    if (e.touches.length === 1 && !touchStart.distance) {
      // Single touch - pan horizontal
      const deltaX = e.touches[0].clientX - touchStart.x;
      const sensitivity = 0.5;
      
      if (Math.abs(deltaX) > 10) { // Threshold mínimo
        if (deltaX > 0) {
          handlePanLeft();
        } else {
          handlePanRight();
        }
        setTouchStart({
          x: e.touches[0].clientX,
          y: e.touches[0].clientY
        });
      }
    } else if (e.touches.length === 2 && touchStart.distance) {
      // Multi-touch - zoom pinch
      const currentDistance = getTouchDistance(e.touches);
      const distanceRatio = currentDistance / touchStart.distance;
      
      if (distanceRatio > 1.1) {
        handleZoomIn();
        setTouchStart({
          x: (e.touches[0].clientX + e.touches[1].clientX) / 2,
          y: (e.touches[0].clientY + e.touches[1].clientY) / 2,
          distance: currentDistance
        });
      } else if (distanceRatio < 0.9) {
        handleZoomOut();
        setTouchStart({
          x: (e.touches[0].clientX + e.touches[1].clientX) / 2,
          y: (e.touches[0].clientY + e.touches[1].clientY) / 2,
          distance: currentDistance
        });
      }
    }
  };

  const handleTouchEnd = (e: React.TouchEvent) => {
    setIsTouching(false);
    setTouchStart(null);
  };

  // Funciones para calcular indicadores técnicos reales
  const calculateSMA = (data: any[], period: number) => {
    if (data.length < period) return null;
    const sum = data.slice(-period).reduce((acc, item) => acc + item.close, 0);
    return sum / period;
  };

  const calculateEMA = (data: any[], period: number) => {
    if (data.length < period) return null;
    const multiplier = 2 / (period + 1);
    let ema = data.slice(0, period).reduce((acc, item) => acc + item.close, 0) / period;
    
    for (let i = period; i < data.length; i++) {
      ema = (data[i].close * multiplier) + (ema * (1 - multiplier));
    }
    return ema;
  };

  const calculateRSI = (data: any[], period: number = 14) => {
    if (data.length < period + 1) return null;
    
    let gains = 0;
    let losses = 0;
    
    // Calcular cambios iniciales
    for (let i = 1; i <= period; i++) {
      const change = data[i].close - data[i - 1].close;
      if (change >= 0) {
        gains += change;
      } else {
        losses += Math.abs(change);
      }
    }
    
    let avgGain = gains / period;
    let avgLoss = losses / period;
    
    // Calcular RSI para los datos restantes
    for (let i = period + 1; i < data.length; i++) {
      const change = data[i].close - data[i - 1].close;
      const gain = change >= 0 ? change : 0;
      const loss = change < 0 ? Math.abs(change) : 0;
      
      avgGain = (avgGain * (period - 1) + gain) / period;
      avgLoss = (avgLoss * (period - 1) + loss) / period;
    }
    
    if (avgLoss === 0) return 100;
    const rs = avgGain / avgLoss;
    return 100 - (100 / (1 + rs));
  };

  const calculateMACD = (data: any[]) => {
    const ema12 = calculateEMA(data, 12);
    const ema26 = calculateEMA(data, 26);
    
    if (!ema12 || !ema26) return null;
    return ema12 - ema26;
  };

  const calculateADX = (data: any[], period: number = 14) => {
    if (data.length < period + 1) return null;
    
    const trueRanges: number[] = [];
    const plusDMs: number[] = [];
    const minusDMs: number[] = [];
    
    // Calcular True Range, +DM y -DM
    for (let i = 1; i < data.length; i++) {
      const current = data[i];
      const previous = data[i - 1];
      
      // True Range
      const tr = Math.max(
        current.high - current.low,
        Math.abs(current.high - previous.close),
        Math.abs(current.low - previous.close)
      );
      trueRanges.push(tr);
      
      // Directional Movement
      const upMove = current.high - previous.high;
      const downMove = previous.low - current.low;
      
      const plusDM = (upMove > downMove && upMove > 0) ? upMove : 0;
      const minusDM = (downMove > upMove && downMove > 0) ? downMove : 0;
      
      plusDMs.push(plusDM);
      minusDMs.push(minusDM);
    }
    
    if (trueRanges.length < period) return null;
    
    // Calcular ATR (Average True Range) suavizado
    let atr = trueRanges.slice(0, period).reduce((sum, tr) => sum + tr, 0) / period;
    let plusDI = plusDMs.slice(0, period).reduce((sum, dm) => sum + dm, 0) / period;
    let minusDI = minusDMs.slice(0, period).reduce((sum, dm) => sum + dm, 0) / period;
    
    // Suavizar para el resto de los datos
    for (let i = period; i < trueRanges.length; i++) {
      atr = (atr * (period - 1) + trueRanges[i]) / period;
      plusDI = (plusDI * (period - 1) + plusDMs[i]) / period;
      minusDI = (minusDI * (period - 1) + minusDMs[i]) / period;
    }
    
    // Calcular +DI% y -DI%
    const plusDIPercent = atr !== 0 ? (plusDI / atr) * 100 : 0;
    const minusDIPercent = atr !== 0 ? (minusDI / atr) * 100 : 0;
    
    // Calcular DX
    const diSum = plusDIPercent + minusDIPercent;
    const diDiff = Math.abs(plusDIPercent - minusDIPercent);
    const dx = diSum !== 0 ? (diDiff / diSum) * 100 : 0;
    
    // Para simplificar, retornamos el DX como aproximación del ADX
    // En una implementación completa, necesitaríamos suavizar múltiples valores DX
    return dx;
  };

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
    { id: 'adx', name: 'ADX 14', color: '#06b6d4' },
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
                {timeframe} • Yahoo Finance • {lastUpdateTime.toLocaleTimeString()}
                {isInitialLoading && <span className="text-blue-400 ml-2">🔄 Actualizando...</span>}
                {!isInitialLoading && (candleLoading || marketLoading) && (
                  <span className="text-green-400 ml-2 text-xs animate-pulse">● Live</span>
                )}
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
            {/* Controles de Zoom */}
            <div className="flex items-center space-x-1 bg-gray-800/50 rounded-lg p-1">
              {/* Controles completos para desktop */}
              <div className="hidden sm:flex items-center space-x-1">
                <button
                  onClick={handlePanLeft}
                  title="Mover a la izquierda"
                  className="p-1.5 rounded text-gray-400 hover:text-white hover:bg-gray-600/50 transition-all duration-200"
                  disabled={dataRange.start <= 0}
                >
                  <span className="text-sm">◀</span>
                </button>
                <button
                  onClick={handleZoomOut}
                  title="Alejar (Zoom Out)"
                  className="p-1.5 rounded text-gray-400 hover:text-white hover:bg-gray-600/50 transition-all duration-200"
                  disabled={zoomLevel <= 0.5}
                >
                  <span className="text-sm font-bold">−</span>
                </button>
                <button
                  onClick={handleResetZoom}
                  title="Resetear zoom"
                  className="px-2 py-1 rounded text-xs font-medium text-gray-400 hover:text-white hover:bg-gray-600/50 transition-all duration-200"
                >
                  {zoomLevel.toFixed(1)}x
                </button>
                <button
                  onClick={handleZoomIn}
                  title="Acercar (Zoom In)"
                  className="p-1.5 rounded text-gray-400 hover:text-white hover:bg-gray-600/50 transition-all duration-200"
                  disabled={zoomLevel >= 10}
                >
                  <span className="text-sm font-bold">+</span>
                </button>
                <button
                  onClick={handlePanRight}
                  title="Mover a la derecha"
                  className="p-1.5 rounded text-gray-400 hover:text-white hover:bg-gray-600/50 transition-all duration-200"
                  disabled={dataRange.end >= 100}
                >
                  <span className="text-sm">▶</span>
                </button>
              </div>

              {/* Controles simplificados para móvil */}
              <div className="flex sm:hidden items-center space-x-1">
                <button
                  onClick={handleZoomOut}
                  title="Alejar"
                  className="p-2 rounded text-gray-400 hover:text-white hover:bg-gray-600/50 transition-all duration-200"
                  disabled={zoomLevel <= 0.5}
                >
                  <span className="text-base font-bold">−</span>
                </button>
                <button
                  onClick={handleResetZoom}
                  title="Resetear zoom"
                  className="px-3 py-2 rounded text-xs font-medium text-gray-400 hover:text-white hover:bg-gray-600/50 transition-all duration-200"
                >
                  {zoomLevel.toFixed(1)}x
                </button>
                <button
                  onClick={handleZoomIn}
                  title="Acercar"
                  className="p-2 rounded text-gray-400 hover:text-white hover:bg-gray-600/50 transition-all duration-200"
                  disabled={zoomLevel >= 10}
                >
                  <span className="text-base font-bold">+</span>
                </button>
              </div>
            </div>

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
          {zoomedChartData.length > 0 && (
            <div className="text-xs text-gray-400">
              Mostrando {zoomedChartData.length} de {zoomedResult.totalItems} velas • Zoom {zoomLevel.toFixed(1)}x 
              {zoomLevel > 1 && <span className="text-blue-400"> (Vista ampliada)</span>}
              {zoomLevel < 1 && <span className="text-yellow-400"> (Vista panorámica)</span>}
              • Rango: {zoomedResult.visibleRange.start + 1}-{zoomedResult.visibleRange.end} 
              • Última actualización: {new Date(chartData[chartData.length - 1]?.date).toLocaleTimeString()}
            </div>
          )}
          
          {/* Instrucciones de uso móvil */}
          <div className="sm:hidden mt-2 text-xs text-gray-500 italic">
            💡 Desliza horizontalmente para navegar • Pellizca para zoom visual {zoomLevel > 1 ? '(ampliado)' : zoomLevel < 1 ? '(panorámico)' : '(normal)'}
          </div>
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
        ) : zoomedChartData.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-64 text-gray-400">
            <div className="text-lg font-medium mb-2">No hay datos disponibles</div>
            <div className="text-sm text-gray-500">Símbolo: {symbol}</div>
          </div>
        ) : (
          chartType === 'candlestick' ? (
            <CandlestickChart 
              data={zoomedChartData} 
              zoomLevel={zoomLevel}
              onTouchStart={handleTouchStart} 
              onTouchMove={handleTouchMove} 
              onTouchEnd={handleTouchEnd}
            />
          ) : chartType === 'line' ? (
            <div 
              onTouchStart={handleTouchStart} 
              onTouchMove={handleTouchMove} 
              onTouchEnd={handleTouchEnd}
              className="touch-none select-none"
            >
                            <ResponsiveContainer width="100%" height={300}>
                <LineChart data={zoomedChartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#2d3748" />
                  <XAxis 
                    dataKey="date" 
                    stroke="#a0aec0" 
                    fontSize={12}
                    domain={['dataMin', 'dataMax']}
                    type="category"
                    interval={Math.max(0, Math.floor(zoomedChartData.length / Math.min(8, zoomedChartData.length)))}
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
            </div>
          ) : (
            <div 
              onTouchStart={handleTouchStart} 
              onTouchMove={handleTouchMove} 
              onTouchEnd={handleTouchEnd}
              className="touch-none select-none"
            >
                            <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={zoomedChartData}>
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
                    domain={['dataMin', 'dataMax']}
                    type="category"
                    interval={Math.max(0, Math.floor(zoomedChartData.length / Math.min(8, zoomedChartData.length)))}
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
            </div>
          )
        )}

        {/* Indicadores técnicos */}
        {showIndicators && !isInitialLoading && (
          <div className="mt-4 sm:mt-6">
            <div className="flex items-center justify-between mb-4">
              <h4 className="text-sm font-semibold text-white">Indicadores Técnicos</h4>
              <div className="flex items-center space-x-2">
                {(candleLoading || marketLoading) && !isInitialLoading && (
                  <span className="text-green-400 text-xs animate-pulse">● Actualizando</span>
                )}
                <div className="text-xs text-gray-400">
                  Última actualización: {lastUpdateTime.toLocaleTimeString()}
                </div>
              </div>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3 sm:gap-4">
              {indicators.map((indicator) => {
                // Calcular indicadores técnicos reales
                const lastCandle = chartData[chartData.length - 1];
                let value, signal, signalColor;
                
                switch(indicator.id) {
                  case 'rsi':
                    const rsiValue = calculateRSI(chartData);
                    value = rsiValue ? rsiValue.toFixed(1) : 'N/A';
                    if (rsiValue) {
                      signal = rsiValue > 70 ? 'Sobrecompra' : rsiValue < 30 ? 'Sobreventa' : 'Neutral';
                      signalColor = rsiValue > 70 ? 'text-red-400' : rsiValue < 30 ? 'text-green-400' : 'text-yellow-400';
                    } else {
                      signal = 'Insuficientes datos';
                      signalColor = 'text-gray-400';
                    }
                    break;
                  case 'macd':
                    const macdValue = calculateMACD(chartData);
                    value = macdValue ? macdValue.toFixed(4) : 'N/A';
                    if (macdValue) {
                      signal = macdValue > 0 ? 'Alcista' : 'Bajista';
                      signalColor = macdValue > 0 ? 'text-green-400' : 'text-red-400';
                    } else {
                      signal = 'Insuficientes datos';
                      signalColor = 'text-gray-400';
                    }
                    break;
                  case 'sma':
                    const smaValue = calculateSMA(chartData, 20);
                    value = smaValue ? smaValue.toFixed(4) : 'N/A';
                    if (smaValue && lastCandle) {
                      signal = lastCandle.close > smaValue ? 'Por encima' : 'Por debajo';
                      signalColor = lastCandle.close > smaValue ? 'text-green-400' : 'text-red-400';
                    } else {
                      signal = 'Insuficientes datos';
                      signalColor = 'text-gray-400';
                    }
                    break;
                  case 'ema':
                    const emaValue = calculateEMA(chartData, 50);
                    value = emaValue ? emaValue.toFixed(4) : 'N/A';
                    if (emaValue && lastCandle) {
                      signal = lastCandle.close > emaValue ? 'Por encima' : 'Por debajo';
                      signalColor = lastCandle.close > emaValue ? 'text-green-400' : 'text-red-400';
                    } else {
                      signal = 'Insuficientes datos';
                      signalColor = 'text-gray-400';
                    }
                    break;
                  case 'adx':
                    const adxValue = calculateADX(chartData);
                    value = adxValue ? adxValue.toFixed(1) : 'N/A';
                    if (adxValue) {
                      if (adxValue >= 50) {
                        signal = 'Tendencia muy fuerte';
                        signalColor = 'text-purple-400';
                      } else if (adxValue >= 25) {
                        signal = 'Tendencia fuerte';
                        signalColor = 'text-blue-400';
                      } else if (adxValue >= 20) {
                        signal = 'Tendencia moderada';
                        signalColor = 'text-yellow-400';
                      } else {
                        signal = 'Sin tendencia';
                        signalColor = 'text-gray-400';
                      }
                    } else {
                      signal = 'Insuficientes datos';
                      signalColor = 'text-gray-400';
                    }
                    break;
                  default:
                    value = 'N/A';
                    signal = 'No disponible';
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
                {(() => {
                  // Calcular señal general basada en indicadores reales
                  let bullishSignals = 0;
                  let bearishSignals = 0;
                  let totalSignals = 0;
                  
                  if (chartData.length > 0) {
                    const lastCandle = chartData[chartData.length - 1];
                    const rsi = calculateRSI(chartData);
                    const macd = calculateMACD(chartData);
                    const sma = calculateSMA(chartData, 20);
                    const ema = calculateEMA(chartData, 50);
                    const adx = calculateADX(chartData);
                    
                    // RSI
                    if (rsi !== null) {
                      totalSignals++;
                      if (rsi < 30) bullishSignals++; // Sobreventa = oportunidad de compra
                      else if (rsi > 70) bearishSignals++; // Sobrecompra = oportunidad de venta
                    }
                    
                    // MACD
                    if (macd !== null) {
                      totalSignals++;
                      if (macd > 0) bullishSignals++;
                      else bearishSignals++;
                    }
                    
                    // SMA
                    if (sma !== null && lastCandle) {
                      totalSignals++;
                      if (lastCandle.close > sma) bullishSignals++;
                      else bearishSignals++;
                    }
                    
                    // EMA
                    if (ema !== null && lastCandle) {
                      totalSignals++;
                      if (lastCandle.close > ema) bullishSignals++;
                      else bearishSignals++;
                    }
                    
                    // ADX - Solo influye si hay tendencia fuerte (>=25)
                    // ADX no da señal direccional pero refuerza las otras señales
                    if (adx !== null && adx >= 25) {
                      // Si ADX es fuerte, damos peso extra a las señales existentes
                      const multiplier = adx >= 50 ? 1.5 : 1.2;
                      
                      // Determinar dirección basada en MACD y precio vs medias
                      let trendDirection = 0;
                      if (macd !== null) trendDirection += macd > 0 ? 1 : -1;
                      if (sma !== null && lastCandle) trendDirection += lastCandle.close > sma ? 1 : -1;
                      if (ema !== null && lastCandle) trendDirection += lastCandle.close > ema ? 1 : -1;
                      
                      if (trendDirection > 0) {
                        bullishSignals += 0.5 * multiplier;
                      } else if (trendDirection < 0) {
                        bearishSignals += 0.5 * multiplier;
                      }
                      totalSignals += 0.5 * multiplier;
                    }
                  }
                  
                  const bullishPercentage = totalSignals > 0 ? (bullishSignals / totalSignals) * 100 : 0;
                  let overallSignal, signalColor, signalBg;
                  
                  if (bullishPercentage >= 60) {
                    overallSignal = 'ALCISTA';
                    signalColor = 'text-green-400';
                    signalBg = 'bg-green-400';
                  } else if (bullishPercentage <= 40) {
                    overallSignal = 'BAJISTA';
                    signalColor = 'text-red-400';
                    signalBg = 'bg-red-400';
                  } else {
                    overallSignal = 'NEUTRAL';
                    signalColor = 'text-yellow-400';
                    signalBg = 'bg-yellow-400';
                  }
                  
                  return (
                    <div className="flex items-center space-x-2">
                      <div className={`w-2 h-2 rounded-full ${signalBg}`}></div>
                      <span className={`text-sm font-medium ${signalColor}`}>
                        {overallSignal} ({bullishPercentage.toFixed(0)}%)
                      </span>
                    </div>
                  );
                })()}
              </div>
              <p className="text-xs text-gray-500 mt-1">
                Basado en {chartData.length > 0 ? 'análisis en tiempo real' : 'datos insuficientes'} de {indicators.length} indicadores técnicos
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}; 