import React, { useState, useEffect, useMemo, useRef } from 'react';
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
  Zap,
  Clock,
  Calendar,
  CandlestickChart as CandlestickChartIcon,
  Activity,
  AreaChart as AreaChartIcon,
  BarChart3,
  Target
} from 'lucide-react';
import { useCandles } from '../../hooks/useCandles';
import { useMarketData } from '../../hooks/useMarketData';
import { useIntelligentAnalysis, TradingType } from '../../hooks/useIntelligentAnalysis';
import { AnalysisResults } from './AnalysisResults';

// Tipos simples para lightweight-charts
declare const createChart: any;

interface TradingChartProps {
  symbol: string;
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

// Opciones de tipo de trading (declarar antes del useState)
const tradingTypes: (TradingType & { icon: any })[] = [
  {
    id: 'scalping',
    name: 'Scalping',
    description: '1-5 minutos Máxima precisión',
    timeframe: '1-5m',
    reason: 'Operaciones rápidas de alta precisión',
    icon: Zap,
    color: 'text-pink-400'
  },
  {
    id: 'day_trading',
    name: 'Day Trading',
    description: '15-30 minutos Balance óptimo',
    timeframe: '15-30m',
    reason: 'Operaciones intradía con balance riesgo/beneficio',
    icon: Clock,
    color: 'text-blue-400'
  },
  {
    id: 'swing_trading',
    name: 'Swing Trading',
    description: '1-4 horas Tendencias medias',
    timeframe: '1-4h',
    reason: 'Captura de tendencias de mediano plazo',
    icon: TrendingUp,
    color: 'text-green-400'
  },
  {
    id: 'position_trading',
    name: 'Position Trading',
    description: '1 día Tendencias largas',
    timeframe: '1d',
    reason: 'Posiciones de largo plazo',
    icon: Calendar,
    color: 'text-yellow-400'
  }
];

export const YahooTradingChart: React.FC<TradingChartProps> = ({ symbol }) => {
  const [chartType, setChartType] = useState<'candlestick' | 'line' | 'area'>('candlestick');
  const [showIndicators, setShowIndicators] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [tradingType, setTradingType] = useState<(TradingType & { icon: any })>(tradingTypes[0]);
  const [timeframe, setTimeframe] = useState(tradingTypes[0].timeframe); // Estado local para timeframe
  const [showTradingTypeMenu, setShowTradingTypeMenu] = useState(false);
  const { executeAnalysis, lastAnalysis, isAnalyzing } = useIntelligentAnalysis();
  const [lastSymbol, setLastSymbol] = useState(symbol);
  const [initialLoadingCandles, setInitialLoadingCandles] = useState(true);
  const [initialLoadingMarket, setInitialLoadingMarket] = useState(true);
  const [lastUpdateTime, setLastUpdateTime] = useState<Date>(new Date());
  const [zoomLevel, setZoomLevel] = useState(1);
  const [dataRange, setDataRange] = useState({ start: 0, end: 100 });
  const [touchStart, setTouchStart] = useState<{ x: number; y: number; distance?: number } | null>(null);
  const [isTouching, setIsTouching] = useState(false);
  const [showSmartAnalysis, setShowSmartAnalysis] = useState(true);
  const [showAnalysisModal, setShowAnalysisModal] = useState(false);

  // Función para mapear timeframe de trading type a intervalo del backend
  const mapTimeframeToInterval = (timeframe: string): string => {
    switch (timeframe) {
      case '1-5m':
        return '5'; // 5 minutos para scalping
      case '15-30m':
        return '15'; // 15 minutos para day trading
      case '1-4h':
        return '60'; // 1 hora para swing trading
      case '1d':
        return 'D'; // 1 día para position trading (usar 'D' en lugar de '1440')
      default:
        return '15'; // Default a 15 minutos
    }
  };

  // Hook para datos de velas y mercado con timeframe dinámico
  const { data: candleData, loading: candleLoading, error: candleError } = useCandles(symbol, mapTimeframeToInterval(timeframe), 100);
  const { data: marketData, loading: marketLoading, error: marketError } = useMarketData([symbol]);
  const realPrice = marketData[symbol]?.price;

  // Log para debugging de timeframe
  useEffect(() => {
    console.log(`[YahooTradingChart] Timeframe changed to: ${timeframe} (mapped to: ${mapTimeframeToInterval(timeframe)})`);
  }, [timeframe]);

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

  // 🤖 FUNCIONES INTELIGENTES DE TRADING
  
  // Detectar patrones de candlestick
  const detectCandlestickPatterns = (data: any[]) => {
    if (data.length < 3) return [];
    
    const patterns = [];
    
    for (let i = 2; i < data.length; i++) {
      const current = data[i];
      const previous = data[i - 1];
      const twoBefore = data[i - 2];
      
      const currentBody = Math.abs(current.close - current.open);
      const currentUpper = current.high - Math.max(current.open, current.close);
      const currentLower = Math.min(current.open, current.close) - current.low;
      const previousBody = Math.abs(previous.close - previous.open);
      
      // Doji Pattern
      if (currentBody < (current.high - current.low) * 0.1) {
        patterns.push({
          index: i,
          name: 'Doji',
          type: 'neutral',
          strength: 0.7,
          description: 'Indecisión del mercado',
          probability: 65
        });
      }
      
      // Hammer Pattern
      if (current.close > current.open && // Vela alcista
          currentLower > currentBody * 2 && // Sombra inferior larga
          currentUpper < currentBody * 0.5) { // Sombra superior corta
        patterns.push({
          index: i,
          name: 'Hammer',
          type: 'bullish',
          strength: 0.8,
          description: 'Posible reversión alcista',
          probability: 75
        });
      }
      
      // Shooting Star Pattern
      if (current.close < current.open && // Vela bajista
          currentUpper > currentBody * 2 && // Sombra superior larga
          currentLower < currentBody * 0.5) { // Sombra inferior corta
        patterns.push({
          index: i,
          name: 'Shooting Star',
          type: 'bearish',
          strength: 0.8,
          description: 'Posible reversión bajista',
          probability: 75
        });
      }
      
      // Bullish Engulfing
      if (previous.close < previous.open && // Vela anterior bajista
          current.close > current.open && // Vela actual alcista
          current.open < previous.close && // Apertura por debajo del cierre anterior
          current.close > previous.open) { // Cierre por encima de la apertura anterior
        patterns.push({
          index: i,
          name: 'Bullish Engulfing',
          type: 'bullish',
          strength: 0.9,
          description: 'Fuerte señal alcista',
          probability: 85
        });
      }
      
      // Bearish Engulfing
      if (previous.close > previous.open && // Vela anterior alcista
          current.close < current.open && // Vela actual bajista
          current.open > previous.close && // Apertura por encima del cierre anterior
          current.close < previous.open) { // Cierre por debajo de la apertura anterior
        patterns.push({
          index: i,
          name: 'Bearish Engulfing',
          type: 'bearish',
          strength: 0.9,
          description: 'Fuerte señal bajista',
          probability: 85
        });
      }
    }
    
    return patterns;
  };

  // Detectar niveles de soporte y resistencia
  const detectSupportResistanceLevels = (data: any[]) => {
    if (data.length < 10) return [];
    
    const levels = [];
    const minTouches = 2; // Mínimo de toques para considerar un nivel válido
    const tolerance = 0.001; // Tolerancia para considerar que el precio "toca" un nivel
    
    // Buscar máximos y mínimos locales
    for (let i = 5; i < data.length - 5; i++) {
      const current = data[i];
      let isLocalHigh = true;
      let isLocalLow = true;
      
      // Verificar si es máximo local
      for (let j = i - 5; j <= i + 5; j++) {
        if (j !== i && data[j].high >= current.high) {
          isLocalHigh = false;
          break;
        }
      }
      
      // Verificar si es mínimo local
      for (let j = i - 5; j <= i + 5; j++) {
        if (j !== i && data[j].low <= current.low) {
          isLocalLow = false;
          break;
        }
      }
      
      if (isLocalHigh) {
        // Contar cuántas veces el precio ha tocado este nivel
        let touches = 1;
        const level = current.high;
        
        for (let k = 0; k < data.length; k++) {
          if (k !== i && Math.abs(data[k].high - level) <= level * tolerance) {
            touches++;
          }
        }
        
        if (touches >= minTouches) {
          levels.push({
            price: level,
            type: 'resistance',
            touches: touches,
            strength: Math.min(touches / 5, 1), // Máximo strength de 1
            lastTouch: i
          });
        }
      }
      
      if (isLocalLow) {
        // Contar cuántas veces el precio ha tocado este nivel
        let touches = 1;
        const level = current.low;
        
        for (let k = 0; k < data.length; k++) {
          if (k !== i && Math.abs(data[k].low - level) <= level * tolerance) {
            touches++;
          }
        }
        
        if (touches >= minTouches) {
          levels.push({
            price: level,
            type: 'support',
            touches: touches,
            strength: Math.min(touches / 5, 1), // Máximo strength de 1
            lastTouch: i
          });
        }
      }
    }
    
    // Eliminar niveles duplicados y ordenar por strength
    const uniqueLevels = levels.filter((level, index, self) => 
      index === self.findIndex(l => 
        Math.abs(l.price - level.price) <= level.price * tolerance && l.type === level.type
      )
    ).sort((a, b) => b.strength - a.strength).slice(0, 10); // Top 10 niveles más fuertes
    
    return uniqueLevels;
  };

  // Generar señales de trading inteligentes
  const generateTradingSignals = (data: any[], rsiPeriod: number = 14, smaPeriod: number = 20, emaPeriod: number = 50) => {
    if (data.length < 50) return [];
    
    const signals = [];
    const lastCandle = data[data.length - 1];
    const rsi = calculateRSI(data, rsiPeriod);
    const macd = calculateMACD(data);
    const sma = calculateSMA(data, smaPeriod);
    const ema = calculateEMA(data, emaPeriod);
    const adx = calculateADX(data);
    const patterns = detectCandlestickPatterns(data);
    const levels = detectSupportResistanceLevels(data);
    
    let bullishSignals = 0;
    let bearishSignals = 0;
    let signalStrength = 0;
    
    // Análisis RSI
    if (rsi !== null) {
      if (rsi < 30) {
        bullishSignals++;
        signalStrength += 0.8;
      } else if (rsi > 70) {
        bearishSignals++;
        signalStrength += 0.8;
      }
    }
    
    // Análisis MACD
    if (macd !== null) {
      if (macd > 0) {
        bullishSignals++;
        signalStrength += 0.7;
      } else {
        bearishSignals++;
        signalStrength += 0.7;
      }
    }
    
    // Análisis de Medias Móviles
    if (sma !== null && ema !== null && lastCandle) {
      if (lastCandle.close > sma && sma > ema) {
        bullishSignals++;
        signalStrength += 0.6;
      } else if (lastCandle.close < sma && sma < ema) {
        bearishSignals++;
        signalStrength += 0.6;
      }
    }
    
    // Análisis de patrones recientes
    const recentPatterns = patterns.filter(p => p.index >= data.length - 3);
    recentPatterns.forEach(pattern => {
      if (pattern.type === 'bullish') {
        bullishSignals++;
        signalStrength += pattern.strength;
      } else if (pattern.type === 'bearish') {
        bearishSignals++;
        signalStrength += pattern.strength;
      }
    });
    
    // Análisis de soporte/resistencia
    const currentPrice = lastCandle.close;
    const nearLevels = levels.filter(level => 
      Math.abs(level.price - currentPrice) / currentPrice < 0.005 // Dentro del 0.5%
    );
    
    nearLevels.forEach(level => {
      if (level.type === 'support' && currentPrice > level.price) {
        bullishSignals++;
        signalStrength += level.strength * 0.5;
      } else if (level.type === 'resistance' && currentPrice < level.price) {
        bearishSignals++;
        signalStrength += level.strength * 0.5;
      }
    });
    
    // Generar señal final
    if (bullishSignals > bearishSignals && signalStrength > 2) {
      signals.push({
        type: 'BUY',
        strength: Math.min(signalStrength / 5, 1),
        confidence: Math.min((bullishSignals / (bullishSignals + bearishSignals)) * 100, 95),
        reasons: [
          ...(rsi !== null && rsi < 30 ? [`RSI sobreventa (${rsi.toFixed(1)})`] : []),
          ...(macd !== null && macd > 0 ? ['MACD alcista'] : []),
          ...(sma !== null && ema !== null && lastCandle && lastCandle.close > sma && sma > ema ? ['Tendencia alcista'] : []),
          ...recentPatterns.filter(p => p.type === 'bullish').map(p => p.name),
          ...nearLevels.filter(l => l.type === 'support').map(l => `Soporte en ${l.price.toFixed(4)}`)
        ],
        adxStrength: adx,
        suggestedStopLoss: currentPrice * 0.98, // 2% stop loss
        suggestedTakeProfit: currentPrice * 1.04 // 4% take profit
      });
    } else if (bearishSignals > bullishSignals && signalStrength > 2) {
      signals.push({
        type: 'SELL',
        strength: Math.min(signalStrength / 5, 1),
        confidence: Math.min((bearishSignals / (bullishSignals + bearishSignals)) * 100, 95),
        reasons: [
          ...(rsi !== null && rsi > 70 ? [`RSI sobrecompra (${rsi.toFixed(1)})`] : []),
          ...(macd !== null && macd < 0 ? ['MACD bajista'] : []),
          ...(sma !== null && ema !== null && lastCandle && lastCandle.close < sma && sma < ema ? ['Tendencia bajista'] : []),
          ...recentPatterns.filter(p => p.type === 'bearish').map(p => p.name),
          ...nearLevels.filter(l => l.type === 'resistance').map(l => `Resistencia en ${l.price.toFixed(4)}`)
        ],
        adxStrength: adx,
        suggestedStopLoss: currentPrice * 1.02, // 2% stop loss
        suggestedTakeProfit: currentPrice * 0.96 // 4% take profit
      });
    }
    
    return signals;
  };

  // 🤖 Calcular análisis inteligente
  const smartAnalysis = useMemo(() => {
    if (chartData.length < 50) return null;
    
    // Ajustar parámetros según el tipo de trading
    let rsiPeriod = 14;
    let smaPeriod = 20;
    let emaPeriod = 50;
    let riskRewardRatio = 2;
    let stopLossPercent = 0.02; // 2%
    let takeProfitPercent = 0.04; // 4%
    
    // Ajustar parámetros según el tipo de trading
    switch (tradingType.id) {
      case 'scalping':
        rsiPeriod = 7; // RSI más rápido para scalping
        smaPeriod = 10;
        emaPeriod = 20;
        riskRewardRatio = 1.5; // Ratio más conservador
        stopLossPercent = 0.01; // 1% stop loss
        takeProfitPercent = 0.015; // 1.5% take profit
        break;
      case 'day_trading':
        rsiPeriod = 14; // RSI estándar
        smaPeriod = 20;
        emaPeriod = 50;
        riskRewardRatio = 2;
        stopLossPercent = 0.02;
        takeProfitPercent = 0.04;
        break;
      case 'swing_trading':
        rsiPeriod = 21; // RSI más lento para swing
        smaPeriod = 50;
        emaPeriod = 100;
        riskRewardRatio = 3; // Ratio más agresivo
        stopLossPercent = 0.03; // 3% stop loss
        takeProfitPercent = 0.09; // 9% take profit
        break;
      case 'position_trading':
        rsiPeriod = 30; // RSI muy lento para position
        smaPeriod = 100;
        emaPeriod = 200;
        riskRewardRatio = 4; // Ratio muy agresivo
        stopLossPercent = 0.05; // 5% stop loss
        takeProfitPercent = 0.20; // 20% take profit
        break;
    }
    
    // Generar señales con parámetros ajustados
    const tradingSignals = generateTradingSignals(chartData, rsiPeriod, smaPeriod, emaPeriod);
    const candlestickPatterns = detectCandlestickPatterns(chartData.slice(-20));
    const supportResistanceLevels = detectSupportResistanceLevels(chartData);
    
    const currentPrice = chartData[chartData.length - 1]?.close || 0;
    
    return {
      tradingSignals,
      candlestickPatterns,
      supportResistanceLevels,
      riskRewardSuggestion: {
        currentPrice,
        suggestedStopLoss: currentPrice * (1 - stopLossPercent),
        suggestedTakeProfit: currentPrice * (1 + takeProfitPercent),
        riskRewardRatio,
        tradingType: tradingType.name,
        timeframe: timeframe
      }
    };
  }, [chartData, tradingType, timeframe]); // Agregar dependencias

  const chartTypes = [
    {
      id: 'candlestick',
      name: 'Candlestick',
      icon: <CandlestickChartIcon className="w-5 h-5" />, 
      description: 'Gráfico de velas japonesas'
    },
    {
      id: 'line',
      name: 'Línea',
      icon: <Activity className="w-5 h-5" />, 
      description: 'Gráfico de líneas del precio de cierre'
    },
    {
      id: 'area',
      name: 'Área',
      icon: <AreaChartIcon className="w-5 h-5" />, 
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

  // Manejar cambio de tipo de trading
  const handleTradingTypeChange = async (type: TradingType & { icon: any }) => {
    console.log(`[YahooTradingChart] Changing trading type to: ${type.name} with timeframe: ${type.timeframe}`);
    setTradingType(type);
    setTimeframe(type.timeframe);
    setShowTradingTypeMenu(false);
    console.log(`[YahooTradingChart] Executing intelligent analysis for ${symbol} with type: ${type.name}`);
    const result = await executeAnalysis(type, symbol);
    console.log(`[YahooTradingChart] Analysis completed for ${type.name}`);
    
    // Los indicadores técnicos y el gráfico se actualizan automáticamente por el nuevo timeframe
  };

  // Cerrar menú al hacer click fuera
  React.useEffect(() => {
    if (!showTradingTypeMenu) return;
    const handleClick = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      if (!target.closest('.trading-type-menu-container')) {
        setShowTradingTypeMenu(false);
      }
    };
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [showTradingTypeMenu]);

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
        <div className="flex flex-col gap-y-2 sm:flex-row sm:items-center sm:justify-between">
          {/* Fila 1: Tipo de gráfico + Tipo de trading */}
          <div className="flex flex-row flex-wrap gap-x-2 w-full sm:w-auto">
            {/* Tipo de gráfico */}
            <div className="flex flex-row items-center bg-gray-800/50 rounded-lg p-1">
              <span className="text-xs font-medium text-gray-400 hidden xs:inline">Tipo de gráfico:</span>
              {chartTypes.map((type) => (
                <button
                  key={type.id}
                  onClick={() => setChartType(type.id as any)}
                  title={type.description}
                  className={`px-2 sm:px-3 py-1 sm:py-2 rounded text-xs sm:text-sm font-medium transition-colors ${
                    chartType === type.id
                      ? 'bg-blue-500 text-white'
                      : 'text-gray-400 hover:text-white'
                  }`}
                >
                  <span className="flex items-center space-x-1">
                    <span>{type.icon}</span>
                    <span className="hidden sm:inline">{type.name}</span>
                  </span>
                </button>
              ))}
            </div>
            {/* Tipo de trading */}
            <div className="flex flex-row items-center gap-x-1">
              <span className="text-xs font-medium text-gray-400 hidden xs:inline">Tipo de trading:</span>
              <div className="relative trading-type-menu-container">
                <button
                  onClick={() => setShowTradingTypeMenu((v) => !v)}
                  className={`flex flex-col items-center justify-center text-center p-1.5 rounded-xl transition-all duration-150 ${tradingType.color} bg-gray-800 hover:bg-gray-700 shadow min-w-[80px]`}
                  title={tradingType.description}
                  aria-haspopup="listbox"
                  aria-expanded={showTradingTypeMenu}
                >
                  <div className="flex flex-col items-center justify-center gap-y-0.5">
                    {React.createElement(tradingType.icon, { className: "w-5 h-5 mb-0.5" })}
                    <span className="text-[10px] text-white font-semibold">Tipo de Trading</span>
                    <span className={`text-xs font-bold ${tradingType.color}`}>{tradingType.name}</span>
                  </div>
                </button>
                {showTradingTypeMenu && (
                  <div className="absolute z-50 mt-2 w-full sm:w-44 right-0 bg-gray-900 border border-gray-700 rounded-lg shadow-xl animate-fade-in">
                    {tradingTypes.map((type) => {
                      const Icon = type.icon;
                      return (
                        <button
                          key={type.id}
                          onClick={() => handleTradingTypeChange(type)}
                          className={`w-full flex items-center px-3 py-2 text-left transition-colors group
                            ${tradingType.id === type.id
                              ? `bg-blue-900/40 border-l-4 ${type.color} font-bold`
                              : 'hover:bg-gray-800'}
                          `}
                          title={type.description}
                          role="option"
                          aria-selected={tradingType.id === type.id}
                        >
                          <Icon className={`w-4 h-4 mr-2 ${type.color}`} />
                          <span className={`text-xs ${type.color}`}>{type.name}</span>
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>
          </div>
          {/* Fila 2: Controles adicionales */}
          <div className="flex flex-row flex-wrap gap-x-2 w-full sm:w-auto">
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
            {/* Análisis Inteligente */}
            <button
              onClick={() => setShowSmartAnalysis(!showSmartAnalysis)}
              title={showSmartAnalysis ? 'Ocultar análisis inteligente' : 'Mostrar análisis inteligente'}
              className={`p-2 rounded-lg transition-all duration-200 ${
                showSmartAnalysis 
                  ? 'bg-purple-500/20 text-purple-400 shadow-lg' 
                  : 'bg-gray-700/50 text-gray-400 hover:text-white hover:bg-gray-600/50'
              }`}
            >
              <span className="text-sm">🤖</span>
            </button>
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

        {/* 🤖 ANÁLISIS INTELIGENTE */}
        {showSmartAnalysis && !isInitialLoading && smartAnalysis && (
          <div className="mt-4 sm:mt-6">
            <div className="flex items-center justify-between mb-4">
              <h4 className="text-sm font-semibold text-white flex items-center space-x-2">
                <span>🤖</span>
                <span>Análisis Inteligente de Trading</span>
              </h4>
              <div className="flex items-center space-x-2">
                <div className="text-xs text-purple-400">
                  AI-Powered • Tiempo real
                </div>
                <button
                  onClick={async () => {
                    try {
                      const result = await executeAnalysis(tradingType, symbol);
                      if (result) {
                        setShowAnalysisModal(true);
                      }
                    } catch (error) {
                      console.error('Error ejecutando análisis:', error);
                    }
                  }}
                  disabled={isAnalyzing}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                    isAnalyzing 
                      ? 'bg-gray-600 text-gray-400 cursor-not-allowed' 
                      : 'bg-blue-500 hover:bg-blue-600 text-white'
                  }`}
                >
                  {isAnalyzing ? (
                    <span className="flex items-center space-x-1">
                      <div className="w-3 h-3 border border-white border-t-transparent rounded-full animate-spin"></div>
                      <span>Analizando...</span>
                    </span>
                  ) : (
                    <span className="flex items-center space-x-1">
                      <Target className="w-3 h-3" />
                      <span>Ver Análisis Completo</span>
                    </span>
                  )}
                </button>
              </div>
            </div>

            {/* Información del Tipo de Trading */}
            <div className="mb-4 p-3 bg-gradient-to-r from-blue-900/20 to-purple-900/20 rounded-lg border border-blue-500/30">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className={`p-2 rounded-lg ${tradingType.color} bg-gray-800/50`}>
                    {React.createElement(tradingType.icon, { className: "w-4 h-4" })}
                  </div>
                  <div>
                    <div className="text-sm font-semibold text-white">{tradingType.name}</div>
                    <div className="text-xs text-gray-400">{tradingType.description}</div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-xs text-gray-400">Timeframe</div>
                  <div className="text-sm font-semibold text-blue-400">{timeframe}</div>
                </div>
              </div>
              <div className="mt-2 text-xs text-gray-400">
                <strong>Razón:</strong> {tradingType.reason}
              </div>
            </div>

            {/* Señales de Trading */}
            {smartAnalysis.tradingSignals.length > 0 && (
              <div className="mb-6 p-4 bg-gradient-to-r from-purple-900/20 to-blue-900/20 rounded-lg border border-purple-500/30">
                <h5 className="text-sm font-semibold text-purple-300 mb-3 flex items-center space-x-2">
                  <span>⚡</span>
                  <span>Señal de Trading Detectada</span>
                </h5>
                {smartAnalysis.tradingSignals.map((signal, index) => (
                  <div key={index} className="space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <div className={`px-3 py-1 rounded-full text-sm font-bold ${
                          signal.type === 'BUY' 
                            ? 'bg-green-500/20 text-green-400 border border-green-500/50' 
                            : 'bg-red-500/20 text-red-400 border border-red-500/50'
                        }`}>
                          {signal.type === 'BUY' ? '📈 COMPRAR' : '📉 VENDER'}
                        </div>
                        <div className="text-xs text-gray-400">
                          Confianza: <span className="text-white font-semibold">{signal.confidence.toFixed(0)}%</span>
                        </div>
                        <div className="text-xs text-gray-400">
                          Fuerza: <span className="text-white font-semibold">{(signal.strength * 100).toFixed(0)}%</span>
                        </div>
                      </div>
                      {signal.adxStrength && signal.adxStrength > 25 && (
                        <div className="text-xs text-yellow-400 font-semibold">
                          🔥 Tendencia Fuerte (ADX: {signal.adxStrength.toFixed(1)})
                        </div>
                      )}
                    </div>
                    
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
                      <div className="bg-gray-800/50 rounded p-2">
                        <div className="text-gray-400 mb-1">Razones:</div>
                        <div className="space-y-1">
                          {signal.reasons.map((reason, i) => (
                            <div key={i} className="text-white">• {reason}</div>
                          ))}
                        </div>
                      </div>
                      <div className="bg-gray-800/50 rounded p-2">
                        <div className="text-gray-400 mb-1">Stop Loss Sugerido:</div>
                        <div className="text-red-400 font-semibold">{signal.suggestedStopLoss.toFixed(4)}</div>
                        <div className="text-gray-500 text-xs mt-1">
                          Riesgo: {Math.abs(((signal.suggestedStopLoss / (chartData[chartData.length - 1]?.close || 1)) - 1) * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div className="bg-gray-800/50 rounded p-2">
                        <div className="text-gray-400 mb-1">Take Profit Sugerido:</div>
                        <div className="text-green-400 font-semibold">{signal.suggestedTakeProfit.toFixed(4)}</div>
                        <div className="text-gray-500 text-xs mt-1">
                          Beneficio: {Math.abs(((signal.suggestedTakeProfit / (chartData[chartData.length - 1]?.close || 1)) - 1) * 100).toFixed(1)}%
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {/* Patrones de Candlestick */}
              <div className="bg-gray-800/30 rounded-lg p-4 border border-gray-700/30">
                <h5 className="text-sm font-semibold text-orange-300 mb-3 flex items-center space-x-2">
                  <span>🕯️</span>
                  <span>Patrones Detectados</span>
                </h5>
                {smartAnalysis.candlestickPatterns.length > 0 ? (
                  <div className="space-y-2 max-h-32 overflow-y-auto">
                    {smartAnalysis.candlestickPatterns.slice(-5).map((pattern, index) => (
                      <div key={index} className="flex items-center justify-between bg-gray-700/50 rounded px-3 py-2">
                        <div>
                          <div className={`text-sm font-medium ${
                            pattern.type === 'bullish' ? 'text-green-400' : 
                            pattern.type === 'bearish' ? 'text-red-400' : 'text-yellow-400'
                          }`}>
                            {pattern.name}
                          </div>
                          <div className="text-xs text-gray-400">{pattern.description}</div>
                        </div>
                        <div className="text-right">
                          <div className="text-xs text-gray-300">{pattern.probability}%</div>
                          <div className="text-xs text-gray-500">éxito</div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-gray-500 text-sm italic">No hay patrones recientes detectados</div>
                )}
              </div>

              {/* Niveles de Soporte y Resistencia */}
              <div className="bg-gray-800/30 rounded-lg p-4 border border-gray-700/30">
                <h5 className="text-sm font-semibold text-cyan-300 mb-3 flex items-center space-x-2">
                  <span>🎯</span>
                  <span>Niveles Clave</span>
                </h5>
                {smartAnalysis.supportResistanceLevels.length > 0 ? (
                  <div className="space-y-2 max-h-32 overflow-y-auto">
                    {smartAnalysis.supportResistanceLevels.slice(0, 5).map((level, index) => (
                      <div key={index} className="flex items-center justify-between bg-gray-700/50 rounded px-3 py-2">
                        <div className="flex items-center space-x-2">
                          <div className={`w-2 h-2 rounded-full ${
                            level.type === 'resistance' ? 'bg-red-400' : 'bg-green-400'
                          }`}></div>
                          <div>
                            <div className="text-sm font-medium text-white">{level.price.toFixed(4)}</div>
                            <div className="text-xs text-gray-400 capitalize">{level.type}</div>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="text-xs text-gray-300">{level.touches} toques</div>
                          <div className="text-xs text-gray-500">
                            Fuerza: {(level.strength * 100).toFixed(0)}%
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-gray-500 text-sm italic">Calculando niveles...</div>
                )}
              </div>
            </div>

            {/* Risk/Reward Calculator */}
            <div className="mt-4 bg-gradient-to-r from-gray-800/50 to-gray-700/50 rounded-lg p-4 border border-gray-600/30">
              <h5 className="text-sm font-semibold text-green-300 mb-3 flex items-center space-x-2">
                <span>⚖️</span>
                <span>Calculadora Risk/Reward</span>
              </h5>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-sm">
                <div className="text-center">
                  <div className="text-gray-400 mb-1">Precio Actual</div>
                  <div className="text-white font-semibold text-lg">
                    {smartAnalysis.riskRewardSuggestion.currentPrice.toFixed(4)}
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-gray-400 mb-1">Stop Loss Sugerido</div>
                  <div className="text-red-400 font-semibold">
                    {smartAnalysis.riskRewardSuggestion.suggestedStopLoss.toFixed(4)}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    -{Math.abs(((smartAnalysis.riskRewardSuggestion.suggestedStopLoss / smartAnalysis.riskRewardSuggestion.currentPrice) - 1) * 100).toFixed(1)}% riesgo
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-gray-400 mb-1">Take Profit Sugerido</div>
                  <div className="text-green-400 font-semibold">
                    {smartAnalysis.riskRewardSuggestion.suggestedTakeProfit.toFixed(4)}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    +{Math.abs(((smartAnalysis.riskRewardSuggestion.suggestedTakeProfit / smartAnalysis.riskRewardSuggestion.currentPrice) - 1) * 100).toFixed(1)}% beneficio
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-gray-400 mb-1">Ratio R/R</div>
                  <div className="text-yellow-400 font-semibold text-lg">
                    1:{smartAnalysis.riskRewardSuggestion.riskRewardRatio}
                  </div>
                  <div className="text-xs text-green-500 mt-1">
                    ✅ Ratio saludable
                  </div>
                </div>
              </div>
            </div>

            {/* Disclaimer */}
            <div className="mt-4 p-3 bg-yellow-900/20 border border-yellow-500/30 rounded-lg">
              <p className="text-xs text-yellow-200">
                ⚠️ <strong>Disclaimer:</strong> Este análisis es generado por algoritmos y debe usarse solo como referencia. 
                Siempre realiza tu propio análisis antes de tomar decisiones de trading. El trading conlleva riesgos significativos.
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Modal de Análisis Inteligente */}
      {showAnalysisModal && lastAnalysis && (
        <AnalysisResults 
          result={lastAnalysis} 
          onClose={() => setShowAnalysisModal(false)} 
        />
      )}
    </div>
  );
}; 