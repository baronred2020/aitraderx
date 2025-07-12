import { useState, useEffect } from 'react';

interface CandlesResponse {
  symbol: string;
  interval: string;
  values: Array<{
    datetime: string;
    open: string;
    high: string;
    low: string;
    close: string;
    volume: string;
  }>;
  error?: string;
}

interface CacheEntry {
  data: CandlesResponse;
  timestamp: number;
  callCount: number;
}

// Cache global para velas de Yahoo Finance
const yahooCandlesCache = new Map<string, CacheEntry>();
const YAHOO_CANDLES_CACHE_TTL = 60 * 1000; // 1 minuto para velas de Yahoo Finance (reducido de 5 minutos)

// Función para detectar si es fin de semana
const isWeekend = (): boolean => {
  const now = new Date();
  const dayOfWeek = now.getDay(); // 0 = Domingo, 6 = Sábado
  return dayOfWeek === 0 || dayOfWeek === 6;
};

// Función para detectar si el mercado está abierto
const isMarketOpen = (): boolean => {
  const now = new Date();
  const dayOfWeek = now.getDay();
  const hour = now.getHours();
  
  // Fin de semana - mercado cerrado
  if (dayOfWeek === 0 || dayOfWeek === 6) {
    return false;
  }
  
  // Días de semana - verificar horario de trading
  // Para simplificar, asumimos que está abierto de 9 AM a 4 PM
  return hour >= 9 && hour < 16;
};

export const useYahooCandles = (symbol: string, timeInterval: string = '15', count: number = 100) => {
  const [data, setData] = useState<CandlesResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isInitialLoad, setIsInitialLoad] = useState(true);
  const [marketStatus, setMarketStatus] = useState<'open' | 'closed'>('open');

  useEffect(() => {
    if (!symbol) return;

    // Verificar si el mercado está abierto
    const marketOpen = isMarketOpen();
    const weekend = isWeekend();
    
    setMarketStatus(marketOpen ? 'open' : 'closed');
    
    // Si es fin de semana, usar cache existente o datos vacíos
    if (weekend) {
      console.log(`[Yahoo Candles] Weekend detected for ${symbol} - using cached data or empty candles`);
      
      const cacheKey = `yahoo_${symbol}-${timeInterval}-${count}`;
      const cached = yahooCandlesCache.get(cacheKey);
      
      if (cached) {
        console.log(`[Yahoo Candles] Using cached weekend data for ${symbol}`);
        setData(cached.data);
        setIsInitialLoad(false);
        return;
      } else {
        // Datos vacíos para fines de semana
        const emptyData: CandlesResponse = {
          symbol: symbol,
          interval: timeInterval,
          values: []
        };
        
        setData(emptyData);
        setIsInitialLoad(false);
        return;
      }
    }

    const fetchCandles = async (isBackgroundUpdate = false) => {
      // Verificar cache primero
      const cacheKey = `yahoo_${symbol}-${timeInterval}-${count}`;
      const cached = yahooCandlesCache.get(cacheKey);
      const now = Date.now();
      
      // Cache válido por 1 minuto para velas de Yahoo Finance
      if (cached && (now - cached.timestamp) < YAHOO_CANDLES_CACHE_TTL) {
        console.log(`[Yahoo Candles] Using cached data for ${symbol}`);
        setData(cached.data);
        if (isInitialLoad) {
          setIsInitialLoad(false);
        }
        return;
      }

      console.log(`[Yahoo Candles] Fetching ${isBackgroundUpdate ? 'background update' : 'fresh data'} for ${symbol}`);
      
      // Solo mostrar loading en carga inicial, no en actualizaciones background
      if (!isBackgroundUpdate && isInitialLoad) {
        setLoading(true);
      }
      setError(null);

      try {
        const response = await fetch(
          `http://localhost:8000/api/candles?symbol=${symbol}&interval=${timeInterval}&count=${count}`
        );
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        console.log(`[Yahoo Candles] Received ${isBackgroundUpdate ? 'background update' : 'data'} for ${symbol}:`, result);
        
        // Actualizar cache
        yahooCandlesCache.set(cacheKey, {
          data: result,
          timestamp: now,
          callCount: (cached?.callCount || 0) + 1
        });

        setData(result);
        if (isInitialLoad) {
          setIsInitialLoad(false);
        }
      } catch (err) {
        console.error('Error fetching Yahoo Finance candles:', err);
        setError(err instanceof Error ? err.message : 'Unknown error');
        
        // En caso de error, usar cache si está disponible
        if (cached) {
          console.log(`[Yahoo Candles] Using cached data due to error`);
          setData(cached.data);
          if (isInitialLoad) {
            setIsInitialLoad(false);
          }
        }
      } finally {
        // Solo ocultar loading si lo habíamos mostrado
        if (!isBackgroundUpdate && isInitialLoad) {
          setLoading(false);
        }
      }
    };

    // Limpiar cache anterior si el símbolo cambió
    const currentCacheKey = `yahoo_${symbol}-${timeInterval}-${count}`;
    const previousCacheKeys = Array.from(yahooCandlesCache.keys()).filter(key => 
      key.includes(symbol) && key !== currentCacheKey
    );
    
    if (previousCacheKeys.length > 0) {
      console.log(`[Yahoo Candles] Clearing old cache for symbol change: ${symbol} - ${previousCacheKeys.join(', ')}`);
      previousCacheKeys.forEach(key => yahooCandlesCache.delete(key));
      setIsInitialLoad(true); // Reset cuando cambia el símbolo
    }

    // Fetch inicial inmediato
    fetchCandles(false);

    // Configurar intervalo - velas de Yahoo Finance se actualizan cada 1 minuto en background
    // Solo si el mercado está abierto
    const updateInterval = setInterval(() => {
      if (isMarketOpen() && !isWeekend()) {
        fetchCandles(true); // Marcar como actualización background
      } else {
        console.log(`[Yahoo Candles] Market closed for ${symbol} - skipping background update`);
      }
    }, 60 * 1000); // 1 minuto

    return () => clearInterval(updateInterval);
  }, [symbol, timeInterval, count, isInitialLoad]);

  return { data, loading, error, marketStatus };
}; 