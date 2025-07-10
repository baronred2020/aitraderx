import { useState, useEffect } from 'react';

interface CandleData {
  datetime: string;
  open: string;
  high: string;
  low: string;
  close: string;
  volume: string;
}

interface CandlesResponse {
  values: CandleData[];
  meta: {
    symbol: string;
    interval: string;
    currency_base: string;
    currency_quote: string;
  };
}

interface CacheEntry {
  data: CandlesResponse;
  timestamp: number;
  callCount: number;
}

// Cache global para velas
const candlesCache = new Map<string, CacheEntry>();
const DAILY_CALL_LIMIT = 800;
const CANDLES_CALL_INTERVAL = 24 * 60 * 60 * 1000 / DAILY_CALL_LIMIT; // ~108 segundos

// Contador compartido con useMarketData
let dailyCallCount = 0;
let lastCallTime = 0;

// Cargar estadísticas guardadas (compartido con useMarketData)
const loadStats = () => {
  const stored = localStorage.getItem('api_stats');
  if (stored) {
    const stats = JSON.parse(stored);
    const now = Date.now();
    const timeSinceLastCall = now - stats.lastCallTime;
    
    // Si han pasado más de 24 horas, resetear
    if (timeSinceLastCall > 24 * 60 * 60 * 1000) {
      dailyCallCount = 0;
      lastCallTime = now;
    } else {
      dailyCallCount = stats.dailyCalls || 0;
      lastCallTime = stats.lastCallTime || now;
    }
  }
};

// Cargar estadísticas al inicializar
loadStats();

// Función para verificar si podemos hacer una nueva llamada
const canMakeCall = (): boolean => {
  const now = Date.now();
  const timeSinceLastCall = now - lastCallTime;
  
  // Si han pasado más de 24 horas, resetear contador
  if (timeSinceLastCall > 24 * 60 * 60 * 1000) {
    dailyCallCount = 0;
    lastCallTime = now;
    return true;
  }
  
  // Si no hemos alcanzado el límite y ha pasado el tiempo mínimo
  if (dailyCallCount < DAILY_CALL_LIMIT && timeSinceLastCall >= CANDLES_CALL_INTERVAL) {
    return true;
  }
  
  return false;
};

// Función para registrar una llamada
const registerCall = (): void => {
  dailyCallCount++;
  lastCallTime = Date.now();
  
  // Guardar estadísticas en localStorage (compartido con useMarketData)
  const stats = {
    dailyCalls: dailyCallCount,
    lastCallTime: lastCallTime,
    timestamp: Date.now()
  };
  localStorage.setItem('api_stats', JSON.stringify(stats));
};

export const useCandles = (symbol: string, timeInterval: string = '15', count: number = 100) => {
  const [data, setData] = useState<CandlesResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!symbol) return;

    const fetchCandles = async () => {
      // Verificar cache primero
      const cacheKey = `${symbol}-${timeInterval}-${count}`;
      const cached = candlesCache.get(cacheKey);
      const now = Date.now();
      
      // Cache válido por 5 minutos para velas (menos frecuente que precios)
      if (cached && (now - cached.timestamp) < 5 * 60 * 1000) {
        setData(cached.data);
        return;
      }

      // Verificar si podemos hacer una llamada
      if (!canMakeCall()) {
        console.log('Rate limit reached for candles, using cached data');
        if (cached) {
          setData(cached.data);
        }
        return;
      }

      setLoading(true);
      setError(null);

      try {
        const response = await fetch(
          `http://localhost:8000/api/candles?symbol=${symbol}&interval=${timeInterval}&count=${count}`
        );
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        
        // Registrar la llamada exitosa
        registerCall();
        
        // Actualizar cache
        candlesCache.set(cacheKey, {
          data: result,
          timestamp: now,
          callCount: (cached?.callCount || 0) + 1
        });

        setData(result);
      } catch (err) {
        console.error('Error fetching candles:', err);
        setError(err instanceof Error ? err.message : 'Unknown error');
        
        // En caso de error, usar cache si está disponible
        if (cached) {
          setData(cached.data);
        }
      } finally {
        setLoading(false);
      }
    };

    // Fetch inicial
    fetchCandles();

    // Configurar intervalo inteligente - velas se actualizan menos frecuentemente
    const updateInterval = setInterval(() => {
      fetchCandles();
    }, Math.max(CANDLES_CALL_INTERVAL, 5 * 60 * 1000)); // Mínimo 5 minutos

    return () => clearInterval(updateInterval);
  }, [symbol, timeInterval, count]);

  return { data, loading, error };
}; 