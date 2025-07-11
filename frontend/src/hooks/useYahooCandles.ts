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

// Cache global para velas de Yahoo Finance
const yahooCandlesCache = new Map<string, CacheEntry>();
const YAHOO_CANDLES_CACHE_TTL = 60 * 1000; // 1 minuto para velas de Yahoo Finance (reducido de 5 minutos)

export const useYahooCandles = (symbol: string, timeInterval: string = '15', count: number = 100) => {
  const [data, setData] = useState<CandlesResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!symbol) return;

    const fetchCandles = async () => {
      // Verificar cache primero
      const cacheKey = `yahoo_${symbol}-${timeInterval}-${count}`;
      const cached = yahooCandlesCache.get(cacheKey);
      const now = Date.now();
      
      // Cache válido por 1 minuto para velas de Yahoo Finance (reducido para actualizaciones más frecuentes)
      if (cached && (now - cached.timestamp) < YAHOO_CANDLES_CACHE_TTL) {
        console.log(`[Yahoo Candles] Using cached data for ${symbol}`);
        setData(cached.data);
        return;
      }

      console.log(`[Yahoo Candles] Fetching fresh data for ${symbol}`);
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
        console.log(`[Yahoo Candles] Received data for ${symbol}:`, result);
        
        // Actualizar cache
        yahooCandlesCache.set(cacheKey, {
          data: result,
          timestamp: now,
          callCount: (cached?.callCount || 0) + 1
        });

        setData(result);
      } catch (err) {
        console.error('Error fetching Yahoo Finance candles:', err);
        setError(err instanceof Error ? err.message : 'Unknown error');
        
        // En caso de error, usar cache si está disponible
        if (cached) {
          console.log(`[Yahoo Candles] Using cached data due to error`);
          setData(cached.data);
        }
      } finally {
        setLoading(false);
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
    }

    // Fetch inicial inmediato
    fetchCandles();

    // Configurar intervalo - velas de Yahoo Finance se actualizan cada 1 minuto (reducido para cambios más frecuentes)
    const updateInterval = setInterval(() => {
      fetchCandles();
    }, 60 * 1000); // 1 minuto (reducido de 5 minutos)

    return () => clearInterval(updateInterval);
  }, [symbol, timeInterval, count]);

  return { data, loading, error };
}; 