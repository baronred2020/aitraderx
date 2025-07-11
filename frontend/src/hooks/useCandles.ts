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

// Cache simple para evitar llamadas duplicadas
const candlesCache = new Map<string, { data: CandlesResponse; timestamp: number }>();

export const useCandles = (symbol: string, timeInterval: string = '15', count: number = 100) => {
  const [data, setData] = useState<CandlesResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!symbol) return;

    const cacheKey = `${symbol}-${timeInterval}-${count}`;
    const cached = candlesCache.get(cacheKey);
    const now = Date.now();

    // Usar cache si es válido (menos de 10 segundos para carga más rápida)
    if (cached && (now - cached.timestamp) < 10000) {
      setData(cached.data);
      return;
    }

    const fetchCandles = async () => {
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
        
        // Guardar en cache
        candlesCache.set(cacheKey, { data: result, timestamp: now });
        setData(result);
      } catch (err) {
        console.error('Error fetching candles:', err);
        setError(err instanceof Error ? err.message : 'Unknown error');
        setData(null);
      } finally {
        setLoading(false);
      }
    };

    fetchCandles();
  }, [symbol, timeInterval, count]);

  return { data, loading, error };
}; 