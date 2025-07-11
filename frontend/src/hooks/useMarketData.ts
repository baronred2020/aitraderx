import { useState, useEffect } from 'react';

interface MarketData {
  [symbol: string]: {
    price: string;
    change: string;
    changePercent: string;
    volume: string;
    high: string;
    low: string;
    open: string;
    previousClose: string;
  };
}

// Cache simple para evitar llamadas duplicadas
const marketCache = new Map<string, { data: MarketData; timestamp: number }>();

export const useMarketData = (symbols: string[]) => {
  const [data, setData] = useState<MarketData>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!symbols.length) return;

    const cacheKey = symbols.sort().join(',');
    const cached = marketCache.get(cacheKey);
    const now = Date.now();

    // Usar cache si es válido (menos de 15 segundos)
    if (cached && (now - cached.timestamp) < 15000) {
      setData(cached.data);
      return;
    }

    const fetchData = async () => {
      setLoading(true);
      setError(null);

      try {
        const response = await fetch(`http://localhost:8000/api/market-data?symbols=${symbols.join(',')}`);
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        
        // Guardar en cache
        marketCache.set(cacheKey, { data: result, timestamp: now });
        setData(result);
      } catch (err) {
        console.error('Error fetching market data:', err);
        setError(err instanceof Error ? err.message : 'Unknown error');
        setData({});
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [symbols]);

  return { data, loading, error };
}; 