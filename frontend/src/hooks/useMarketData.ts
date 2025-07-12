import { useState, useEffect, useMemo } from 'react';

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

// Prefetch de símbolos comunes
const commonSymbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD'];
const prefetchCommonSymbols = () => {
  if (marketCache.size === 0) {
    // Prefetch solo si no hay datos en cache
    fetch(`http://localhost:8000/api/market-data?symbols=${commonSymbols.join(',')}`)
      .then(response => response.json())
      .then(data => {
        commonSymbols.forEach(symbol => {
          if (data[symbol]) {
            marketCache.set(symbol, { data: { [symbol]: data[symbol] }, timestamp: Date.now() });
          }
        });
      })
      .catch(err => console.log('Prefetch failed:', err));
  }
};

// Ejecutar prefetch al cargar el módulo
if (typeof window !== 'undefined') {
  setTimeout(prefetchCommonSymbols, 1000);
}

export const useMarketData = (symbols: string[]) => {
  const [data, setData] = useState<MarketData>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Memoizar el array de símbolos para evitar re-renders innecesarios
  const stableSymbols = useMemo(() => symbols.sort(), [symbols.join(',')]);

  useEffect(() => {
    if (!stableSymbols.length) return;

    const cacheKey = stableSymbols.join(',');
    const cached = marketCache.get(cacheKey);
    const now = Date.now();

    // Usar cache si es válido (menos de 5 segundos para carga más rápida)
    if (cached && (now - cached.timestamp) < 5000) {
      setData(cached.data);
      return;
    }

    const fetchData = async () => {
      setLoading(true);
      setError(null);

      try {
        const response = await fetch(`http://localhost:8000/api/market-data?symbols=${stableSymbols.join(',')}`);
        
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
  }, [stableSymbols]);

  return { data, loading, error };
}; 