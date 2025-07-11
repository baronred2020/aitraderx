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

interface CacheEntry {
  data: MarketData;
  timestamp: number;
  callCount: number;
}

// Cache global para Yahoo Finance
const yahooCache = new Map<string, CacheEntry>();
const YAHOO_CACHE_TTL = 30 * 1000; // 30 segundos para Yahoo Finance (reducido de 2 minutos)

export const useYahooMarketData = (symbols: string[]) => {
  const [data, setData] = useState<MarketData>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!symbols.length) return;

    const fetchData = async () => {
      // Verificar cache primero
      const cacheKey = symbols.sort().join(',');
      const cached = yahooCache.get(cacheKey);
      const now = Date.now();
      
      // Cache válido por 30 segundos (reducido para actualizaciones más frecuentes)
      if (cached && (now - cached.timestamp) < YAHOO_CACHE_TTL) {
        console.log(`[Yahoo Market Data] Using cached data for ${symbols.join(',')}`);
        setData(cached.data);
        return;
      }

      console.log(`[Yahoo Market Data] Fetching fresh data for ${symbols.join(',')}`);
      setLoading(true);
      setError(null);

      try {
        const response = await fetch(`http://localhost:8000/api/market-data?symbols=${symbols.join(',')}`);
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        console.log(`[Yahoo Market Data] Received data:`, result);
        
        // Actualizar cache
        yahooCache.set(cacheKey, {
          data: result,
          timestamp: now,
          callCount: (cached?.callCount || 0) + 1
        });

        setData(result);
      } catch (err) {
        console.error('Error fetching Yahoo Finance market data:', err);
        setError(err instanceof Error ? err.message : 'Unknown error');
        
        // En caso de error, usar cache si está disponible
        if (cached) {
          console.log(`[Yahoo Market Data] Using cached data due to error`);
          setData(cached.data);
        }
      } finally {
        setLoading(false);
      }
    };

    // Limpiar cache anterior si los símbolos cambiaron
    const currentCacheKey = symbols.sort().join(',');
    const previousCacheKeys = Array.from(yahooCache.keys()).filter(key => 
      key !== currentCacheKey && symbols.some(s => key.includes(s))
    );
    
    if (previousCacheKeys.length > 0) {
      console.log(`[Yahoo Market Data] Clearing old cache for symbol change: ${previousCacheKeys.join(', ')}`);
      previousCacheKeys.forEach(key => yahooCache.delete(key));
    }

    // Fetch inicial inmediato
    fetchData();

    // Configurar intervalo - actualizar cada 30 segundos para cambios más frecuentes
    const interval = setInterval(() => {
      fetchData();
    }, 30 * 1000); // 30 segundos (reducido de 2 minutos)

    return () => clearInterval(interval);
  }, [symbols]);

  return { data, loading, error };
}; 