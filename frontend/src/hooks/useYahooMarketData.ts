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
    marketStatus?: 'open' | 'closed' | 'error';
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

export const useYahooMarketData = (symbols: string[]) => {
  const [data, setData] = useState<MarketData>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isInitialLoad, setIsInitialLoad] = useState(true);
  const [marketStatus, setMarketStatus] = useState<'open' | 'closed'>('open');

  useEffect(() => {
    if (!symbols.length) return;

    // Verificar si el mercado está abierto
    const marketOpen = isMarketOpen();
    const weekend = isWeekend();
    
    setMarketStatus(marketOpen ? 'open' : 'closed');
    
    // Si es fin de semana, usar cache existente o datos de fallback
    if (weekend) {
      console.log('[Yahoo Market Data] Weekend detected - using cached data or fallback');
      
      const cacheKey = symbols.sort().join(',');
      const cached = yahooCache.get(cacheKey);
      
      if (cached) {
        console.log('[Yahoo Market Data] Using cached weekend data');
        setData(cached.data);
        setIsInitialLoad(false);
        return;
      } else {
        // Datos de fallback para fines de semana
        const fallbackData: MarketData = {};
        symbols.forEach(symbol => {
          const fallbackPrices: { [key: string]: string } = {
            'EURUSD': '1.0850',
            'GBPUSD': '1.2650',
            'USDJPY': '148.50',
            'AUDUSD': '0.6550',
            'USDCAD': '1.3550',
            'AAPL': '150.00',
            'MSFT': '300.00',
            'TSLA': '200.00',
            'BTCUSD': '45000.00',
            'ETHUSD': '2500.00'
          };
          
          fallbackData[symbol] = {
            price: fallbackPrices[symbol] || '100.00',
            change: '0.000',
            changePercent: '0.00',
            volume: '0',
            high: fallbackPrices[symbol] || '100.00',
            low: fallbackPrices[symbol] || '100.00',
            open: fallbackPrices[symbol] || '100.00',
            previousClose: fallbackPrices[symbol] || '100.00',
            marketStatus: 'closed'
          };
        });
        
        setData(fallbackData);
        setIsInitialLoad(false);
        return;
      }
    }

    const fetchData = async (isBackgroundUpdate = false) => {
      // Verificar cache primero
      const cacheKey = symbols.sort().join(',');
      const cached = yahooCache.get(cacheKey);
      const now = Date.now();
      
      // Cache válido por 30 segundos (reducido para actualizaciones más frecuentes)
      if (cached && (now - cached.timestamp) < YAHOO_CACHE_TTL) {
        console.log(`[Yahoo Market Data] Using cached data for ${symbols.join(',')}`);
        setData(cached.data);
        if (isInitialLoad) {
          setIsInitialLoad(false);
        }
        return;
      }

      console.log(`[Yahoo Market Data] Fetching ${isBackgroundUpdate ? 'background update' : 'fresh data'} for ${symbols.join(',')}`);
      
      // Solo mostrar loading en carga inicial, no en actualizaciones background
      if (!isBackgroundUpdate && isInitialLoad) {
        setLoading(true);
      }
      setError(null);

      try {
        const response = await fetch(`http://localhost:8000/api/market-data?symbols=${symbols.join(',')}`);
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        console.log(`[Yahoo Market Data] Received ${isBackgroundUpdate ? 'background update' : 'data'}:`, result);
        
        // Actualizar cache
        yahooCache.set(cacheKey, {
          data: result,
          timestamp: now,
          callCount: (cached?.callCount || 0) + 1
        });

        setData(result);
        if (isInitialLoad) {
          setIsInitialLoad(false);
        }
      } catch (err) {
        console.error('Error fetching Yahoo Finance market data:', err);
        setError(err instanceof Error ? err.message : 'Unknown error');
        
        // En caso de error, usar cache si está disponible
        if (cached) {
          console.log(`[Yahoo Market Data] Using cached data due to error`);
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

    // Limpiar cache anterior si los símbolos cambiaron
    const currentCacheKey = symbols.sort().join(',');
    const previousCacheKeys = Array.from(yahooCache.keys()).filter(key => 
      key !== currentCacheKey && symbols.some(s => key.includes(s))
    );
    
    if (previousCacheKeys.length > 0) {
      console.log(`[Yahoo Market Data] Clearing old cache for symbol change: ${previousCacheKeys.join(', ')}`);
      previousCacheKeys.forEach(key => yahooCache.delete(key));
      setIsInitialLoad(true); // Reset cuando cambian los símbolos
    }

    // Fetch inicial inmediato
    fetchData(false);

    // Configurar intervalo - solo actualizar si el mercado está abierto
    const interval = setInterval(() => {
      if (isMarketOpen() && !isWeekend()) {
        fetchData(true); // Marcar como actualización background
      } else {
        console.log('[Yahoo Market Data] Market closed - skipping background update');
      }
    }, 30 * 1000); // 30 segundos

    return () => clearInterval(interval);
  }, [symbols, isInitialLoad]);

  return { data, loading, error, marketStatus };
}; 