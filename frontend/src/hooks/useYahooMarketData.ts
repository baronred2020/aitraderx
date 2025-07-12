import { useState, useEffect, useMemo, useCallback } from 'react';

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
const YAHOO_CACHE_TTL = 30 * 1000; // 30 segundos para Yahoo Finance

// Función para detectar si es fin de semana
const isWeekend = (): boolean => {
  const now = new Date();
  const dayOfWeek = now.getDay(); // 0 = Domingo, 6 = Sábado
  return dayOfWeek === 0 || dayOfWeek === 6;
};

// Función para detectar si el mercado está abierto
const isMarketOpen = (): boolean => {
  const now = new Date();
  const hour = now.getHours();
  const dayOfWeek = now.getDay();
  
  // Lunes a Viernes, 9:00 AM - 5:00 PM EST (simplificado)
  return dayOfWeek >= 1 && dayOfWeek <= 5 && hour >= 9 && hour < 17;
};

// Datos de fallback para cuando no hay datos disponibles
const getFallbackData = (symbols: string[]): MarketData => {
  const fallbackPrices: { [key: string]: number } = {
    'EURUSD': 1.0850,
    'GBPUSD': 1.2650,
    'USDJPY': 148.50,
    'AUDUSD': 0.6650,
    'USDCAD': 1.3550,
    'USDCHF': 0.8850,
    'NZDUSD': 0.6150,
    'EURGBP': 0.8580,
    'GBPJPY': 187.80,
    'EURJPY': 161.20,
    'AAPL': 175.50,
    'GOOGL': 140.20,
    'MSFT': 380.80,
    'TSLA': 240.50,
    'AMZN': 150.30,
    'BTCUSD': 42000,
    'ETHUSD': 2500,
  };

  const result: MarketData = {};
  
  symbols.forEach(symbol => {
    const basePrice = fallbackPrices[symbol] || 100.00;
    const change = (Math.random() - 0.5) * 0.01; // ±0.5% cambio
    const currentPrice = basePrice * (1 + change);
    const changePercent = (change * 100).toFixed(2);
    
    result[symbol] = {
      price: currentPrice.toFixed(4),
      change: (change * basePrice).toFixed(4),
      changePercent: changePercent,
      volume: (Math.random() * 1000000 + 500000).toFixed(0),
      high: (currentPrice * 1.002).toFixed(4),
      low: (currentPrice * 0.998).toFixed(4),
      open: (currentPrice * (1 + (Math.random() - 0.5) * 0.001)).toFixed(4),
      previousClose: basePrice.toFixed(4),
      marketStatus: 'closed'
    };
  });
  
  return result;
};

export const useYahooMarketData = (symbols: string[]) => {
  const [data, setData] = useState<MarketData>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [marketStatus, setMarketStatus] = useState<'open' | 'closed' | 'error'>('closed');
  const [isInitialLoad, setIsInitialLoad] = useState(true);

  // Memoizar el array de símbolos para evitar re-renders innecesarios
  const stableSymbols = useMemo(() => symbols.sort(), [symbols.join(',')]);

  // Memoizar la función de fetch para evitar recreaciones
  const fetchData = useCallback(async () => {
    console.log('[useYahooMarketData] Hook iniciado con símbolos:', stableSymbols);
    
    const marketOpen = isMarketOpen();
    const weekend = isWeekend();
    
    console.log('[useYahooMarketData] Estado del mercado:', { marketOpen, weekend });
    
    // Siempre intentar obtener datos reales primero, independientemente del estado del mercado
    setLoading(true);
    setError(null);

    try {
      console.log('[useYahooMarketData] Intentando obtener datos reales de Yahoo Finance...');
      const response = await fetch(`http://localhost:8000/api/market-data?symbols=${stableSymbols.join(',')}`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      console.log('[useYahooMarketData] Datos reales recibidos:', result);
      
      // Verificar que los datos son válidos (no están vacíos)
      const hasValidData = Object.keys(result).length > 0 && 
                          Object.values(result).some((symbolData: any) => 
                            symbolData && symbolData.price && symbolData.price !== '0' && symbolData.price !== '0.0000'
                          );
      
      if (hasValidData) {
        console.log('[useYahooMarketData] Usando datos reales de Yahoo Finance');
        
        // Cachear los datos reales
        const cacheKey = stableSymbols.join(',');
        yahooCache.set(cacheKey, { 
          data: result, 
          timestamp: Date.now(), 
          callCount: (yahooCache.get(cacheKey)?.callCount || 0) + 1 
        });
        
        setData(result);
        setMarketStatus(marketOpen ? 'open' : 'closed');
        setLoading(false);
        setIsInitialLoad(false);
        return;
      } else {
        console.log('[useYahooMarketData] Datos reales no válidos, usando fallback');
        throw new Error('No valid data received from Yahoo Finance');
      }
      
    } catch (err) {
      console.error('[useYahooMarketData] Error obteniendo datos reales:', err);
      
      // Solo usar fallback si es fin de semana o si no hay datos reales disponibles
      if (weekend) {
        console.log('[useYahooMarketData] Weekend detected - using fallback data');
        
        const cacheKey = stableSymbols.join(',');
        const cached = yahooCache.get(cacheKey);
        const now = Date.now();
        
        if (cached && (now - cached.timestamp) < YAHOO_CACHE_TTL) {
          console.log('[useYahooMarketData] Using cached fallback data');
          setData(cached.data);
          setMarketStatus('closed');
          setLoading(false);
          setIsInitialLoad(false);
          return;
        }
        
        console.log('[useYahooMarketData] No cached data, generating fallback');
        const fallbackData = getFallbackData(stableSymbols);
        console.log('[useYahooMarketData] Fallback data set:', fallbackData);
        
        // Cachear los datos de fallback
        yahooCache.set(cacheKey, { 
          data: fallbackData, 
          timestamp: now, 
          callCount: (cached?.callCount || 0) + 1 
        });
        
        setData(fallbackData);
        setMarketStatus('closed');
        setError('Mercado cerrado - datos de fin de semana');
      } else {
        // Si no es fin de semana, mostrar error pero intentar con datos de fallback
        console.log('[useYahooMarketData] Market should be open but no data available');
        const fallbackData = getFallbackData(stableSymbols);
        setData(fallbackData);
        setMarketStatus('error');
        setError('Error obteniendo datos de mercado');
      }
    } finally {
      setLoading(false);
      setIsInitialLoad(false);
    }
  }, [stableSymbols]);

  useEffect(() => {
    if (stableSymbols.length === 0) return;
    
    fetchData();
  }, [fetchData]);

  // Log del estado actual para debugging
  useEffect(() => {
    console.log('[useYahooMarketData] Current state:', { 
      data, 
      loading, 
      error, 
      marketStatus, 
      isInitialLoad 
    });
  }, [data, loading, error, marketStatus, isInitialLoad]);

  return { 
    data, 
    loading, 
    error, 
    marketStatus, 
    isInitialLoad,
    refetch: fetchData 
  };
}; 