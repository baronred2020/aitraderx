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

// Cache global para evitar llamadas duplicadas
const globalCache = new Map<string, CacheEntry>();
const DAILY_CALL_LIMIT = 800;
const CALL_INTERVAL = 24 * 60 * 60 * 1000 / DAILY_CALL_LIMIT; // ~108 segundos entre llamadas

// Contador de llamadas diarias
let dailyCallCount = 0;
let lastCallTime = 0;

// Cargar estadísticas guardadas
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
  if (dailyCallCount < DAILY_CALL_LIMIT && timeSinceLastCall >= CALL_INTERVAL) {
    return true;
  }
  
  return false;
};

// Función para registrar una llamada
const registerCall = (): void => {
  dailyCallCount++;
  lastCallTime = Date.now();
  
  // Guardar estadísticas en localStorage
  const stats = {
    dailyCalls: dailyCallCount,
    lastCallTime: lastCallTime,
    timestamp: Date.now()
  };
  localStorage.setItem('api_stats', JSON.stringify(stats));
};

export const useMarketData = (symbols: string[]) => {
  const [data, setData] = useState<MarketData>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!symbols.length) return;

    const fetchData = async () => {
      // Verificar cache primero
      const cacheKey = symbols.sort().join(',');
      const cached = globalCache.get(cacheKey);
      const now = Date.now();
      
      // Cache válido por 2 minutos
      if (cached && (now - cached.timestamp) < 2 * 60 * 1000) {
        setData(cached.data);
        return;
      }

      // Verificar si podemos hacer una llamada
      if (!canMakeCall()) {
        console.log('Rate limit reached, using cached data');
        if (cached) {
          setData(cached.data);
        }
        return;
      }

      setLoading(true);
      setError(null);

      try {
        const response = await fetch(`http://localhost:8000/api/market-data?symbols=${symbols.join(',')}`);
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        
        // Registrar la llamada exitosa
        registerCall();
        
        // Actualizar cache
        globalCache.set(cacheKey, {
          data: result,
          timestamp: now,
          callCount: (cached?.callCount || 0) + 1
        });

        setData(result);
      } catch (err) {
        console.error('Error fetching market data:', err);
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
    fetchData();

    // Configurar intervalo inteligente
    const interval = setInterval(() => {
      fetchData();
    }, Math.max(CALL_INTERVAL, 2 * 60 * 1000)); // Mínimo 2 minutos

    return () => clearInterval(interval);
  }, [symbols]);

  return { data, loading, error };
}; 