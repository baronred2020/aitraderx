# 🚀 Setup Completo del Proyecto AI Trading App

## 📋 **Paso 1: Crear el Proyecto Base**

```bash
# Crear el proyecto React con TypeScript
npx create-react-app ai-trading-app --template typescript
cd aitraderx

# Instalar dependencias necesarias
npm install recharts lucide-react axios date-fns

# Instalar Tailwind CSS
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p

# Instalar tipos para TypeScript
npm install -D @types/node @types/react @types/react-dom
```

## ⚙️ **Paso 2: Configurar Tailwind CSS**

Edita el archivo `tailwind.config.js`:

```javascript
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
```

Reemplaza el contenido de `src/index.css`:

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

/* Estilos personalizados */
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
    sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

code {
  font-family: source-code-pro, Menlo, Monaco, Consolas, 'Courier New',
    monospace;
}

/* Animaciones personalizadas */
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

.animate-fade-in {
  animation: fadeIn 0.3s ease-out;
}
```

## 📁 **Paso 3: Crear la Estructura de Directorios**

```bash
# Crear directorios
mkdir -p src/components/Dashboard
mkdir -p src/components/Analysis
mkdir -p src/components/Portfolio
mkdir -p src/components/AIMonitor
mkdir -p src/components/Alerts
mkdir -p src/components/Common
mkdir -p src/hooks
mkdir -p src/services
mkdir -p src/utils
mkdir -p src/types
mkdir -p src/contexts
```

## 🏗️ **Paso 4: Crear Archivos Base**

### **package.json (añadir scripts adicionales)**

```json
{
  "name": "ai-trading-app",
  "version": "0.1.0",
  "private": true,
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1",
    "typescript": "^4.9.5",
    "recharts": "^2.8.0",
    "lucide-react": "^0.263.1",
    "axios": "^1.4.0",
    "date-fns": "^2.30.0"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject",
    "lint": "eslint src/**/*.{ts,tsx}",
    "format": "prettier --write src/**/*.{ts,tsx}"
  },
  "devDependencies": {
    "tailwindcss": "^3.3.0",
    "postcss": "^8.4.24",
    "autoprefixer": "^10.4.14",
    "@types/node": "^20.4.0",
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0"
  }
}
```

### **src/types/index.ts**

```typescript
export interface Asset {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  signal: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  targetPrice: number;
  timeframe: string;
  reasoning: string;
  volume: number;
  marketCap: string;
  pe: number;
  volatility: number;
}

export interface PriceData {
  time: string;
  price: number;
  volume: number;
  rsi: number;
  macd: number;
}

export interface PortfolioPosition {
  symbol: string;
  shares: number;
  avgPrice: number;
  currentPrice: number;
  value: number;
  pnl: number;
  pnlPercent: number;
  allocation: number;
}

export interface Alert {
  id: number;
  type: 'success' | 'warning' | 'error' | 'info';
  message: string;
  time: string;
  symbol: string;
  priority: 'high' | 'medium' | 'low';
}

export interface AutoTrainingStatus {
  isTraining: boolean;
  lastTraining: string | null;
  modelVersion: string;
  driftDetected: boolean;
  accuracy: number;
}

export interface ModelMetric {
  metric: string;
  current: number;
  previous: number;
  target: number;
}
```

### **src/services/api.ts**

```typescript
import axios from 'axios';
import { Asset, Alert, AutoTrainingStatus } from '../types';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Interceptor para manejo de errores
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error);
    return Promise.reject(error);
  }
);

export const apiService = {
  // Assets
  async getAssets(): Promise<Asset[]> {
    const response = await apiClient.get('/api/assets');
    return response.data;
  },

  // Technical Analysis
  async getTechnicalAnalysis(symbol: string) {
    const response = await apiClient.get(`/api/technical-analysis/${symbol}`);
    return response.data;
  },

  // Fundamental Analysis
  async getFundamentalAnalysis(symbol: string) {
    const response = await apiClient.get(`/api/fundamental-analysis/${symbol}`);
    return response.data;
  },

  // Price Prediction
  async getPricePrediction(symbol: string, timeframe: number = 5) {
    const response = await apiClient.post('/api/predict-price', {
      symbol,
      timeframe
    });
    return response.data;
  },

  // Price Data
  async getPriceData(symbol: string, period: string = '1d') {
    const response = await apiClient.get(`/api/price-data/${symbol}?period=${period}`);
    return response.data;
  },

  // Alerts
  async getAlerts(): Promise<Alert[]> {
    const response = await apiClient.get('/api/alerts');
    return response.data;
  },

  async createAlert(alert: Omit<Alert, 'id'>) {
    const response = await apiClient.post('/api/alerts', alert);
    return response.data;
  },

  async deleteAlert(alertId: number) {
    const response = await apiClient.delete(`/api/alerts/${alertId}`);
    return response.data;
  },

  // Auto-training
  async getModelStatus(): Promise<AutoTrainingStatus> {
    const response = await apiClient.get('/api/model/status');
    return response.data;
  },

  async forceRetrain() {
    const response = await apiClient.post('/api/model/force-retrain');
    return response.data;
  },

  async rollbackModel() {
    const response = await apiClient.post('/api/model/rollback');
    return response.data;
  },

  async addModelFeedback(data: {
    symbol: string;
    predicted_price: number;
    actual_price: number;
    user_feedback?: number;
  }) {
    const response = await apiClient.post('/api/model/add-feedback', data);
    return response.data;
  }
};
```

### **src/services/websocket.ts**

```typescript
class WebSocketService {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectInterval = 1000;
  private listeners: { [key: string]: Function[] } = {};

  connect(url: string = 'ws://localhost:8000/ws') {
    try {
      this.ws = new WebSocket(url);
      
      this.ws.onopen = () => {
        console.log('WebSocket connected');
        this.reconnectAttempts = 0;
        this.emit('connected', true);
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.emit('message', data);
          
          // Emit specific event types
          if (data.type) {
            this.emit(data.type, data.data);
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      this.ws.onclose = () => {
        console.log('WebSocket disconnected');
        this.emit('connected', false);
        this.handleReconnect();
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.emit('error', error);
      };

    } catch (error) {
      console.error('Error connecting to WebSocket:', error);
    }
  }

  private handleReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      setTimeout(() => {
        console.log(`Reconnecting... attempt ${this.reconnectAttempts}`);
        this.connect();
      }, this.reconnectInterval * this.reconnectAttempts);
    }
  }

  on(event: string, callback: Function) {
    if (!this.listeners[event]) {
      this.listeners[event] = [];
    }
    this.listeners[event].push(callback);
  }

  off(event: string, callback: Function) {
    if (this.listeners[event]) {
      this.listeners[event] = this.listeners[event].filter(cb => cb !== callback);
    }
  }

  private emit(event: string, data: any) {
    if (this.listeners[event]) {
      this.listeners[event].forEach(callback => callback(data));
    }
  }

  send(data: any) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}

export const websocketService = new WebSocketService();
```

### **src/hooks/useApi.ts**

```typescript
import { useState, useEffect } from 'react';
import { apiService } from '../services/api';

export function useApi<T>(
  apiCall: () => Promise<T>,
  dependencies: any[] = []
) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;

    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);
        const result = await apiCall();
        
        if (mounted) {
          setData(result);
        }
      } catch (err) {
        if (mounted) {
          setError(err instanceof Error ? err.message : 'Unknown error');
        }
      } finally {
        if (mounted) {
          setLoading(false);
        }
      }
    };

    fetchData();

    return () => {
      mounted = false;
    };
  }, dependencies);

  const refetch = async () => {
    try {
      setLoading(true);
      setError(null);
      const result = await apiCall();
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  return { data, loading, error, refetch };
}
```

### **src/hooks/useWebSocket.ts**

```typescript
import { useState, useEffect, useRef } from 'react';
import { websocketService } from '../services/websocket';

export function useWebSocket(url?: string) {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<any>(null);
  const [error, setError] = useState<any>(null);
  const callbacksRef = useRef<{ [key: string]: Function[] }>({});

  useEffect(() => {
    // Connect to WebSocket
    websocketService.connect(url);

    // Set up event listeners
    const handleConnected = (connected: boolean) => setIsConnected(connected);
    const handleMessage = (message: any) => setLastMessage(message);
    const handleError = (error: any) => setError(error);

    websocketService.on('connected', handleConnected);
    websocketService.on('message', handleMessage);
    websocketService.on('error', handleError);

    return () => {
      websocketService.off('connected', handleConnected);
      websocketService.off('message', handleMessage);
      websocketService.off('error', handleError);
    };
  }, [url]);

  const subscribe = (event: string, callback: Function) => {
    websocketService.on(event, callback);
    
    // Keep track of callbacks for cleanup
    if (!callbacksRef.current[event]) {
      callbacksRef.current[event] = [];
    }
    callbacksRef.current[event].push(callback);
  };

  const unsubscribe = (event: string, callback: Function) => {
    websocketService.off(event, callback);
    
    // Remove from tracked callbacks
    if (callbacksRef.current[event]) {
      callbacksRef.current[event] = callbacksRef.current[event].filter(cb => cb !== callback);
    }
  };

  const sendMessage = (data: any) => {
    websocketService.send(data);
  };

  // Cleanup all subscriptions on unmount
  useEffect(() => {
    return () => {
      Object.entries(callbacksRef.current).forEach(([event, callbacks]) => {
        callbacks.forEach(callback => {
          websocketService.off(event, callback);
        });
      });
    };
  }, []);

  return {
    isConnected,
    lastMessage,
    error,
    subscribe,
    unsubscribe,
    sendMessage
  };
}
```

### **src/utils/formatters.ts**

```typescript
// Formateo de números
export const formatCurrency = (value: number, decimals: number = 2): string => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value);
};

export const formatNumber = (value: number, decimals: number = 2): string => {
  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value);
};

export const formatPercentage = (value: number, decimals: number = 2): string => {
  return `${value > 0 ? '+' : ''}${value.toFixed(decimals)}%`;
};

export const formatLargeNumber = (value: number): string => {
  if (value >= 1e12) return `${(value / 1e12).toFixed(1)}T`;
  if (value >= 1e9) return `${(value / 1e9).toFixed(1)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
  return value.toString();
};

// Formateo de fechas
export const formatDate = (date: Date | string): string => {
  const d = typeof date === 'string' ? new Date(date) : date;
  return d.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  });
};

export const formatTime = (date: Date | string): string => {
  const d = typeof date === 'string' ? new Date(date) : date;
  return d.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
  });
};

export const formatDateTime = (date: Date | string): string => {
  const d = typeof date === 'string' ? new Date(date) : date;
  return d.toLocaleString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
};

// Utilidades de colores
export const getChangeColor = (value: number): string => {
  if (value > 0) return 'text-green-600';
  if (value < 0) return 'text-red-600';
  return 'text-gray-600';
};

export const getSignalColor = (signal: string): string => {
  switch (signal) {
    case 'BUY': return 'text-green-600 bg-green-50';
    case 'SELL': return 'text-red-600 bg-red-50';
    case 'HOLD': return 'text-yellow-600 bg-yellow-50';
    default: return 'text-gray-600 bg-gray-50';
  }
};

// Validaciones
export const isValidEmail = (email: string): boolean => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
};

export const isValidSymbol = (symbol: string): boolean => {
  const symbolRegex = /^[A-Z]{1,5}$/;
  return symbolRegex.test(symbol);
};

// Cálculos financieros
export const calculatePercentageChange = (current: number, previous: number): number => {
  return ((current - previous) / previous) * 100;
};

export const calculateSharpeRatio = (returns: number[], riskFreeRate: number = 0.02): number => {
  const avgReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length;
  const stdDev = Math.sqrt(
    returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length
  );
  return (avgReturn - riskFreeRate) / stdDev;
};

export const calculateMaxDrawdown = (values: number[]): number => {
  let maxDrawdown = 0;
  let peak = values[0];
  
  for (const value of values) {
    if (value > peak) {
      peak = value;
    }
    const drawdown = (peak - value) / peak;
    if (drawdown > maxDrawdown) {
      maxDrawdown = drawdown;
    }
  }
  
  return maxDrawdown;
};
```

### **src/contexts/AppContext.tsx**

```typescript
import React, { createContext, useContext, useReducer, ReactNode } from 'react';
import { Asset, Alert, AutoTrainingStatus } from '../types';

interface AppState {
  assets: Asset[];
  alerts: Alert[];
  autoTrainingStatus: AutoTrainingStatus;
  selectedAsset: string;
  isConnected: boolean;
  lastUpdate: Date;
}

type AppAction =
  | { type: 'SET_ASSETS'; payload: Asset[] }
  | { type: 'SET_ALERTS'; payload: Alert[] }
  | { type: 'ADD_ALERT'; payload: Alert }
  | { type: 'REMOVE_ALERT'; payload: number }
  | { type: 'SET_AUTO_TRAINING_STATUS'; payload: AutoTrainingStatus }
  | { type: 'SET_SELECTED_ASSET'; payload: string }
  | { type: 'SET_CONNECTION_STATUS'; payload: boolean }
  | { type: 'UPDATE_TIMESTAMP' };

const initialState: AppState = {
  assets: [],
  alerts: [],
  autoTrainingStatus: {
    isTraining: false,
    lastTraining: null,
    modelVersion: '1.0.0',
    driftDetected: false,
    accuracy: 0,
  },
  selectedAsset: 'AAPL',
  isConnected: false,
  lastUpdate: new Date(),
};

function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case 'SET_ASSETS':
      return { ...state, assets: action.payload };
    case 'SET_ALERTS':
      return { ...state, alerts: action.payload };
    case 'ADD_ALERT':
      return { ...state, alerts: [...state.alerts, action.payload] };
    case 'REMOVE_ALERT':
      return { 
        ...state, 
        alerts: state.alerts.filter(alert => alert.id !== action.payload) 
      };
    case 'SET_AUTO_TRAINING_STATUS':
      return { ...state, autoTrainingStatus: action.payload };
    case 'SET_SELECTED_ASSET':
      return { ...state, selectedAsset: action.payload };
    case 'SET_CONNECTION_STATUS':
      return { ...state, isConnected: action.payload };
    case 'UPDATE_TIMESTAMP':
      return { ...state, lastUpdate: new Date() };
    default:
      return state;
  }
}

const AppContext = createContext<{
  state: AppState;
  dispatch: React.Dispatch<AppAction>;
} | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(appReducer, initialState);

  return (
    <AppContext.Provider value={{ state, dispatch }}>
      {children}
    </AppContext.Provider>
  );
}

export function useAppContext() {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useAppContext must be used within AppProvider');
  }
  return context;
}
```

## 🎯 **Paso 5: Comando para Crear Todo Automáticamente**

Crea un script `setup.sh` para automatizar todo:

```bash
#!/bin/bash

echo "🚀 Configurando AI Trading App..."

# Crear proyecto
npx create-react-app ai-trading-app --template typescript
cd ai-trading-app

# Instalar dependencias
echo "📦 Instalando dependencias..."
npm install recharts lucide-react axios date-fns
npm install -D tailwindcss postcss autoprefixer

# Configurar Tailwind
npx tailwindcss init -p

# Crear estructura de directorios
echo "📁 Creando estructura de directorios..."
mkdir -p src/components/{Dashboard,Analysis,Portfolio,AIMonitor,Alerts,Common}
mkdir -p src/{hooks,services,utils,types,contexts}

echo "✅ ¡Proyecto configurado exitosamente!"
echo "📝 Próximos pasos:"
echo "1. cd ai-trading-app"
echo "2. Copiar los archivos de código"
echo "3. npm start"
```

## 🔥 **Paso 6: Iniciar el Proyecto**

```bash
# Hacer ejecutable el script
chmod +x setup.sh

# Ejecutar setup
./setup.sh

# O manualmente:
cd ai-trading-app
npm start
```

**¡Ahora tienes toda la estructura completa y lista para usar!** 🎉

Los próximos pasos serán:
1. ✅ Copiar el código de los componentes
2. ✅ Configurar las variables de entorno
3. ✅ Conectar con el backend
4. ✅ ¡Empezar a tradear! 📈