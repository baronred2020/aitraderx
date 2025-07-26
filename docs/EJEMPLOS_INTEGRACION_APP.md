# 💻 EJEMPLOS DE INTEGRACIÓN - MODELO AI EN LA APP

## 📋 ÍNDICE
1. [Servicio AI Trading](#servicio-ai-trading)
2. [API Endpoints](#api-endpoints)
3. [Base de Datos](#base-de-datos)
4. [Frontend Integration](#frontend-integration)
5. [WebSocket Real-time](#websocket-real-time)
6. [Alertas y Notificaciones](#alertas-y-notificaciones)

---

## 🤖 SERVICIO AI TRADING

### 1. Clase Principal del Servicio

```python
# src/services/ai_trading_service.py

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from models.Modelo_AI_Ultra import UniversalMultiStrategyAI
from utils.data_fetcher import get_real_market_data
from utils.signal_processor import SignalProcessor
from database.models import Signal, Performance, ModelConfig

logger = logging.getLogger(__name__)

class AITradingService:
    """
    Servicio principal para el modelo AI de trading
    """
    
    def __init__(self):
        self.models: Dict[str, UniversalMultiStrategyAI] = {}
        self.signal_processor = SignalProcessor()
        self.active_signals: Dict[str, List[Dict]] = {}
        self.performance_cache: Dict[str, Dict] = {}
        
    async def initialize_models(self, symbols: List[str]):
        """Inicializa modelos para todos los símbolos"""
        for symbol in symbols:
            try:
                logger.info(f"Inicializando modelo para {symbol}")
                self.models[symbol] = UniversalMultiStrategyAI(symbol=symbol)
                
                # Cargar modelos entrenados si existen
                if self.models[symbol].load_all_models():
                    logger.info(f"Modelos cargados para {symbol}")
                else:
                    logger.warning(f"Modelos no encontrados para {symbol}, entrenando...")
                    await self.train_models(symbol)
                    
            except Exception as e:
                logger.error(f"Error inicializando {symbol}: {e}")
    
    async def train_models(self, symbol: str, force_retrain: bool = False):
        """Entrena modelos para un símbolo específico"""
        try:
            logger.info(f"Entrenando modelos para {symbol}")
            
            # Obtener datos
            data_dict = await self._get_training_data(symbol)
            
            # Entrenar modelos
            results = self.models[symbol].train_all_strategies(data_dict)
            
            # Guardar modelos
            self.models[symbol].save_all_models()
            
            # Actualizar métricas de rendimiento
            await self._update_performance_metrics(symbol, results)
            
            logger.info(f"Entrenamiento completado para {symbol}")
            return results
            
        except Exception as e:
            logger.error(f"Error entrenando {symbol}: {e}")
            raise
    
    async def get_signals(self, symbol: str, strategies: List[str] = None) -> Dict:
        """Obtiene señales en tiempo real para un símbolo"""
        try:
            if symbol not in self.models:
                raise ValueError(f"Modelo no inicializado para {symbol}")
            
            # Obtener datos recientes
            recent_data = await self._get_recent_data(symbol)
            
            # Generar señales
            signals = {}
            if strategies is None:
                strategies = ['scalping', 'day_trading', 'swing_trading', 'position_trading']
            
            for strategy in strategies:
                if strategy in recent_data:
                    strategy_signals = self.models[symbol].generate_signals_strategy(
                        recent_data[strategy], strategy
                    )
                    signals[strategy] = strategy_signals
            
            # Procesar y filtrar señales
            processed_signals = self.signal_processor.process_signals(signals)
            
            # Guardar señales en base de datos
            await self._save_signals_to_db(symbol, processed_signals)
            
            return {
                'symbol': symbol,
                'timestamp': datetime.utcnow().isoformat(),
                'signals': processed_signals
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo señales para {symbol}: {e}")
            raise
    
    async def get_performance(self, symbol: str) -> Dict:
        """Obtiene métricas de rendimiento"""
        try:
            if symbol in self.performance_cache:
                return self.performance_cache[symbol]
            
            # Obtener métricas de la base de datos
            performance = await self._get_performance_from_db(symbol)
            
            # Cachear resultados
            self.performance_cache[symbol] = performance
            
            return performance
            
        except Exception as e:
            logger.error(f"Error obteniendo rendimiento para {symbol}: {e}")
            raise
    
    async def update_config(self, symbol: str, config: Dict):
        """Actualiza configuración de estrategias"""
        try:
            # Validar configuración
            self._validate_config(config)
            
            # Actualizar thresholds
            for strategy, settings in config.items():
                if 'confidence_threshold' in settings:
                    self.models[symbol].strategies[strategy]['confidence_threshold'] = settings['confidence_threshold']
            
            # Guardar configuración en base de datos
            await self._save_config_to_db(symbol, config)
            
            logger.info(f"Configuración actualizada para {symbol}")
            
        except Exception as e:
            logger.error(f"Error actualizando configuración para {symbol}: {e}")
            raise
    
    async def _get_training_data(self, symbol: str) -> Dict:
        """Obtiene datos para entrenamiento"""
        data_dict = {}
        
        # Configuraciones por estrategia
        strategies_config = {
            'scalping': {'timeframe': '5m', 'periods': '60d'},
            'day_trading': {'timeframe': '1h', 'periods': '6mo'},
            'swing_trading': {'timeframe': '1h', 'periods': '1y'},
            'position_trading': {'timeframe': '1d', 'periods': '2y'}
        }
        
        for strategy, config in strategies_config.items():
            try:
                data = get_real_market_data(
                    symbol, 
                    config['timeframe'], 
                    config['periods']
                )
                data_dict[strategy] = data
            except Exception as e:
                logger.error(f"Error obteniendo datos para {strategy}: {e}")
        
        return data_dict
    
    async def _get_recent_data(self, symbol: str) -> Dict:
        """Obtiene datos recientes para predicción"""
        data_dict = {}
        
        # Obtener datos recientes para cada estrategia
        recent_configs = {
            'scalping': {'timeframe': '5m', 'periods': 50},
            'day_trading': {'timeframe': '1h', 'periods': 100},
            'swing_trading': {'timeframe': '1h', 'periods': 250},
            'position_trading': {'timeframe': '1d', 'periods': 520}
        }
        
        for strategy, config in recent_configs.items():
            try:
                data = get_real_market_data(
                    symbol,
                    config['timeframe'],
                    config['periods']
                )
                data_dict[strategy] = data
            except Exception as e:
                logger.error(f"Error obteniendo datos recientes para {strategy}: {e}")
        
        return data_dict
    
    async def _save_signals_to_db(self, symbol: str, signals: Dict):
        """Guarda señales en base de datos"""
        try:
            for strategy, strategy_signals in signals.items():
                for signal_data in strategy_signals:
                    signal = Signal(
                        symbol=symbol,
                        strategy=strategy,
                        signal_type=signal_data['signal'],
                        confidence=signal_data['confidence'],
                        current_price=signal_data['current_price'],
                        target_price=signal_data.get('take_profit'),
                        stop_loss=signal_data.get('stop_loss'),
                        timestamp=datetime.utcnow()
                    )
                    await signal.save()
                    
        except Exception as e:
            logger.error(f"Error guardando señales en DB: {e}")
    
    async def _update_performance_metrics(self, symbol: str, results: Dict):
        """Actualiza métricas de rendimiento"""
        try:
            for strategy, metrics in results.items():
                performance = Performance(
                    symbol=symbol,
                    strategy=strategy,
                    accuracy=metrics.get('accuracy', 0),
                    error_pips=metrics.get('error_pips', 0),
                    signals_generated=metrics.get('signals_generated', 0),
                    timestamp=datetime.utcnow()
                )
                await performance.save()
                
        except Exception as e:
            logger.error(f"Error actualizando métricas: {e}")
    
    def _validate_config(self, config: Dict):
        """Valida configuración de estrategias"""
        valid_strategies = ['scalping', 'day_trading', 'swing_trading', 'position_trading']
        
        for strategy, settings in config.items():
            if strategy not in valid_strategies:
                raise ValueError(f"Estrategia inválida: {strategy}")
            
            if 'confidence_threshold' in settings:
                threshold = settings['confidence_threshold']
                if not (0 <= threshold <= 100):
                    raise ValueError(f"Threshold inválido para {strategy}: {threshold}")

```

### 2. Procesador de Señales

```python
# src/utils/signal_processor.py

import logging
from typing import Dict, List
from datetime import datetime

logger = logging.getLogger(__name__)

class SignalProcessor:
    """
    Procesa y filtra señales de trading
    """
    
    def __init__(self):
        self.signal_history: Dict[str, List] = {}
        self.confidence_thresholds = {
            'scalping': 60,
            'day_trading': 70,
            'swing_trading': 80,
            'position_trading': 85
        }
    
    def process_signals(self, raw_signals: Dict) -> Dict:
        """Procesa y filtra señales"""
        processed_signals = {}
        
        for strategy, signals in raw_signals.items():
            if not signals:
                continue
            
            # Filtrar señales por confianza
            filtered_signals = self._filter_by_confidence(signals, strategy)
            
            # Validar señales
            validated_signals = self._validate_signals(filtered_signals, strategy)
            
            # Formatear señales
            formatted_signals = self._format_signals(validated_signals)
            
            processed_signals[strategy] = formatted_signals
        
        return processed_signals
    
    def _filter_by_confidence(self, signals: List[Dict], strategy: str) -> List[Dict]:
        """Filtra señales por nivel de confianza"""
        threshold = self.confidence_thresholds.get(strategy, 70)
        
        filtered = []
        for signal in signals:
            if signal['confidence'] >= threshold:
                filtered.append(signal)
        
        return filtered
    
    def _validate_signals(self, signals: List[Dict], strategy: str) -> List[Dict]:
        """Valida señales según reglas de negocio"""
        validated = []
        
        for signal in signals:
            # Validar que la señal no sea muy reciente
            if self._is_signal_too_recent(signal, strategy):
                continue
            
            # Validar que no haya conflicto con señales anteriores
            if self._has_signal_conflict(signal, strategy):
                continue
            
            # Validar rangos de precio
            if self._is_price_in_range(signal):
                validated.append(signal)
        
        return validated
    
    def _format_signals(self, signals: List[Dict]) -> List[Dict]:
        """Formatea señales para la API"""
        formatted = []
        
        for signal in signals:
            formatted_signal = {
                'signal': signal['signal'],
                'confidence': round(signal['confidence'], 1),
                'current_price': signal['current_price'],
                'target_price': signal.get('take_profit'),
                'stop_loss': signal.get('stop_loss'),
                'risk_reward_ratio': signal.get('risk_reward_ratio', 0),
                'timestamp': signal.get('timestamp', datetime.utcnow().isoformat())
            }
            formatted.append(formatted_signal)
        
        return formatted
    
    def _is_signal_too_recent(self, signal: Dict, strategy: str) -> bool:
        """Verifica si la señal es muy reciente"""
        # Implementar lógica de validación temporal
        return False
    
    def _has_signal_conflict(self, signal: Dict, strategy: str) -> bool:
        """Verifica conflictos con señales anteriores"""
        # Implementar lógica de detección de conflictos
        return False
    
    def _is_price_in_range(self, signal: Dict) -> bool:
        """Verifica que el precio esté en rango válido"""
        # Implementar validación de rangos de precio
        return True
```

---

## 🔌 API ENDPOINTS

### 1. FastAPI Application

```python
# src/main.py

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocket, WebSocketDisconnect
import asyncio
import json
from typing import Dict, List

from services.ai_trading_service import AITradingService
from database.connection import get_database
from utils.auth import get_current_user
from utils.rate_limiter import RateLimiter

app = FastAPI(title="AI Trading API", version="1.0.0")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar servicios
ai_service = AITradingService()
rate_limiter = RateLimiter()

# WebSocket connections
active_connections: List[WebSocket] = []

@app.on_event("startup")
async def startup_event():
    """Inicializar servicios al arrancar"""
    # Inicializar modelos AI
    symbols = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X']
    await ai_service.initialize_models(symbols)
    
    # Inicializar base de datos
    await get_database().connect()

@app.on_event("shutdown")
async def shutdown_event():
    """Limpiar recursos al cerrar"""
    await get_database().disconnect()

# Endpoints principales

@app.get("/api/v1/signals/{symbol}")
async def get_signals(
    symbol: str,
    strategies: List[str] = None,
    current_user = Depends(get_current_user)
):
    """Obtener señales de trading"""
    try:
        # Rate limiting
        if not rate_limiter.check_limit(current_user.id):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        signals = await ai_service.get_signals(symbol, strategies)
        return signals
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/models/train")
async def train_models(
    request: Dict,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user)
):
    """Entrenar modelos para un símbolo"""
    try:
        symbol = request.get('symbol')
        force_retrain = request.get('force_retrain', False)
        
        if not symbol:
            raise HTTPException(status_code=400, detail="Symbol required")
        
        # Ejecutar entrenamiento en background
        background_tasks.add_task(ai_service.train_models, symbol, force_retrain)
        
        return {"message": f"Training started for {symbol}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/v1/strategies/{symbol}")
async def update_strategy_config(
    symbol: str,
    config: Dict,
    current_user = Depends(get_current_user)
):
    """Actualizar configuración de estrategias"""
    try:
        await ai_service.update_config(symbol, config)
        return {"message": "Configuration updated successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/performance/{symbol}")
async def get_performance(
    symbol: str,
    current_user = Depends(get_current_user)
):
    """Obtener métricas de rendimiento"""
    try:
        performance = await ai_service.get_performance(symbol)
        return performance
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/symbols")
async def get_available_symbols():
    """Obtener símbolos disponibles"""
    return {
        "symbols": [
            {"code": "EURUSD", "name": "Euro/Dólar"},
            {"code": "GBPUSD", "name": "Libra/Dólar"},
            {"code": "USDJPY", "name": "Dólar/Yen"},
            {"code": "AUDUSD", "name": "Dólar Australiano/Dólar"},
            {"code": "USDCAD", "name": "Dólar/Dólar Canadiense"}
        ]
    }

# WebSocket para señales en tiempo real

@app.websocket("/ws/signals/{symbol}")
async def websocket_signals(websocket: WebSocket, symbol: str):
    """WebSocket para señales en tiempo real"""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            # Enviar señales cada 30 segundos
            signals = await ai_service.get_signals(symbol)
            await websocket.send_text(json.dumps(signals))
            await asyncio.sleep(30)
            
    except WebSocketDisconnect:
        active_connections.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        active_connections.remove(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 2. Modelos de Base de Datos

```python
# src/database/models.py

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Signal(Base):
    """Modelo para señales de trading"""
    __tablename__ = "signals"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), index=True)
    strategy = Column(String(20), index=True)
    signal_type = Column(String(10))  # BUY, SELL, HOLD
    confidence = Column(Float)
    current_price = Column(Float)
    target_price = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    executed = Column(Boolean, default=False)
    user_id = Column(Integer, nullable=True)

class Performance(Base):
    """Modelo para métricas de rendimiento"""
    __tablename__ = "performance"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), index=True)
    strategy = Column(String(20), index=True)
    accuracy = Column(Float)
    error_pips = Column(Float)
    signals_generated = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)

class ModelConfig(Base):
    """Modelo para configuración de modelos"""
    __tablename__ = "model_config"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), index=True)
    strategy = Column(String(20), index=True)
    confidence_threshold = Column(Float)
    enabled = Column(Boolean, default=True)
    config_data = Column(Text)  # JSON string
    timestamp = Column(DateTime, default=datetime.utcnow)

class User(Base):
    """Modelo para usuarios"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(100), unique=True, index=True)
    hashed_password = Column(String(100))
    is_active = Column(Boolean, default=True)
    subscription_plan = Column(String(20), default="free")
    created_at = Column(DateTime, default=datetime.utcnow)
```

---

## 📱 FRONTEND INTEGRATION

### 1. React Component para Señales

```typescript
// src/components/TradingSignals.tsx

import React, { useState, useEffect } from 'react';
import { Card, Badge, Button, Alert } from 'antd';
import { TrendingUp, TrendingDown, MinusOutlined } from '@ant-design/icons';

interface Signal {
  signal: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  current_price: number;
  target_price?: number;
  stop_loss?: number;
  risk_reward_ratio: number;
  timestamp: string;
}

interface SignalsData {
  symbol: string;
  timestamp: string;
  signals: {
    scalping?: Signal[];
    day_trading?: Signal[];
    swing_trading?: Signal[];
    position_trading?: Signal[];
  };
}

const TradingSignals: React.FC<{ symbol: string }> = ({ symbol }) => {
  const [signals, setSignals] = useState<SignalsData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchSignals = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch(`/api/v1/signals/${symbol}`);
      if (!response.ok) {
        throw new Error('Failed to fetch signals');
      }
      
      const data = await response.json();
      setSignals(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSignals();
    const interval = setInterval(fetchSignals, 30000); // Actualizar cada 30s
    return () => clearInterval(interval);
  }, [symbol]);

  const getSignalIcon = (signal: string) => {
    switch (signal) {
      case 'BUY':
        return <TrendingUp style={{ color: '#52c41a' }} />;
      case 'SELL':
        return <TrendingDown style={{ color: '#ff4d4f' }} />;
      default:
        return <MinusOutlined style={{ color: '#8c8c8c' }} />;
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 80) return 'success';
    if (confidence >= 60) return 'warning';
    return 'error';
  };

  const renderStrategySignals = (strategyName: string, strategySignals: Signal[]) => {
    if (!strategySignals || strategySignals.length === 0) {
      return <p>No signals available</p>;
    }

    return strategySignals.map((signal, index) => (
      <Card key={index} size="small" style={{ marginBottom: 8 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            {getSignalIcon(signal.signal)}
            <span style={{ marginLeft: 8, fontWeight: 'bold' }}>
              {signal.signal}
            </span>
          </div>
          <Badge 
            status={getConfidenceColor(signal.confidence) as any}
            text={`${signal.confidence}%`}
          />
        </div>
        
        <div style={{ marginTop: 8 }}>
          <p>Price: ${signal.current_price.toFixed(5)}</p>
          {signal.target_price && (
            <p>Target: ${signal.target_price.toFixed(5)}</p>
          )}
          {signal.stop_loss && (
            <p>Stop Loss: ${signal.stop_loss.toFixed(5)}</p>
          )}
          <p>R:R Ratio: {signal.risk_reward_ratio.toFixed(1)}:1</p>
        </div>
      </Card>
    ));
  };

  if (loading) {
    return <div>Loading signals...</div>;
  }

  if (error) {
    return <Alert message="Error" description={error} type="error" />;
  }

  if (!signals) {
    return <div>No signals available</div>;
  }

  return (
    <div>
      <h2>Trading Signals - {symbol}</h2>
      <p>Last updated: {new Date(signals.timestamp).toLocaleString()}</p>
      
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 16 }}>
        <Card title="Scalping" size="small">
          {renderStrategySignals('scalping', signals.signals.scalping || [])}
        </Card>
        
        <Card title="Day Trading" size="small">
          {renderStrategySignals('day_trading', signals.signals.day_trading || [])}
        </Card>
        
        <Card title="Swing Trading" size="small">
          {renderStrategySignals('swing_trading', signals.signals.swing_trading || [])}
        </Card>
        
        <Card title="Position Trading" size="small">
          {renderStrategySignals('position_trading', signals.signals.position_trading || [])}
        </Card>
      </div>
      
      <Button onClick={fetchSignals} style={{ marginTop: 16 }}>
        Refresh Signals
      </Button>
    </div>
  );
};

export default TradingSignals;
```

### 2. WebSocket Hook para Tiempo Real

```typescript
// src/hooks/useWebSocket.ts

import { useState, useEffect, useRef } from 'react';

interface WebSocketMessage {
  symbol: string;
  timestamp: string;
  signals: any;
}

export const useWebSocket = (symbol: string) => {
  const [messages, setMessages] = useState<WebSocketMessage[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const connect = () => {
      const ws = new WebSocket(`ws://localhost:8000/ws/signals/${symbol}`);
      
      ws.onopen = () => {
        setIsConnected(true);
        setError(null);
        console.log('WebSocket connected');
      };
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setMessages(prev => [...prev.slice(-10), data]); // Mantener últimos 10
        } catch (err) {
          console.error('Error parsing WebSocket message:', err);
        }
      };
      
      ws.onclose = () => {
        setIsConnected(false);
        console.log('WebSocket disconnected');
      };
      
      ws.onerror = (event) => {
        setError('WebSocket error');
        console.error('WebSocket error:', event);
      };
      
      wsRef.current = ws;
    };

    connect();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [symbol]);

  const sendMessage = (message: any) => {
    if (wsRef.current && isConnected) {
      wsRef.current.send(JSON.stringify(message));
    }
  };

  return {
    messages,
    isConnected,
    error,
    sendMessage
  };
};
```

---

## 🔔 ALERTAS Y NOTIFICACIONES

### 1. Servicio de Notificaciones

```python
# src/services/notification_service.py

import asyncio
import logging
from typing import Dict, List
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)

class NotificationService:
    """
    Servicio para enviar alertas y notificaciones
    """
    
    def __init__(self):
        self.email_config = {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'username': 'your-email@gmail.com',
            'password': 'your-app-password'
        }
        
        self.push_config = {
            'api_key': 'your-push-api-key',
            'api_secret': 'your-push-api-secret'
        }
    
    async def send_signal_alert(self, user_id: int, signal: Dict):
        """Envía alerta por señal de trading"""
        try:
            # Obtener configuración del usuario
            user_config = await self._get_user_notification_config(user_id)
            
            if user_config.get('email_enabled'):
                await self._send_email_alert(user_id, signal)
            
            if user_config.get('push_enabled'):
                await self._send_push_notification(user_id, signal)
            
            if user_config.get('sms_enabled'):
                await self._send_sms_alert(user_id, signal)
                
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
    
    async def _send_email_alert(self, user_id: int, signal: Dict):
        """Envía alerta por email"""
        try:
            user_email = await self._get_user_email(user_id)
            
            msg = MIMEMultipart()
            msg['From'] = self.email_config['username']
            msg['To'] = user_email
            msg['Subject'] = f"Trading Signal: {signal['signal']} {signal['symbol']}"
            
            body = self._create_email_body(signal)
            msg.attach(MIMEText(body, 'html'))
            
            with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
                server.starttls()
                server.login(self.email_config['username'], self.email_config['password'])
                server.send_message(msg)
                
            logger.info(f"Email alert sent to {user_email}")
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
    
    async def _send_push_notification(self, user_id: int, signal: Dict):
        """Envía notificación push"""
        try:
            # Implementar lógica de push notifications
            # (Firebase, OneSignal, etc.)
            pass
            
        except Exception as e:
            logger.error(f"Error sending push notification: {e}")
    
    async def _send_sms_alert(self, user_id: int, signal: Dict):
        """Envía alerta por SMS"""
        try:
            # Implementar lógica de SMS
            # (Twilio, AWS SNS, etc.)
            pass
            
        except Exception as e:
            logger.error(f"Error sending SMS: {e}")
    
    def _create_email_body(self, signal: Dict) -> str:
        """Crea el cuerpo del email"""
        return f"""
        <html>
        <body>
            <h2>Trading Signal Alert</h2>
            <p><strong>Symbol:</strong> {signal['symbol']}</p>
            <p><strong>Signal:</strong> {signal['signal']}</p>
            <p><strong>Confidence:</strong> {signal['confidence']}%</p>
            <p><strong>Current Price:</strong> ${signal['current_price']}</p>
            <p><strong>Target Price:</strong> ${signal.get('target_price', 'N/A')}</p>
            <p><strong>Stop Loss:</strong> ${signal.get('stop_loss', 'N/A')}</p>
            <p><strong>Time:</strong> {signal['timestamp']}</p>
        </body>
        </html>
        """
    
    async def _get_user_notification_config(self, user_id: int) -> Dict:
        """Obtiene configuración de notificaciones del usuario"""
        # Implementar consulta a base de datos
        return {
            'email_enabled': True,
            'push_enabled': True,
            'sms_enabled': False
        }
    
    async def _get_user_email(self, user_id: int) -> str:
        """Obtiene email del usuario"""
        # Implementar consulta a base de datos
        return "user@example.com"
```

### 2. Configuración de Alertas

```typescript
// src/components/AlertSettings.tsx

import React, { useState, useEffect } from 'react';
import { Form, Switch, InputNumber, Button, Card, message } from 'antd';

interface AlertSettings {
  email_enabled: boolean;
  push_enabled: boolean;
  sms_enabled: boolean;
  confidence_threshold: number;
  min_risk_reward: number;
  strategies: string[];
}

const AlertSettings: React.FC = () => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    try {
      const response = await fetch('/api/v1/user/alert-settings');
      const settings = await response.json();
      form.setFieldsValue(settings);
    } catch (error) {
      message.error('Failed to load settings');
    }
  };

  const onFinish = async (values: AlertSettings) => {
    try {
      setLoading(true);
      
      const response = await fetch('/api/v1/user/alert-settings', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(values),
      });
      
      if (response.ok) {
        message.success('Settings saved successfully');
      } else {
        message.error('Failed to save settings');
      }
    } catch (error) {
      message.error('Error saving settings');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card title="Alert Settings" style={{ maxWidth: 600 }}>
      <Form
        form={form}
        layout="vertical"
        onFinish={onFinish}
      >
        <Form.Item label="Email Notifications" name="email_enabled" valuePropName="checked">
          <Switch />
        </Form.Item>
        
        <Form.Item label="Push Notifications" name="push_enabled" valuePropName="checked">
          <Switch />
        </Form.Item>
        
        <Form.Item label="SMS Notifications" name="sms_enabled" valuePropName="checked">
          <Switch />
        </Form.Item>
        
        <Form.Item 
          label="Minimum Confidence (%)" 
          name="confidence_threshold"
          rules={[{ required: true, message: 'Please enter confidence threshold' }]}
        >
          <InputNumber min={0} max={100} />
        </Form.Item>
        
        <Form.Item 
          label="Minimum Risk/Reward Ratio" 
          name="min_risk_reward"
          rules={[{ required: true, message: 'Please enter risk/reward ratio' }]}
        >
          <InputNumber min={0} step={0.1} />
        </Form.Item>
        
        <Form.Item>
          <Button type="primary" htmlType="submit" loading={loading}>
            Save Settings
          </Button>
        </Form.Item>
      </Form>
    </Card>
  );
};

export default AlertSettings;
```

---

## 📊 DASHBOARD DE MONITOREO

### 1. Componente de Dashboard

```typescript
// src/components/TradingDashboard.tsx

import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Statistic, Progress, Table, Button } from 'antd';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface PerformanceData {
  symbol: string;
  strategy: string;
  accuracy: number;
  error_pips: number;
  signals_generated: number;
  timestamp: string;
}

const TradingDashboard: React.FC = () => {
  const [performanceData, setPerformanceData] = useState<PerformanceData[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadPerformanceData();
  }, []);

  const loadPerformanceData = async () => {
    try {
      setLoading(true);
      const symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD'];
      
      const allData = await Promise.all(
        symbols.map(async (symbol) => {
          const response = await fetch(`/api/v1/performance/${symbol}`);
          return response.json();
        })
      );
      
      setPerformanceData(allData.flat());
    } catch (error) {
      console.error('Error loading performance data:', error);
    } finally {
      setLoading(false);
    }
  };

  const columns = [
    {
      title: 'Symbol',
      dataIndex: 'symbol',
      key: 'symbol',
    },
    {
      title: 'Strategy',
      dataIndex: 'strategy',
      key: 'strategy',
    },
    {
      title: 'Accuracy',
      dataIndex: 'accuracy',
      key: 'accuracy',
      render: (accuracy: number) => (
        <Progress percent={accuracy} size="small" />
      ),
    },
    {
      title: 'Error (Pips)',
      dataIndex: 'error_pips',
      key: 'error_pips',
    },
    {
      title: 'Signals',
      dataIndex: 'signals_generated',
      key: 'signals_generated',
    },
  ];

  return (
    <div>
      <h1>Trading AI Dashboard</h1>
      
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="Total Signals"
              value={performanceData.reduce((sum, item) => sum + item.signals_generated, 0)}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Average Accuracy"
              value={performanceData.reduce((sum, item) => sum + item.accuracy, 0) / performanceData.length}
              suffix="%"
              precision={1}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Active Symbols"
              value={new Set(performanceData.map(item => item.symbol)).size}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Average Error"
              value={performanceData.reduce((sum, item) => sum + item.error_pips, 0) / performanceData.length}
              suffix="pips"
              precision={1}
            />
          </Card>
        </Col>
      </Row>
      
      <Row gutter={16}>
        <Col span={12}>
          <Card title="Performance by Strategy">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="strategy" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="accuracy" stroke="#8884d8" />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </Col>
        
        <Col span={12}>
          <Card title="Performance by Symbol">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="symbol" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="accuracy" stroke="#82ca9d" />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </Col>
      </Row>
      
      <Card title="Detailed Performance" style={{ marginTop: 24 }}>
        <Table
          columns={columns}
          dataSource={performanceData}
          loading={loading}
          rowKey={(record) => `${record.symbol}-${record.strategy}`}
        />
      </Card>
      
      <Button onClick={loadPerformanceData} style={{ marginTop: 16 }}>
        Refresh Data
      </Button>
    </div>
  );
};

export default TradingDashboard;
```

---

## 🚀 DEPLOYMENT Y CONFIGURACIÓN

### 1. Docker Compose

```yaml
# docker-compose.yml

version: '3.8'

services:
  ai-trading-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=mysql://user:password@db:3306/trading_db
      - REDIS_URL=redis://redis:6379
      - AI_MODEL_PATH=/app/models/trained_models/Brain_Ultra
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    depends_on:
      - db
      - redis

  db:
    image: mysql:8.0
    environment:
      MYSQL_ROOT_PASSWORD: rootpassword
      MYSQL_DATABASE: trading_db
      MYSQL_USER: user
      MYSQL_PASSWORD: password
    ports:
      - "3306:3306"
    volumes:
      - mysql_data:/var/lib/mysql

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - ai-trading-api

volumes:
  mysql_data:
  redis_data:
```

### 2. Nginx Configuration

```nginx
# nginx.conf

events {
    worker_connections 1024;
}

http {
    upstream ai_trading_api {
        server ai-trading-api:8000;
    }

    server {
        listen 80;
        server_name localhost;

        location / {
            proxy_pass http://ai_trading_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        location /ws/ {
            proxy_pass http://ai_trading_api;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
        }
    }
}
```

---

*Documento de ejemplos de integración completado - Julio 2025* 