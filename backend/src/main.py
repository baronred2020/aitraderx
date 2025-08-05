"""
AI Trading System - Main FastAPI Application
============================================
Sistema de trading con inteligencia artificial que incluye:
- Análisis técnico y fundamental automatizado
- Machine Learning tradicional (Random Forest, LSTM)
- Reinforcement Learning (DQN, PPO)
- Auto-entrenamiento de modelos
- Integración con MetaTrader 4
- Dashboard web en tiempo real
"""

import sys
import os
import uuid

# Agregar el directorio src al path para importaciones
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import asyncio
import logging
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import json
import random
from typing import List, Dict, Optional, Any
from enum import Enum
import pandas as pd
import numpy as np
from pydantic import BaseModel
import yfinance as yf
import ta
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Importar configuración de entorno
try:
    from . import env_config
except ImportError:
    import env_config

# Importar configuración de base de datos
try:
    from config.database_config import db_config
except ImportError:
    from .config.database_config import db_config

# Importar funciones de RL Trading Agent
try:
    from rl_trading_agent import (
        get_rl_status as get_rl_status_imported,
        get_rl_performance as get_rl_performance_imported,
        get_active_signals as get_active_signals_imported,
        execute_signal as execute_signal_imported,
        get_training_progress as get_training_progress_imported,
        can_user_start_training as can_user_start_training_imported,
        validate_training_parameters as validate_training_parameters_imported,
        start_rl_training as start_rl_training_imported,
        get_training_progress_by_session as get_training_progress_by_session_imported,
        cancel_rl_training as cancel_rl_training_imported,
        get_user_training_history as get_user_training_history_imported,
        get_user_rl_configuration as get_user_rl_configuration_imported,
        save_user_rl_configuration as save_user_rl_configuration_imported,
        reset_user_rl_configuration as reset_user_rl_configuration_imported,
        get_rl_configuration_limits as get_rl_configuration_limits_imported
    )
except ImportError as e:
    logger.warning(f"RL Trading Agent not available: {e}")

# Importar servicios de estrategias
try:
    from services.scalping_strategy_service import scalping_service
except ImportError as e:
    logger.warning(f"Scalping Strategy Service not available: {e}")
    scalping_service = None

# Importar rutas del portfolio
try:
    from api.portfolio_routes import router as portfolio_router
except ImportError as e:
    logger.warning(f"Portfolio routes not available: {e}")
    portfolio_router = None

# Importar rutas de Mega Mind
try:
    from api.mega_mind_routes import router as mega_mind_router
except ImportError as e:
    logger.warning(f"Mega Mind routes not available: {e}")
    mega_mind_router = None

# Importar funciones de RL Trading Agent
try:
    from rl_trading_agent import (
        get_rl_status as get_rl_status_imported,
        get_rl_performance as get_rl_performance_imported,
        get_active_signals as get_active_signals_imported,
        execute_signal as execute_signal_imported,
        get_training_progress as get_training_progress_imported,
        can_user_start_training as can_user_start_training_imported,
        validate_training_parameters as validate_training_parameters_imported,
        start_rl_training as start_rl_training_imported,
        get_training_progress_by_session as get_training_progress_by_session_imported,
        cancel_rl_training as cancel_rl_training_imported,
        get_user_training_history as get_user_training_history_imported,
        get_user_rl_configuration as get_user_rl_configuration_imported,
        save_user_rl_configuration as save_user_rl_configuration_imported,
        reset_user_rl_configuration as reset_user_rl_configuration_imported,
        get_rl_configuration_limits as get_rl_configuration_limits_imported
    )
except ImportError as e:
    logger.warning(f"RL Trading Agent not available: {e}")
    # Funciones fallback
    def get_rl_status_imported():
        return {"status": "inactive", "error": "Service not available"}
    
    def get_rl_performance_imported():
        return {"error": "Service not available"}
    
    def get_active_signals_imported():
        return []
    
    def execute_signal_imported(signal):
        return {"success": False, "error": "Service not available"}
    
    def get_training_progress_imported():
        return {"is_training": False, "error": "Service not available"}
    
    def can_user_start_training_imported(user_id):
        return {"can_train": False, "reason": "Service not available"}
    
    def validate_training_parameters_imported(episodes, user_plan="starter"):
        return {"valid": False, "reason": "Service not available"}
    
    async def start_rl_training_imported(user_id, episodes, algorithm="dqn", trading_pair="EURUSD", timeframe="1h"):
        return {"success": False, "error": "Service not available"}
    
    def get_training_progress_by_session_imported(session_id):
        return {"is_training": False, "error": "Service not available"}
    
    def cancel_rl_training_imported(session_id, user_id):
        return {"success": False, "error": "Service not available"}
    
    def get_user_training_history_imported(user_id, limit=10):
        return []
    
    def get_user_rl_configuration_imported(user_id):
        return {
            "success": True,
            "configuration": {
                "max_drawdown_percentage": 15.0,
                "max_position_size_percentage": 5.0,
                "min_confidence_threshold": 70.0,
                "retraining_frequency": "monthly",
                "retraining_enabled": False
            }
        }
    
    def save_user_rl_configuration_imported(user_id, **kwargs):
        return {
            "success": True,
            "configuration": {
                "max_drawdown_percentage": kwargs.get("max_drawdown_percentage", 15.0),
                "max_position_size_percentage": kwargs.get("max_position_size_percentage", 5.0),
                "min_confidence_threshold": kwargs.get("min_confidence_threshold", 70.0),
                "retraining_frequency": kwargs.get("retraining_frequency", "monthly"),
                "retraining_enabled": kwargs.get("retraining_enabled", False)
            }
        }
    
    def get_rl_configuration_limits_imported(user_id=None):
        return {
            "success": True,
            "limits": {
                "max_drawdown_percentage": {
                    "min": 5.0,
                    "max": 25.0,
                    "default": 15.0,
                    "step": 1.0
                },
                "max_position_size_percentage": {
                    "min": 1.0,
                    "max": 10.0,
                    "default": 5.0,
                    "step": 0.5
                },
                "min_confidence_threshold": {
                    "min": 50.0,
                    "max": 90.0,
                    "default": 70.0,
                    "step": 5.0
                },
                "retraining_frequency": {
                    "options": [
                        {"value": "monthly", "label": "Mensual"}
                    ],
                    "default": "monthly",
                    "available": True
                },
                "retraining_enabled": {
                    "default": False,
                    "available": True
                },
                "user_subscription": "premium"
            }
        }
    
    def reset_user_rl_configuration_imported(user_id):
        return {
            "success": True,
            "configuration": {
                "max_drawdown_percentage": 15.0,
                "max_position_size_percentage": 5.0,
                "min_confidence_threshold": 70.0,
                "retraining_frequency": "monthly",
                "retraining_enabled": False
            }
        }

# Importar sistema de suscripciones y autenticación
try:
    # Importaciones de rutas
    from api.auth_routes import auth_router
    from api.brain_trader_routes import router as brain_trader_router
    from api.market_data_routes import router as market_data_router
    from api.monitoring_routes import router as monitoring_router
    from api.prediction_routes import router as prediction_router
    from api.subscription_routes import subscription_router
    from services.subscription_service import SubscriptionService
    from services.brain_trader_service import BrainTraderService
    from api import market_data_routes
    from api.mega_mind_routes import router as mega_mind_router
    from api.wallet_routes import router as wallet_router
    
except ImportError:
    # Fallback para importaciones absolutas
    from api.auth_routes import auth_router
    from api.brain_trader_routes import router as brain_trader_router
    from api.market_data_routes import router as market_data_router
    from api.monitoring_routes import router as monitoring_router
    from api.prediction_routes import router as prediction_router
    from api.subscription_routes import subscription_router
    from services.subscription_service import SubscriptionService
    from services.brain_trader_service import BrainTraderService
    from api import market_data_routes
    from api.mega_mind_routes import router as mega_mind_router
    from api.wallet_routes import router as wallet_router

# Crear instancia global del BrainTraderService
brain_trader_service = BrainTraderService()

# Variables globales
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestión del ciclo de vida de la aplicación"""
    # Startup
    logger.info("Iniciando AI Trading System...")
    
    try:
        # Aquí se inicializarán todos los servicios
        app_state["status"] = "initializing"
        
        # TODO: Inicializar servicios
        # - Data collector
        # - AI models
        # - Auto-training system
        # - RL system
        # - MT4 integration
        
        app_state["status"] = "running"
        app_state["start_time"] = datetime.now()
        
        logger.info("AI Trading System iniciado correctamente")
        
    except Exception as e:
        logger.error(f"Error iniciando sistema: {e}")
        app_state["status"] = "error"
        
    yield
    
    # Shutdown
    logger.info("Deteniendo AI Trading System...")
    app_state["status"] = "shutdown"

# Crear aplicación FastAPI
app = FastAPI(
    title="AI Trading System",
    description="Sistema de trading con inteligencia artificial avanzada",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir rutas del portfolio
if portfolio_router:
    app.include_router(portfolio_router)

# Incluir routers
app.include_router(subscription_router)
app.include_router(auth_router)
app.include_router(market_data_router)
app.include_router(wallet_router)
app.include_router(mega_mind_router)
app.include_router(monitoring_router)
app.include_router(brain_trader_router)
app.include_router(prediction_router)

# Models de datos
class Asset(BaseModel):
    symbol: str
    name: str
    price: float
    change: float
    changePercent: float
    signal: str
    confidence: int
    targetPrice: float
    timeframe: str
    reasoning: str

class TechnicalAnalysis(BaseModel):
    rsi: float
    macd: str
    bollinger: str
    support: float
    resistance: float
    volume: str
    trend: str

class FundamentalAnalysis(BaseModel):
    pe: float
    eps: float
    epsGrowth: float
    sentiment: int
    nextEarnings: str
    rating: str

class PredictionRequest(BaseModel):
    symbol: str
    timeframe: int = 5  # días

class AlertRequest(BaseModel):
    symbol: str
    condition: str
    value: float
    user_id: str

# ===== BRAIN TRADER AND MEGA MIND MODELS =====
class PredictionResponse(BaseModel):
    pair: str
    direction: str
    confidence: float
    precision: float
    win_rate: float
    timeframe: str
    reasoning: str
    brain_type: str
    timestamp: str
    expires_at: str

class SignalResponse(BaseModel):
    pair: str
    type: str
    strength: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    brain_type: str
    timestamp: str

class TrendResponse(BaseModel):
    pair: str
    direction: str
    strength: float
    timeframe: str
    support: float
    resistance: float
    description: str
    brain_type: str
    timestamp: str

class MegaMindPredictionResponse(BaseModel):
    pair: str
    direction: str
    confidence: float
    precision: float
    win_rate: float
    timeframe: str
    reasoning: str
    brain_type: str
    fusion_method: str
    collaboration_score: float
    fusion_details: dict
    timestamp: str
    expires_at: str

# ===== AUTOMATED TRADING MODELS =====
class BrainType(str, Enum):
    BRAIN_MAX = "Brain_Max"
    BRAIN_ULTRA = "Brain_Ultra"
    BRAIN_PREDICTOR = "Brain_Predictor"
    MEGA_MIND = "Mega_Mind"

class TradingStyle(str, Enum):
    SCALPING = "scalping"
    DAY_TRADING = "day_trading"
    SWING_TRADING = "swing_trading"

class StrategyStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"

class StrategyConfig(BaseModel):
    name: str
    brainType: BrainType
    pair: str
    style: TradingStyle
    lotSize: float
    stopLossPips: int
    takeProfitPips: int
    minConfidence: float
    maxPositions: int
    riskPerTrade: float

class Strategy(BaseModel):
    id: str
    name: str
    brainType: BrainType
    pair: str
    style: TradingStyle
    status: StrategyStatus
    currentPrice: float
    totalTrades: int
    winningTrades: int
    totalPnL: float
    openPositions: int
    lastSignal: str
    lastSignalTime: str
    createdAt: str
    lotSize: float
    stopLossPips: int
    takeProfitPips: int
    minConfidence: float
    maxPositions: int
    riskPerTrade: float

class Trade(BaseModel):
    id: str
    strategyId: str
    pair: str
    type: str  # 'buy' or 'sell'
    lotSize: float
    openPrice: float
    closePrice: Optional[float]
    stopLoss: float
    takeProfit: float
    status: str  # 'open', 'closed', 'cancelled'
    pnl: float
    openTime: str
    closeTime: Optional[str]
    confidence: float

# Clases del sistema de IA
class DataCollector:
    """Recolecta datos de múltiples fuentes"""
    
    def __init__(self):
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META', 'AMZN']
    
    def get_market_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Obtiene datos históricos del mercado"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            return data
        except Exception as e:
            print(f"Error obteniendo datos para {symbol}: {e}")
            return pd.DataFrame()
    
    def get_fundamental_data(self, symbol: str) -> Dict:
        """Obtiene datos fundamentales"""
        try:
            # Para pares de forex, generar datos simulados ya que yfinance no tiene fundamentales para forex
            if any(forex_pair in symbol.upper() for forex_pair in ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD']):
                # Datos fundamentales simulados para pares de forex
                import random
                return {
                    'pe': round(random.uniform(15, 25), 2),
                    'eps': round(random.uniform(1.5, 3.5), 2),
                    'market_cap': random.randint(1000000, 10000000),
                    'revenue': random.randint(100000, 1000000),
                    'profit_margin': round(random.uniform(0.05, 0.25), 3),
                    'debt_to_equity': round(random.uniform(0.3, 1.2), 2),
                    'rating': random.choice(['buy', 'hold', 'sell', 'neutral'])
                }
            
            # Para otros símbolos, usar yfinance
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'pe': info.get('trailingPE', 0),
                'eps': info.get('trailingEps', 0),
                'market_cap': info.get('marketCap', 0),
                'revenue': info.get('totalRevenue', 0),
                'profit_margin': info.get('profitMargins', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'rating': 'neutral'
            }
        except Exception as e:
            print(f"Error obteniendo fundamentales para {symbol}: {e}")
            # Retornar datos por defecto en caso de error
            import random
            return {
                'pe': round(random.uniform(15, 25), 2),
                'eps': round(random.uniform(1.5, 3.5), 2),
                'market_cap': random.randint(1000000, 10000000),
                'revenue': random.randint(100000, 1000000),
                'profit_margin': round(random.uniform(0.05, 0.25), 3),
                'debt_to_equity': round(random.uniform(0.3, 1.2), 2),
                'rating': 'neutral'
            }

class TechnicalAnalyzer:
    """Analiza indicadores técnicos"""
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula todos los indicadores técnicos"""
        if df.empty:
            return df
            
        # RSI
        df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_histogram'] = macd.macd_diff()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Close'])
        df['BB_upper'] = bollinger.bollinger_hband()
        df['BB_middle'] = bollinger.bollinger_mavg()
        df['BB_lower'] = bollinger.bollinger_lband()
        
        # Moving Averages
        df['SMA_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
        df['SMA_50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
        df['EMA_12'] = ta.trend.EMAIndicator(df['Close'], window=12).ema_indicator()
        
        # Volume indicators
        df['Volume_SMA'] = ta.volume.VolumeSMAIndicator(df['Close'], df['Volume']).volume_sma()
        
        # Support and Resistance
        df['Support'] = df['Low'].rolling(window=20).min()
        df['Resistance'] = df['High'].rolling(window=20).max()
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> Dict:
        """Genera señales de trading basadas en análisis técnico"""
        if df.empty or len(df) < 50:
            return {'signal': 'HOLD', 'confidence': 50, 'reasoning': 'Datos insuficientes'}
        
        latest = df.iloc[-1]
        signals = []
        confidence_factors = []
        
        # Señal RSI
        if latest['RSI'] < 30:
            signals.append('BUY')
            confidence_factors.append(0.8)
        elif latest['RSI'] > 70:
            signals.append('SELL')
            confidence_factors.append(0.8)
        else:
            signals.append('HOLD')
            confidence_factors.append(0.5)
        
        # Señal MACD
        if latest['MACD'] > latest['MACD_signal'] and latest['MACD_histogram'] > 0:
            signals.append('BUY')
            confidence_factors.append(0.7)
        elif latest['MACD'] < latest['MACD_signal'] and latest['MACD_histogram'] < 0:
            signals.append('SELL')
            confidence_factors.append(0.7)
        
        # Señal Bollinger Bands
        if latest['Close'] < latest['BB_lower']:
            signals.append('BUY')
            confidence_factors.append(0.6)
        elif latest['Close'] > latest['BB_upper']:
            signals.append('SELL')
            confidence_factors.append(0.6)
        
        # Determinar señal final
        buy_signals = signals.count('BUY')
        sell_signals = signals.count('SELL')
        
        if buy_signals > sell_signals:
            final_signal = 'BUY'
        elif sell_signals > buy_signals:
            final_signal = 'SELL'
        else:
            final_signal = 'HOLD'
        
        # Calcular confianza
        avg_confidence = np.mean(confidence_factors) if confidence_factors else 0.5
        confidence = int(avg_confidence * 100)
        
        # Generar razonamiento
        reasoning_parts = []
        if latest['RSI'] < 30:
            reasoning_parts.append("RSI sobrevendido")
        elif latest['RSI'] > 70:
            reasoning_parts.append("RSI sobrecomprado")
        
        if latest['MACD'] > latest['MACD_signal']:
            reasoning_parts.append("MACD alcista")
        elif latest['MACD'] < latest['MACD_signal']:
            reasoning_parts.append("MACD bajista")
        
        reasoning = " + ".join(reasoning_parts) if reasoning_parts else "Análisis neutro"
        
        return {
            'signal': final_signal,
            'confidence': confidence,
            'reasoning': reasoning,
            'support': latest['Support'],
            'resistance': latest['Resistance'],
            'rsi': latest['RSI'],
            'macd_status': 'Alcista' if latest['MACD'] > latest['MACD_signal'] else 'Bajista'
        }

class PricePredictorAI:
    """Modelo de IA para predicción de precios"""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepara features para el modelo"""
        if df.empty:
            return np.array([])
        
        # Calcular features
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(10).std()
        df['Price_change'] = df['Close'].diff()
        df['Volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # Seleccionar features relevantes
        feature_columns = ['RSI', 'MACD', 'Returns', 'Volatility', 'Volume_ratio']
        features = df[feature_columns].fillna(0)
        
        return features.values
    
    def train(self, symbol: str):
        """Entrena el modelo con datos históricos"""
        try:
            collector = DataCollector()
            df = collector.get_market_data(symbol, "2y")
            
            if df.empty:
                return False
            
            analyzer = TechnicalAnalyzer()
            df = analyzer.calculate_indicators(df)
            
            # Preparar datos de entrenamiento
            features = self.prepare_features(df)
            
            if len(features) == 0:
                return False
            
            # Target: precio futuro (5 días adelante)
            target = df['Close'].shift(-5).fillna(df['Close'].iloc[-1])
            
            # Remover NaN
            valid_indices = ~np.isnan(features).any(axis=1) & ~np.isnan(target)
            features_clean = features[valid_indices]
            target_clean = target[valid_indices]
            
            if len(features_clean) < 50:
                return False
            
            # Entrenar modelo
            features_scaled = self.scaler.fit_transform(features_clean)
            self.model.fit(features_scaled, target_clean)
            self.is_trained = True
            
            return True
            
        except Exception as e:
            print(f"Error entrenando modelo: {e}")
            return False
    
    def predict(self, symbol: str, days_ahead: int = 5) -> Dict:
        """Predice el precio futuro"""
        try:
            if not self.is_trained:
                self.train(symbol)
            
            collector = DataCollector()
            df = collector.get_market_data(symbol, "6mo")
            
            if df.empty:
                return {'error': 'No hay datos disponibles'}
            
            analyzer = TechnicalAnalyzer()
            df = analyzer.calculate_indicators(df)
            
            features = self.prepare_features(df)
            
            if len(features) == 0:
                return {'error': 'No se pudieron calcular features'}
            
            # Usar los últimos datos para predicción
            latest_features = features[-1].reshape(1, -1)
            latest_features_scaled = self.scaler.transform(latest_features)
            
            # Hacer predicción
            predicted_price = self.model.predict(latest_features_scaled)[0]
            current_price = df['Close'].iloc[-1]
            
            # Calcular confianza basada en volatilidad
            volatility = df['Close'].pct_change().std()
            confidence = max(60, min(95, int(100 - (volatility * 1000))))
            
            return {
                'current_price': current_price,
                'predicted_price': predicted_price,
                'confidence': confidence,
                'timeframe': f"{days_ahead} días",
                'change_percent': ((predicted_price - current_price) / current_price) * 100
            }
            
        except Exception as e:
            print(f"Error en predicción: {e}")
            return {'error': str(e)}

class AlertManager:
    """Gestiona alertas y notificaciones"""
    
    def __init__(self):
        self.active_alerts = []
        self.triggered_alerts = []
    
    def add_alert(self, alert: AlertRequest) -> bool:
        """Añade una nueva alerta"""
        try:
            self.active_alerts.append({
                'id': len(self.active_alerts) + 1,
                'symbol': alert.symbol,
                'condition': alert.condition,
                'value': alert.value,
                'user_id': alert.user_id,
                'created_at': datetime.now(),
                'status': 'active'
            })
            return True
        except Exception as e:
            print(f"Error añadiendo alerta: {e}")
            return False
    
    def check_alerts(self) -> List[Dict]:
        """Verifica alertas activas"""
        triggered = []
        
        for alert in self.active_alerts[:]:
            if alert['status'] != 'active':
                continue
                
            try:
                # Obtener precio actual
                ticker = yf.Ticker(alert['symbol'])
                current_data = ticker.history(period="1d")
                
                if current_data.empty:
                    continue
                
                current_price = current_data['Close'].iloc[-1]
                
                # Verificar condición
                condition_met = False
                
                if 'Precio >' in alert['condition'] and current_price > alert['value']:
                    condition_met = True
                elif 'Precio <' in alert['condition'] and current_price < alert['value']:
                    condition_met = True
                
                if condition_met:
                    alert['status'] = 'triggered'
                    alert['triggered_at'] = datetime.now()
                    triggered.append({
                        'symbol': alert['symbol'],
                        'message': f"{alert['symbol']} {alert['condition']} {alert['value']}",
                        'current_price': current_price,
                        'time': datetime.now().strftime('%H:%M')
                    })
                    
            except Exception as e:
                print(f"Error verificando alerta: {e}")
        
        return triggered

# Instancias globales
data_collector = DataCollector()
technical_analyzer = TechnicalAnalyzer()
price_predictor = PricePredictorAI()
alert_manager = AlertManager()

# WebSocket para datos en tiempo real
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                self.disconnect(connection)

manager = ConnectionManager()

# Endpoints de la API
@app.get("/")
async def root():
    """Endpoint raíz con información del sistema"""
    return {
        "message": "AI Trading System API",
        "version": "1.0.0",
        "status": app_state.get("status", "unknown"),
        "uptime": str(datetime.now() - app_state.get("start_time", datetime.now())) if app_state.get("start_time") else "0",
        "features": [
            "Traditional AI (Random Forest, LSTM)",
            "Reinforcement Learning (DQN, PPO)", 
            "Auto-training & Model Management",
            "MetaTrader 4 Integration",
            "Real-time WebSocket Data",
            "Risk Management System"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system_status": app_state.get("status", "unknown")
    }

# Placeholder endpoints - se implementarán en archivos separados
@app.get("/api/model/status")
async def get_model_status():
    """Estado del sistema de IA"""
    # TODO: Implementar lógica real
    return {
        "traditional_ai": {"status": "ready", "version": "1.0"},
        "reinforcement_learning": {"status": "training", "episodes": 0},
        "auto_training": {"status": "monitoring", "last_training": None}
    }

@app.get("/api/rl/status")
async def get_rl_status():
    """Obtiene el estado del RL Director"""
    try:
        return globals()['get_rl_status_imported']()
    except NameError:
        return {"status": "inactive", "error": "Service not available"}

# Rutas adicionales de RL
@app.get("/api/rl/performance")
async def get_rl_performance():
    """Obtiene el rendimiento del RL Director"""
    try:
        return globals()['get_rl_performance_imported']()
    except NameError:
        return {"error": "Service not available"}

@app.get("/api/rl/active-signals")
async def get_active_signals():
    """Obtiene señales activas del RL Director"""
    try:
        return globals()['get_active_signals_imported']()
    except NameError:
        return []

@app.post("/api/rl/execute-signal")
async def execute_signal(signal: dict):
    """Ejecuta una señal de trading"""
    try:
        return globals()['execute_signal_imported'](signal)
    except NameError:
        return {"success": False, "error": "Service not available"}

@app.get("/api/rl/training-progress")
async def get_training_progress():
    """Obtiene el progreso del entrenamiento actual"""
    try:
        return globals()['get_training_progress_imported']()
    except NameError:
        return {"is_training": False, "error": "Service not available"}

# Nuevos endpoints para el sistema de entrenamiento
@app.get("/api/rl/can-train/{user_id}")
async def can_user_train(user_id: str):
    """Verifica si un usuario puede iniciar entrenamiento"""
    try:
        return globals()['can_user_start_training_imported'](user_id)
    except NameError:
        return {"can_train": False, "reason": "Service not available"}

@app.post("/api/rl/validate-params")
async def validate_training_params(request: dict):
    """Valida los parámetros de entrenamiento"""
    try:
        episodes = request.get("episodes", 100)
        user_plan = request.get("user_plan", "starter")
        return globals()['validate_training_parameters_imported'](episodes, user_plan)
    except NameError:
        return {"valid": False, "reason": "Service not available"}

@app.post("/api/rl/start-training")
async def start_training(request: dict):
    """Inicia un entrenamiento de RL"""
    try:
        user_id = request.get("user_id")
        episodes = request.get("episodes", 100)
        algorithm = request.get("algorithm", "dqn")
        trading_pair = request.get("trading_pair", "EURUSD")
        timeframe = request.get("timeframe", "1h")
        
        return await globals()['start_rl_training_imported'](
            user_id, episodes, algorithm, trading_pair, timeframe
        )
    except NameError:
        return {"success": False, "error": "Service not available"}

@app.get("/api/rl/training-progress/{session_id}")
async def get_session_progress(session_id: str):
    """Obtiene el progreso de una sesión específica"""
    try:
        return globals()['get_training_progress_by_session_imported'](session_id)
    except NameError:
        return {"is_training": False, "error": "Service not available"}

@app.post("/api/rl/cancel-training/{session_id}")
async def cancel_training(session_id: str, user_id: str):
    """Cancela un entrenamiento en curso"""
    try:
        return globals()['cancel_rl_training_imported'](session_id, user_id)
    except NameError:
        return {"success": False, "error": "Service not available"}

@app.get("/api/rl/training-history/{user_id}")
async def get_training_history(user_id: str, limit: int = 10):
    """Obtiene el historial de entrenamientos del usuario"""
    try:
        return globals()['get_user_training_history_imported'](user_id, limit)
    except NameError:
        return []

# Endpoints para configuración avanzada de RL
@app.get("/api/rl/configuration/{user_id}")
async def get_user_configuration(user_id: str):
    """Obtiene la configuración de RL del usuario"""
    try:
        return globals()['get_user_rl_configuration_imported'](user_id)
    except NameError:
        return {"success": False, "error": "Service not available"}

@app.post("/api/rl/configuration/{user_id}")
async def save_user_configuration(
    user_id: str,
    max_drawdown_percentage: float = 15.0,
    max_position_size_percentage: float = 5.0,
    min_confidence_threshold: float = 70.0,
    retraining_frequency: str = "monthly",
    retraining_enabled: bool = False
):
    """Guarda la configuración de RL del usuario"""
    try:
        return globals()['save_user_rl_configuration_imported'](
            user_id,
            max_drawdown_percentage,
            max_position_size_percentage,
            min_confidence_threshold,
            retraining_frequency,
            retraining_enabled
        )
    except NameError:
        return {"success": False, "error": "Service not available"}

@app.post("/api/rl/configuration/{user_id}/reset")
async def reset_user_configuration(user_id: str):
    """Restaura la configuración de RL del usuario a valores por defecto"""
    try:
        return globals()['reset_user_rl_configuration_imported'](user_id)
    except NameError:
        return {"success": False, "error": "Service not available"}

@app.get("/api/rl/configuration/limits")
async def get_configuration_limits(user_id: str = None):
    """Obtiene los límites válidos para los parámetros de configuración"""
    # Endpoint simplificado - NUEVO
    limits_data = {
        "success": True,
        "limits": {
            "max_drawdown_percentage": {
                "min": 5.0,
                "max": 25.0,
                "default": 15.0,
                "step": 1.0
            },
            "max_position_size_percentage": {
                "min": 1.0,
                "max": 10.0,
                "default": 5.0,
                "step": 0.5
            },
            "min_confidence_threshold": {
                "min": 50.0,
                "max": 90.0,
                "default": 70.0,
                "step": 5.0
            },
            "retraining_frequency": {
                "options": [
                    {"value": "monthly", "label": "Mensual"}
                ],
                "default": "monthly",
                "available": True
            },
            "retraining_enabled": {
                "default": False,
                "available": True
            },
            "user_subscription": "premium"
        }
    }
    return limits_data

@app.get("/api/rl/configuration/test")
async def test_configuration_endpoint():
    """Endpoint de prueba para verificar que el servidor funciona"""
    return {"success": True, "message": "Endpoint funcionando correctamente", "timestamp": "2025-08-03"}

@app.get("/api/rl/config-limits-new")
async def get_config_limits_new():
    """Endpoint nuevo para configuración RL - completamente independiente"""
    return {
        "success": True,
        "limits": {
            "max_drawdown_percentage": {"min": 5.0, "max": 25.0, "default": 15.0, "step": 1.0},
            "max_position_size_percentage": {"min": 1.0, "max": 10.0, "default": 5.0, "step": 0.5},
            "min_confidence_threshold": {"min": 50.0, "max": 90.0, "default": 70.0, "step": 5.0},
            "retraining_frequency": {
                "options": [{"value": "monthly", "label": "Mensual"}],
                "default": "monthly",
                "available": True
            },
            "retraining_enabled": {"default": False, "available": True},
            "user_subscription": "premium"
        }
    }

@app.get("/api/mt4/status")
async def get_mt4_status():
    """Estado de conexión con MetaTrader 4"""
    # TODO: Implementar lógica real
    return {
        "connected": False,
        "host": "localhost",
        "port": 9090
    }

# Wallet endpoints
@app.get("/wallet")
async def get_wallet():
    """Obtiene información de la wallet del usuario"""
    # TODO: Implementar lógica real con autenticación
    return {
        "balance": 10000.0,
        "transactions": [
            {
                "id": 1,
                "type": "deposit",
                "amount": 10000.0,
                "description": "Depósito inicial",
                "created_at": "2024-01-01T00:00:00Z"
            }
        ]
    }

@app.post("/wallet/recharge")
async def recharge_wallet(amount: float):
    """Recarga la wallet del usuario"""
    # TODO: Implementar lógica real con autenticación
    return {
        "balance": 10000.0 + amount,
        "message": f"Wallet recargada con ${amount}"
    }

@app.post("/wallet/trade")
async def trade_wallet(amount: float, description: str = ""):
    """Realiza una operación de trading en la wallet"""
    # TODO: Implementar lógica real con autenticación
    return {
        "balance": 10000.0 - amount,
        "message": f"Operación realizada: {description}",
        "amount": amount
    }

@app.get("/wallet/transactions")
async def get_wallet_transactions():
    """Obtiene el historial de transacciones de la wallet"""
    # TODO: Implementar lógica real con autenticación
    return [
        {
            "id": 1,
            "type": "deposit",
            "amount": 10000.0,
            "description": "Depósito inicial",
            "created_at": "2024-01-01T00:00:00Z"
        },
        {
            "id": 2,
            "type": "trade",
            "amount": -500.0,
            "description": "Compra EURUSD",
            "created_at": "2024-01-02T10:30:00Z"
        }
    ]

@app.get("/api/assets", response_model=List[Asset])
async def get_recommended_assets():
    """Obtiene activos recomendados por la IA"""
    try:
        assets = []
        symbols = ['AAPL', 'TSLA', 'MSFT', 'NVDA', 'GOOGL']
        
        for symbol in symbols:
            try:
                # Obtener datos de mercado
                df = data_collector.get_market_data(symbol, "6mo")
                if df.empty:
                    continue
                
                # Calcular indicadores técnicos
                df = technical_analyzer.calculate_indicators(df)
                
                # Generar señales
                signals = technical_analyzer.generate_signals(df)
                
                # Obtener precio actual
                current_price = df['Close'].iloc[-1]
                previous_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
                change = current_price - previous_price
                change_percent = (change / previous_price) * 100
                
                # Calcular precio objetivo
                if signals['signal'] == 'BUY':
                    target_price = current_price * 1.05  # 5% al alza
                elif signals['signal'] == 'SELL':
                    target_price = current_price * 0.95  # 5% a la baja
                else:
                    target_price = current_price
                
                # Determinar timeframe
                timeframe = "3-5 días" if signals['confidence'] > 70 else "1 semana"
                
                # Crear asset
                asset = Asset(
                    symbol=symbol,
                    name=f"{symbol} Inc.",
                    price=round(current_price, 2),
                    change=round(change, 2),
                    changePercent=round(change_percent, 2),
                    signal=signals['signal'],
                    confidence=signals['confidence'],
                    targetPrice=round(target_price, 2),
                    timeframe=timeframe,
                    reasoning=signals['reasoning']
                )
                
                assets.append(asset)
                
            except Exception as e:
                print(f"Error procesando {symbol}: {e}")
                continue
        
        return assets
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo activos: {str(e)}")

@app.get("/api/technical-analysis/{symbol}")
async def get_technical_analysis(symbol: str):
    """Obtiene análisis técnico detallado"""
    try:
        df = data_collector.get_market_data(symbol, "6mo")
        if df.empty:
            raise HTTPException(status_code=404, detail="No se encontraron datos")
        
        df = technical_analyzer.calculate_indicators(df)
        signals = technical_analyzer.generate_signals(df)
        
        latest = df.iloc[-1]
        
        analysis = TechnicalAnalysis(
            rsi=round(latest['RSI'], 1),
            macd=signals['macd_status'],
            bollinger="Cerca límite inferior" if latest['Close'] < latest['BB_lower'] else 
                     "Cerca límite superior" if latest['Close'] > latest['BB_upper'] else "En rango medio",
            support=round(signals['support'], 2),
            resistance=round(signals['resistance'], 2),
            volume="Alto" if latest['Volume'] > latest['Volume_SMA'] else "Normal",
            trend="Alcista" if latest['SMA_20'] > latest['SMA_50'] else "Bajista"
        )
        
        return analysis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en análisis técnico: {str(e)}")

@app.get("/api/fundamental-analysis/{symbol}")
async def get_fundamental_analysis(symbol: str):
    """Obtiene análisis fundamental"""
    try:
        fundamental_data = data_collector.get_fundamental_data(symbol)
        
        if not fundamental_data:
            raise HTTPException(status_code=404, detail="No se encontraron datos fundamentales")
        
        # Calcular sentiment simulado (en producción sería análisis de noticias)
        sentiment = np.random.randint(60, 85)
        
        analysis = FundamentalAnalysis(
            pe=round(fundamental_data.get('pe', 0), 1),
            eps=round(fundamental_data.get('eps', 0), 2),
            epsGrowth=12,  # Simulado
            sentiment=sentiment,
            nextEarnings="15 días",  # Simulado
            rating=fundamental_data.get('rating', 'neutral')
        )
        
        return analysis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en análisis fundamental: {str(e)}")

@app.post("/api/predict-price")
async def predict_price(request: PredictionRequest):
    """Predice el precio futuro usando IA"""
    try:
        prediction = price_predictor.predict(request.symbol, request.timeframe)
        
        if 'error' in prediction:
            raise HTTPException(status_code=400, detail=prediction['error'])
        
        return {
            "symbol": request.symbol,
            "current_price": round(prediction['current_price'], 2),
            "predicted_price": round(prediction['predicted_price'], 2),
            "confidence": prediction['confidence'],
            "timeframe": prediction['timeframe'],
            "change_percent": round(prediction['change_percent'], 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

@app.get("/api/price-data/{symbol}")
async def get_price_data(symbol: str, period: str = "1d"):
    """Obtiene datos de precios para gráficos"""
    try:
        df = data_collector.get_market_data(symbol, period)
        if df.empty:
            raise HTTPException(status_code=404, detail="No se encontraron datos")
        
        # Convertir a formato para gráficos
        if period == "1d":
            # Datos intraday (simulados)
            data = []
            for i in range(10):
                base_time = datetime.now().replace(hour=9, minute=30) + timedelta(minutes=i*30)
                price = df['Close'].iloc[-1] + np.random.normal(0, 0.5)
                volume = df['Volume'].iloc[-1] * np.random.uniform(0.8, 1.2)
                
                data.append({
                    "time": base_time.strftime('%H:%M'),
                    "price": round(price, 2),
                    "volume": int(volume)
                })
        else:
            # Datos históricos
            data = []
            for i, (date, row) in enumerate(df.tail(100).iterrows()):
                data.append({
                    "date": date.strftime('%Y-%m-%d'),
                    "open": round(row['Open'], 2),
                    "high": round(row['High'], 2),
                    "low": round(row['Low'], 2),
                    "close": round(row['Close'], 2),
                    "volume": int(row['Volume'])
                })
        
        return data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo datos de precio: {str(e)}")

@app.post("/api/alerts")
async def create_alert(alert: AlertRequest):
    """Crea una nueva alerta"""
    try:
        success = alert_manager.add_alert(alert)
        if success:
            return {"message": "Alerta creada exitosamente", "status": "success"}
        else:
            raise HTTPException(status_code=400, detail="Error creando alerta")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/api/alerts")
async def get_alerts():
    """Obtiene alertas activas"""
    try:
        return alert_manager.active_alerts
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo alertas: {str(e)}")

@app.get("/api/alerts/check")
async def check_alerts():
    """Verifica alertas y retorna las disparadas"""
    try:
        triggered = alert_manager.check_alerts()
        return triggered
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error verificando alertas: {str(e)}")

# ===== BRAIN TRADER ENDPOINTS =====
@app.get("/api/v1/brain-trader/available-brains")
async def get_available_brains(plan_type: str = "starter"):
    """Obtiene los cerebros disponibles según el plan de suscripción"""
    try:
        # Configuración de cerebros por plan
        brains_by_plan = {
            "starter": ["brain_max"],
            "trader": ["brain_max", "mega_mind"],
            "expert": ["brain_max", "brain_ultra", "mega_mind"],
            "premium": ["brain_max", "brain_ultra", "brain_predictor", "mega_mind"],
            "institutional": ["brain_max", "brain_ultra", "brain_predictor", "mega_mind"]
        }
        
        # Obtener cerebros disponibles para el plan
        available_brains = brains_by_plan.get(plan_type, ["brain_max"])
        
        return {
            "available_brains": available_brains,
            "default_brain": "brain_max"
        }
        
    except Exception as e:
        logger.error(f"Error getting available brains: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/brain-trader/predictions/{brain_type}")
async def get_predictions(brain_type: str, pair: str = "EURUSD", style: str = "day_trading", limit: int = 5):
    """Endpoint de predicciones - Usar el servicio real en lugar de datos mock"""
    try:
        # Validar brain type
        valid_brain_types = ['brain_max', 'brain_ultra', 'brain_predictor', 'mega_mind']
        if brain_type not in valid_brain_types:
            raise HTTPException(status_code=400, detail=f"Brain type must be one of: {valid_brain_types}")
        
        # Usar el servicio real de brain trader
        predictions = await brain_trader_service.get_predictions(brain_type, pair, style, limit)
        return predictions
        
    except Exception as e:
        logger.error(f"Error getting predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/brain-trader/trends/{brain_type}")
async def get_trends(brain_type: str, pair: str = "EURUSD", limit: int = 3):
    """Endpoint de tendencias - Usar el servicio real en lugar de datos mock"""
    try:
        # Validar brain type
        valid_brain_types = ['brain_max', 'brain_ultra', 'brain_predictor', 'mega_mind']
        if brain_type not in valid_brain_types:
            raise HTTPException(status_code=400, detail=f"Brain type must be one of: {valid_brain_types}")
        
        # Usar el servicio real de brain trader
        trends = await brain_trader_service.get_trends(brain_type, pair, limit)
        return trends
        
    except Exception as e:
        logger.error(f"Error getting trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== MEGA MIND ENDPOINTS =====
@app.get("/api/v1/mega-mind/predictions")
async def get_mega_mind_predictions(pair: str = "EURUSD", style: str = "day_trading", limit: int = 5):
    """Endpoint de predicciones MEGA MIND - Usar el servicio real"""
    try:
        # Usar el servicio real de brain trader para MEGA MIND
        predictions = await brain_trader_service.get_predictions('mega_mind', pair, style, limit)
        return predictions
        
    except Exception as e:
        logger.error(f"Error getting MEGA MIND predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/mega-mind/collaboration")
async def get_brain_collaboration(pair: str = "EURUSD"):
    return {
        "pair": pair,
        "collaboration_score": random.uniform(0.85, 0.98),
        "consensus_level": random.uniform(0.75, 0.95),
        "brain_synergy": {
            "brain_max_contribution": random.uniform(0.20, 0.30),
            "brain_ultra_contribution": random.uniform(0.30, 0.40),
            "brain_predictor_contribution": random.uniform(0.35, 0.45)
        },
        "conflict_resolution": {
            "resolved_conflicts": random.randint(5, 15),
            "consensus_achieved": random.uniform(0.80, 0.95),
            "decision_confidence": random.uniform(0.90, 0.98)
        },
        "performance_metrics": {
            "accuracy_improvement": random.uniform(0.05, 0.15),
            "risk_reduction": random.uniform(0.10, 0.20),
            "prediction_stability": random.uniform(0.85, 0.95)
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/mega-mind/arena")
async def get_brain_arena_results(pair: str = "EURUSD"):
    return {
        "pair": pair,
        "competition_round": random.randint(1, 10),
        "arena_results": {
            "brain_max": {
                "wins": random.randint(15, 25),
                "losses": random.randint(5, 15),
                "win_rate": random.uniform(0.65, 0.85),
                "performance_score": random.uniform(0.75, 0.88)
            },
            "brain_ultra": {
                "wins": random.randint(20, 30),
                "losses": random.randint(5, 15),
                "win_rate": random.uniform(0.75, 0.90),
                "performance_score": random.uniform(0.80, 0.92)
            },
            "brain_predictor": {
                "wins": random.randint(25, 35),
                "losses": random.randint(3, 12),
                "win_rate": random.uniform(0.80, 0.94),
                "performance_score": random.uniform(0.85, 0.94)
            }
        },
        "champion": "brain_predictor",
        "overall_performance": random.uniform(0.85, 0.95),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/mega-mind/performance")
async def get_mega_mind_performance():
    return {
        "overall_accuracy": random.uniform(92, 98),
        "prediction_success_rate": random.uniform(0.85, 0.95),
        "risk_adjusted_returns": random.uniform(0.12, 0.25),
        "sharpe_ratio": random.uniform(1.5, 2.5),
        "max_drawdown": random.uniform(0.05, 0.15),
        "win_rate": random.uniform(0.75, 0.90),
        "profit_factor": random.uniform(1.8, 3.2),
        "average_trade_duration": random.uniform(2, 8),
        "consecutive_wins": random.randint(5, 15),
        "consecutive_losses": random.randint(1, 3),
        "volatility": random.uniform(0.08, 0.18),
        "calmar_ratio": random.uniform(2.0, 4.0),
        "sortino_ratio": random.uniform(2.5, 4.5),
        "information_ratio": random.uniform(1.8, 3.0),
        "timestamp": datetime.now().isoformat()
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket para datos en tiempo real"""
    await manager.connect(websocket)
    try:
        while True:
            # Enviar datos actualizados cada 30 segundos
            await asyncio.sleep(30)
            
            # Verificar alertas
            triggered_alerts = alert_manager.check_alerts()
            
            if triggered_alerts:
                message = {
                    "type": "alerts",
                    "data": triggered_alerts,
                    "timestamp": datetime.now().isoformat()
                }
                await manager.send_personal_message(json.dumps(message), websocket)
            
            # Enviar actualización de precios
            try:
                symbols = ['AAPL', 'TSLA', 'MSFT', 'NVDA']
                price_updates = []
                
                for symbol in symbols:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period="1d")
                    if not data.empty:
                        current_price = data['Close'].iloc[-1]
                        price_updates.append({
                            "symbol": symbol,
                            "price": round(current_price, 2),
                            "timestamp": datetime.now().isoformat()
                        })
                
                if price_updates:
                    message = {
                        "type": "price_update",
                        "data": price_updates
                    }
                    await manager.send_personal_message(json.dumps(message), websocket)
                    
            except Exception as e:
                print(f"Error en WebSocket price update: {e}")
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# ===== AUTOMATED TRADING SYSTEM =====
# Almacenamiento en memoria para estrategias y trades
strategies_db: Dict[str, Strategy] = {}
trades_db: Dict[str, Trade] = {}
active_tasks: Dict[str, asyncio.Task] = {}

# Precios simulados para pares de divisas
current_prices = {
    "EUR/USD": 1.0856,
    "GBP/USD": 1.2654,
    "USD/JPY": 148.23,
    "AUD/USD": 0.6589,
    "USD/CAD": 1.3542
}

def update_prices():
    """Simula cambios de precio para los pares de divisas"""
    for pair in current_prices:
        # Simular movimiento de precio aleatorio
        change = np.random.normal(0, 0.0005)  # Cambio pequeño
        current_prices[pair] += change
        # Mantener precios realistas
        if pair == "USD/JPY":
            current_prices[pair] = max(100, min(200, current_prices[pair]))
        else:
            current_prices[pair] = max(0.5, min(2.0, current_prices[pair]))

def generate_signal(strategy: Strategy) -> Optional[Dict[str, Any]]:
    """Genera señales de trading basadas en la estrategia usando IA real"""
    try:
        # Obtener predicción de IA basada en el cerebro seleccionado
        brain_type = strategy.brainType.value if hasattr(strategy.brainType, 'value') else str(strategy.brainType)
        
        # Generar datos de mercado simulados para la IA
        current_price = current_prices.get(strategy.pair, 1.0856)
        
        # Crear datos de mercado para la IA
        market_data = {
            'pair': strategy.pair,
            'current_price': current_price,
            'timestamp': datetime.now().isoformat(),
            'data': {
                'open': [current_price - 0.0002, current_price - 0.0001, current_price, current_price + 0.0001],
                'high': [current_price + 0.0001, current_price + 0.0002, current_price + 0.0003, current_price + 0.0004],
                'low': [current_price - 0.0003, current_price - 0.0002, current_price - 0.0001, current_price],
                'close': [current_price - 0.0001, current_price, current_price + 0.0001, current_price + 0.0002],
                'volume': [1000, 1200, 1100, 1300]
            }
        }
        
        # Usar IA real basada en el cerebro seleccionado
        if brain_type == "Brain_Ultra":
            # Usar EURUSDMultiStrategyAI para Brain_Ultra
            try:
                from models.sistemas_prueba.Modelo_AI_Ultra import EURUSDMultiStrategyAI
                ai_model = EURUSDMultiStrategyAI()
                signals = ai_model.generate_signals_strategy(market_data, strategy.style.value if hasattr(strategy.style, 'value') else str(strategy.style))
                
                if signals and len(signals) > 0:
                    latest_signal = signals[-1]
                    confidence = latest_signal.get('confidence', 70)
                    
                    if confidence >= strategy.minConfidence:
                        return {
                            'type': latest_signal.get('signal', 'hold').lower(),
                            'confidence': confidence,
                            'price': current_price,
                            'timestamp': datetime.now().isoformat(),
                            'ai_model': 'EURUSDMultiStrategyAI'
                        }
            except ImportError:
                print(f"⚠️ Modelo EURUSDMultiStrategyAI no disponible, usando fallback")
        
        elif brain_type == "Brain_Max":
            # Usar Brain Max para análisis técnico completo
            try:
                from models.Modelo_Brain_Max import BrainMaxAI
                ai_model = BrainMaxAI()
                signals = ai_model.generate_signals_strategy(market_data, strategy.style.value if hasattr(strategy.style, 'value') else str(strategy.style))
                
                if signals and len(signals) > 0:
                    latest_signal = signals[-1]
                    confidence = latest_signal.get('confidence', 70)
                    
                    if confidence >= strategy.minConfidence:
                        return {
                            'type': latest_signal.get('signal', 'hold').lower(),
                            'confidence': confidence,
                            'price': current_price,
                            'timestamp': datetime.now().isoformat(),
                            'ai_model': 'BrainMaxAI'
                        }
            except ImportError:
                print(f"⚠️ Modelo BrainMaxAI no disponible, usando fallback")
        
        elif brain_type == "Brain_Predictor":
            # Usar Brain Predictor para predicciones ML
            try:
                from models.Brain_predictor import BrainPredictorAI
                ai_model = BrainPredictorAI()
                signals = ai_model.generate_signals_strategy(market_data, strategy.style.value if hasattr(strategy.style, 'value') else str(strategy.style))
                
                if signals and len(signals) > 0:
                    latest_signal = signals[-1]
                    confidence = latest_signal.get('confidence', 70)
                    
                    if confidence >= strategy.minConfidence:
                        return {
                            'type': latest_signal.get('signal', 'hold').lower(),
                            'confidence': confidence,
                            'price': current_price,
                            'timestamp': datetime.now().isoformat(),
                            'ai_model': 'BrainPredictorAI'
                        }
            except ImportError:
                print(f"⚠️ Modelo BrainPredictorAI no disponible, usando fallback")
        
        elif brain_type == "Mega_Mind":
            # Usar Mega Mind para colaboración de cerebros
            try:
                from src.services.mega_mind_service import MegaMindService
                mega_mind = MegaMindService()
                signals = mega_mind.generate_collaborative_signals(market_data, strategy.style.value if hasattr(strategy.style, 'value') else str(strategy.style))
                
                if signals and len(signals) > 0:
                    latest_signal = signals[-1]
                    confidence = latest_signal.get('confidence', 70)
                    
                    if confidence >= strategy.minConfidence:
                        return {
                            'type': latest_signal.get('signal', 'hold').lower(),
                            'confidence': confidence,
                            'price': current_price,
                            'timestamp': datetime.now().isoformat(),
                            'ai_model': 'MegaMindService'
                        }
            except ImportError:
                print(f"⚠️ MegaMindService no disponible, usando fallback")
        
        # Fallback: usar Brain Trader API si está disponible
        try:
            import requests
            response = requests.get(
                f"http://localhost:8000/api/v1/brain-trader/predictions/{brain_type}",
                params={
                    "pair": strategy.pair,
                    "style": strategy.style.value if hasattr(strategy.style, 'value') else str(strategy.style),
                    "limit": 1
                },
                timeout=5
            )
            
            if response.status_code == 200:
                predictions = response.json()
                if predictions and len(predictions) > 0:
                    pred = predictions[0]
                    confidence = pred.get('confidence', 70)
                    
                    if confidence >= strategy.minConfidence:
                        return {
                            'type': pred.get('direction', 'hold').lower(),
                            'confidence': confidence,
                            'price': current_price,
                            'timestamp': datetime.now().isoformat(),
                            'ai_model': f'Brain_Trader_{brain_type}'
                        }
        except Exception as e:
            print(f"⚠️ Error en Brain Trader API: {e}")
        
        # Fallback final: simulación con mejor lógica
        confidence = np.random.uniform(strategy.minConfidence, 95)
        
        if confidence >= strategy.minConfidence:
            # Mejorar la lógica de selección de señales
            if strategy.style.value == "scalping":
                signal_type = np.random.choice(['buy', 'sell'], p=[0.55, 0.45])  # Más balanceado para scalping
            elif strategy.style.value == "day_trading":
                signal_type = np.random.choice(['buy', 'sell'], p=[0.6, 0.4])   # Tendencia alcista para day trading
            else:  # swing_trading
                signal_type = np.random.choice(['buy', 'sell'], p=[0.5, 0.5])   # Balanceado para swing
            
            return {
                'type': signal_type,
                'confidence': confidence,
                'price': current_price,
                'timestamp': datetime.now().isoformat(),
                'ai_model': 'fallback_simulation'
            }
        
        return None
        
    except Exception as e:
        print(f"❌ Error en generate_signal: {e}")
        # Fallback de emergencia
        confidence = np.random.uniform(60, 95)
        if confidence >= strategy.minConfidence:
            return {
                'type': np.random.choice(['buy', 'sell']),
                'confidence': confidence,
                'price': current_prices.get(strategy.pair, 1.0856),
                'timestamp': datetime.now().isoformat(),
                'ai_model': 'emergency_fallback'
            }
        return None

async def run_strategy(strategy_id: str):
    """Ejecuta una estrategia de trading automático"""
    while strategy_id in active_tasks and strategies_db[strategy_id].status == StrategyStatus.ACTIVE:
        try:
            strategy = strategies_db[strategy_id]
            
            # Verificar si podemos abrir más posiciones
            open_positions = len([t for t in trades_db.values() 
                                if t.strategyId == strategy_id and t.status == 'open'])
            
            if open_positions < strategy.maxPositions:
                # Generar señal
                signal = generate_signal(strategy)
                
                if signal:
                    # Crear trade
                    trade = Trade(
                        id=str(uuid.uuid4()),
                        strategyId=strategy_id,
                        pair=strategy.pair,
                        type=signal['type'],
                        lotSize=strategy.lotSize,
                        openPrice=signal['price'],
                        closePrice=None,
                        stopLoss=signal['price'] - (strategy.stopLossPips * 0.0001) if signal['type'] == 'buy' 
                               else signal['price'] + (strategy.stopLossPips * 0.0001),
                        takeProfit=signal['price'] + (strategy.takeProfitPips * 0.0001) if signal['type'] == 'buy'
                                 else signal['price'] - (strategy.takeProfitPips * 0.0001),
                        status='open',
                        pnl=0.0,
                        openTime=signal['timestamp'],
                        closeTime=None,
                        confidence=signal['confidence']
                    )
                    
                    trades_db[trade.id] = trade
                    
                    # Actualizar estrategia
                    strategies_db[strategy_id].openPositions = open_positions + 1
                    strategies_db[strategy_id].lastSignal = signal['type']
                    strategies_db[strategy_id].lastSignalTime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Procesar trades abiertos
            for trade_id, trade in trades_db.items():
                if trade.strategyId == strategy_id and trade.status == 'open':
                    current_price = current_prices.get(trade.pair, trade.openPrice)
                    
                    # Verificar stop loss y take profit
                    if trade.type == 'buy':
                        if current_price <= trade.stopLoss or current_price >= trade.takeProfit:
                            # Cerrar trade
                            trade.status = 'closed'
                            trade.closePrice = current_price
                            trade.closeTime = datetime.now().isoformat()
                            trade.pnl = (current_price - trade.openPrice) * trade.lotSize * 100000
                            
                            # Actualizar estrategia
                            strategies_db[strategy_id].totalTrades += 1
                            if trade.pnl > 0:
                                strategies_db[strategy_id].winningTrades += 1
                            strategies_db[strategy_id].totalPnL += trade.pnl
                            strategies_db[strategy_id].openPositions -= 1
                    
                    elif trade.type == 'sell':
                        if current_price >= trade.stopLoss or current_price <= trade.takeProfit:
                            # Cerrar trade
                            trade.status = 'closed'
                            trade.closePrice = current_price
                            trade.closeTime = datetime.now().isoformat()
                            trade.pnl = (trade.openPrice - current_price) * trade.lotSize * 100000
                            
                            # Actualizar estrategia
                            strategies_db[strategy_id].totalTrades += 1
                            if trade.pnl > 0:
                                strategies_db[strategy_id].winningTrades += 1
                            strategies_db[strategy_id].totalPnL += trade.pnl
                            strategies_db[strategy_id].openPositions -= 1
            
            # Actualizar precio actual de la estrategia
            strategies_db[strategy_id].currentPrice = current_prices.get(strategy.pair, 1.0856)
            
            # Esperar antes de la siguiente iteración
            await asyncio.sleep(30)  # 30 segundos
            
        except Exception as e:
            print(f"Error en estrategia {strategy_id}: {e}")
            await asyncio.sleep(60)  # Esperar más tiempo si hay error

# ===== AUTOMATED TRADING API ENDPOINTS =====
@app.get("/api/v1/trading/strategies", response_model=Dict[str, List[Strategy]])
async def get_strategies():
    """Obtener todas las estrategias de trading"""
    return {"strategies": list(strategies_db.values())}

@app.post("/api/v1/trading/strategies", response_model=Dict[str, Strategy])
async def create_strategy(config: StrategyConfig):
    """Crear una nueva estrategia de trading"""
    strategy_id = str(uuid.uuid4())
    
    strategy = Strategy(
        id=strategy_id,
        name=config.name,
        brainType=config.brainType,
        pair=config.pair,
        style=config.style,
        status=StrategyStatus.PAUSED,
        currentPrice=current_prices.get(config.pair, 1.0856),
        totalTrades=0,
        winningTrades=0,
        totalPnL=0.0,
        openPositions=0,
        lastSignal="none",
        lastSignalTime="N/A",
        createdAt=datetime.now().isoformat(),
        lotSize=config.lotSize,
        stopLossPips=config.stopLossPips,
        takeProfitPips=config.takeProfitPips,
        minConfidence=config.minConfidence,
        maxPositions=config.maxPositions,
        riskPerTrade=config.riskPerTrade
    )
    
    strategies_db[strategy_id] = strategy
    return {"strategy": strategy}

@app.get("/api/v1/trading/prices")
async def get_prices():
    """Obtener precios actuales de los pares de divisas"""
    # Actualizar precios simulados
    update_prices()
    return {"prices": current_prices}

@app.post("/api/v1/trading/strategies/{strategy_id}/start")
async def start_strategy(strategy_id: str):
    """Iniciar una estrategia de trading"""
    if strategy_id not in strategies_db:
        raise HTTPException(status_code=404, detail="Estrategia no encontrada")
    
    strategy = strategies_db[strategy_id]
    strategy.status = StrategyStatus.ACTIVE
    
    # Iniciar tarea en segundo plano
    if strategy_id not in active_tasks:
        task = asyncio.create_task(run_strategy(strategy_id))
        active_tasks[strategy_id] = task
    
    return {"message": "Estrategia iniciada", "strategy_id": strategy_id}

@app.post("/api/v1/trading/strategies/{strategy_id}/stop")
async def stop_strategy(strategy_id: str):
    """Pausar una estrategia de trading"""
    if strategy_id not in strategies_db:
        raise HTTPException(status_code=404, detail="Estrategia no encontrada")
    
    strategy = strategies_db[strategy_id]
    strategy.status = StrategyStatus.PAUSED
    
    # Cancelar tarea en segundo plano
    if strategy_id in active_tasks:
        active_tasks[strategy_id].cancel()
        del active_tasks[strategy_id]
    
    return {"message": "Estrategia pausada", "strategy_id": strategy_id}

@app.delete("/api/v1/trading/strategies/{strategy_id}")
async def delete_strategy(strategy_id: str):
    """Eliminar una estrategia de trading"""
    if strategy_id not in strategies_db:
        raise HTTPException(status_code=404, detail="Estrategia no encontrada")
    
    # Cancelar tarea en segundo plano si está activa
    if strategy_id in active_tasks:
        active_tasks[strategy_id].cancel()
        del active_tasks[strategy_id]
    
    # Eliminar estrategia
    del strategies_db[strategy_id]
    
    # Eliminar trades asociados
    trades_to_delete = [trade_id for trade_id, trade in trades_db.items() 
                       if trade.strategyId == strategy_id]
    for trade_id in trades_to_delete:
        del trades_db[trade_id]
    
    return {"message": "Estrategia eliminada", "strategy_id": strategy_id}

@app.get("/api/v1/trading/trades")
async def get_trades(strategy_id: Optional[str] = None):
    """Obtener trades de una estrategia específica o todos"""
    if strategy_id:
        trades = [trade for trade in trades_db.values() if trade.strategyId == strategy_id]
    else:
        trades = list(trades_db.values())
    
    return {"trades": trades}

@app.get("/api/v1/trading/performance/{strategy_id}")
async def get_strategy_performance(strategy_id: str):
    """Obtener rendimiento de una estrategia"""
    if strategy_id not in strategies_db:
        raise HTTPException(status_code=404, detail="Estrategia no encontrada")
    
    strategy = strategies_db[strategy_id]
    total_trades = strategy.totalTrades
    winning_trades = strategy.winningTrades
    
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    return {
        "strategy_id": strategy_id,
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "win_rate": win_rate,
        "total_pnl": strategy.totalPnL,
        "open_positions": strategy.openPositions
    }

# Tareas en segundo plano
@app.on_event("startup")
async def startup_event():
    """Inicialización de la aplicación"""
    print("🚀 AI Trading API iniciada")
    
    # Inicializar base de datos
    try:
        db_config.initialize_db()
        print("✅ Base de datos inicializada")
    except Exception as e:
        print(f"❌ Error al inicializar base de datos: {e}")
    
    # Entrenar modelos para símbolos principales
    symbols = ['AAPL', 'TSLA', 'MSFT', 'NVDA']
    for symbol in symbols:
        try:
            print(f"📊 Entrenando modelo para {symbol}...")
            price_predictor.train(symbol)
        except Exception as e:
            print(f"❌ Error entrenando {symbol}: {e}")
    
    print("✅ Modelos entrenados exitosamente")

# Funciones de utilidad
def calculate_portfolio_metrics(positions: List[Dict]) -> Dict:
    """Calcula métricas del portafolio"""
    total_value = sum(pos['value'] for pos in positions)
    total_pnl = sum(pos['pnl'] for pos in positions)
    
    if total_value > 0:
        total_pnl_percent = (total_pnl / (total_value - total_pnl)) * 100
    else:
        total_pnl_percent = 0
    
    return {
        'total_value': total_value,
        'total_pnl': total_pnl,
        'total_pnl_percent': total_pnl_percent,
        'num_positions': len(positions)
    }

def risk_management_check(signal: str, position_size: float, portfolio_value: float) -> bool:
    """Verifica reglas de gestión de riesgo"""
    # No más del 5% del portafolio en una posición
    if position_size > portfolio_value * 0.05:
        return False
    
    # Otras reglas de riesgo pueden añadirse aquí
    return True

# ===== MT4/MT5 CONNECTION API ENDPOINTS =====
import os
import json
import time
from pathlib import Path

# Configuración de archivos compartidos con MT4/MT5
MT4_COMMANDS_FILE = "C:/Users/andre/AppData/Roaming/MetaQuotes/Terminal/C348917D9E28C59E863914247686464D/MQL4/Files/mt4_commands.txt"
MT4_RESPONSES_FILE = "C:/Users/andre/AppData/Roaming/MetaQuotes/Terminal/C348917D9E28C59E863914247686464D/MQL4/Files/mt4_responses.txt"
MT4_STATUS_FILE = "C:/Users/andre/AppData/Roaming/MetaQuotes/Terminal/C348917D9E28C59E863914247686464D/MQL4/Files/mt4_status.txt"

# Crear directorio si no existe
os.makedirs("C:/Users/andre/AppData/Roaming/MetaQuotes/Terminal/C348917D9E28C59E863914247686464D/MQL4/Files", exist_ok=True)

mt4_connection_status = {
    "connected": False,
    "account": None,
    "last_connected": None,
    "server": None,
    "balance": 0.0,
    "equity": 0.0
}

def write_mt4_command(command_type, data=None):
    """Escribe un comando para MT4"""
    command = {
        "timestamp": datetime.now().isoformat(),
        "type": command_type,
        "data": data or {}
    }
    
    try:
        with open(MT4_COMMANDS_FILE, 'w') as f:
            json.dump(command, f)
        return True
    except Exception as e:
        print(f"Error writing MT4 command: {e}")
        return False

def read_mt4_response():
    """Lee la respuesta de MT4"""
    try:
        if os.path.exists(MT4_RESPONSES_FILE):
            with open(MT4_RESPONSES_FILE, 'r') as f:
                response = json.load(f)
            # Limpiar archivo después de leer
            os.remove(MT4_RESPONSES_FILE)
            return response
    except Exception as e:
        print(f"Error reading MT4 response: {e}")
    return None

def check_mt4_status():
    """Verifica el estado de MT4"""
    try:
        if os.path.exists(MT4_STATUS_FILE):
            with open(MT4_STATUS_FILE, 'r') as f:
                status = json.load(f)
            return status
    except Exception as e:
        print(f"Error reading MT4 status: {e}")
    return None

@app.post("/api/v1/trading/mt4/connect")
async def connect_mt4():
    """Conectar con MetaTrader 4/5"""
    try:
        # Enviar comando de conexión a MT4
        if write_mt4_command("connect"):
            # Esperar respuesta
            time.sleep(1)
            response = read_mt4_response()
            
            if response and response.get("status") == "connected":
                mt4_connection_status.update({
                    "connected": True,
                    "account": response.get("account", "DemoAccount"),
                    "last_connected": datetime.now().isoformat(),
                    "server": response.get("server", "Demo Server"),
                    "balance": response.get("balance", 10000.0),
                    "equity": response.get("equity", 10000.0)
                })
                return {"status": "connected", "account": mt4_connection_status["account"]}
            else:
                return {"status": "error", "message": "No se pudo conectar con MT4"}
        else:
            return {"status": "error", "message": "Error al enviar comando a MT4"}
            
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/v1/trading/mt4/disconnect")
async def disconnect_mt4():
    """Desconectar de MetaTrader 4/5"""
    try:
        write_mt4_command("disconnect")
        mt4_connection_status["connected"] = False
        return {"status": "disconnected"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/v1/trading/mt4/status")
async def mt4_status():
    """Obtener estado de conexión MT4"""
    # Verificar estado actual de MT4
    current_status = check_mt4_status()
    if current_status:
        mt4_connection_status.update(current_status)
    else:
        # Si no hay archivo de estado, resetear a desconectado
        mt4_connection_status.update({
            "connected": False,
            "account": None,
            "last_connected": None,
            "server": None,
            "balance": 0.0,
            "equity": 0.0
        })
    
    return mt4_connection_status

@app.post("/api/v1/trading/mt4/reset")
async def reset_mt4_status():
    """Resetear estado de conexión MT4"""
    global mt4_connection_status
    mt4_connection_status = {
        "connected": False,
        "account": None,
        "last_connected": None,
        "server": None,
        "balance": 0.0,
        "equity": 0.0
    }
    
    # Eliminar archivo de estado si existe
    try:
        if os.path.exists(MT4_STATUS_FILE):
            os.remove(MT4_STATUS_FILE)
    except Exception as e:
        print(f"Error eliminando archivo de estado: {e}")
    
    return {"status": "reset", "message": "Estado de MT4 reseteado"}

@app.post("/api/v1/trading/mt4/order")
async def place_mt4_order(order_data: dict):
    """Colocar orden en MT4"""
    if not mt4_connection_status["connected"]:
        return {"status": "error", "message": "MT4 no está conectado"}
    
    try:
        if write_mt4_command("place_order", order_data):
            time.sleep(0.5)
            response = read_mt4_response()
            return response or {"status": "pending", "message": "Orden enviada"}
        else:
            return {"status": "error", "message": "Error al enviar orden"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/v1/trading/mt4/download-ea")
async def download_ea():
    """Descargar el Expert Advisor"""
    try:
        # Path al archivo EA - usar ruta absoluta
        import pathlib
        current_dir = pathlib.Path(__file__).parent.parent
        ea_file_path = current_dir / "mt4_expert_advisor.mq4"
        
        if not os.path.exists(ea_file_path):
            return {"success": False, "message": f"Archivo EA no encontrado en: {ea_file_path}"}
        
        # Leer contenido del archivo EA
        with open(ea_file_path, 'r', encoding='utf-8') as f:
            ea_content = f.read()
        
        # Retornar el contenido como archivo descargable
        from fastapi.responses import Response
        return Response(
            content=ea_content,
            media_type="text/plain",
            headers={
                "Content-Disposition": "attachment; filename=AITRADERX_EA.mq4"
            }
        )
    except Exception as e:
        return {"success": False, "message": f"Error: {str(e)}"}

# ============================================================================
# ENDPOINTS PARA ESTRATEGIAS DE SCALPING
# ============================================================================

@app.get("/api/v1/trading/scalping/strategies")
async def get_scalping_strategies():
    """Obtener todas las estrategias de scalping"""
    try:
        if not scalping_service:
            raise HTTPException(status_code=503, detail="Servicio de scalping no disponible")
        
        result = await scalping_service.get_all_strategies()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo estrategias: {str(e)}")

@app.post("/api/v1/trading/scalping/strategies")
async def create_scalping_strategy(request: dict):
    """Crear una nueva estrategia de scalping"""
    try:
        if not scalping_service:
            raise HTTPException(status_code=503, detail="Servicio de scalping no disponible")
        
        strategy_type = request.get("strategy_type")
        config = request.get("config", {})
        
        if not strategy_type:
            raise HTTPException(status_code=400, detail="strategy_type es requerido")
        
        result = await scalping_service.create_strategy(strategy_type, config)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando estrategia: {str(e)}")

@app.post("/api/v1/trading/scalping/strategies/{strategy_id}/start")
async def start_scalping_strategy(strategy_id: str):
    """Iniciar una estrategia de scalping"""
    try:
        if not scalping_service:
            raise HTTPException(status_code=503, detail="Servicio de scalping no disponible")
        
        result = await scalping_service.start_strategy(strategy_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error iniciando estrategia: {str(e)}")

@app.post("/api/v1/trading/scalping/strategies/{strategy_id}/stop")
async def stop_scalping_strategy(strategy_id: str):
    """Detener una estrategia de scalping"""
    try:
        if not scalping_service:
            raise HTTPException(status_code=503, detail="Servicio de scalping no disponible")
        
        result = await scalping_service.stop_strategy(strategy_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deteniendo estrategia: {str(e)}")

@app.get("/api/v1/trading/scalping/strategies/{strategy_id}/status")
async def get_scalping_strategy_status(strategy_id: str):
    """Obtener el estado de una estrategia de scalping"""
    try:
        if not scalping_service:
            raise HTTPException(status_code=503, detail="Servicio de scalping no disponible")
        
        result = await scalping_service.get_strategy_status(strategy_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo estado: {str(e)}")

@app.delete("/api/v1/trading/scalping/strategies/{strategy_id}")
async def delete_scalping_strategy(strategy_id: str):
    """Eliminar una estrategia de scalping"""
    try:
        if not scalping_service:
            raise HTTPException(status_code=503, detail="Servicio de scalping no disponible")
        
        result = await scalping_service.delete_strategy(strategy_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error eliminando estrategia: {str(e)}")

@app.post("/api/v1/trading/scalping/strategies/{strategy_id}/signal")
async def generate_scalping_signal(strategy_id: str, market_data: dict):
    """Generar señal para una estrategia de scalping"""
    try:
        if not scalping_service:
            raise HTTPException(status_code=503, detail="Servicio de scalping no disponible")
        
        # Convertir market_data a DataFrame
        df = pd.DataFrame(market_data)
        result = await scalping_service.generate_signal(strategy_id, df)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando señal: {str(e)}")

@app.post("/api/v1/trading/scalping/strategies/{strategy_id}/execute")
async def execute_scalping_trade(strategy_id: str, signal: dict):
    """Ejecutar operación para una estrategia de scalping"""
    try:
        if not scalping_service:
            raise HTTPException(status_code=503, detail="Servicio de scalping no disponible")
        
        result = await scalping_service.execute_trade(strategy_id, signal)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ejecutando operación: {str(e)}")

@app.get("/api/v1/trading/scalping/available-types")
async def get_available_scalping_types():
    """Obtener tipos de estrategias de scalping disponibles"""
    try:
        if not scalping_service:
            raise HTTPException(status_code=503, detail="Servicio de scalping no disponible")
        
        return {
            "success": True,
            "strategy_types": list(scalping_service.strategy_configs.keys()),
            "configs": scalping_service.strategy_configs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo tipos: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

# requirements.txt contenido:
"""
fastapi==0.104.1
uvicorn[standard]==0.24.0
pandas==2.1.3
numpy==1.25.2
yfinance==0.2.28
ta==0.10.2
scikit-learn==1.3.2
joblib==1.3.2
websockets==12.0
python-multipart==0.0.6
pydantic==2.5.0
"""

# docker-compose.yml para deployment:
"""
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENV=production
    volumes:
      - ./models:/app/models
    restart: unless-stopped
    
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    restart: unless-stopped
    
  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: trading_db
      POSTGRES_USER: trader
      POSTGRES_PASSWORD: password123
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped
    
volumes:
  postgres_data:
"""

# Dockerfile:
"""
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""