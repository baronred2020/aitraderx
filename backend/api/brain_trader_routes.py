from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from pydantic import BaseModel
import asyncio
from datetime import datetime

# Importar servicios (se crearán después)
# from ..services.brain_trader_service import BrainTraderService
# from ..utils.model_loader import ModelLoader

router = APIRouter(prefix="/brain-trader", tags=["Brain Trader"])

# Modelos Pydantic para las respuestas
class PredictionResponse(BaseModel):
    pair: str
    direction: str  # 'up', 'down', 'sideways'
    confidence: float
    target_price: float
    timeframe: str
    reasoning: str
    brain_type: str
    timestamp: datetime

class SignalResponse(BaseModel):
    pair: str
    type: str  # 'buy', 'sell', 'hold'
    strength: str  # 'strong', 'medium', 'weak'
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    brain_type: str
    timestamp: datetime

class TrendResponse(BaseModel):
    pair: str
    direction: str  # 'bullish', 'bearish', 'neutral'
    strength: float
    timeframe: str
    support: float
    resistance: float
    description: str
    brain_type: str
    timestamp: datetime

class ModelInfoResponse(BaseModel):
    brain_type: str
    pair: str
    style: str
    accuracy: float
    last_update: datetime
    status: str  # 'active', 'training', 'error'

# Instancia del servicio (se inicializará después)
# brain_trader_service = BrainTraderService()

@router.get("/predictions/{brain_type}")
async def get_predictions(
    brain_type: str,
    pair: str,
    style: str = "day_trading",
    limit: int = 10
) -> List[PredictionResponse]:
    """
    Obtener predicciones según el cerebro activo
    """
    try:
        # Validar brain_type
        valid_brain_types = ['brain_max', 'brain_ultra', 'brain_predictor', 'mega_mind']
        if brain_type not in valid_brain_types:
            raise HTTPException(status_code=400, detail=f"Brain type must be one of: {valid_brain_types}")
        
        # Validar pair
        valid_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'EURGBP', 'GBPJPY', 'EURJPY']
        if pair not in valid_pairs:
            raise HTTPException(status_code=400, detail=f"Pair must be one of: {valid_pairs}")
        
        # Validar style
        valid_styles = ['scalping', 'day_trading', 'swing_trading', 'position_trading']
        if style not in valid_styles:
            raise HTTPException(status_code=400, detail=f"Style must be one of: {valid_styles}")
        
        # Simular llamada al servicio (se implementará después)
        # predictions = await brain_trader_service.get_predictions(brain_type, pair, style)
        
        # Datos simulados por ahora
        import random
        predictions = []
        for i in range(min(limit, 5)):
            direction = random.choice(['up', 'down', 'sideways'])
            confidence = random.uniform(70, 95)
            base_price = 1.0925 if pair == 'EURUSD' else 1.2500
            target_price = base_price + (random.uniform(-0.01, 0.01))
            
            prediction = PredictionResponse(
                pair=pair,
                direction=direction,
                confidence=confidence,
                target_price=target_price,
                timeframe='1H',
                reasoning=f'Análisis técnico basado en {brain_type} - {direction.upper()}',
                brain_type=brain_type,
                timestamp=datetime.now()
            )
            predictions.append(prediction)
        
        return predictions
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting predictions: {str(e)}")

@router.get("/signals/{brain_type}")
async def get_signals(
    brain_type: str,
    pair: str,
    limit: int = 10
) -> List[SignalResponse]:
    """
    Obtener señales de trading
    """
    try:
        # Validaciones similares
        valid_brain_types = ['brain_max', 'brain_ultra', 'brain_predictor', 'mega_mind']
        if brain_type not in valid_brain_types:
            raise HTTPException(status_code=400, detail=f"Brain type must be one of: {valid_brain_types}")
        
        # Simular señales
        import random
        signals = []
        for i in range(min(limit, 5)):
            signal_type = random.choice(['buy', 'sell', 'hold'])
            strength = random.choice(['strong', 'medium', 'weak'])
            confidence = random.uniform(60, 90)
            base_price = 1.0925 if pair == 'EURUSD' else 1.2500
            entry_price = base_price + (random.uniform(-0.005, 0.005))
            
            signal = SignalResponse(
                pair=pair,
                type=signal_type,
                strength=strength,
                confidence=confidence,
                entry_price=entry_price,
                stop_loss=entry_price - 0.005,
                take_profit=entry_price + 0.015,
                brain_type=brain_type,
                timestamp=datetime.now()
            )
            signals.append(signal)
        
        return signals
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting signals: {str(e)}")

@router.get("/trends/{brain_type}")
async def get_trends(
    brain_type: str,
    pair: str,
    limit: int = 10
) -> List[TrendResponse]:
    """
    Obtener análisis de tendencias
    """
    try:
        # Validaciones
        valid_brain_types = ['brain_max', 'brain_ultra', 'brain_predictor', 'mega_mind']
        if brain_type not in valid_brain_types:
            raise HTTPException(status_code=400, detail=f"Brain type must be one of: {valid_brain_types}")
        
        # Simular tendencias
        import random
        trends = []
        for i in range(min(limit, 3)):
            direction = random.choice(['bullish', 'bearish', 'neutral'])
            strength = random.uniform(50, 100)
            base_price = 1.0925 if pair == 'EURUSD' else 1.2500
            support = base_price - 0.01
            resistance = base_price + 0.01
            
            trend = TrendResponse(
                pair=pair,
                direction=direction,
                strength=strength,
                timeframe='4H',
                support=support,
                resistance=resistance,
                description=f'Tendencia {direction} con soporte en {support:.4f}',
                brain_type=brain_type,
                timestamp=datetime.now()
            )
            trends.append(trend)
        
        return trends
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting trends: {str(e)}")

@router.get("/model-info/{brain_type}")
async def get_model_info(
    brain_type: str,
    pair: str,
    style: str = "day_trading"
) -> ModelInfoResponse:
    """
    Obtener información del modelo
    """
    try:
        # Validaciones
        valid_brain_types = ['brain_max', 'brain_ultra', 'brain_predictor', 'mega_mind']
        if brain_type not in valid_brain_types:
            raise HTTPException(status_code=400, detail=f"Brain type must be one of: {valid_brain_types}")
        
        # Simular información del modelo
        import random
        accuracy = random.uniform(80, 95)
        if brain_type == 'mega_mind':
            accuracy = random.uniform(90, 98)  # Mega Mind tiene mayor precisión
        
        model_info = ModelInfoResponse(
            brain_type=brain_type,
            pair=pair,
            style=style,
            accuracy=accuracy,
            last_update=datetime.now(),
            status='active'
        )
        
        return model_info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

@router.get("/available-brains")
async def get_available_brains() -> dict:
    """
    Obtener cerebros disponibles según el plan de suscripción
    """
    try:
        # Por ahora retornamos todos los cerebros
        # En el futuro esto dependerá del plan de suscripción del usuario
        return {
            "available_brains": [
                "brain_max",
                "brain_ultra", 
                "brain_predictor",
                "mega_mind"
            ],
            "default_brain": "brain_max"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting available brains: {str(e)}")

@router.get("/health")
async def health_check() -> dict:
    """
    Health check para el servicio Brain Trader
    """
    return {
        "status": "healthy",
        "service": "Brain Trader API",
        "timestamp": datetime.now(),
        "version": "1.0.0"
    } 