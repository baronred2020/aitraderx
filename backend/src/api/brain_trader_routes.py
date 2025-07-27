"""
Brain Trader API Routes
=======================
Rutas para el sistema de Brain Trader
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from datetime import datetime
import asyncio
import logging

# Configurar logging
logger = logging.getLogger(__name__)

# Importar el servicio Brain Trader
try:
    from services.brain_trader_service import BrainTraderService
    brain_trader_service = BrainTraderService()
except ImportError as e:
    logger.error(f"Error importing BrainTraderService: {e}")
    # Crear un servicio mock para desarrollo
    class MockBrainTraderService:
        def __init__(self):
            pass
            
        async def get_predictions(self, brain_type: str, pair: str, style: str, limit: int, plan_type: str = 'starter'):
            return []
            
        async def get_signals(self, brain_type: str, pair: str, limit: int):
            return []
            
        async def get_trends(self, brain_type: str, pair: str, limit: int):
            return []
    
    brain_trader_service = MockBrainTraderService()

# Crear router
router = APIRouter(prefix="/api/v1/brain-trader", tags=["Brain Trader"])

# Modelos Pydantic para requests/responses
class PredictionResponse(BaseModel):
    pair: str
    direction: str
    confidence: float
    target_price: float
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

@router.get("/available-brains")
async def get_available_brains():
    """Obtiene los cerebros disponibles"""
    return {
        "available_brains": [
            "brain_max",
            "brain_ultra", 
            "brain_predictor",
            "mega_mind"
        ],
        "default_brain": "brain_max"
    }

@router.get("/predictions/{brain_type}")
async def get_predictions(
    brain_type: str,
    pair: str = "EURUSD",
    style: str = "day_trading",
    limit: int = 5,
    plan_type: str = "starter"
) -> List[PredictionResponse]:
    """Obtiene predicciones del cerebro especificado"""
    try:
        predictions = await brain_trader_service.get_predictions(brain_type, pair, style, limit, plan_type)
        return predictions
    except Exception as e:
        logger.error(f"Error getting predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/signals/{brain_type}")
async def get_signals(
    brain_type: str,
    pair: str = "EURUSD",
    limit: int = 5
) -> List[SignalResponse]:
    """Obtiene señales del cerebro especificado"""
    try:
        signals = await brain_trader_service.get_signals(brain_type, pair, limit)
        return signals
    except Exception as e:
        logger.error(f"Error getting signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trends/{brain_type}")
async def get_trends(
    brain_type: str,
    pair: str = "EURUSD",
    limit: int = 3
) -> List[TrendResponse]:
    """Obtiene tendencias del cerebro especificado"""
    try:
        trends = await brain_trader_service.get_trends(brain_type, pair, limit)
        return trends
    except Exception as e:
        logger.error(f"Error getting trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check para Brain Trader"""
    return {
        "status": "healthy",
        "service": "brain_trader",
        "timestamp": datetime.now().isoformat()
    } 