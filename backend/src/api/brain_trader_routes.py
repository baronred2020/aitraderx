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
    import sys
    import os
    # Agregar el directorio src al path si no está
    src_path = os.path.join(os.path.dirname(__file__), '..')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    from services.brain_trader_service import BrainTraderService
    brain_trader_service = BrainTraderService()
    logger.info("BrainTraderService importado correctamente")
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
            
        def _is_valid_signal_time(self, style: str) -> bool:
            """Verificar si es un momento válido para generar señal"""
            from datetime import datetime
            current_time = datetime.now()
            minutes = current_time.minute
            
            if style == 'day_trading':
                # Para day trading: intervalos de 15 minutos (00, 15, 30, 45)
                valid_minutes = [0, 15, 30, 45]
                return minutes in valid_minutes
            elif style == 'scalping':
                # Para scalping: intervalos de 5 minutos
                valid_minutes = list(range(0, 60, 5))
                return minutes in valid_minutes
            elif style == 'swing_trading':
                # Para swing trading: intervalos de 1 hora
                return minutes == 0
            elif style == 'position_trading':
                # Para position trading: intervalos de 4 horas
                return current_time.hour % 4 == 0 and minutes == 0
            
            return False
            
        def _get_next_valid_interval(self, style: str):
            """Obtener el próximo intervalo válido para generar señal"""
            from datetime import datetime, timedelta
            current_time = datetime.now()
            
            if style == 'day_trading':
                # Próximo intervalo de 15 minutos
                minutes = current_time.minute
                next_minute = ((minutes // 15) + 1) * 15
                if next_minute >= 60:
                    next_minute = 0
                    current_time = current_time.replace(hour=current_time.hour + 1)
                return current_time.replace(minute=next_minute, second=0, microsecond=0)
            elif style == 'scalping':
                # Próximo intervalo de 5 minutos
                minutes = current_time.minute
                next_minute = ((minutes // 5) + 1) * 5
                if next_minute >= 60:
                    next_minute = 0
                    current_time = current_time.replace(hour=current_time.hour + 1)
                return current_time.replace(minute=next_minute, second=0, microsecond=0)
            elif style == 'swing_trading':
                # Próximo intervalo de 1 hora
                return current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            elif style == 'position_trading':
                # Próximo intervalo de 4 horas
                next_hour = ((current_time.hour // 4) + 1) * 4
                if next_hour >= 24:
                    next_hour = 0
                    current_time = current_time + timedelta(days=1)
                return current_time.replace(hour=next_hour, minute=0, second=0, microsecond=0)
            
            return current_time + timedelta(minutes=15)
            
        def get_timeframe_for_style(self, style: str) -> str:
            """Obtener el timeframe para el estilo especificado"""
            timeframes = {
                'scalping': '5M',
                'day_trading': '15M',
                'swing_trading': '1H',
                'position_trading': '1D'
            }
            return timeframes.get(style, '15M')
            
        def get_duration_for_style(self, style: str) -> int:
            """Obtener la duración en minutos para el estilo especificado"""
            durations = {
                'scalping': 5,
                'day_trading': 15,
                'swing_trading': 60,
                'position_trading': 240
            }
            return durations.get(style, 15)
            
        async def generate_quality_signal(self, brain_type: str, pair: str, style: str) -> Dict[str, Any]:
            """Generar una señal de calidad para el par y estilo especificados"""
            from datetime import datetime
            
            # Verificar si es un momento válido para generar señal
            if not self._is_valid_signal_time(style):
                return {
                    "error": "No es un momento válido para generar señal",
                    "next_valid_interval": self._get_next_valid_interval(style).isoformat(),
                    "current_time": datetime.now().isoformat()
                }
            
            # Retornar mensaje de que no hay señales disponibles en modo mock
            return {
                "message": "No hay señales disponibles en modo desarrollo",
                "signal_quality": 0,
                "signal_type": "HOLD",
                "current_price": 0,
                "pair": pair,
                "style": style,
                "brain_type": brain_type,
                "timestamp": datetime.now().isoformat(),
                "timeframe": self.get_timeframe_for_style(style)
            }
            
        def _get_time_intervals(self, style: str):
            """Obtener intervalos de tiempo para el estilo especificado"""
            from datetime import datetime, timedelta
            intervals = []
            current_time = datetime.now()
            
            if style == 'day_trading':
                # Generar próximos 8 intervalos de 15 minutos
                for i in range(8):
                    minutes = current_time.minute
                    next_minute = ((minutes // 15) + i + 1) * 15
                    if next_minute >= 60:
                        next_minute = 0
                        current_time = current_time.replace(hour=current_time.hour + 1)
                    interval_time = current_time.replace(minute=next_minute, second=0, microsecond=0)
                    intervals.append(interval_time)
            else:
                # Para otros estilos, generar intervalos básicos
                for i in range(8):
                    interval_time = current_time + timedelta(minutes=15 * (i + 1))
                    intervals.append(interval_time)
            
            return intervals
    
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

@router.get("/predictions/{brain_type}")
async def get_predictions(
    brain_type: str,
    pair: str = "EURUSD",
    style: str = "day_trading",
    limit: int = 5,
    plan_type: str = "starter"
) -> List[PredictionResponse]:
    """Obtiene predicciones del cerebro especificado (método original)"""
    try:
        predictions = await brain_trader_service.get_predictions(brain_type, pair, style, limit, plan_type)
        return predictions
    except Exception as e:
        logger.error(f"Error getting predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predictions/{brain_type}/with-intervals")
async def get_predictions_with_intervals(
    brain_type: str,
    pair: str = "EURUSD",
    style: str = "day_trading",
    limit: int = 5,
    plan_type: str = "starter"
) -> List[PredictionResponse]:
    """Obtener predicciones respetando intervalos de tiempo específicos"""
    try:
        logger.info(f"Obteniendo predicciones con intervalos para {brain_type} - {pair} - {style}")
        
        predictions = await brain_trader_service.get_predictions_with_intervals(
            brain_type=brain_type,
            pair=pair,
            style=style,
            limit=limit,
            plan_type=plan_type
        )
        
        logger.info(f"Predicciones con intervalos obtenidas: {len(predictions)}")
        return predictions
        
    except Exception as e:
        logger.error(f"Error obteniendo predicciones con intervalos: {e}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo predicciones con intervalos: {str(e)}")

@router.get("/predictions/{brain_type}/next-interval")
async def get_next_prediction_time(
    brain_type: str,
    style: str = "day_trading"
) -> Dict[str, Any]:
    """Obtener información sobre el próximo momento válido para predicción"""
    try:
        logger.info(f"Obteniendo próximo tiempo de predicción para {brain_type} - {style}")
        
        next_time_info = await brain_trader_service.get_next_prediction_time(style)
        
        logger.info(f"Información de próximo intervalo obtenida")
        return next_time_info
        
    except Exception as e:
        logger.error(f"Error obteniendo próximo tiempo de predicción: {e}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo próximo tiempo de predicción: {str(e)}")

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
        
        # Intentar obtener señales reales del servicio
        try:
            signals = await brain_trader_service.get_signals(brain_type, pair, limit)
            return signals
        except Exception as e:
            logger.error(f"Error getting real signals: {e}")
            # Retornar lista vacía en lugar de señales mock
            return []
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting signals: {str(e)}")

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

@router.post("/signals/{brain_type}/generate")
async def generate_signal(
    brain_type: str,
    pair: str = "EURUSD",
    style: str = "day_trading"
) -> Dict[str, Any]:
    """Generar una señal manual en el intervalo de tiempo correcto"""
    try:
        # Validar parámetros
        valid_brain_types = ['brain_max', 'brain_ultra', 'brain_predictor', 'mega_mind']
        if brain_type not in valid_brain_types:
            raise HTTPException(status_code=400, detail=f"Brain type must be one of: {valid_brain_types}")
        
        valid_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
        if pair not in valid_pairs:
            raise HTTPException(status_code=400, detail=f"Pair must be one of: {valid_pairs}")
        
        valid_styles = ['scalping', 'day_trading', 'swing_trading', 'position_trading']
        if style not in valid_styles:
            raise HTTPException(status_code=400, detail=f"Style must be one of: {valid_styles}")
        
        # Verificar si es momento válido para generar señal
        if not brain_trader_service._is_valid_signal_time(style):
            current_time = datetime.now()
            next_interval = brain_trader_service._get_next_valid_interval(style)
            
            return {
                "success": False,
                "message": f"No es momento de generar señal. Próximo intervalo: {next_interval.strftime('%H:%M')}",
                "current_time": current_time.strftime('%H:%M:%S'),
                "next_interval": next_interval.strftime('%H:%M'),
                "style": style,
                "timeframe": brain_trader_service.get_timeframe_for_style(style)
            }
        
        # Generar señal de calidad
        signal_result = await brain_trader_service.generate_quality_signal(
            brain_type, pair, style
        )
        
        # Verificar si hay error en la generación
        if "error" in signal_result:
            return {
                "success": False,
                "message": signal_result["error"],
                "next_valid_interval": signal_result.get("next_valid_interval"),
                "current_time": signal_result.get("current_time"),
                "style": style,
                "timeframe": brain_trader_service.get_timeframe_for_style(style)
            }
        
        # Verificar si la señal no cumple el umbral de calidad
        if "message" in signal_result and "signal_quality" in signal_result:
            # Extraer datos del objeto signal si existe
            signal = signal_result.get("signal")
            signal_type = signal.type if signal else "unknown"
            current_price = signal.entry_price if signal else 0.0
            
            return {
                "success": False,
                "message": signal_result["message"],
                "signal_quality": signal_result["signal_quality"],
                "signal_type": signal_type,
                "current_price": current_price,
                "threshold": signal_result.get("threshold", 70),
                "style": style,
                "timeframe": brain_trader_service.get_timeframe_for_style(style)
            }
        
        # Señal exitosa
        if "signal_quality" in signal_result and signal_result["signal_quality"] >= 70:
            # Extraer datos del objeto signal
            signal = signal_result.get("signal")
            if signal:
                return {
                    "success": True,
                    "signal_type": signal.type,
                    "signal_quality": signal_result["signal_quality"],
                    "current_price": signal.entry_price,
                    "stop_loss": signal.stop_loss,
                    "take_profit": signal.take_profit,
                    "pair": signal.pair,
                    "style": style,
                    "brain_type": signal.brain_type,
                    "timestamp": signal.timestamp,
                    "timeframe": brain_trader_service.get_timeframe_for_style(style),
                    "generated_at": datetime.now().isoformat(),
                    "next_interval": brain_trader_service._get_next_valid_interval(style).strftime('%H:%M')
                }
            else:
                return {
                    "success": False,
                    "message": "Error: No se pudo generar la señal",
                    "signal_quality": signal_result.get("signal_quality", 0),
                    "next_interval": brain_trader_service._get_next_valid_interval(style).strftime('%H:%M'),
                    "style": style,
                    "timeframe": brain_trader_service.get_timeframe_for_style(style)
                }
        else:
            # Obtener el score real de la señal
            actual_quality = signal_result.get("signal_quality", 0)
            return {
                "success": False,
                "message": f"Señal de baja calidad (Score: {actual_quality:.1f}%). Intente en el próximo intervalo.",
                "signal_quality": actual_quality,
                "next_interval": brain_trader_service._get_next_valid_interval(style).strftime('%H:%M'),
                "style": style,
                "timeframe": brain_trader_service.get_timeframe_for_style(style)
            }
            
    except Exception as e:
        logger.error(f"Error generating signal: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/signals/{brain_type}/intervals")
async def get_signal_intervals(
    brain_type: str,
    style: str = "day_trading"
) -> Dict[str, Any]:
    """Obtener información sobre los intervalos de señales"""
    try:
        # Validar parámetros
        valid_brain_types = ['brain_max', 'brain_ultra', 'brain_predictor', 'mega_mind']
        if brain_type not in valid_brain_types:
            raise HTTPException(status_code=400, detail=f"Brain type must be one of: {valid_brain_types}")
        
        valid_styles = ['scalping', 'day_trading', 'swing_trading', 'position_trading']
        if style not in valid_styles:
            raise HTTPException(status_code=400, detail=f"Style must be one of: {valid_styles}")
        
        # Obtener información básica primero
        current_time = datetime.now()
        
        try:
            # Obtener intervalos de tiempo
            intervals = brain_trader_service._get_time_intervals(style)
            is_valid_time = brain_trader_service._is_valid_signal_time(style)
            next_interval = brain_trader_service._get_next_valid_interval(style)
            
            return {
                "style": style,
                "timeframe": brain_trader_service.get_timeframe_for_style(style),
                "current_time": current_time.strftime('%H:%M:%S'),
                "is_valid_time": is_valid_time,
                "next_interval": next_interval.strftime('%H:%M'),
                "upcoming_intervals": [interval.strftime('%H:%M') for interval in intervals[:5]],
                "duration_minutes": brain_trader_service.get_duration_for_style(style)
            }
        except Exception as interval_error:
            logger.error(f"Error calculating intervals: {interval_error}")
            # Fallback con valores básicos
            return {
                "style": style,
                "timeframe": "15M",
                "current_time": current_time.strftime('%H:%M:%S'),
                "is_valid_time": True,
                "next_interval": "00:15",
                "upcoming_intervals": ["00:15", "00:30", "00:45", "01:00", "01:15"],
                "duration_minutes": 15
            }
        
    except Exception as e:
        logger.error(f"Error getting signal intervals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check para Brain Trader"""
    return {
        "status": "healthy",
        "service": "brain_trader",
        "timestamp": datetime.now().isoformat()
    }

@router.post("/signals/{brain_type}/test-generate")
async def test_generate_signal(
    brain_type: str,
    pair: str = "EURUSD",
    style: str = "day_trading"
) -> Dict[str, Any]:
    """Endpoint de prueba para generar señal sin validación de tiempo"""
    try:
        logger.info(f"Generando señal de prueba para {pair} con {brain_type}")
        
        # Generar señal sin validación de tiempo
        signal_result = await brain_trader_service.generate_quality_signal(brain_type, pair, style)
        
        logger.info(f"Resultado de señal de prueba: {signal_result}")
        
        return {
            "success": True,
            "test_mode": True,
            "signal_result": signal_result,
            "message": "Señal generada en modo de prueba"
        }
        
    except Exception as e:
        logger.error(f"Error en señal de prueba: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 