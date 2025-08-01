"""
Prediction Routes for Day Trading System
"""
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime, timedelta
import logging
import numpy as np

from services.prediction_service import PredictionService
from config.database_config import db_config

# Función temporal para obtener usuario actual
async def get_current_user(request: Request):
    """Función temporal para obtener usuario actual"""
    # Por ahora, usar un usuario de prueba
    from models.auth_models import User
    return User(id=1, username="test_user", email="test@example.com", plan_type="starter")
from models.auth_models import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/predictions", tags=["predictions"])

# Pydantic models for request/response
class GeneratePredictionRequest(BaseModel):
    pair: str
    brain_type: str = "brain_max"
    style: str = "day_trading"

class PredictionResponse(BaseModel):
    id: int
    pair: str
    direction: str
    current_price: float
    target_price: float
    confidence: float
    timeframe: str
    reasoning: str
    brain_type: str  # ✅ Agregada propiedad brain_type
    created_at: str
    expires_at: str
    time_remaining: Optional[float] = None

class PredictionHistoryResponse(BaseModel):
    id: int
    pair: str
    direction: str
    current_price: float
    target_price: float
    confidence: float
    timeframe: str
    reasoning: str
    brain_type: str  # ✅ Agregada propiedad brain_type
    created_at: str
    expires_at: str
    is_completed: bool
    actual_price_at_expiry: Optional[float] = None
    prediction_success: Optional[bool] = None
    success_percentage: Optional[float] = None

class UserStatsResponse(BaseModel):
    total_predictions: int
    successful_predictions: int
    success_rate: float
    average_success_percentage: float
    best_pair: Optional[str]
    total_predictions_today: int

class RealMetricsResponse(BaseModel):
    total_predictions: int
    successful_predictions: int
    win_rate: float
    precision: float
    average_confidence: float
    average_success_percentage: float
    best_pair: Optional[str]
    best_brain_type: Optional[str]
    recent_performance: List[Dict[str, Any]]
    metrics_by_pair: Dict[str, Dict[str, Any]]
    metrics_by_brain: Dict[str, Dict[str, Any]]

class LimitsResponse(BaseModel):
    can_generate: bool
    remaining_predictions: int
    max_predictions_per_day: int
    has_active_prediction: bool
    active_prediction_expires: Optional[str] = None
    plan_type: str
    analysis_type: str
    timeframe: str
    duration_minutes: int

@router.get("/limits", response_model=LimitsResponse)
async def get_prediction_limits(
    style: str = "day_trading",
    request: Request = None, 
    current_user: User = Depends(get_current_user)
):
    """Get user's prediction limits and status"""
    try:
        # Verificar conexión a la base de datos
        if not db_config.test_connection():
            raise HTTPException(
                status_code=503, 
                detail="Database connection not available"
            )
        
        prediction_service = PredictionService()
        # Usar el plan_type del usuario actual, por defecto 'starter'
        plan_type = getattr(current_user, 'plan_type', 'starter')
        user_id = getattr(current_user, 'user_id', '4dabfd30-483d-4fa0-a8d0-bd151a46340f')
        limits_info = await prediction_service.can_generate_prediction(user_id, style, plan_type)
        
        return LimitsResponse(**limits_info)
        
    except Exception as e:
        logger.error(f"Error getting prediction limits: {e}")
        raise HTTPException(status_code=500, detail="Error getting prediction limits")

@router.post("/generate", response_model=Dict[str, Any])
async def generate_prediction(
    request_data: GeneratePredictionRequest,
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """Generate a new prediction for the user"""
    try:
        # Verificar conexión a la base de datos
        if not db_config.test_connection():
            raise HTTPException(
                status_code=503, 
                detail="Database connection not available"
            )
        
        prediction_service = PredictionService()
        
        # Check if user can generate prediction
        plan_type = getattr(current_user, 'plan_type', 'starter')
        user_id = getattr(current_user, 'user_id', '4dabfd30-483d-4fa0-a8d0-bd151a46340f')
        
        # Verificar si puede generar predicción
        can_generate_result = await prediction_service.can_generate_prediction(user_id, request_data.style, plan_type)
        if not can_generate_result.get('can_generate', False):
            return JSONResponse(
                status_code=400,
                content={
                    'success': False,
                    'error': 'Cannot generate prediction - limit reached',
                    'details': {
                        'plan_type': plan_type,
                        'style': request_data.style
                    }
                }
            )
        
        # Generar predicción real usando el servicio
        result = await prediction_service.generate_prediction(
            user_id=user_id,
            pair=request_data.pair,
            brain_type=request_data.brain_type,
            style=request_data.style
        )
        
        if not result:
            return JSONResponse(
                status_code=500,
                content={
                    'success': False,
                    'error': 'Failed to generate prediction'
                }
            )
        
        # Obtener límites actualizados después de generar predicción
        updated_limits = await prediction_service.can_generate_prediction(user_id, request_data.style, plan_type)
        
        return JSONResponse(content={
            'success': True,
            'prediction': result,
            'limits': updated_limits
        })
        
    except Exception as e:
        logger.error(f"Error generating prediction: {e}")
        raise HTTPException(status_code=500, detail="Error generating prediction")

@router.get("/active", response_model=Optional[PredictionResponse])
async def get_active_prediction(
    style: str = "day_trading",
    request: Request = None,
    current_user: User = Depends(get_current_user)
):
    """Get user's active prediction"""
    try:
        # Verificar conexión a la base de datos
        if not db_config.test_connection():
            raise HTTPException(
                status_code=503, 
                detail="Database connection not available"
            )
        
        prediction_service = PredictionService()
        # Por ahora, retornar None ya que no hay predicciones activas implementadas
        # TODO: Implementar lógica de predicciones activas
        return None
        
    except Exception as e:
        logger.error(f"Error getting active prediction: {e}")
        raise HTTPException(status_code=500, detail="Error getting active prediction")

@router.get("/history", response_model=List[Dict[str, Any]])
async def get_prediction_history(
    limit: int = 20,
    request: Request = None,
    current_user: User = Depends(get_current_user)
):
    """Get user's prediction history"""
    try:
        # Verificar conexión a la base de datos
        if not db_config.test_connection():
            raise HTTPException(
                status_code=503, 
                detail="Database connection not available"
            )
        
        prediction_service = PredictionService()
        # Usar el UUID del usuario
        user_id = getattr(current_user, 'user_id', '4dabfd30-483d-4fa0-a8d0-bd151a46340f')
        history = await prediction_service.get_prediction_history(user_id, limit)
        
        return history
        
    except Exception as e:
        logger.error(f"Error getting prediction history: {e}")
        raise HTTPException(status_code=500, detail="Error getting prediction history")

@router.get("/stats", response_model=UserStatsResponse)
async def get_user_stats(request: Request, current_user: User = Depends(get_current_user)):
    """Get user's prediction statistics"""
    try:
        # Verificar conexión a la base de datos
        if not db_config.test_connection():
            raise HTTPException(
                status_code=503, 
                detail="Database connection not available"
            )
        
        prediction_service = PredictionService()
        # Usar el UUID del usuario
        user_id = getattr(current_user, 'user_id', '4dabfd30-483d-4fa0-a8d0-bd151a46340f')
        stats = await prediction_service.get_user_stats(user_id)
        
        return UserStatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Error getting user stats: {e}")
        raise HTTPException(status_code=500, detail="Error getting user stats")

@router.get("/real-metrics", response_model=RealMetricsResponse)
async def get_real_metrics(
    brain_type: Optional[str] = None,
    pair: Optional[str] = None,
    style: Optional[str] = None,
    request: Request = None,
    current_user: User = Depends(get_current_user)
):
    """Get real metrics based on completed predictions"""
    try:
        # Verificar conexión a la base de datos
        if not db_config.test_connection():
            raise HTTPException(
                status_code=503, 
                detail="Database connection not available"
            )
        
        prediction_service = PredictionService()
        # Usar el UUID del usuario
        user_id = getattr(current_user, 'user_id', '4dabfd30-483d-4fa0-a8d0-bd151a46340f')
        
        # Obtener métricas reales
        real_metrics = await prediction_service.get_real_metrics(
            user_id, brain_type, pair, style
        )
        
        if not real_metrics:
            # Retornar métricas vacías si no hay datos
            return RealMetricsResponse(
                total_predictions=0,
                successful_predictions=0,
                win_rate=0.0,
                precision=0.0,
                average_confidence=0.0,
                average_success_percentage=0.0,
                best_pair=None,
                best_brain_type=None,
                recent_performance=[],
                metrics_by_pair={},
                metrics_by_brain={}
            )
        
        return RealMetricsResponse(**real_metrics)
        
    except Exception as e:
        logger.error(f"Error getting real metrics: {e}")
        raise HTTPException(status_code=500, detail="Error getting real metrics")

@router.post("/complete-expired-with-real-results")
async def complete_expired_predictions_with_real_results(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """Complete expired predictions with real market results"""
    try:
        # Verificar conexión a la base de datos
        if not db_config.test_connection():
            raise HTTPException(
                status_code=503, 
                detail="Database connection not available"
            )
        
        prediction_service = PredictionService()
        # Usar el UUID del usuario
        user_id = getattr(current_user, 'user_id', '4dabfd30-483d-4fa0-a8d0-bd151a46340f')
        
        # Completar predicciones expiradas con resultados reales
        result = await prediction_service.complete_expired_predictions_with_real_results(user_id)
        
        return {
            "message": "Expired predictions completed with real results",
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Error completing expired predictions: {e}")
        raise HTTPException(status_code=500, detail="Error completing expired predictions")

@router.post("/complete-expired")
async def complete_expired_predictions(request: Request, current_user: User = Depends(get_current_user)):
    """Complete expired predictions for the current user"""
    try:
        # Verificar conexión a la base de datos
        if not db_config.test_connection():
            raise HTTPException(
                status_code=503, 
                detail="Database connection not available"
            )
        
        prediction_service = PredictionService()
        # ✅ Usar el UUID del usuario
        user_id = getattr(current_user, 'user_id', '4dabfd30-483d-4fa0-a8d0-bd151a46340f')
        result = await prediction_service.complete_expired_predictions(user_id)
        
        return result
        
    except Exception as e:
        logger.error(f"Error completing expired predictions: {e}")
        raise HTTPException(status_code=500, detail="Error completing expired predictions") 

@router.post("/reset-daily", response_model=Dict[str, Any])
async def reset_daily_predictions_manual(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """Reset daily predictions counter (admin only)"""
    try:
        # Verificar conexión a la base de datos
        if not db_config.test_connection():
            raise HTTPException(
                status_code=503, 
                detail="Database connection not available"
            )
        
        # Verificar si el usuario es administrador
        # Por ahora, permitir a todos los usuarios para pruebas
        # TODO: Implementar verificación real de rol de administrador
        logger.info(f"Usuario solicitando reinicio: {getattr(current_user, 'username', 'unknown')}")
        
        # Importar y ejecutar el script de reinicio
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent.parent))
        
        from reset_daily_predictions import reset_daily_predictions
        
        success = reset_daily_predictions()
        
        if success:
            return {
                "success": True,
                "message": "Reinicio diario de predicciones ejecutado exitosamente",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=500, 
                detail="Error ejecutando el reinicio diario"
            )
        
    except Exception as e:
        logger.error(f"Error in manual reset: {e}")
        raise HTTPException(status_code=500, detail="Error en reinicio manual") 