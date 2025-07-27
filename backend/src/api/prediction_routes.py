"""
Prediction Routes for Day Trading System
"""
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime, timedelta
import logging

from services.prediction_service import PredictionService
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
        # Get database session from request state
        db_session = request.state.db if hasattr(request.state, 'db') else None
        
        if not db_session:
            # Mock response for testing
            return LimitsResponse(
                can_generate=True,
                remaining_predictions=10,
                max_predictions_per_day=10,
                has_active_prediction=False,
                plan_type="starter",
                analysis_type="rsi_only",
                timeframe="15M" if style == "day_trading" else "1H",
                duration_minutes=15 if style == "day_trading" else 60
            )
        
        prediction_service = PredictionService(db_session)
        limits_info = prediction_service.can_generate_prediction(current_user.id, style)
        
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
        # Get database session from request state
        db_session = request.state.db if hasattr(request.state, 'db') else None
        
        if not db_session:
            # Mock response for testing
            return JSONResponse(content={
                'success': True,
                'prediction': {
                    'id': 1,
                    'pair': request_data.pair,
                    'direction': 'up',
                    'current_price': 1.0925,
                    'target_price': 1.0935,
                    'confidence': 85.5,
                    'timeframe': '15M' if request_data.style == 'day_trading' else '1H',
                    'reasoning': f'{request_data.style.replace("_", " ").title()}: RSI indica sobreventa - señal de compra - RSI: 25.3',
                    'created_at': datetime.now().isoformat(),
                    'expires_at': (datetime.now() + timedelta(minutes=15)).isoformat(),
                    'time_remaining': 15.0
                },
                'limits': {
                    'can_generate': True,
                    'remaining_predictions': 9,
                    'max_predictions_per_day': 10,
                    'has_active_prediction': True,
                    'active_prediction_expires': (datetime.now() + timedelta(minutes=15)).isoformat(),
                    'plan_type': 'starter',
                    'analysis_type': 'rsi_only',
                    'timeframe': '15M' if request_data.style == 'day_trading' else '1H',
                    'duration_minutes': 15 if request_data.style == 'day_trading' else 60
                }
            })
        
        prediction_service = PredictionService(db_session)
        
        # Check if user can generate prediction
        limits_info = prediction_service.can_generate_prediction(current_user.id, request_data.style)
        if not limits_info['can_generate']:
            return JSONResponse(
                status_code=400,
                content={
                    'success': False,
                    'error': 'Cannot generate prediction',
                    'details': limits_info
                }
            )
        
        # Generate prediction
        result = await prediction_service.generate_prediction(
            user_id=current_user.id,
            pair=request_data.pair,
            brain_type=request_data.brain_type,
            style=request_data.style
        )
        
        if not result['success']:
            return JSONResponse(
                status_code=400,
                content=result
            )
        
        return JSONResponse(content=result)
        
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
        # Get database session from request state
        db_session = request.state.db if hasattr(request.state, 'db') else None
        
        if not db_session:
            return None
        
        prediction_service = PredictionService(db_session)
        active_prediction = prediction_service.get_active_prediction(current_user.id, style)
        
        if not active_prediction:
            return None
        
        return PredictionResponse(**active_prediction)
        
    except Exception as e:
        logger.error(f"Error getting active prediction: {e}")
        raise HTTPException(status_code=500, detail="Error getting active prediction")

@router.get("/history", response_model=List[PredictionHistoryResponse])
async def get_prediction_history(
    limit: int = 20,
    request: Request = None,
    current_user: User = Depends(get_current_user)
):
    """Get user's prediction history"""
    try:
        # Get database session from request state
        db_session = request.state.db if hasattr(request.state, 'db') else None
        
        if not db_session:
            return []
        
        prediction_service = PredictionService(db_session)
        predictions = prediction_service.get_user_predictions(current_user.id, limit)
        
        return [PredictionHistoryResponse(**prediction) for prediction in predictions]
        
    except Exception as e:
        logger.error(f"Error getting prediction history: {e}")
        raise HTTPException(status_code=500, detail="Error getting prediction history")

@router.get("/stats", response_model=UserStatsResponse)
async def get_user_stats(request: Request, current_user: User = Depends(get_current_user)):
    """Get user's prediction statistics"""
    try:
        # Get database session from request state
        db_session = request.state.db if hasattr(request.state, 'db') else None
        
        if not db_session:
            return UserStatsResponse(
                total_predictions=0,
                successful_predictions=0,
                success_rate=0.0,
                average_success_percentage=0.0,
                best_pair=None,
                total_predictions_today=0
            )
        
        prediction_service = PredictionService(db_session)
        stats = prediction_service.get_user_stats(current_user.id)
        
        return UserStatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Error getting user stats: {e}")
        raise HTTPException(status_code=500, detail="Error getting user stats")

@router.post("/complete-expired")
async def complete_expired_predictions(request: Request, current_user: User = Depends(get_current_user)):
    """Complete expired predictions (admin function)"""
    try:
        # Get database session from request state
        db_session = request.state.db if hasattr(request.state, 'db') else None
        
        if not db_session:
            return {"success": True, "message": "No database connection", "completed": 0}
        
        prediction_service = PredictionService(db_session)
        completed_count = await prediction_service.complete_expired_predictions()
        
        return {"success": True, "message": "Expired predictions completed", "completed": completed_count}
        
    except Exception as e:
        logger.error(f"Error completing expired predictions: {e}")
        raise HTTPException(status_code=500, detail="Error completing expired predictions") 