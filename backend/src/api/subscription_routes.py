"""
Subscription Routes for AI Trading System
"""
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
from datetime import datetime
import logging
import jwt
from config.auth_config import SECRET_KEY, ALGORITHM, get_current_user

from models.auth_models import User
from services.user_service import UserService
from config.database_config import db_config

logger = logging.getLogger(__name__)

subscription_router = APIRouter(prefix="/api/subscriptions", tags=["subscriptions"])

# Función para obtener usuario actual desde token
async def get_current_user_from_token(request: Request):
    """Obtener usuario actual desde token JWT"""
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return None
        
        token = auth_header.split(' ')[1]
        
        # Decodificar token JWT
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get('sub')
        
        if not username:
            return None
        
        # Obtener usuario de la base de datos
        user_service = UserService()
        user = user_service.get_user_by_username(username)
        
        return user
    except Exception as e:
        logger.error(f"Error decoding token: {e}")
        return None

@subscription_router.get("/me")
async def get_user_subscription(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """Get current user's subscription"""
    try:
        # Verificar conexión a la base de datos
        if not db_config.test_connection():
            raise HTTPException(
                status_code=503,
                detail="Database connection not available"
            )
        
        # Obtener usuario real desde token
        user = await get_current_user_from_token(request)
        if not user:
            raise HTTPException(status_code=401, detail="Usuario no autenticado")
        
        # Obtener información del plan
        plan_type = user.get('plan_type', 'starter')
        
        # Crear respuesta de suscripción real
        subscription_data = {
            "id": f"sub_{user.get('user_id')}",
            "planType": plan_type,
            "status": "active",
            "startDate": user.get('created_at', datetime.now()).isoformat(),
            "endDate": (datetime.now().replace(year=datetime.now().year + 1)).isoformat(),
            "isTrial": plan_type == "starter"
        }
        
        return {
            "subscription": subscription_data,
            "user": {
                "id": user.get('user_id'),
                "username": user.get('username'),
                "email": user.get('email'),
                "firstName": user.get('first_name'),
                "lastName": user.get('last_name'),
                "role": user.get('role', 'user'),
                "isActive": user.get('is_active', True)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user subscription: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor") 