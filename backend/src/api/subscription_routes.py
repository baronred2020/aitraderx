"""
Rutas de Suscripciones
======================
Endpoints para gestión de suscripciones
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from datetime import datetime
import logging

from models.subscription import SubscriptionPlan, UserSubscription
from services.subscription_service import SubscriptionService
from services.user_service import UserService
from config.auth_config import get_current_user

# Configurar logging
logger = logging.getLogger(__name__)

# Router para suscripciones
subscription_router = APIRouter(prefix="/api/subscriptions", tags=["subscriptions"])

# Instancia del servicio
subscription_service = SubscriptionService()
user_service = UserService()

@subscription_router.get("/plans")
async def get_subscription_plans():
    """Obtiene todos los planes de suscripción disponibles"""
    try:
        plans = subscription_service.get_all_plans()
        return plans
    except Exception as e:
        logger.error(f"Error obteniendo planes: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@subscription_router.get("/me")
async def get_current_user_subscription(current_user: dict = Depends(get_current_user)):
    """Obtiene la suscripción del usuario actual"""
    try:
        username = current_user.get("username")
        if not username:
            raise HTTPException(status_code=401, detail="Usuario no autenticado")
        
        # Obtener usuario de la base de datos
        user = user_service.get_user_by_username(username)
        if not user:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        
        # Obtener suscripción del usuario
        subscription = user_service.get_user_subscription(user.user_id)
        
        if not subscription:
            # Si no tiene suscripción, crear una freemium por defecto
            subscription = user_service.create_subscription(user.user_id, "freemium")
        
        # Preparar respuesta
        subscription_response = {
            "id": subscription.subscription_id,
            "planType": subscription.plan_type,
            "status": subscription.status,
            "startDate": subscription.start_date.isoformat(),
            "endDate": subscription.end_date.isoformat(),
            "isTrial": subscription.is_trial
        }
        
        return {
            "subscription": subscription_response,
            "user": {
                "id": user.user_id,
                "username": user.username,
                "email": user.email,
                "firstName": user.first_name,
                "lastName": user.last_name,
                "role": user.role,
                "isActive": user.is_active
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo suscripción: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@subscription_router.post("/upgrade")
async def upgrade_subscription(
    plan_type: str,
    payment_method: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Actualiza la suscripción del usuario"""
    try:
        username = current_user.get("username")
        if not username:
            raise HTTPException(status_code=401, detail="Usuario no autenticado")
        
        # Obtener usuario
        user = user_service.get_user_by_username(username)
        if not user:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        
        # Cancelar suscripción actual si existe
        current_subscription = user_service.get_user_subscription(user.user_id)
        if current_subscription:
            current_subscription.status = "cancelled"
            user_service.db.commit()
        
        # Crear nueva suscripción
        new_subscription = user_service.create_subscription(user.user_id, plan_type, payment_method)
        if not new_subscription:
            raise HTTPException(status_code=500, detail="Error al crear suscripción")
        
        return {
            "message": "Suscripción actualizada exitosamente",
            "subscription": {
                "id": new_subscription.subscription_id,
                "planType": new_subscription.plan_type,
                "status": new_subscription.status,
                "startDate": new_subscription.start_date.isoformat(),
                "endDate": new_subscription.end_date.isoformat(),
                "isTrial": new_subscription.is_trial
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error actualizando suscripción: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@subscription_router.get("/usage")
async def get_usage_metrics(current_user: dict = Depends(get_current_user)):
    """Obtiene métricas de uso del usuario"""
    try:
        username = current_user.get("username")
        if not username:
            raise HTTPException(status_code=401, detail="Usuario no autenticado")
        
        # Obtener métricas de uso (simulado por ahora)
        usage_data = {
            "api_requests_today": 15,
            "predictions_made_today": 8,
            "backtests_run_today": 2,
            "alerts_created": 3,
            "ai_models_used": ["traditional_ai", "reinforcement_learning"],
            "rl_episodes_trained": 1250,
            "custom_models_created": 0,
            "trades_executed": 5,
            "portfolio_value": 125430.50,
            "profit_loss": 1234.75
        }
        
        return usage_data
        
    except Exception as e:
        logger.error(f"Error obteniendo métricas: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor") 