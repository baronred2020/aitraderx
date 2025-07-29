"""
Subscription Routes for AI Trading System
"""
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import jwt
from config.auth_config import SECRET_KEY, ALGORITHM, get_current_user

from models.auth_models import User
from services.user_service import UserService
from services.subscription_mysql_service import SubscriptionMySQLService
from config.database_config import db_config

logger = logging.getLogger(__name__)

subscription_router = APIRouter(prefix="/api/subscriptions", tags=["subscriptions"])

# Instancia del servicio de suscripción
subscription_service = SubscriptionMySQLService("mysql://root:root@localhost:3306/trading_db")

@subscription_router.get("/plans")
async def get_all_plans():
    """Obtener todos los planes de suscripción disponibles"""
    try:
        # Obtener todos los planes desde la base de datos
        plans = subscription_service.get_all_plans()
        
        if not plans:
            raise HTTPException(status_code=404, detail="No se encontraron planes")
        
        # Convertir los planes de la base de datos al formato esperado por el frontend
        formatted_plans = []
        for plan in plans:
            # Convertir benefits de string JSON a lista
            benefits = []
            if hasattr(plan, 'benefits') and plan.benefits:
                try:
                    import json
                    benefits = json.loads(plan.benefits)
                except:
                    # Si no se puede parsear JSON, usar beneficios por defecto
                    if plan.plan_type == 'starter':
                        benefits = ["Dashboard básico", "Trading básico", "Portfolio básico", "Análisis básico", "Brain Trader básico", "Acceso a comunidad"]
                    elif plan.plan_type == 'trader':
                        benefits = ["Todo del plan Starter", "Sistema de alertas avanzado", "Comunidad de traders premium", "Monitoreo de agentes", "Análisis técnico avanzado", "Backtesting mejorado"]
                    elif plan.plan_type == 'expert':
                        benefits = ["Todo del plan Trader", "AI Monitor completo", "Reinforcement Learning", "Reportes avanzados", "Configuración de monitoreo", "Análisis fundamental"]
                    elif plan.plan_type == 'premium':
                        benefits = ["Todo del plan Expert", "Mega Mind Institutional", "Acceso completo a API", "Modelos personalizados", "Auto-entrenamiento", "Soporte prioritario"]
                    elif plan.plan_type == 'institutional':
                        benefits = ["Todo del plan Premium", "Acceso institucional completo", "API sin límites", "Modelos exclusivos", "Soporte 24/7", "Consultoría personalizada"]
            
            formatted_plan = {
                "id": plan.plan_id,
                "name": plan.name,
                "plan_type": plan.plan_type,
                "price": float(plan.price),
                "currency": plan.currency,
                "description": plan.description,
                "benefits": benefits,
                "ai_capabilities": {
                    "traditional_ai": plan.traditional_ai,
                    "reinforcement_learning": plan.reinforcement_learning,
                    "ensemble_ai": plan.ensemble_ai,
                    "lstm_predictions": plan.lstm_predictions,
                    "custom_models": plan.custom_models,
                    "auto_training": plan.auto_training
                },
                "api_limits": {
                    "daily_requests": plan.daily_requests,
                    "prediction_days": plan.prediction_days,
                    "backtest_days": plan.backtest_days,
                    "trading_pairs": plan.trading_pairs,
                    "alerts_limit": plan.alerts_limit,
                    "portfolio_size": plan.portfolio_size
                }
            }
            formatted_plans.append(formatted_plan)
        
        return formatted_plans
        
    except Exception as e:
        logger.error(f"Error getting plans: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

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