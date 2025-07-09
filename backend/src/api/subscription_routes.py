"""
API Routes para Sistema de Suscripciones
========================================
Endpoints para gestionar planes, usuarios y verificar permisos
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging

from models.subscription import (
    SubscriptionPlan, UserSubscription, UsageMetrics, PlanType,
    SubscriptionStatus, SubscriptionUpgrade
)
from services.subscription_service import SubscriptionService

logger = logging.getLogger(__name__)

# Router para suscripciones
subscription_router = APIRouter(prefix="/api/subscriptions", tags=["subscriptions"])

# Instancia del servicio
subscription_service = SubscriptionService()

# Dependencia para obtener el servicio
def get_subscription_service() -> SubscriptionService:
    return subscription_service

# ============================================================================
# ENDPOINTS DE PLANES
# ============================================================================

@subscription_router.get("/plans", response_model=List[SubscriptionPlan])
async def get_all_plans():
    """Obtiene todos los planes disponibles"""
    try:
        plans = subscription_service.get_all_plans()
        return plans
    except Exception as e:
        logger.error(f"Error obteniendo planes: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@subscription_router.get("/plans/{plan_type}", response_model=SubscriptionPlan)
async def get_plan_by_type(plan_type: PlanType):
    """Obtiene un plan específico por tipo"""
    try:
        plan = subscription_service.get_plan_by_type(plan_type)
        if not plan:
            raise HTTPException(status_code=404, detail=f"Plan {plan_type} no encontrado")
        return plan
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo plan {plan_type}: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

# ============================================================================
# ENDPOINTS DE USUARIOS
# ============================================================================

@subscription_router.post("/users/{user_id}/subscribe", response_model=UserSubscription)
async def create_subscription(
    user_id: str,
    plan_type: PlanType,
    trial_days: int = Query(default=0, ge=0, le=30, description="Días de prueba")
):
    """Crea una nueva suscripción para un usuario"""
    try:
        # Verificar si ya tiene una suscripción activa
        existing_sub = subscription_service.get_user_subscription(user_id)
        if existing_sub:
            raise HTTPException(
                status_code=400, 
                detail="Usuario ya tiene una suscripción activa"
            )
        
        subscription = subscription_service.create_user_subscription(
            user_id=user_id,
            plan_type=plan_type,
            trial_days=trial_days
        )
        
        return subscription
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creando suscripción para {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@subscription_router.get("/users/{user_id}/subscription", response_model=UserSubscription)
async def get_user_subscription(user_id: str):
    """Obtiene la suscripción activa de un usuario"""
    try:
        subscription = subscription_service.get_user_subscription(user_id)
        if not subscription:
            raise HTTPException(
                status_code=404, 
                detail="Usuario sin suscripción activa"
            )
        return subscription
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo suscripción de {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@subscription_router.get("/users/{user_id}/plan", response_model=SubscriptionPlan)
async def get_user_plan(user_id: str):
    """Obtiene el plan actual de un usuario"""
    try:
        plan = subscription_service.get_user_plan(user_id)
        if not plan:
            raise HTTPException(
                status_code=404, 
                detail="Usuario sin plan activo"
            )
        return plan
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo plan de {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@subscription_router.post("/users/{user_id}/upgrade", response_model=UserSubscription)
async def upgrade_subscription(
    user_id: str,
    upgrade_request: SubscriptionUpgrade
):
    """Actualiza la suscripción de un usuario a un plan superior"""
    try:
        subscription = subscription_service.upgrade_user_subscription(
            user_id=user_id,
            new_plan_type=upgrade_request.target_plan
        )
        return subscription
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error actualizando suscripción de {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@subscription_router.delete("/users/{user_id}/subscription")
async def cancel_subscription(user_id: str):
    """Cancela la suscripción de un usuario"""
    try:
        success = subscription_service.cancel_user_subscription(user_id)
        if not success:
            raise HTTPException(
                status_code=404, 
                detail="Usuario sin suscripción activa para cancelar"
            )
        return {"message": "Suscripción cancelada exitosamente"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelando suscripción de {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

# ============================================================================
# ENDPOINTS DE PERMISOS
# ============================================================================

@subscription_router.post("/users/{user_id}/permissions/check")
async def check_user_permissions(
    user_id: str,
    feature: str = Query(..., description="Característica a verificar"),
    resource_count: int = Query(default=1, ge=1, description="Cantidad de recursos")
):
    """Verifica si un usuario tiene permisos para una característica"""
    try:
        allowed, message = subscription_service.check_user_permissions(
            user_id=user_id,
            feature=feature,
            resource_count=resource_count
        )
        
        return {
            "user_id": user_id,
            "feature": feature,
            "resource_count": resource_count,
            "allowed": allowed,
            "message": message
        }
    except Exception as e:
        logger.error(f"Error verificando permisos de {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@subscription_router.get("/users/{user_id}/permissions/features")
async def get_user_features(user_id: str):
    """Obtiene todas las características disponibles para un usuario"""
    try:
        plan = subscription_service.get_user_plan(user_id)
        if not plan:
            raise HTTPException(
                status_code=404, 
                detail="Usuario sin plan activo"
            )
        
        # Construir lista de características disponibles
        features = {
            "ai_capabilities": {
                "traditional_ai": plan.ai_capabilities.traditional_ai,
                "reinforcement_learning": plan.ai_capabilities.reinforcement_learning,
                "ensemble_ai": plan.ai_capabilities.ensemble_ai,
                "lstm_predictions": plan.ai_capabilities.lstm_predictions,
                "custom_models": plan.ai_capabilities.custom_models,
                "auto_training": plan.ai_capabilities.auto_training
            },
            "ui_features": {
                "advanced_charts": plan.ui_features.advanced_charts,
                "multiple_timeframes": plan.ui_features.multiple_timeframes,
                "rl_dashboard": plan.ui_features.rl_dashboard,
                "ai_monitor": plan.ui_features.ai_monitor,
                "mt4_integration": plan.ui_features.mt4_integration,
                "api_access": plan.ui_features.api_access,
                "custom_reports": plan.ui_features.custom_reports,
                "priority_support": plan.ui_features.priority_support
            },
            "api_limits": {
                "daily_requests": plan.api_limits.daily_requests,
                "prediction_days": plan.api_limits.prediction_days,
                "backtest_days": plan.api_limits.backtest_days,
                "trading_pairs": plan.api_limits.trading_pairs,
                "alerts_limit": plan.api_limits.alerts_limit,
                "portfolio_size": plan.api_limits.portfolio_size
            },
            "limits": {
                "max_indicators": plan.max_indicators,
                "max_predictions_per_day": plan.max_predictions_per_day,
                "max_backtests_per_month": plan.max_backtests_per_month,
                "max_portfolios": plan.max_portfolios
            }
        }
        
        return {
            "user_id": user_id,
            "plan_type": plan.plan_type,
            "features": features
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo características de {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

# ============================================================================
# ENDPOINTS DE MÉTRICAS
# ============================================================================

@subscription_router.get("/users/{user_id}/usage", response_model=UsageMetrics)
async def get_user_usage(user_id: str):
    """Obtiene las métricas de uso de un usuario"""
    try:
        usage = subscription_service.get_user_usage(user_id)
        return usage
    except Exception as e:
        logger.error(f"Error obteniendo uso de {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@subscription_router.post("/users/{user_id}/usage/update")
async def update_user_usage(
    user_id: str,
    metric: str = Query(..., description="Métrica a actualizar"),
    value: int = Query(default=1, ge=1, description="Valor a incrementar")
):
    """Actualiza las métricas de uso de un usuario"""
    try:
        subscription_service.update_usage_metrics(user_id, metric, value)
        return {"message": f"Métrica {metric} actualizada exitosamente"}
    except Exception as e:
        logger.error(f"Error actualizando uso de {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

# ============================================================================
# ENDPOINTS DE ADMINISTRACIÓN
# ============================================================================

@subscription_router.get("/admin/stats")
async def get_subscription_stats():
    """Obtiene estadísticas de suscripciones (solo admin)"""
    try:
        stats = subscription_service.get_subscription_stats()
        return stats
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@subscription_router.post("/admin/maintenance/check-expiry")
async def check_subscription_expiry():
    """Verifica y actualiza suscripciones expiradas (solo admin)"""
    try:
        subscription_service.check_subscription_expiry()
        return {"message": "Verificación de expiración completada"}
    except Exception as e:
        logger.error(f"Error verificando expiración: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@subscription_router.post("/admin/maintenance/reset-usage")
async def reset_daily_usage():
    """Resetea las métricas diarias de uso (solo admin)"""
    try:
        subscription_service.reset_daily_usage()
        return {"message": "Métricas diarias reseteadas"}
    except Exception as e:
        logger.error(f"Error reseteando métricas: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

# ============================================================================
# ENDPOINTS DE UTILIDAD
# ============================================================================

@subscription_router.get("/health")
async def subscription_health_check():
    """Verifica el estado del sistema de suscripciones"""
    try:
        # Verificar que el servicio está funcionando
        plans = subscription_service.get_all_plans()
        stats = subscription_service.get_subscription_stats()
        
        return {
            "status": "healthy",
            "plans_count": len(plans),
            "total_users": stats["total_users"],
            "active_subscriptions": stats["active_subscriptions"],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error en health check: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        } 