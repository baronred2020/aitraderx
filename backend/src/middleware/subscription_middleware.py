"""
Middleware para Verificación de Suscripciones
============================================
Middleware que verifica automáticamente los permisos de suscripción
"""

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
import logging
from datetime import datetime

from services.subscription_service import SubscriptionService

logger = logging.getLogger(__name__)

class SubscriptionMiddleware:
    """Middleware para verificar permisos de suscripción"""
    
    def __init__(self, subscription_service: SubscriptionService):
        self.subscription_service = subscription_service
        
        # Mapeo de endpoints a características requeridas
        self.endpoint_features = {
            # Endpoints de AI Tradicional
            "/api/predict-price": "traditional_ai",
            "/api/technical-analysis": "traditional_ai",
            "/api/fundamental-analysis": "traditional_ai",
            
            # Endpoints de Reinforcement Learning
            "/api/rl/status": "reinforcement_learning",
            "/api/rl/train": "reinforcement_learning",
            "/api/rl/predict": "reinforcement_learning",
            "/api/rl/performance": "reinforcement_learning",
            "/api/rl/evaluate": "reinforcement_learning",
            
            # Endpoints de LSTM
            "/api/predict-price": "lstm_predictions",  # Para predicciones > 7 días
            
            # Endpoints de Auto-Training
            "/api/auto-training": "auto_training",
            "/api/models/train": "auto_training",
            
            # Endpoints de Custom Models
            "/api/models/custom": "custom_models",
            "/api/models/create": "custom_models",
            
            # Endpoints de MT4
            "/api/mt4": "mt4_integration",
            "/api/mt4/connect": "mt4_integration",
            "/api/mt4/orders": "mt4_integration",
            
            # Endpoints de API Access
            "/api/v1": "api_access",
            "/api/external": "api_access",
        }
        
        # Endpoints que requieren verificación de límites
        self.limited_endpoints = {
            "/api/predict-price": "predictions",
            "/api/backtest": "backtests",
            "/api/alerts": "alerts",
            "/api/portfolios": "portfolios",
            "/api/indicators": "indicators",
        }
    
    async def __call__(self, request: Request, call_next):
        """Procesa la request y verifica permisos"""
        
        # Obtener user_id del header o query params
        user_id = self._extract_user_id(request)
        
        if not user_id:
            # Si no hay user_id, permitir acceso (para endpoints públicos)
            response = await call_next(request)
            return response
        
        # Verificar si el endpoint requiere verificación
        path = request.url.path
        feature = self._get_required_feature(path)
        
        if feature:
            try:
                # Verificar permisos
                allowed, message = self.subscription_service.check_user_permissions(
                    user_id=user_id,
                    feature=feature
                )
                
                if not allowed:
                    return JSONResponse(
                        status_code=403,
                        content={
                            "error": "Permiso denegado",
                            "message": message,
                            "feature": feature,
                            "upgrade_required": True
                        }
                    )
                
                # Actualizar métricas de uso
                self._update_usage_metrics(request, user_id, path)
                
            except Exception as e:
                logger.error(f"Error verificando permisos para {user_id}: {e}")
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": "Error interno del servidor",
                        "message": "Error verificando permisos"
                    }
                )
        
        # Continuar con la request
        response = await call_next(request)
        return response
    
    def _extract_user_id(self, request: Request) -> Optional[str]:
        """Extrae el user_id de la request"""
        # Intentar obtener de headers
        user_id = request.headers.get("X-User-ID")
        if user_id:
            return user_id
        
        # Intentar obtener de query params
        user_id = request.query_params.get("user_id")
        if user_id:
            return user_id
        
        # Intentar obtener de body (para POST requests)
        if request.method == "POST":
            try:
                body = request.json()
                if isinstance(body, dict) and "user_id" in body:
                    return body["user_id"]
            except:
                pass
        
        return None
    
    def _get_required_feature(self, path: str) -> Optional[str]:
        """Obtiene la característica requerida para un endpoint"""
        for endpoint, feature in self.endpoint_features.items():
            if path.startswith(endpoint):
                return feature
        return None
    
    def _update_usage_metrics(self, request: Request, user_id: str, path: str):
        """Actualiza las métricas de uso del usuario"""
        try:
            # Determinar qué métrica actualizar basado en el endpoint
            if "/api/predict-price" in path:
                self.subscription_service.update_usage_metrics(user_id, "predictions")
            elif "/api/backtest" in path:
                self.subscription_service.update_usage_metrics(user_id, "backtests")
            elif "/api/alerts" in path and request.method == "POST":
                self.subscription_service.update_usage_metrics(user_id, "alerts")
            elif "/api/rl/train" in path:
                self.subscription_service.update_usage_metrics(user_id, "rl_episodes")
            elif "/api/models/custom" in path:
                self.subscription_service.update_usage_metrics(user_id, "custom_models")
            else:
                # Métrica general de requests
                self.subscription_service.update_usage_metrics(user_id, "api_requests")
                
        except Exception as e:
            logger.error(f"Error actualizando métricas para {user_id}: {e}")

def create_subscription_middleware(subscription_service: SubscriptionService):
    """Factory para crear el middleware de suscripciones"""
    return SubscriptionMiddleware(subscription_service) 