"""
Servicio de Gestión de Suscripciones
====================================
Servicio para gestionar planes, usuarios y verificar permisos
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging

from models.subscription import (
    SubscriptionPlan, UserSubscription, UsageMetrics, PlanType,
    SubscriptionStatus, DEFAULT_PLANS
)

logger = logging.getLogger(__name__)

class SubscriptionService:
    """Servicio para gestionar suscripciones y permisos"""
    
    def __init__(self, data_dir: str = "data/subscriptions"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Archivos de datos
        self.plans_file = self.data_dir / "plans.json"
        self.subscriptions_file = self.data_dir / "subscriptions.json"
        self.usage_file = self.data_dir / "usage.json"
        
        # Cache en memoria
        self._plans_cache: Dict[str, SubscriptionPlan] = {}
        self._subscriptions_cache: Dict[str, UserSubscription] = {}
        self._usage_cache: Dict[str, UsageMetrics] = {}
        
        # Inicializar
        self._load_data()
    
    def _load_data(self):
        """Carga datos desde archivos"""
        try:
            # Cargar planes
            if self.plans_file.exists():
                with open(self.plans_file, 'r') as f:
                    plans_data = json.load(f)
                    for plan_data in plans_data.values():
                        plan = SubscriptionPlan(**plan_data)
                        self._plans_cache[plan.plan_id] = plan
            else:
                # Crear planes por defecto
                self._create_default_plans()
            
            # Cargar suscripciones
            if self.subscriptions_file.exists():
                with open(self.subscriptions_file, 'r') as f:
                    subs_data = json.load(f)
                    for sub_data in subs_data.values():
                        sub = UserSubscription(**sub_data)
                        self._subscriptions_cache[sub.subscription_id] = sub
            
            # Cargar métricas de uso
            if self.usage_file.exists():
                with open(self.usage_file, 'r') as f:
                    usage_data = json.load(f)
                    for usage_data_item in usage_data.values():
                        usage = UsageMetrics(**usage_data_item)
                        self._usage_cache[usage.user_id] = usage
                        
        except Exception as e:
            logger.error(f"Error cargando datos de suscripciones: {e}")
    
    def _save_data(self):
        """Guarda datos a archivos"""
        try:
            # Guardar planes
            plans_data = {plan.plan_id: plan.dict() for plan in self._plans_cache.values()}
            with open(self.plans_file, 'w') as f:
                json.dump(plans_data, f, indent=2, default=str)
            
            # Guardar suscripciones
            subs_data = {sub.subscription_id: sub.dict() for sub in self._subscriptions_cache.values()}
            with open(self.subscriptions_file, 'w') as f:
                json.dump(subs_data, f, indent=2, default=str)
            
            # Guardar métricas
            usage_data = {usage.user_id: usage.dict() for usage in self._usage_cache.values()}
            with open(self.usage_file, 'w') as f:
                json.dump(usage_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error guardando datos de suscripciones: {e}")
    
    def _create_default_plans(self):
        """Crea planes por defecto"""
        for plan_type, plan in DEFAULT_PLANS.items():
            self._plans_cache[plan.plan_id] = plan
        self._save_data()
    
    def get_all_plans(self) -> List[SubscriptionPlan]:
        """Obtiene todos los planes disponibles"""
        return list(self._plans_cache.values())
    
    def get_plan_by_type(self, plan_type: PlanType) -> Optional[SubscriptionPlan]:
        """Obtiene un plan por tipo"""
        for plan in self._plans_cache.values():
            if plan.plan_type == plan_type:
                return plan
        return None
    
    def get_plan_by_id(self, plan_id: str) -> Optional[SubscriptionPlan]:
        """Obtiene un plan por ID"""
        return self._plans_cache.get(plan_id)
    
    def create_user_subscription(
        self, 
        user_id: str, 
        plan_type: PlanType,
        trial_days: int = 0
    ) -> UserSubscription:
        """Crea una nueva suscripción de usuario"""
        plan = self.get_plan_by_type(plan_type)
        if not plan:
            raise ValueError(f"Plan {plan_type} no encontrado")
        
        now = datetime.now()
        start_date = now
        
        if trial_days > 0:
            end_date = now + timedelta(days=trial_days)
            status = SubscriptionStatus.TRIAL
            is_trial = True
        else:
            end_date = now + timedelta(days=30)  # Mensual por defecto
            status = SubscriptionStatus.ACTIVE
            is_trial = False
        
        subscription = UserSubscription(
            user_id=user_id,
            plan_id=plan.plan_id,
            plan_type=plan_type,
            start_date=start_date,
            end_date=end_date,
            status=status,
            is_trial=is_trial,
            trial_end_date=end_date if is_trial else None
        )
        
        self._subscriptions_cache[subscription.subscription_id] = subscription
        
        # Crear métricas de uso iniciales
        usage = UsageMetrics(user_id=user_id)
        self._usage_cache[user_id] = usage
        
        self._save_data()
        logger.info(f"Suscripción creada para usuario {user_id}: {plan_type}")
        
        return subscription
    
    def get_user_subscription(self, user_id: str) -> Optional[UserSubscription]:
        """Obtiene la suscripción activa de un usuario"""
        for sub in self._subscriptions_cache.values():
            if sub.user_id == user_id and sub.status == SubscriptionStatus.ACTIVE:
                return sub
        return None
    
    def get_user_plan(self, user_id: str) -> Optional[SubscriptionPlan]:
        """Obtiene el plan actual de un usuario"""
        subscription = self.get_user_subscription(user_id)
        if subscription:
            return self.get_plan_by_id(subscription.plan_id)
        return None
    
    def check_user_permissions(
        self, 
        user_id: str, 
        feature: str,
        resource_count: int = 1
    ) -> Tuple[bool, str]:
        """
        Verifica si un usuario tiene permisos para una característica
        
        Args:
            user_id: ID del usuario
            feature: Característica a verificar
            resource_count: Cantidad de recursos solicitados
            
        Returns:
            (permitido, mensaje)
        """
        subscription = self.get_user_subscription(user_id)
        if not subscription:
            return False, "Usuario sin suscripción activa"
        
        plan = self.get_plan_by_id(subscription.plan_id)
        if not plan:
            return False, "Plan no encontrado"
        
        # Verificar límites específicos
        if feature == "api_requests":
            usage = self._usage_cache.get(user_id, UsageMetrics(user_id=user_id))
            if usage.api_requests_today >= plan.api_limits.daily_requests:
                return False, f"Límite diario de requests alcanzado ({plan.api_limits.daily_requests})"
        
        elif feature == "predictions":
            usage = self._usage_cache.get(user_id, UsageMetrics(user_id=user_id))
            if usage.predictions_made_today >= plan.max_predictions_per_day:
                return False, f"Límite diario de predicciones alcanzado ({plan.max_predictions_per_day})"
        
        elif feature == "backtests":
            usage = self._usage_cache.get(user_id, UsageMetrics(user_id=user_id))
            if usage.backtests_run_today >= plan.max_backtests_per_month:
                return False, f"Límite mensual de backtests alcanzado ({plan.max_backtests_per_month})"
        
        elif feature == "alerts":
            usage = self._usage_cache.get(user_id, UsageMetrics(user_id=user_id))
            if usage.alerts_created >= plan.api_limits.alerts_limit:
                return False, f"Límite de alertas alcanzado ({plan.api_limits.alerts_limit})"
        
        elif feature == "trading_pairs":
            if resource_count > plan.api_limits.trading_pairs:
                return False, f"Límite de pares de trading alcanzado ({plan.api_limits.trading_pairs})"
        
        elif feature == "portfolios":
            if resource_count > plan.max_portfolios:
                return False, f"Límite de portafolios alcanzado ({plan.max_portfolios})"
        
        elif feature == "indicators":
            if resource_count > plan.max_indicators:
                return False, f"Límite de indicadores alcanzado ({plan.max_indicators})"
        
        # Verificar capacidades de IA
        elif feature == "traditional_ai":
            if not plan.ai_capabilities.traditional_ai:
                return False, "AI Tradicional no disponible en este plan"
        
        elif feature == "reinforcement_learning":
            if not plan.ai_capabilities.reinforcement_learning:
                return False, "Reinforcement Learning no disponible en este plan"
        
        elif feature == "ensemble_ai":
            if not plan.ai_capabilities.ensemble_ai:
                return False, "Ensemble AI no disponible en este plan"
        
        elif feature == "lstm_predictions":
            if not plan.ai_capabilities.lstm_predictions:
                return False, "Predicciones LSTM no disponibles en este plan"
        
        elif feature == "custom_models":
            if not plan.ai_capabilities.custom_models:
                return False, "Modelos personalizados no disponibles en este plan"
        
        elif feature == "auto_training":
            if not plan.ai_capabilities.auto_training:
                return False, "Auto-entrenamiento no disponible en este plan"
        
        # Verificar características de UI
        elif feature == "advanced_charts":
            if not plan.ui_features.advanced_charts:
                return False, "Gráficos avanzados no disponibles en este plan"
        
        elif feature == "multiple_timeframes":
            if not plan.ui_features.multiple_timeframes:
                return False, "Múltiples timeframes no disponibles en este plan"
        
        elif feature == "rl_dashboard":
            if not plan.ui_features.rl_dashboard:
                return False, "Dashboard RL no disponible en este plan"
        
        elif feature == "ai_monitor":
            if not plan.ui_features.ai_monitor:
                return False, "Monitor AI no disponible en este plan"
        
        elif feature == "mt4_integration":
            if not plan.ui_features.mt4_integration:
                return False, "Integración MT4 no disponible en este plan"
        
        elif feature == "api_access":
            if not plan.ui_features.api_access:
                return False, "Acceso API no disponible en este plan"
        
        elif feature == "custom_reports":
            if not plan.ui_features.custom_reports:
                return False, "Reportes personalizados no disponibles en este plan"
        
        elif feature == "priority_support":
            if not plan.ui_features.priority_support:
                return False, "Soporte prioritario no disponible en este plan"
        
        return True, "Permitido"
    
    def update_usage_metrics(
        self, 
        user_id: str, 
        metric: str, 
        value: int = 1
    ):
        """Actualiza métricas de uso del usuario"""
        usage = self._usage_cache.get(user_id, UsageMetrics(user_id=user_id))
        
        if metric == "api_requests":
            usage.api_requests_today += value
        elif metric == "predictions":
            usage.predictions_made_today += value
        elif metric == "backtests":
            usage.backtests_run_today += value
        elif metric == "alerts":
            usage.alerts_created += value
        elif metric == "rl_episodes":
            usage.rl_episodes_trained += value
        elif metric == "custom_models":
            usage.custom_models_created += value
        elif metric == "trades":
            usage.trades_executed += value
        
        usage.date = datetime.now()
        self._usage_cache[user_id] = usage
        self._save_data()
    
    def get_user_usage(self, user_id: str) -> UsageMetrics:
        """Obtiene las métricas de uso de un usuario"""
        return self._usage_cache.get(user_id, UsageMetrics(user_id=user_id))
    
    def upgrade_user_subscription(
        self, 
        user_id: str, 
        new_plan_type: PlanType
    ) -> UserSubscription:
        """Actualiza la suscripción de un usuario a un plan superior"""
        current_sub = self.get_user_subscription(user_id)
        if not current_sub:
            raise ValueError("Usuario sin suscripción activa")
        
        new_plan = self.get_plan_by_type(new_plan_type)
        if not new_plan:
            raise ValueError(f"Plan {new_plan_type} no encontrado")
        
        # Verificar que es un upgrade válido
        plan_hierarchy = {
            PlanType.FREEMIUM: 0,
            PlanType.BASIC: 1,
            PlanType.PRO: 2,
            PlanType.ELITE: 3
        }
        
        current_level = plan_hierarchy.get(current_sub.plan_type, 0)
        new_level = plan_hierarchy.get(new_plan_type, 0)
        
        if new_level <= current_level:
            raise ValueError("Solo se permiten upgrades a planes superiores")
        
        # Crear nueva suscripción
        new_subscription = UserSubscription(
            user_id=user_id,
            plan_id=new_plan.plan_id,
            plan_type=new_plan_type,
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=30),
            status=SubscriptionStatus.ACTIVE,
            is_trial=False
        )
        
        # Desactivar suscripción anterior
        current_sub.status = SubscriptionStatus.CANCELLED
        current_sub.updated_at = datetime.now()
        
        self._subscriptions_cache[new_subscription.subscription_id] = new_subscription
        self._save_data()
        
        logger.info(f"Usuario {user_id} actualizado de {current_sub.plan_type} a {new_plan_type}")
        
        return new_subscription
    
    def cancel_user_subscription(self, user_id: str) -> bool:
        """Cancela la suscripción de un usuario"""
        subscription = self.get_user_subscription(user_id)
        if not subscription:
            return False
        
        subscription.status = SubscriptionStatus.CANCELLED
        subscription.updated_at = datetime.now()
        self._save_data()
        
        logger.info(f"Suscripción cancelada para usuario {user_id}")
        return True
    
    def check_subscription_expiry(self):
        """Verifica y actualiza suscripciones expiradas"""
        now = datetime.now()
        expired_count = 0
        
        for sub in self._subscriptions_cache.values():
            if sub.status == SubscriptionStatus.ACTIVE and sub.end_date < now:
                sub.status = SubscriptionStatus.EXPIRED
                sub.updated_at = now
                expired_count += 1
        
        if expired_count > 0:
            self._save_data()
            logger.info(f"{expired_count} suscripciones marcadas como expiradas")
    
    def reset_daily_usage(self):
        """Resetea las métricas diarias de uso"""
        today = datetime.now().date()
        
        for usage in self._usage_cache.values():
            if usage.date.date() < today:
                usage.api_requests_today = 0
                usage.predictions_made_today = 0
                usage.backtests_run_today = 0
                usage.alerts_created = 0
                usage.date = datetime.now()
        
        self._save_data()
        logger.info("Métricas diarias de uso reseteadas")
    
    def get_subscription_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de suscripciones"""
        total_users = len(set(sub.user_id for sub in self._subscriptions_cache.values()))
        active_subscriptions = len([sub for sub in self._subscriptions_cache.values() 
                                 if sub.status == SubscriptionStatus.ACTIVE])
        
        plan_counts = {}
        for sub in self._subscriptions_cache.values():
            if sub.status == SubscriptionStatus.ACTIVE:
                plan_counts[sub.plan_type] = plan_counts.get(sub.plan_type, 0) + 1
        
        return {
            "total_users": total_users,
            "active_subscriptions": active_subscriptions,
            "plan_distribution": plan_counts,
            "total_revenue": self._calculate_monthly_revenue()
        }
    
    def _calculate_monthly_revenue(self) -> float:
        """Calcula el ingreso mensual proyectado"""
        revenue = 0.0
        for sub in self._subscriptions_cache.values():
            if sub.status == SubscriptionStatus.ACTIVE:
                plan = self.get_plan_by_id(sub.plan_id)
                if plan:
                    revenue += plan.price
        return revenue 