"""
Servicio de Suscripciones con MySQL
===================================
Servicio para gestionar suscripciones usando MySQL como base de datos
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
import logging

from models.database_models import (
    SubscriptionPlan, UserSubscription, UsageMetrics, 
    SubscriptionUpgrade, SubscriptionPayment, SubscriptionAudit
)

logger = logging.getLogger(__name__)

class SubscriptionMySQLService:
    """Servicio para gestionar suscripciones con MySQL"""
    
    def __init__(self, database_url: str = None):
        # Obtener URL de base de datos
        self.database_url = database_url or os.getenv(
            'DATABASE_URL', 
            'mysql://trader:password123@localhost:3306/trading_db'
        )
        
        # Crear engine y session
        self.engine = create_engine(self.database_url, pool_pre_ping=True)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Verificar conexión
        self._test_connection()
    
    def _test_connection(self):
        """Verificar conexión a la base de datos"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("✅ Conexión a MySQL establecida")
        except Exception as e:
            logger.error(f"❌ Error conectando a MySQL: {e}")
            raise
    
    def get_db_session(self) -> Session:
        """Obtener sesión de base de datos"""
        return self.SessionLocal()
    
    def get_all_plans(self) -> List[SubscriptionPlan]:
        """Obtiene todos los planes disponibles"""
        try:
            with self.get_db_session() as session:
                plans = session.query(SubscriptionPlan).all()
                return plans
        except Exception as e:
            logger.error(f"Error obteniendo planes: {e}")
            return []
    
    def get_plan_by_type(self, plan_type: str) -> Optional[SubscriptionPlan]:
        """Obtiene un plan por tipo"""
        try:
            with self.get_db_session() as session:
                plan = session.query(SubscriptionPlan).filter(
                    SubscriptionPlan.plan_type == plan_type
                ).first()
                return plan
        except Exception as e:
            logger.error(f"Error obteniendo plan {plan_type}: {e}")
            return None
    
    def get_plan_by_id(self, plan_id: str) -> Optional[SubscriptionPlan]:
        """Obtiene un plan por ID"""
        try:
            with self.get_db_session() as session:
                plan = session.query(SubscriptionPlan).filter(
                    SubscriptionPlan.plan_id == plan_id
                ).first()
                return plan
        except Exception as e:
            logger.error(f"Error obteniendo plan {plan_id}: {e}")
            return None
    
    def create_user_subscription(
        self, 
        user_id: str, 
        plan_type: str,
        trial_days: int = 0
    ) -> Optional[UserSubscription]:
        """Crea una nueva suscripción de usuario"""
        try:
            # Obtener plan
            plan = self.get_plan_by_type(plan_type)
            if not plan:
                raise ValueError(f"Plan {plan_type} no encontrado")
            
            # Verificar si ya tiene suscripción activa
            existing_sub = self.get_user_subscription(user_id)
            if existing_sub:
                raise ValueError("Usuario ya tiene una suscripción activa")
            
            now = datetime.utcnow()
            start_date = now
            
            if trial_days > 0:
                end_date = now + timedelta(days=trial_days)
                status = "trial"
                is_trial = True
            else:
                end_date = now + timedelta(days=30)  # Mensual por defecto
                status = "active"
                is_trial = False
            
            # Crear suscripción
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
            
            with self.get_db_session() as session:
                session.add(subscription)
                session.commit()
                session.refresh(subscription)
            
            # Crear métricas de uso iniciales
            self._create_usage_metrics(user_id, subscription.subscription_id)
            
            # Registrar en auditoría
            self._log_audit(user_id, subscription.subscription_id, "created", None, plan_type)
            
            logger.info(f"Suscripción creada para usuario {user_id}: {plan_type}")
            return subscription
            
        except Exception as e:
            logger.error(f"Error creando suscripción para {user_id}: {e}")
            raise
    
    def get_user_subscription(self, user_id: str) -> Optional[UserSubscription]:
        """Obtiene la suscripción activa de un usuario"""
        try:
            with self.get_db_session() as session:
                subscription = session.query(UserSubscription).filter(
                    UserSubscription.user_id == user_id,
                    UserSubscription.status == "active"
                ).first()
                return subscription
        except Exception as e:
            logger.error(f"Error obteniendo suscripción de {user_id}: {e}")
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
        """Verifica si un usuario tiene permisos para una característica"""
        try:
            subscription = self.get_user_subscription(user_id)
            if not subscription:
                return False, "Usuario sin suscripción activa"
            
            plan = self.get_plan_by_id(subscription.plan_id)
            if not plan:
                return False, "Plan no encontrado"
            
            # Verificar límites específicos
            if feature == "api_requests":
                usage = self._get_today_usage(user_id)
                if usage and usage.api_requests_today >= plan.daily_requests:
                    return False, f"Límite diario de requests alcanzado ({plan.daily_requests})"
            
            elif feature == "predictions":
                usage = self._get_today_usage(user_id)
                if usage and usage.predictions_made_today >= plan.max_predictions_per_day:
                    return False, f"Límite diario de predicciones alcanzado ({plan.max_predictions_per_day})"
            
            elif feature == "backtests":
                usage = self._get_today_usage(user_id)
                if usage and usage.backtests_run_today >= plan.max_backtests_per_month:
                    return False, f"Límite mensual de backtests alcanzado ({plan.max_backtests_per_month})"
            
            elif feature == "alerts":
                usage = self._get_today_usage(user_id)
                if usage and usage.alerts_created >= plan.alerts_limit:
                    return False, f"Límite de alertas alcanzado ({plan.alerts_limit})"
            
            elif feature == "trading_pairs":
                if resource_count > plan.trading_pairs:
                    return False, f"Límite de pares de trading alcanzado ({plan.trading_pairs})"
            
            elif feature == "portfolios":
                if resource_count > plan.max_portfolios:
                    return False, f"Límite de portafolios alcanzado ({plan.max_portfolios})"
            
            elif feature == "indicators":
                if resource_count > plan.max_indicators:
                    return False, f"Límite de indicadores alcanzado ({plan.max_indicators})"
            
            # Verificar capacidades de IA
            elif feature == "traditional_ai":
                if not plan.traditional_ai:
                    return False, "AI Tradicional no disponible en este plan"
            
            elif feature == "reinforcement_learning":
                if not plan.reinforcement_learning:
                    return False, "Reinforcement Learning no disponible en este plan"
            
            elif feature == "ensemble_ai":
                if not plan.ensemble_ai:
                    return False, "Ensemble AI no disponible en este plan"
            
            elif feature == "lstm_predictions":
                if not plan.lstm_predictions:
                    return False, "Predicciones LSTM no disponibles en este plan"
            
            elif feature == "custom_models":
                if not plan.custom_models:
                    return False, "Modelos personalizados no disponibles en este plan"
            
            elif feature == "auto_training":
                if not plan.auto_training:
                    return False, "Auto-entrenamiento no disponible en este plan"
            
            # Verificar características de UI
            elif feature == "advanced_charts":
                if not plan.advanced_charts:
                    return False, "Gráficos avanzados no disponibles en este plan"
            
            elif feature == "multiple_timeframes":
                if not plan.multiple_timeframes:
                    return False, "Múltiples timeframes no disponibles en este plan"
            
            elif feature == "rl_dashboard":
                if not plan.rl_dashboard:
                    return False, "Dashboard RL no disponible en este plan"
            
            elif feature == "ai_monitor":
                if not plan.ai_monitor:
                    return False, "Monitor AI no disponible en este plan"
            
            elif feature == "mt4_integration":
                if not plan.mt4_integration:
                    return False, "Integración MT4 no disponible en este plan"
            
            elif feature == "api_access":
                if not plan.api_access:
                    return False, "Acceso API no disponible en este plan"
            
            elif feature == "custom_reports":
                if not plan.custom_reports:
                    return False, "Reportes personalizados no disponibles en este plan"
            
            elif feature == "priority_support":
                if not plan.priority_support:
                    return False, "Soporte prioritario no disponible en este plan"
            
            return True, "Permitido"
            
        except Exception as e:
            logger.error(f"Error verificando permisos de {user_id}: {e}")
            return False, "Error interno del servidor"
    
    def update_usage_metrics(
        self, 
        user_id: str, 
        metric: str, 
        value: int = 1
    ):
        """Actualiza métricas de uso del usuario"""
        try:
            usage = self._get_today_usage(user_id)
            if not usage:
                # Crear métricas si no existen
                subscription = self.get_user_subscription(user_id)
                if subscription:
                    usage = self._create_usage_metrics(user_id, subscription.subscription_id)
            
            if usage:
                with self.get_db_session() as session:
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
                    
                    usage.updated_at = datetime.utcnow()
                    session.commit()
                    
        except Exception as e:
            logger.error(f"Error actualizando métricas para {user_id}: {e}")
    
    def get_user_usage(self, user_id: str) -> Optional[UsageMetrics]:
        """Obtiene las métricas de uso de un usuario"""
        return self._get_today_usage(user_id)
    
    def upgrade_user_subscription(
        self, 
        user_id: str, 
        new_plan_type: str
    ) -> Optional[UserSubscription]:
        """Actualiza la suscripción de un usuario a un plan superior"""
        try:
            current_sub = self.get_user_subscription(user_id)
            if not current_sub:
                raise ValueError("Usuario sin suscripción activa")
            
            new_plan = self.get_plan_by_type(new_plan_type)
            if not new_plan:
                raise ValueError(f"Plan {new_plan_type} no encontrado")
            
            # Verificar que es un upgrade válido
            plan_hierarchy = {
                "freemium": 0,
                "basic": 1,
                "pro": 2,
                "elite": 3
            }
            
            current_level = plan_hierarchy.get(current_sub.plan_type, 0)
            new_level = plan_hierarchy.get(new_plan_type, 0)
            
            if new_level <= current_level:
                raise ValueError("Solo se permiten upgrades a planes superiores")
            
            # Cancelar suscripción anterior
            old_plan_type = current_sub.plan_type
            current_sub.status = "cancelled"
            current_sub.updated_at = datetime.utcnow()
            
            # Crear nueva suscripción
            new_subscription = UserSubscription(
                user_id=user_id,
                plan_id=new_plan.plan_id,
                plan_type=new_plan_type,
                start_date=datetime.utcnow(),
                end_date=datetime.utcnow() + timedelta(days=30),
                status="active",
                is_trial=False
            )
            
            with self.get_db_session() as session:
                session.add(new_subscription)
                session.commit()
                session.refresh(new_subscription)
            
            # Crear métricas de uso para nuevo plan
            self._create_usage_metrics(user_id, new_subscription.subscription_id)
            
            # Registrar en auditoría
            self._log_audit(user_id, new_subscription.subscription_id, "upgraded", old_plan_type, new_plan_type)
            
            logger.info(f"Usuario {user_id} actualizado de {old_plan_type} a {new_plan_type}")
            return new_subscription
            
        except Exception as e:
            logger.error(f"Error actualizando suscripción de {user_id}: {e}")
            raise
    
    def cancel_user_subscription(self, user_id: str) -> bool:
        """Cancela la suscripción de un usuario"""
        try:
            subscription = self.get_user_subscription(user_id)
            if not subscription:
                return False
            
            subscription.status = "cancelled"
            subscription.updated_at = datetime.utcnow()
            
            with self.get_db_session() as session:
                session.commit()
            
            # Registrar en auditoría
            self._log_audit(user_id, subscription.subscription_id, "cancelled", subscription.plan_type, None)
            
            logger.info(f"Suscripción cancelada para usuario {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelando suscripción de {user_id}: {e}")
            return False
    
    def check_subscription_expiry(self):
        """Verifica y actualiza suscripciones expiradas"""
        try:
            with self.get_db_session() as session:
                expired_subs = session.query(UserSubscription).filter(
                    UserSubscription.status == "active",
                    UserSubscription.end_date < datetime.utcnow()
                ).all()
                
                for sub in expired_subs:
                    sub.status = "expired"
                    sub.updated_at = datetime.utcnow()
                    self._log_audit(sub.user_id, sub.subscription_id, "expired", sub.plan_type, None)
                
                session.commit()
                logger.info(f"{len(expired_subs)} suscripciones marcadas como expiradas")
                
        except Exception as e:
            logger.error(f"Error verificando expiración: {e}")
    
    def reset_daily_usage(self):
        """Resetea las métricas diarias de uso"""
        try:
            with self.get_db_session() as session:
                # Obtener métricas de ayer
                yesterday = datetime.utcnow().date() - timedelta(days=1)
                
                # Crear nuevas métricas para hoy
                usage_records = session.query(UsageMetrics).filter(
                    UsageMetrics.date < datetime.utcnow().date()
                ).all()
                
                for usage in usage_records:
                    # Crear nueva métrica para hoy con valores en 0
                    new_usage = UsageMetrics(
                        user_id=usage.user_id,
                        subscription_id=usage.subscription_id,
                        date=datetime.utcnow().date(),
                        api_requests_today=0,
                        predictions_made_today=0,
                        backtests_run_today=0,
                        alerts_created=0,
                        rl_episodes_trained=0,
                        custom_models_created=0,
                        trades_executed=0,
                        portfolio_value=0.0,
                        profit_loss=0.0
                    )
                    session.add(new_usage)
                
                session.commit()
                logger.info("Métricas diarias de uso reseteadas")
                
        except Exception as e:
            logger.error(f"Error reseteando métricas: {e}")
    
    def get_subscription_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de suscripciones"""
        try:
            with self.get_db_session() as session:
                # Total usuarios
                total_users = session.query(UserSubscription.user_id).distinct().count()
                
                # Suscripciones activas
                active_subscriptions = session.query(UserSubscription).filter(
                    UserSubscription.status == "active"
                ).count()
                
                # Distribución por plan
                plan_distribution = {}
                for sub in session.query(UserSubscription).filter(
                    UserSubscription.status == "active"
                ).all():
                    plan_type = sub.plan_type
                    plan_distribution[plan_type] = plan_distribution.get(plan_type, 0) + 1
                
                # Calcular ingresos
                total_revenue = 0
                for sub in session.query(UserSubscription).filter(
                    UserSubscription.status == "active"
                ).all():
                    plan = self.get_plan_by_id(sub.plan_id)
                    if plan:
                        total_revenue += plan.price
                
                return {
                    "total_users": total_users,
                    "active_subscriptions": active_subscriptions,
                    "plan_distribution": plan_distribution,
                    "total_revenue": total_revenue
                }
                
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            return {
                "total_users": 0,
                "active_subscriptions": 0,
                "plan_distribution": {},
                "total_revenue": 0.0
            }
    
    def _get_today_usage(self, user_id: str) -> Optional[UsageMetrics]:
        """Obtiene las métricas de uso de hoy para un usuario"""
        try:
            with self.get_db_session() as session:
                today = datetime.utcnow().date()
                usage = session.query(UsageMetrics).filter(
                    UsageMetrics.user_id == user_id,
                    UsageMetrics.date == today
                ).first()
                return usage
        except Exception as e:
            logger.error(f"Error obteniendo uso de {user_id}: {e}")
            return None
    
    def _create_usage_metrics(self, user_id: str, subscription_id: str) -> Optional[UsageMetrics]:
        """Crea métricas de uso iniciales para un usuario"""
        try:
            usage = UsageMetrics(
                user_id=user_id,
                subscription_id=subscription_id,
                date=datetime.utcnow().date(),
                api_requests_today=0,
                predictions_made_today=0,
                backtests_run_today=0,
                alerts_created=0,
                rl_episodes_trained=0,
                custom_models_created=0,
                trades_executed=0,
                portfolio_value=0.0,
                profit_loss=0.0
            )
            
            with self.get_db_session() as session:
                session.add(usage)
                session.commit()
                session.refresh(usage)
            
            return usage
            
        except Exception as e:
            logger.error(f"Error creando métricas para {user_id}: {e}")
            return None
    
    def _log_audit(self, user_id: str, subscription_id: str, action: str, old_plan: str = None, new_plan: str = None):
        """Registra acción en auditoría"""
        try:
            audit = SubscriptionAudit(
                user_id=user_id,
                subscription_id=subscription_id,
                action=action,
                old_plan=old_plan,
                new_plan=new_plan
            )
            
            with self.get_db_session() as session:
                session.add(audit)
                session.commit()
                
        except Exception as e:
            logger.error(f"Error registrando auditoría: {e}") 