"""
Tests para el Sistema de Suscripciones
======================================
Tests unitarios y de integración para el sistema de suscripciones
"""

import pytest
import json
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from models.subscription import (
    SubscriptionPlan, UserSubscription, UsageMetrics, PlanType,
    SubscriptionStatus, DEFAULT_PLANS
)
from services.subscription_service import SubscriptionService

class TestSubscriptionSystem:
    """Tests para el sistema de suscripciones"""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Fixture para crear directorio temporal de datos"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def subscription_service(self, temp_data_dir):
        """Fixture para crear servicio de suscripciones"""
        return SubscriptionService(data_dir=temp_data_dir)
    
    def test_create_default_plans(self, subscription_service):
        """Test para verificar que se crean los planes por defecto"""
        plans = subscription_service.get_all_plans()
        
        # Verificar que se crearon los 4 planes
        assert len(plans) == 4
        
        # Verificar que existen todos los tipos de plan
        plan_types = [plan.plan_type for plan in plans]
        assert PlanType.FREEMIUM in plan_types
        assert PlanType.BASIC in plan_types
        assert PlanType.PRO in plan_types
        assert PlanType.ELITE in plan_types
    
    def test_get_plan_by_type(self, subscription_service):
        """Test para obtener plan por tipo"""
        # Test plan existente
        plan = subscription_service.get_plan_by_type(PlanType.BASIC)
        assert plan is not None
        assert plan.plan_type == PlanType.BASIC
        assert plan.price == 29.0
        
        # Test plan inexistente
        plan = subscription_service.get_plan_by_type("INVALID")
        assert plan is None
    
    def test_create_user_subscription(self, subscription_service):
        """Test para crear suscripción de usuario"""
        user_id = "test_user_123"
        
        # Crear suscripción básica
        subscription = subscription_service.create_user_subscription(
            user_id=user_id,
            plan_type=PlanType.BASIC
        )
        
        assert subscription.user_id == user_id
        assert subscription.plan_type == PlanType.BASIC
        assert subscription.status == SubscriptionStatus.ACTIVE
        assert subscription.is_trial == False
        
        # Verificar que se creó en el cache
        retrieved_sub = subscription_service.get_user_subscription(user_id)
        assert retrieved_sub is not None
        assert retrieved_sub.subscription_id == subscription.subscription_id
    
    def test_create_trial_subscription(self, subscription_service):
        """Test para crear suscripción de prueba"""
        user_id = "trial_user_123"
        
        subscription = subscription_service.create_user_subscription(
            user_id=user_id,
            plan_type=PlanType.PRO,
            trial_days=14
        )
        
        assert subscription.user_id == user_id
        assert subscription.plan_type == PlanType.PRO
        assert subscription.status == SubscriptionStatus.TRIAL
        assert subscription.is_trial == True
        assert subscription.trial_end_date is not None
    
    def test_duplicate_subscription_error(self, subscription_service):
        """Test para verificar error en suscripción duplicada"""
        user_id = "duplicate_user_123"
        
        # Crear primera suscripción
        subscription_service.create_user_subscription(
            user_id=user_id,
            plan_type=PlanType.BASIC
        )
        
        # Intentar crear segunda suscripción
        with pytest.raises(ValueError, match="Usuario ya tiene una suscripción activa"):
            subscription_service.create_user_subscription(
                user_id=user_id,
                plan_type=PlanType.PRO
            )
    
    def test_get_user_plan(self, subscription_service):
        """Test para obtener plan de usuario"""
        user_id = "plan_user_123"
        
        # Crear suscripción
        subscription_service.create_user_subscription(
            user_id=user_id,
            plan_type=PlanType.PRO
        )
        
        # Obtener plan
        plan = subscription_service.get_user_plan(user_id)
        assert plan is not None
        assert plan.plan_type == PlanType.PRO
        assert plan.price == 99.0
    
    def test_check_user_permissions(self, subscription_service):
        """Test para verificar permisos de usuario"""
        user_id = "permission_user_123"
        
        # Crear suscripción básica
        subscription_service.create_user_subscription(
            user_id=user_id,
            plan_type=PlanType.BASIC
        )
        
        # Test permisos permitidos
        allowed, message = subscription_service.check_user_permissions(
            user_id=user_id,
            feature="traditional_ai"
        )
        assert allowed == True
        assert "Permitido" in message
        
        # Test permisos denegados
        allowed, message = subscription_service.check_user_permissions(
            user_id=user_id,
            feature="reinforcement_learning"
        )
        assert allowed == False
        assert "Reinforcement Learning no disponible" in message
    
    def test_check_limits(self, subscription_service):
        """Test para verificar límites de uso"""
        user_id = "limit_user_123"
        
        # Crear suscripción freemium
        subscription_service.create_user_subscription(
            user_id=user_id,
            plan_type=PlanType.FREEMIUM
        )
        
        # Simular uso máximo de predicciones
        for i in range(10):  # Límite freemium
            subscription_service.update_usage_metrics(user_id, "predictions")
        
        # Intentar una predicción más
        allowed, message = subscription_service.check_user_permissions(
            user_id=user_id,
            feature="predictions"
        )
        assert allowed == False
        assert "Límite diario de predicciones alcanzado" in message
    
    def test_upgrade_subscription(self, subscription_service):
        """Test para upgrade de suscripción"""
        user_id = "upgrade_user_123"
        
        # Crear suscripción básica
        subscription_service.create_user_subscription(
            user_id=user_id,
            plan_type=PlanType.BASIC
        )
        
        # Upgrade a PRO
        new_subscription = subscription_service.upgrade_user_subscription(
            user_id=user_id,
            new_plan_type=PlanType.PRO
        )
        
        assert new_subscription.plan_type == PlanType.PRO
        assert new_subscription.status == SubscriptionStatus.ACTIVE
        
        # Verificar que el plan anterior fue cancelado
        old_subscriptions = [
            sub for sub in subscription_service._subscriptions_cache.values()
            if sub.user_id == user_id and sub.status == SubscriptionStatus.CANCELLED
        ]
        assert len(old_subscriptions) == 1
        assert old_subscriptions[0].plan_type == PlanType.BASIC
    
    def test_invalid_upgrade(self, subscription_service):
        """Test para upgrade inválido"""
        user_id = "invalid_upgrade_user_123"
        
        # Crear suscripción PRO
        subscription_service.create_user_subscription(
            user_id=user_id,
            plan_type=PlanType.PRO
        )
        
        # Intentar upgrade a BASIC (downgrade)
        with pytest.raises(ValueError, match="Solo se permiten upgrades a planes superiores"):
            subscription_service.upgrade_user_subscription(
                user_id=user_id,
                new_plan_type=PlanType.BASIC
            )
    
    def test_cancel_subscription(self, subscription_service):
        """Test para cancelar suscripción"""
        user_id = "cancel_user_123"
        
        # Crear suscripción
        subscription_service.create_user_subscription(
            user_id=user_id,
            plan_type=PlanType.BASIC
        )
        
        # Cancelar suscripción
        success = subscription_service.cancel_user_subscription(user_id)
        assert success == True
        
        # Verificar que no hay suscripción activa
        active_sub = subscription_service.get_user_subscription(user_id)
        assert active_sub is None
    
    def test_usage_metrics(self, subscription_service):
        """Test para métricas de uso"""
        user_id = "usage_user_123"
        
        # Crear suscripción
        subscription_service.create_user_subscription(
            user_id=user_id,
            plan_type=PlanType.BASIC
        )
        
        # Actualizar métricas
        subscription_service.update_usage_metrics(user_id, "predictions", 5)
        subscription_service.update_usage_metrics(user_id, "api_requests", 10)
        
        # Obtener métricas
        usage = subscription_service.get_user_usage(user_id)
        assert usage.predictions_made_today == 5
        assert usage.api_requests_today == 10
    
    def test_subscription_expiry(self, subscription_service):
        """Test para expiración de suscripciones"""
        user_id = "expiry_user_123"
        
        # Crear suscripción con fecha de expiración pasada
        subscription = UserSubscription(
            user_id=user_id,
            plan_id="test_plan",
            plan_type=PlanType.BASIC,
            start_date=datetime.now() - timedelta(days=35),
            end_date=datetime.now() - timedelta(days=5),
            status=SubscriptionStatus.ACTIVE
        )
        subscription_service._subscriptions_cache[subscription.subscription_id] = subscription
        
        # Verificar expiración
        subscription_service.check_subscription_expiry()
        
        # Verificar que la suscripción fue marcada como expirada
        expired_subs = [
            sub for sub in subscription_service._subscriptions_cache.values()
            if sub.user_id == user_id and sub.status == SubscriptionStatus.EXPIRED
        ]
        assert len(expired_subs) == 1
    
    def test_reset_daily_usage(self, subscription_service):
        """Test para reset de métricas diarias"""
        user_id = "reset_user_123"
        
        # Crear suscripción
        subscription_service.create_user_subscription(
            user_id=user_id,
            plan_type=PlanType.BASIC
        )
        
        # Actualizar métricas
        subscription_service.update_usage_metrics(user_id, "predictions", 5)
        subscription_service.update_usage_metrics(user_id, "api_requests", 10)
        
        # Verificar métricas antes del reset
        usage_before = subscription_service.get_user_usage(user_id)
        assert usage_before.predictions_made_today == 5
        assert usage_before.api_requests_today == 10
        
        # Reset métricas
        subscription_service.reset_daily_usage()
        
        # Verificar métricas después del reset
        usage_after = subscription_service.get_user_usage(user_id)
        assert usage_after.predictions_made_today == 0
        assert usage_after.api_requests_today == 0
    
    def test_subscription_stats(self, subscription_service):
        """Test para estadísticas de suscripciones"""
        # Crear múltiples suscripciones
        users = ["user1", "user2", "user3", "user4"]
        plans = [PlanType.FREEMIUM, PlanType.BASIC, PlanType.PRO, PlanType.ELITE]
        
        for user_id, plan_type in zip(users, plans):
            subscription_service.create_user_subscription(
                user_id=user_id,
                plan_type=plan_type
            )
        
        # Obtener estadísticas
        stats = subscription_service.get_subscription_stats()
        
        assert stats["total_users"] == 4
        assert stats["active_subscriptions"] == 4
        assert "plan_distribution" in stats
        assert "total_revenue" in stats
    
    def test_persistence(self, temp_data_dir):
        """Test para persistencia de datos"""
        # Crear servicio
        service1 = SubscriptionService(data_dir=temp_data_dir)
        
        # Crear suscripción
        user_id = "persistence_user_123"
        service1.create_user_subscription(
            user_id=user_id,
            plan_type=PlanType.PRO
        )
        
        # Crear nuevo servicio (simula reinicio)
        service2 = SubscriptionService(data_dir=temp_data_dir)
        
        # Verificar que los datos persisten
        subscription = service2.get_user_subscription(user_id)
        assert subscription is not None
        assert subscription.plan_type == PlanType.PRO
        
        plans = service2.get_all_plans()
        assert len(plans) == 4

if __name__ == "__main__":
    # Ejecutar tests
    pytest.main([__file__, "-v"]) 