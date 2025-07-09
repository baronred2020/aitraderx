"""
Demo del Sistema de Suscripciones
=================================
Script de demostración para mostrar el funcionamiento del sistema
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from models.subscription import PlanType, SubscriptionStatus
from services.subscription_service import SubscriptionService

def demo_subscription_system():
    """Demostración del sistema de suscripciones"""
    
    print("🚀 DEMO: Sistema de Suscripciones AI Trading")
    print("=" * 50)
    
    # Inicializar servicio
    service = SubscriptionService()
    
    # 1. Mostrar planes disponibles
    print("\n📋 1. PLANES DISPONIBLES")
    print("-" * 30)
    plans = service.get_all_plans()
    for plan in plans:
        print(f"• {plan.name} (${plan.price}/mes)")
        print(f"  - {plan.description}")
        print(f"  - AI Tradicional: {'✅' if plan.ai_capabilities.traditional_ai else '❌'}")
        print(f"  - RL: {'✅' if plan.ai_capabilities.reinforcement_learning else '❌'}")
        print(f"  - LSTM: {'✅' if plan.ai_capabilities.lstm_predictions else '❌'}")
        print(f"  - MT4: {'✅' if plan.ui_features.mt4_integration else '❌'}")
        print()
    
    # 2. Crear usuarios de ejemplo
    print("\n👥 2. CREANDO USUARIOS DE EJEMPLO")
    print("-" * 30)
    
    users = [
        ("user_freemium", PlanType.FREEMIUM),
        ("user_basic", PlanType.BASIC),
        ("user_pro", PlanType.PRO),
        ("user_elite", PlanType.ELITE)
    ]
    
    for user_id, plan_type in users:
        try:
            subscription = service.create_user_subscription(
                user_id=user_id,
                plan_type=plan_type
            )
            print(f"✅ Usuario {user_id} creado con plan {plan_type.value}")
        except Exception as e:
            print(f"❌ Error creando {user_id}: {e}")
    
    # 3. Demostrar verificación de permisos
    print("\n🔐 3. VERIFICACIÓN DE PERMISOS")
    print("-" * 30)
    
    features_to_test = [
        "traditional_ai",
        "reinforcement_learning", 
        "lstm_predictions",
        "mt4_integration",
        "api_access"
    ]
    
    for user_id, plan_type in users:
        print(f"\n👤 Usuario: {user_id} (Plan: {plan_type.value})")
        for feature in features_to_test:
            allowed, message = service.check_user_permissions(user_id, feature)
            status = "✅" if allowed else "❌"
            print(f"  {status} {feature}: {message}")
    
    # 4. Demostrar límites de uso
    print("\n📊 4. LÍMITES DE USO")
    print("-" * 30)
    
    # Simular uso para usuario freemium
    user_id = "user_freemium"
    print(f"\n👤 Usuario: {user_id}")
    
    # Simular predicciones hasta el límite
    for i in range(12):  # Límite freemium es 10
        allowed, message = service.check_user_permissions(user_id, "predictions")
        if allowed:
            service.update_usage_metrics(user_id, "predictions")
            print(f"  ✅ Predicción {i+1}: Permitida")
        else:
            print(f"  ❌ Predicción {i+1}: {message}")
            break
    
    # 5. Demostrar upgrade de suscripción
    print("\n⬆️ 5. UPGRADE DE SUSCRIPCIÓN")
    print("-" * 30)
    
    user_id = "user_basic"
    print(f"\n👤 Usuario: {user_id} - Upgrade de BASIC a PRO")
    
    try:
        new_subscription = service.upgrade_user_subscription(
            user_id=user_id,
            new_plan_type=PlanType.PRO
        )
        print(f"✅ Upgrade exitoso a {new_subscription.plan_type.value}")
        
        # Verificar nuevos permisos
        allowed, message = service.check_user_permissions(user_id, "reinforcement_learning")
        print(f"  RL disponible: {'✅' if allowed else '❌'}")
        
    except Exception as e:
        print(f"❌ Error en upgrade: {e}")
    
    # 6. Mostrar métricas de uso
    print("\n📈 6. MÉTRICAS DE USO")
    print("-" * 30)
    
    for user_id, plan_type in users:
        usage = service.get_user_usage(user_id)
        print(f"\n👤 {user_id}:")
        print(f"  - Predicciones hoy: {usage.predictions_made_today}")
        print(f"  - Requests API hoy: {usage.api_requests_today}")
        print(f"  - Alertas creadas: {usage.alerts_created}")
    
    # 7. Mostrar estadísticas del sistema
    print("\n📊 7. ESTADÍSTICAS DEL SISTEMA")
    print("-" * 30)
    
    stats = service.get_subscription_stats()
    print(f"👥 Total usuarios: {stats['total_users']}")
    print(f"✅ Suscripciones activas: {stats['active_subscriptions']}")
    print(f"💰 Ingresos mensuales: ${stats['total_revenue']:.2f}")
    
    if 'plan_distribution' in stats:
        print("\n📋 Distribución por plan:")
        for plan_type, count in stats['plan_distribution'].items():
            print(f"  - {plan_type}: {count} usuarios")
    
    # 8. Demostrar cancelación
    print("\n❌ 8. CANCELACIÓN DE SUSCRIPCIÓN")
    print("-" * 30)
    
    user_id = "user_freemium"
    print(f"\n👤 Cancelando suscripción de {user_id}")
    
    success = service.cancel_user_subscription(user_id)
    if success:
        print("✅ Suscripción cancelada exitosamente")
        
        # Verificar que no tiene acceso
        allowed, message = service.check_user_permissions(user_id, "traditional_ai")
        print(f"  Acceso a AI: {'✅' if allowed else '❌'} ({message})")
    
    # 9. Mostrar características por plan
    print("\n🎯 9. CARACTERÍSTICAS POR PLAN")
    print("-" * 30)
    
    for plan_type in [PlanType.FREEMIUM, PlanType.BASIC, PlanType.PRO, PlanType.ELITE]:
        plan = service.get_plan_by_type(plan_type)
        if plan:
            print(f"\n📋 {plan.name} (${plan.price}/mes):")
            print(f"  🧠 AI Tradicional: {'✅' if plan.ai_capabilities.traditional_ai else '❌'}")
            print(f"  🤖 RL: {'✅' if plan.ai_capabilities.reinforcement_learning else '❌'}")
            print(f"  🔄 Ensemble AI: {'✅' if plan.ai_capabilities.ensemble_ai else '❌'}")
            print(f"  📈 LSTM: {'✅' if plan.ai_capabilities.lstm_predictions else '❌'}")
            print(f"  🎨 Custom Models: {'✅' if plan.ai_capabilities.custom_models else '❌'}")
            print(f"  🔧 Auto-Training: {'✅' if plan.ai_capabilities.auto_training else '❌'}")
            print(f"  📊 Charts Avanzados: {'✅' if plan.ui_features.advanced_charts else '❌'}")
            print(f"  ⏰ Multi-Timeframes: {'✅' if plan.ui_features.multiple_timeframes else '❌'}")
            print(f"  🎮 RL Dashboard: {'✅' if plan.ui_features.rl_dashboard else '❌'}")
            print(f"  📱 MT4 Integration: {'✅' if plan.ui_features.mt4_integration else '❌'}")
            print(f"  🔌 API Access: {'✅' if plan.ui_features.api_access else '❌'}")
            print(f"  📋 Custom Reports: {'✅' if plan.ui_features.custom_reports else '❌'}")
            print(f"  🆘 Priority Support: {'✅' if plan.ui_features.priority_support else '❌'}")
    
    print("\n🎉 DEMO COMPLETADA!")
    print("=" * 50)

def demo_api_endpoints():
    """Demostración de endpoints de API"""
    
    print("\n🌐 DEMO: Endpoints de API")
    print("=" * 30)
    
    endpoints = [
        ("GET", "/api/subscriptions/plans", "Obtener todos los planes"),
        ("GET", "/api/subscriptions/plans/basic", "Obtener plan específico"),
        ("POST", "/api/subscriptions/users/{user_id}/subscribe", "Crear suscripción"),
        ("GET", "/api/subscriptions/users/{user_id}/subscription", "Obtener suscripción"),
        ("GET", "/api/subscriptions/users/{user_id}/plan", "Obtener plan de usuario"),
        ("POST", "/api/subscriptions/users/{user_id}/upgrade", "Upgrade de suscripción"),
        ("DELETE", "/api/subscriptions/users/{user_id}/subscription", "Cancelar suscripción"),
        ("POST", "/api/subscriptions/users/{user_id}/permissions/check", "Verificar permisos"),
        ("GET", "/api/subscriptions/users/{user_id}/permissions/features", "Obtener características"),
        ("GET", "/api/subscriptions/users/{user_id}/usage", "Obtener métricas de uso"),
        ("POST", "/api/subscriptions/users/{user_id}/usage/update", "Actualizar métricas"),
        ("GET", "/api/subscriptions/admin/stats", "Estadísticas del sistema"),
        ("POST", "/api/subscriptions/admin/maintenance/check-expiry", "Verificar expiración"),
        ("POST", "/api/subscriptions/admin/maintenance/reset-usage", "Reset métricas"),
        ("GET", "/api/subscriptions/health", "Health check")
    ]
    
    for method, endpoint, description in endpoints:
        print(f"{method:6} {endpoint:<50} - {description}")
    
    print("\n📝 Ejemplos de uso:")
    print("-" * 20)
    
    examples = [
        {
            "endpoint": "POST /api/subscriptions/users/john123/subscribe",
            "body": {
                "plan_type": "basic",
                "trial_days": 7
            },
            "response": "UserSubscription object"
        },
        {
            "endpoint": "POST /api/subscriptions/users/john123/permissions/check",
            "params": {
                "feature": "reinforcement_learning",
                "resource_count": 1
            },
            "response": {
                "user_id": "john123",
                "feature": "reinforcement_learning",
                "allowed": False,
                "message": "Reinforcement Learning no disponible en este plan"
            }
        },
        {
            "endpoint": "GET /api/subscriptions/users/john123/usage",
            "response": {
                "user_id": "john123",
                "api_requests_today": 45,
                "predictions_made_today": 8,
                "backtests_run_today": 2,
                "alerts_created": 3
            }
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['endpoint']}")
        if 'body' in example:
            print(f"   Body: {json.dumps(example['body'], indent=2)}")
        if 'params' in example:
            print(f"   Params: {json.dumps(example['params'], indent=2)}")
        print(f"   Response: {example['response']}")

if __name__ == "__main__":
    # Ejecutar demos
    demo_subscription_system()
    demo_api_endpoints() 