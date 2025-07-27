"""
Script para actualizar nombres de planes de suscripción
====================================================
Actualiza los nombres de los planes en la base de datos
"""

import os
import sys
from pathlib import Path

# Agregar el directorio actual al path
sys.path.append(str(Path(__file__).parent))

from services.subscription_service import SubscriptionService
from models.subscription import PlanType

def update_plan_names():
    """Actualiza los nombres de los planes en la base de datos"""
    
    print("🔄 Actualizando nombres de planes de suscripción...")
    
    # Inicializar servicio
    service = SubscriptionService()
    
    # Mapeo de nombres antiguos a nuevos
    name_mapping = {
        'freemium': 'starter',
        'basic': 'trader', 
        'pro': 'expert',
        'elite': 'premium'
    }
    
    try:
        # Actualizar planes en el cache
        updated_plans = []
        for plan in service.get_all_plans():
            old_type = plan.plan_type
            if old_type in name_mapping:
                new_type = name_mapping[old_type]
                plan.plan_type = PlanType(new_type)
                plan.name = new_type.capitalize()
                updated_plans.append(plan)
                print(f"✅ Actualizado: {old_type} → {new_type}")
        
        # Guardar cambios
        service._save_data()
        
        # Actualizar suscripciones de usuarios
        updated_subs = 0
        for sub in service._subscriptions_cache.values():
            if sub.plan_type in name_mapping:
                old_type = sub.plan_type
                new_type = name_mapping[old_type]
                sub.plan_type = PlanType(new_type)
                updated_subs += 1
                print(f"✅ Usuario {sub.user_id}: {old_type} → {new_type}")
        
        service._save_data()
        
        print(f"\n🎉 Actualización completada:")
        print(f"   - {len(updated_plans)} planes actualizados")
        print(f"   - {updated_subs} suscripciones de usuarios actualizadas")
        
        # Mostrar planes actualizados
        print("\n📋 Planes actualizados:")
        for plan in service.get_all_plans():
            print(f"   • {plan.name} ({plan.plan_type.value}) - ${plan.price}/mes")
        
    except Exception as e:
        print(f"❌ Error actualizando planes: {e}")
        return False
    
    return True

def verify_changes():
    """Verifica que los cambios se aplicaron correctamente"""
    
    print("\n🔍 Verificando cambios...")
    
    service = SubscriptionService()
    
    # Verificar que los nuevos planes existen
    expected_plans = ['starter', 'trader', 'expert', 'premium', 'institutional']
    actual_plans = [plan.plan_type.value for plan in service.get_all_plans()]
    
    print(f"Planes esperados: {expected_plans}")
    print(f"Planes actuales: {actual_plans}")
    
    if set(expected_plans) == set(actual_plans):
        print("✅ Todos los planes están correctamente configurados")
        return True
    else:
        print("❌ Hay discrepancias en los planes")
        return False

if __name__ == "__main__":
    print("🚀 Iniciando actualización de nombres de planes...")
    
    # Ejecutar actualización
    success = update_plan_names()
    
    if success:
        # Verificar cambios
        verify_changes()
        print("\n🎉 ¡Actualización completada exitosamente!")
    else:
        print("\n❌ Error en la actualización")
        sys.exit(1) 