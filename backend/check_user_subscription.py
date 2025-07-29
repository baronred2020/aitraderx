#!/usr/bin/env python3
"""
Script para verificar y crear suscripción del usuario "user"
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from services.subscription_mysql_service import SubscriptionMySQLService
from services.user_service import UserService

def main():
    print("🔍 Verificando suscripción del usuario 'user'...")
    
    # Usar la configuración correcta de la base de datos
    database_url = "mysql://root:root@localhost:3306/trading_db"
    
    # Inicializar servicios
    subscription_service = SubscriptionMySQLService(database_url)
    user_service = UserService()
    
    # Obtener usuario
    user = user_service.get_user_by_username("user")
    if not user:
        print("❌ Usuario 'user' no encontrado")
        return
    
    print(f"✅ Usuario encontrado: {user.get('username')} (ID: {user.get('user_id')})")
    
    # Verificar suscripción existente
    user_id = user.get('user_id')
    existing_subscription = subscription_service.get_user_subscription(user_id)
    
    if existing_subscription:
        plan = subscription_service.get_plan_by_id(existing_subscription.plan_id)
        print(f"✅ Usuario ya tiene suscripción: {plan.plan_type if plan else 'Unknown'}")
        print(f"   Status: {existing_subscription.status}")
        print(f"   Start Date: {existing_subscription.start_date}")
        print(f"   End Date: {existing_subscription.end_date}")
    else:
        print("❌ Usuario no tiene suscripción activa")
        
        # Crear suscripción starter
        print("🔄 Creando suscripción 'starter'...")
        new_subscription = subscription_service.create_user_subscription(
            user_id=user_id,
            plan_type="starter",
            trial_days=0
        )
        
        if new_subscription:
            plan = subscription_service.get_plan_by_id(new_subscription.plan_id)
            print(f"✅ Suscripción creada exitosamente: {plan.plan_type if plan else 'starter'}")
            print(f"   Subscription ID: {new_subscription.subscription_id}")
            print(f"   Status: {new_subscription.status}")
        else:
            print("❌ Error al crear suscripción")

if __name__ == "__main__":
    main() 