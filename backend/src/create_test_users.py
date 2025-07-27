"""
Script para crear usuarios de prueba
==================================
Este script crea usuarios de prueba para el sistema de autenticación
"""

import sys
import os
from pathlib import Path

# Agregar el directorio actual al path
sys.path.append(str(Path(__file__).parent))

from services.user_service import UserService
from config.database_config import create_tables
import hashlib
import uuid
from datetime import datetime, timedelta

def create_test_users():
    """Crea usuarios de prueba en el sistema"""
    try:
        # Crear tablas si no existen
        print("Creando tablas de base de datos...")
        create_tables()
        
        # Crear servicio de usuarios
        user_service = UserService()
        
        # Usuarios de prueba
        test_users = [
            {
                "username": "admin",
                "email": "admin@aitraderx.com",
                "password": "admin123",
                "firstName": "Admin",
                "lastName": "User",
                "role": "admin",
                "plan_type": "premium"
            },
            {
                "username": "user",
                "email": "user@aitraderx.com",
                "password": "user123",
                "firstName": "Normal",
                "lastName": "User",
                "role": "user",
                "plan_type": "starter"
            },
            {
                "username": "trader",
                "email": "trader@aitraderx.com",
                "password": "trader123",
                "firstName": "Trader",
                "lastName": "User",
                "role": "user",
                "plan_type": "trader"
            }
        ]
        
        print("Creando usuarios de prueba...")
        for user_data in test_users:
            # Verificar si el usuario ya existe
            existing_user = user_service.get_user_by_username(user_data["username"])
            if existing_user:
                print(f"Usuario {user_data['username']} ya existe, saltando...")
                continue
            
            # Crear usuario
            user = user_service.create_user({
                "username": user_data["username"],
                "email": user_data["email"],
                "password": user_data["password"],
                "firstName": user_data["firstName"],
                "lastName": user_data["lastName"]
            })
            
            if user:
                # Crear suscripción
                subscription = user_service.create_subscription(
                    user.user_id, 
                    user_data["plan_type"]
                )
                
                if subscription:
                    print(f"✅ Usuario {user_data['username']} creado exitosamente")
                    print(f"   - Plan: {user_data['plan_type']}")
                    print(f"   - Role: {user_data['role']}")
                else:
                    print(f"⚠️  Usuario {user_data['username']} creado pero sin suscripción")
            else:
                print(f"❌ Error creando usuario {user_data['username']}")
        
        print("\n🎉 Usuarios de prueba creados exitosamente!")
        print("\nCredenciales disponibles:")
        print("👑 Admin: admin / admin123 (Plan: Premium)")
        print("👤 User: user / user123 (Plan: Starter)")
        print("📈 Trader: trader / trader123 (Plan: Trader)")
        
    except Exception as e:
        print(f"❌ Error creando usuarios de prueba: {e}")
        print("Asegúrate de que la base de datos esté configurada correctamente")

if __name__ == "__main__":
    create_test_users() 