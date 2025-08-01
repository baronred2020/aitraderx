#!/usr/bin/env python3
"""
Script para verificar qué usuarios existen en la base de datos
"""

import os
import sys

# Agregar el directorio backend/src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'src'))

from config.database_config import db_config

def check_users():
    """
    Verificar qué usuarios existen en la base de datos
    """
    print("🔍 VERIFICACIÓN DE USUARIOS EN LA BASE DE DATOS")
    print("=" * 80)
    
    try:
        with db_config.get_connection() as connection:
            cursor = connection.cursor()
            
            # Verificar si la tabla users existe
            cursor.execute("SHOW TABLES LIKE 'users'")
            if not cursor.fetchone():
                print("❌ La tabla users no existe")
                return
            
            print("✅ Tabla users existe")
            
            # Contar usuarios
            cursor.execute("SELECT COUNT(*) FROM users")
            user_count = cursor.fetchone()[0]
            print(f"📊 Total de usuarios: {user_count}")
            
            # Mostrar todos los usuarios
            cursor.execute("SELECT user_id, username, email, plan_type FROM users")
            users = cursor.fetchall()
            
            if users:
                print("\n📋 Usuarios disponibles:")
                print("-" * 80)
                for user in users:
                    print(f"   - ID: {user[0]}")
                    print(f"     Username: {user[1]}")
                    print(f"     Email: {user[2]}")
                    print(f"     Plan: {user[3]}")
                    print()
            else:
                print("📋 No hay usuarios en la base de datos")
            
            cursor.close()
            
    except Exception as e:
        print(f"❌ Error verificando usuarios: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_users() 