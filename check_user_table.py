#!/usr/bin/env python3
"""
Script para verificar la estructura de la tabla users
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'src'))

from config.database_config import db_config

def check_user_table():
    print("🔍 Verificando estructura de la tabla users...")
    
    try:
        with db_config.get_connection() as connection:
            cursor = connection.cursor()
            
            # Verificar estructura de la tabla
            cursor.execute("DESCRIBE users")
            columns = cursor.fetchall()
            
            print("📋 Estructura de la tabla users:")
            for column in columns:
                print(f"  - {column[0]}: {column[1]} ({column[2]})")
            
            # Verificar datos del usuario "user"
            cursor.execute("SELECT * FROM users WHERE username = 'user'")
            user_data = cursor.fetchone()
            
            if user_data:
                print("\n👤 Datos del usuario 'user':")
                cursor.execute("SHOW COLUMNS FROM users")
                column_names = [col[0] for col in cursor.fetchall()]
                
                for i, value in enumerate(user_data):
                    if i < len(column_names):
                        print(f"  - {column_names[i]}: {value}")
            else:
                print("\n❌ Usuario 'user' no encontrado en la tabla")
            
            cursor.close()
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_user_table() 