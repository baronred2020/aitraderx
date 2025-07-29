#!/usr/bin/env python3
"""
Script para verificar la estructura de la tabla users
"""
import sys
import os
from pathlib import Path

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent / "src"))

from config.database_config import db_config
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_users_table():
    """Verificar la estructura de la tabla users"""
    print("🔍 Verificando estructura de la tabla users...")
    
    try:
        with db_config.get_connection() as connection:
            cursor = connection.cursor()
            
            # Verificar si la tabla users existe
            cursor.execute("SHOW TABLES LIKE 'users'")
            if not cursor.fetchone():
                print("❌ La tabla users no existe")
                return False
            
            # Mostrar estructura de la tabla
            cursor.execute("DESCRIBE users")
            columns = cursor.fetchall()
            print("📋 Estructura de la tabla users:")
            for column in columns:
                print(f"   - {column[0]}: {column[1]}")
            
            # Contar registros
            cursor.execute("SELECT COUNT(*) FROM users")
            count = cursor.fetchone()[0]
            print(f"📊 Registros en users: {count}")
            
            # Mostrar algunos registros de ejemplo
            if count > 0:
                cursor.execute("SELECT * FROM users LIMIT 3")
                sample_records = cursor.fetchall()
                print("📄 Registros de ejemplo:")
                for i, record in enumerate(sample_records, 1):
                    print(f"   {i}. {record}")
            
            cursor.close()
            return True
            
    except Exception as e:
        print(f"❌ Error verificando tabla users: {e}")
        return False

if __name__ == "__main__":
    success = check_users_table()
    sys.exit(0 if success else 1)