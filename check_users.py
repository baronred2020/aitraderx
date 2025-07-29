#!/usr/bin/env python3
"""
Script para verificar los usuarios existentes
"""
import sys
import os
from pathlib import Path

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent / "backend" / "src"))

from config.database_config import db_config
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_users():
    """Verificar los usuarios existentes"""
    print("🔍 Verificando usuarios existentes...")
    
    try:
        with db_config.get_connection() as connection:
            cursor = connection.cursor()
            
            # Mostrar todos los usuarios
            cursor.execute("SELECT * FROM users")
            users = cursor.fetchall()
            print(f"📊 Total usuarios: {len(users)}")
            
            for i, user in enumerate(users, 1):
                print(f"   {i}. {user}")
            
            cursor.close()
            return True
            
    except Exception as e:
        print(f"❌ Error verificando usuarios: {e}")
        return False

if __name__ == "__main__":
    success = check_users()
    sys.exit(0 if success else 1)