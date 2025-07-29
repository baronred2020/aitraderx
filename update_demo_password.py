#!/usr/bin/env python3
import sys
from pathlib import Path
import hashlib
sys.path.append(str(Path(__file__).parent / "backend" / "src"))
from config.database_config import db_config
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def update_demo_password():
    print("🔧 Actualizando contraseña del usuario demo...")
    try:
        with db_config.get_connection() as connection:
            cursor = connection.cursor()
            
            # Hash de la contraseña demo123
            password_hash = hashlib.sha256('demo123'.encode()).hexdigest()
            
            # Actualizar contraseña del usuario demo_user
            update_query = "UPDATE users SET password_hash = %s WHERE username = 'demo_user'"
            cursor.execute(update_query, (password_hash,))
            
            connection.commit()
            cursor.close()
            
            print("✅ Contraseña del usuario demo actualizada exitosamente")
            print(f"   Usuario: demo_user")
            print(f"   Contraseña: demo123")
            print(f"   Hash: {password_hash}")
            return True
            
    except Exception as e:
        print(f"❌ Error actualizando contraseña: {e}")
        return False

if __name__ == "__main__":
    success = update_demo_password()
    sys.exit(0 if success else 1)