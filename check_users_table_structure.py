#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "backend" / "src"))
from config.database_config import db_config
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_users_table_structure():
    print("🔍 Verificando estructura de la tabla users...")
    try:
        with db_config.get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute("DESCRIBE users")
            columns = cursor.fetchall()
            print("📋 Estructura de la tabla users:")
            for col in columns:
                print(f"   - {col[0]}: {col[1]}")
            cursor.close()
            return True
    except Exception as e:
        print(f"❌ Error verificando estructura: {e}")
        return False

if __name__ == "__main__":
    success = check_users_table_structure()
    sys.exit(0 if success else 1)