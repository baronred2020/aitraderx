#!/usr/bin/env python3
"""
Script para verificar la estructura de la tabla predictions
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

def check_table_structure():
    """Verificar la estructura de la tabla predictions"""
    print("🔍 Verificando estructura de la tabla predictions...")
    
    try:
        with db_config.get_connection() as connection:
            cursor = connection.cursor()
            
            # Mostrar estructura de la tabla
            cursor.execute("DESCRIBE predictions")
            columns = cursor.fetchall()
            print("📋 Estructura de la tabla predictions:")
            for column in columns:
                print(f"   - {column[0]}: {column[1]}")
            
            # Contar registros
            cursor.execute("SELECT COUNT(*) FROM predictions")
            count = cursor.fetchone()[0]
            print(f"📊 Registros en predictions: {count}")
            
            # Mostrar algunos registros de ejemplo si existen
            if count > 0:
                cursor.execute("SELECT * FROM predictions LIMIT 3")
                sample_records = cursor.fetchall()
                print("📄 Registros de ejemplo:")
                for i, record in enumerate(sample_records, 1):
                    print(f"   {i}. {record}")
            
            cursor.close()
            return True
            
    except Exception as e:
        print(f"❌ Error verificando estructura: {e}")
        return False

if __name__ == "__main__":
    success = check_table_structure()
    sys.exit(0 if success else 1)