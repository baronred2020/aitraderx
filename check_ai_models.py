#!/usr/bin/env python3
"""
Script para verificar los modelos de IA existentes
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

def check_ai_models():
    """Verificar los modelos de IA existentes"""
    print("🔍 Verificando modelos de IA existentes...")
    
    try:
        with db_config.get_connection() as connection:
            cursor = connection.cursor()
            
            # Verificar si la tabla existe
            cursor.execute("SHOW TABLES LIKE 'ai_models'")
            if not cursor.fetchone():
                print("❌ La tabla ai_models no existe")
                return False
            
            # Mostrar todos los modelos
            cursor.execute("SELECT * FROM ai_models")
            models = cursor.fetchall()
            print(f"📊 Total modelos: {len(models)}")
            
            for i, model in enumerate(models, 1):
                print(f"   {i}. {model}")
            
            cursor.close()
            return True
            
    except Exception as e:
        print(f"❌ Error verificando modelos: {e}")
        return False

if __name__ == "__main__":
    success = check_ai_models()
    sys.exit(0 if success else 1)