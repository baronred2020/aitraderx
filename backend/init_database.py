#!/usr/bin/env python3
"""
Script para inicializar la base de datos
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

def init_database():
    """Inicializar la base de datos"""
    print("🔧 Inicializando base de datos...")
    
    try:
        # 1. Crear pool de conexiones
        print("📡 Creando pool de conexiones...")
        if not db_config.create_connection_pool():
            print("❌ Error creando pool de conexiones")
            return False
        
        # 2. Probar conexión
        print("🔍 Probando conexión...")
        if not db_config.test_connection():
            print("❌ Error probando conexión")
            return False
        
        # 3. Crear tablas si no existen
        print("📋 Creando tablas...")
        if not db_config.create_tables_if_not_exist():
            print("❌ Error creando tablas")
            return False
        
        print("✅ Base de datos inicializada exitosamente")
        return True
        
    except Exception as e:
        print(f"❌ Error inicializando base de datos: {e}")
        return False

if __name__ == "__main__":
    success = init_database()
    sys.exit(0 if success else 1)