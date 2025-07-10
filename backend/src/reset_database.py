"""
Script para Resetear Base de Datos
=================================
Elimina todas las tablas y las recrea
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.database_config import engine, SessionLocal
from sqlalchemy import text

def reset_database():
    """Elimina todas las tablas y las recrea"""
    try:
        # Conectar a la base de datos
        with engine.connect() as connection:
            # Deshabilitar verificación de claves foráneas
            connection.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
            
            # Eliminar tablas en orden
            tables = [
                "subscription_audit",
                "subscription_payments", 
                "usage_metrics",
                "user_subscriptions",
                "subscription_upgrades",
                "subscription_plans",
                "users"
            ]
            
            for table in tables:
                try:
                    connection.execute(text(f"DROP TABLE IF EXISTS {table}"))
                    print(f"Tabla {table} eliminada")
                except Exception as e:
                    print(f"Error eliminando tabla {table}: {e}")
            
            # Habilitar verificación de claves foráneas
            connection.execute(text("SET FOREIGN_KEY_CHECKS = 1"))
            connection.commit()
            
        print("Base de datos reseteada correctamente")
        
        # Recrear tablas
        from init_database import main
        main()
        
    except Exception as e:
        print(f"Error reseteando base de datos: {e}")

if __name__ == "__main__":
    reset_database() 