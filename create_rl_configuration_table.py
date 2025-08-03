#!/usr/bin/env python3
"""
Script para crear la tabla rl_user_configurations
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'src'))

from sqlalchemy import create_engine, text
from config.database_config import DatabaseConfig

def create_rl_configuration_table():
    """Crea la tabla rl_user_configurations"""
    print("🔄 Creando tabla rl_user_configurations...")
    print("=" * 50)
    
    # Configuración de base de datos
    db_config = DatabaseConfig()
    database_url = f"mysql+pymysql://{db_config.user}:{db_config.password}@{db_config.host}:{db_config.port}/{db_config.database}"
    engine = create_engine(database_url)
    
    try:
        with engine.connect() as connection:
            # Crear tabla rl_user_configurations
            create_table_query = text("""
                CREATE TABLE IF NOT EXISTS rl_user_configurations (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id VARCHAR(36) NOT NULL,
                    max_drawdown_percentage FLOAT DEFAULT 15.0,
                    max_position_size_percentage FLOAT DEFAULT 5.0,
                    min_confidence_threshold FLOAT DEFAULT 70.0,
                    retraining_frequency VARCHAR(20) DEFAULT 'monthly',
                    retraining_enabled BOOLEAN DEFAULT FALSE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                    UNIQUE KEY unique_user_config (user_id),
                    INDEX idx_user_id (user_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            
            connection.execute(create_table_query)
            connection.commit()
            
            print("✅ Tabla rl_user_configurations creada exitosamente")
            
            # Verificar que la tabla existe
            verify_query = text("SHOW TABLES LIKE 'rl_user_configurations'")
            result = connection.execute(verify_query)
            table_exists = result.fetchone()
            
            if table_exists:
                print("✅ Verificación: Tabla existe en la base de datos")
                
                # Mostrar estructura de la tabla
                structure_query = text("DESCRIBE rl_user_configurations")
                structure_result = connection.execute(structure_query)
                columns = structure_result.fetchall()
                
                print("\n📋 Estructura de la tabla:")
                for column in columns:
                    print(f"   - {column[0]}: {column[1]} ({column[2]})")
                    
                return True
            else:
                print("❌ Error: La tabla no se creó correctamente")
                return False
                
    except Exception as e:
        print(f"❌ Error creando tabla: {e}")
        return False

if __name__ == "__main__":
    success = create_rl_configuration_table()
    if success:
        print("\n🎉 ¡Tabla rl_user_configurations creada exitosamente!")
    else:
        print("\n💥 Error creando tabla rl_user_configurations") 