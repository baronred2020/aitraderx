#!/usr/bin/env python3
"""
Script para reiniciar el contador de predicciones diarias a las 00:00
Este script debe ejecutarse como un cron job o tarea programada
"""
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent / "src"))

from config.database_config import db_config

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/daily_reset.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def reset_daily_predictions():
    """Reiniciar el contador de predicciones diarias para todos los usuarios"""
    try:
        logger.info("🔄 Iniciando reinicio diario de predicciones...")
        
        with db_config.get_connection() as connection:
            cursor = connection.cursor()
            
            # Obtener la fecha de ayer
            yesterday = datetime.now().date() - timedelta(days=1)
            
            # Verificar si ya se ejecutó el reinicio hoy
            today = datetime.now().date()
            cursor.execute("""
                SELECT COUNT(*) FROM system_logs 
                WHERE action = 'daily_predictions_reset' 
                AND DATE(created_at) = %s
            """, (today,))
            
            already_reset = cursor.fetchone()[0] > 0
            
            if already_reset:
                logger.info("✅ El reinicio ya se ejecutó hoy")
                return True
            
            # Obtener todos los usuarios activos
            cursor.execute("""
                SELECT user_id, username, plan_type 
                FROM users 
                WHERE is_active = 1
            """)
            users = cursor.fetchall()
            
            logger.info(f"👥 Procesando {len(users)} usuarios activos...")
            
            reset_count = 0
            for user in users:
                user_id = user[0]
                username = user[1]
                plan_type = user[2]
                
                # Verificar predicciones de ayer
                cursor.execute("""
                    SELECT COUNT(*) FROM predictions 
                    WHERE user_id = %s 
                    AND DATE(prediction_date) = %s
                """, (user_id, yesterday))
                
                yesterday_predictions = cursor.fetchone()[0]
                
                if yesterday_predictions > 0:
                    logger.info(f"📊 Usuario {username}: {yesterday_predictions} predicciones ayer")
                    reset_count += 1
                
                # El contador se reinicia automáticamente al consultar predicciones de hoy
                # No necesitamos hacer nada más aquí
            
            # Registrar el reinicio en el log
            cursor.execute("""
                INSERT INTO system_logs (action, details, created_at) 
                VALUES (%s, %s, %s)
            """, (
                'daily_predictions_reset',
                f'Reinicio completado para {len(users)} usuarios',
                datetime.now()
            ))
            
            connection.commit()
            cursor.close()
            
            logger.info(f"✅ Reinicio diario completado. {reset_count} usuarios con predicciones procesados")
            return True
            
    except Exception as e:
        logger.error(f"❌ Error en reinicio diario: {e}")
        return False

def create_system_logs_table():
    """Crear tabla de logs del sistema si no existe"""
    try:
        with db_config.get_connection() as connection:
            cursor = connection.cursor()
            
            # Crear tabla system_logs si no existe
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_logs (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    action VARCHAR(100) NOT NULL,
                    details TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_action (action),
                    INDEX idx_created_at (created_at)
                )
            """)
            
            connection.commit()
            cursor.close()
            logger.info("✅ Tabla system_logs verificada/creada")
            return True
            
    except Exception as e:
        logger.error(f"❌ Error creando tabla system_logs: {e}")
        return False

if __name__ == "__main__":
    # Crear tabla de logs si no existe
    if not create_system_logs_table():
        sys.exit(1)
    
    # Ejecutar reinicio
    success = reset_daily_predictions()
    sys.exit(0 if success else 1)