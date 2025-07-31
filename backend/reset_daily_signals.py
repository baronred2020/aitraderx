#!/usr/bin/env python3
"""
Reset Daily Signals Script
==========================
Script para resetear diariamente los contadores de señales de todos los usuarios
"""

import os
import sys
import logging
from datetime import datetime, date
from pathlib import Path

# Agregar el directorio backend al path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/daily_signal_reset.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def reset_daily_signals():
    """Resetear contadores diarios de señales"""
    try:
        logger.info("Iniciando reset diario de señales...")
        
        # Importar después de configurar el path
        from src.services.signal_service import SignalService
        from src.models.database_models import get_db_session
        
        # Crear instancia del servicio
        signal_service = SignalService()
        
        # Obtener sesión de base de datos
        db_session = get_db_session()
        
        try:
            # Resetear contadores de señales
            today = date.today()
            
            # Actualizar todos los registros de límites de señales
            from src.models.signal_models import UserSignalLimit
            
            # Obtener todos los límites de señales
            signal_limits = db_session.query(UserSignalLimit).all()
            
            reset_count = 0
            for signal_limit in signal_limits:
                # Verificar si necesita reset
                if signal_limit.last_reset_date != today:
                    signal_limit.signals_used_today = 0
                    signal_limit.last_reset_date = today
                    reset_count += 1
                    logger.info(f"Reset signal limits for user {signal_limit.user_id}")
            
            # Commit de los cambios
            db_session.commit()
            
            logger.info(f"Reset completado. {reset_count} usuarios actualizados.")
            
            # Registrar en logs del sistema
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "action": "daily_signal_reset",
                "users_updated": reset_count,
                "status": "success"
            }
            
            # Guardar log
            log_file = "logs/system_logs.json"
            os.makedirs("logs", exist_ok=True)
            
            import json
            try:
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                logs = []
            
            logs.append(log_entry)
            
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
            
            logger.info("Reset diario de señales completado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error durante el reset: {e}")
            db_session.rollback()
            return False
            
        finally:
            db_session.close()
            
    except Exception as e:
        logger.error(f"Error crítico en reset diario de señales: {e}")
        return False

def main():
    """Función principal"""
    logger.info("=== INICIO RESET DIARIO DE SEÑALES ===")
    
    success = reset_daily_signals()
    
    if success:
        logger.info("=== RESET DIARIO DE SEÑALES COMPLETADO ===")
        sys.exit(0)
    else:
        logger.error("=== ERROR EN RESET DIARIO DE SEÑALES ===")
        sys.exit(1)

if __name__ == "__main__":
    main() 