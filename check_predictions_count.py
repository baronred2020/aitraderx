#!/usr/bin/env python3
"""
Script para verificar el conteo de predicciones en la base de datos
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

def check_predictions_count():
    """Verificar el conteo de predicciones en la base de datos"""
    print("🔍 Verificando predicciones en la base de datos...")
    
    try:
        with db_config.get_connection() as connection:
            cursor = connection.cursor()
            
            # Contar todas las predicciones
            cursor.execute("SELECT COUNT(*) FROM predictions")
            total_predictions = cursor.fetchone()[0]
            print(f"📊 Total predicciones en la base de datos: {total_predictions}")
            
            # Contar predicciones del usuario demo
            demo_user_id = "0bb94f45-4299-4506-b8c4-9d12d438c79c"
            cursor.execute("SELECT COUNT(*) FROM predictions WHERE user_id = %s", (demo_user_id,))
            user_predictions = cursor.fetchone()[0]
            print(f"👤 Predicciones del usuario demo: {user_predictions}")
            
            # Contar predicciones de hoy
            from datetime import datetime
            today = datetime.now().date()
            cursor.execute("SELECT COUNT(*) FROM predictions WHERE user_id = %s AND DATE(prediction_date) = %s", (demo_user_id, today))
            today_predictions = cursor.fetchone()[0]
            print(f"📅 Predicciones de hoy del usuario demo: {today_predictions}")
            
            # Mostrar algunas predicciones de ejemplo
            if user_predictions > 0:
                cursor.execute("SELECT prediction_id, symbol, predicted_signal, confidence, prediction_date FROM predictions WHERE user_id = %s ORDER BY prediction_date DESC LIMIT 5", (demo_user_id,))
                sample_predictions = cursor.fetchall()
                print("📄 Últimas 5 predicciones del usuario demo:")
                for i, pred in enumerate(sample_predictions, 1):
                    print(f"   {i}. ID: {pred[0][:8]}..., Par: {pred[1]}, Señal: {pred[2]}, Confianza: {pred[3]}%, Fecha: {pred[4]}")
            
            cursor.close()
            return True
            
    except Exception as e:
        print(f"❌ Error verificando predicciones: {e}")
        return False

if __name__ == "__main__":
    success = check_predictions_count()
    sys.exit(0 if success else 1)