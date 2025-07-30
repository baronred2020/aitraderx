#!/usr/bin/env python3
"""
Script para debuggear el problema de guardado de predicciones
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'src'))

from services.prediction_service import PredictionService
from config.database_config import db_config
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_prediction_save():
    """Debuggear el guardado de predicciones"""
    try:
        prediction_service = PredictionService()
        
        # Usuario demo
        user_id = "0bb94f45-4299-4506-b8c4-9d12d438c79c"
        
        # Crear una predicción de prueba
        test_prediction = {
            "id": None,
            "pair": "EURUSD",
            "direction": "up",
            "current_price": 1.15701,
            "target_price": 1.15800,
            "confidence": 85.5,
            "timeframe": "15M",
            "reasoning": "Test prediction",
            "brain_type": "brain_max",
            "created_at": "2025-07-29T22:30:00",
            "expires_at": "2025-07-29T22:45:00",
            "time_remaining": 900,
            "is_completed": False,
            "actual_price_at_expiry": None,
            "prediction_success": None,
            "success_percentage": None
        }
        
        print("🔍 Intentando guardar predicción de prueba...")
        
        # Intentar guardar
        prediction_id = prediction_service._save_prediction_to_db(user_id, test_prediction)
        
        print(f"📊 Resultado del guardado: ID = {prediction_id}")
        
        if prediction_id > 0:
            print("✅ Predicción guardada exitosamente")
            
            # Verificar que se guardó correctamente
            with db_config.get_connection() as connection:
                cursor = connection.cursor()
                
                query = "SELECT * FROM user_predictions WHERE id = %s"
                cursor.execute(query, (prediction_id,))
                result = cursor.fetchone()
                
                if result:
                    print("✅ Predicción encontrada en la base de datos")
                    print(f"   - ID: {result[0]}")
                    print(f"   - Par: {result[2]}")
                    print(f"   - Dirección: {result[3]}")
                    print(f"   - Precio actual: {result[4]}")
                    print(f"   - Precio objetivo: {result[5]}")
                    print(f"   - Confianza: {result[6]}")
                    print(f"   - Brain: {result[9]}")
                else:
                    print("❌ Predicción no encontrada en la base de datos")
                
                cursor.close()
        else:
            print("❌ Error al guardar la predicción")
            
            # Verificar la estructura de la tabla
            with db_config.get_connection() as connection:
                cursor = connection.cursor()
                
                cursor.execute("DESCRIBE user_predictions")
                columns = cursor.fetchall()
                
                print("📋 Estructura de la tabla user_predictions:")
                for col in columns:
                    print(f"   - {col[0]}: {col[1]}")
                
                cursor.close()
        
    except Exception as e:
        logger.error(f"Error en debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🔧 DEBUGGEANDO GUARDADO DE PREDICCIONES")
    print("=" * 50)
    
    debug_prediction_save()
    
    print("\n🏁 Debug completado.") 