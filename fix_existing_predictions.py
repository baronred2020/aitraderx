#!/usr/bin/env python3
"""
Script para corregir predicciones existentes que no tienen porcentaje de éxito
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

def fix_existing_predictions():
    """Corregir predicciones existentes que no tienen porcentaje de éxito"""
    try:
        prediction_service = PredictionService()
        
        # Usuario demo
        user_id = "0bb94f45-4299-4506-b8c4-9d12d438c79c"
        
        with db_config.get_connection() as connection:
            cursor = connection.cursor()
            
            # Buscar predicciones completadas sin porcentaje de éxito
            query = """
                SELECT id, pair, direction, current_price, target_price, actual_price_at_expiry, 
                       prediction_success, success_percentage
                FROM user_predictions 
                WHERE user_id = %s 
                AND is_completed = TRUE 
                AND actual_price_at_expiry IS NOT NULL
                AND (success_percentage IS NULL OR success_percentage = 0)
            """
            
            cursor.execute(query, (user_id,))
            predictions = cursor.fetchall()
            
            print(f"🔍 Encontradas {len(predictions)} predicciones para corregir")
            
            fixed_count = 0
            
            for pred in predictions:
                pred_id, pair, direction, current_price, target_price, actual_price, prediction_success, success_percentage = pred
                
                # Calcular porcentaje de éxito
                new_success_percentage = prediction_service._calculate_success_percentage(
                    direction, current_price, target_price, actual_price
                )
                
                # Actualizar la predicción
                update_query = """
                    UPDATE user_predictions 
                    SET success_percentage = %s
                    WHERE id = %s
                """
                
                cursor.execute(update_query, (new_success_percentage, pred_id))
                fixed_count += 1
                
                print(f"✅ Predicción {pred_id}: {direction} {pair} - Porcentaje: {new_success_percentage:.2f}%")
            
            connection.commit()
            cursor.close()
            
            print(f"\n🎉 Se corrigieron {fixed_count} predicciones")
            return fixed_count
            
    except Exception as e:
        logger.error(f"Error corrigiendo predicciones: {e}")
        return 0

if __name__ == "__main__":
    print("🔧 CORRIGIENDO PREDICCIONES EXISTENTES")
    print("=" * 50)
    
    fixed = fix_existing_predictions()
    
    print(f"\n🏁 Proceso completado. {fixed} predicciones corregidas.") 