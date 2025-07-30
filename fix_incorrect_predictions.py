#!/usr/bin/env python3
"""
Script para corregir predicciones marcadas como incorrectas pero con porcentaje pendiente
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

def fix_incorrect_predictions():
    """Corregir predicciones marcadas como incorrectas pero con porcentaje pendiente"""
    try:
        prediction_service = PredictionService()
        
        # Usuario demo
        user_id = "0bb94f45-4299-4506-b8c4-9d12d438c79c"
        
        with db_config.get_connection() as connection:
            cursor = connection.cursor()
            
            # Buscar predicciones marcadas como incorrectas pero con porcentaje pendiente
            query = """
                SELECT id, pair, direction, current_price, target_price, actual_price_at_expiry, 
                       prediction_success, success_percentage
                FROM user_predictions 
                WHERE user_id = %s 
                AND is_completed = TRUE 
                AND actual_price_at_expiry IS NOT NULL
                AND prediction_success IS NOT NULL
                AND (success_percentage IS NULL OR success_percentage = 0)
            """
            
            cursor.execute(query, (user_id,))
            predictions = cursor.fetchall()
            
            print(f"🔍 Encontradas {len(predictions)} predicciones incorrectas para corregir")
            
            fixed_count = 0
            
            for pred in predictions:
                pred_id, pair, direction, current_price, target_price, actual_price, prediction_success, success_percentage = pred
                
                print(f"\n📊 Predicción {pred_id}:")
                print(f"   - Par: {pair}")
                print(f"   - Dirección: {direction}")
                print(f"   - Precio actual: {current_price}")
                print(f"   - Precio objetivo: {target_price}")
                print(f"   - Precio real: {actual_price}")
                print(f"   - Éxito: {prediction_success}")
                print(f"   - Porcentaje actual: {success_percentage}")
                
                # Calcular porcentaje de éxito
                new_success_percentage = prediction_service._calculate_success_percentage(
                    direction, current_price, target_price, actual_price
                )
                
                print(f"   - Porcentaje calculado: {new_success_percentage:.2f}%")
                
                # Actualizar la predicción
                update_query = """
                    UPDATE user_predictions 
                    SET success_percentage = %s
                    WHERE id = %s
                """
                
                cursor.execute(update_query, (new_success_percentage, pred_id))
                fixed_count += 1
                
                print(f"   ✅ Actualizada")
            
            connection.commit()
            cursor.close()
            
            print(f"\n🎉 Se corrigieron {fixed_count} predicciones")
            return fixed_count
            
    except Exception as e:
        logger.error(f"Error corrigiendo predicciones incorrectas: {e}")
        import traceback
        traceback.print_exc()
        return 0

if __name__ == "__main__":
    print("🔧 CORRIGIENDO PREDICCIONES INCORRECTAS")
    print("=" * 50)
    
    fixed = fix_incorrect_predictions()
    
    print(f"\n🏁 Proceso completado. {fixed} predicciones corregidas.") 