#!/usr/bin/env python3
"""
Script para debuggear el cálculo de porcentaje de éxito
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

def debug_percentage_calculation():
    """Debuggear el cálculo de porcentaje de éxito"""
    try:
        prediction_service = PredictionService()
        
        # Usuario demo
        user_id = "0bb94f45-4299-4506-b8c4-9d12d438c79c"
        
        with db_config.get_connection() as connection:
            cursor = connection.cursor()
            
            # Buscar predicciones completadas sin porcentaje
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
            
            print(f"🔍 Encontradas {len(predictions)} predicciones completadas sin porcentaje")
            
            for pred in predictions:
                pred_id, pair, direction, current_price, target_price, actual_price, prediction_success, success_percentage = pred
                
                print(f"\n📊 Debug Predicción {pred_id}:")
                print(f"   - Par: {pair}")
                print(f"   - Dirección: {direction}")
                print(f"   - Precio actual: {current_price}")
                print(f"   - Precio objetivo: {target_price}")
                print(f"   - Precio real: {actual_price}")
                print(f"   - Éxito: {prediction_success}")
                print(f"   - Porcentaje actual: {success_percentage}")
                
                # Debug del cálculo
                print(f"\n🔧 Debug del cálculo:")
                
                try:
                    # Probar cálculo de éxito
                    calculated_success = prediction_service._calculate_prediction_success(
                        direction, current_price, target_price, actual_price
                    )
                    print(f"   - Éxito calculado: {calculated_success}")
                    
                    # Probar cálculo de porcentaje
                    calculated_percentage = prediction_service._calculate_success_percentage(
                        direction, current_price, target_price, actual_price
                    )
                    print(f"   - Porcentaje calculado: {calculated_percentage:.2f}%")
                    
                    # Verificar si los cálculos coinciden
                    if calculated_success == prediction_success:
                        print(f"   ✅ Éxito coincide")
                    else:
                        print(f"   ❌ Éxito NO coincide: {calculated_success} vs {prediction_success}")
                    
                    if calculated_percentage > 0:
                        print(f"   ✅ Porcentaje calculado correctamente")
                        
                        # Actualizar la predicción
                        update_query = """
                            UPDATE user_predictions 
                            SET success_percentage = %s
                            WHERE id = %s
                        """
                        
                        cursor.execute(update_query, (calculated_percentage, pred_id))
                        print(f"   ✅ Predicción actualizada con porcentaje: {calculated_percentage:.2f}%")
                    else:
                        print(f"   ❌ Error en cálculo de porcentaje")
                        
                except Exception as e:
                    print(f"   ❌ Error en cálculo: {e}")
                    import traceback
                    traceback.print_exc()
            
            connection.commit()
            cursor.close()
            
    except Exception as e:
        logger.error(f"Error en debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🔧 DEBUGGEANDO CÁLCULO DE PORCENTAJE")
    print("=" * 50)
    
    debug_percentage_calculation()
    
    print("\n🏁 Debug completado.") 