#!/usr/bin/env python3
"""
Script para verificar todas las predicciones en la base de datos
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'src'))

from config.database_config import db_config
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_all_predictions():
    """Verificar todas las predicciones en la base de datos"""
    try:
        with db_config.get_connection() as connection:
            cursor = connection.cursor()
            
            # Verificar todas las predicciones
            query = """
                SELECT id, user_id, pair, direction, current_price, target_price, actual_price_at_expiry, 
                       prediction_success, success_percentage, is_completed, created_at, expires_at
                FROM user_predictions 
                ORDER BY created_at DESC
            """
            
            cursor.execute(query)
            predictions = cursor.fetchall()
            
            print(f"🔍 Encontradas {len(predictions)} predicciones totales en la base de datos")
            
            for pred in predictions:
                pred_id, user_id, pair, direction, current_price, target_price, actual_price, prediction_success, success_percentage, is_completed, created_at, expires_at = pred
                
                print(f"\n📊 Predicción {pred_id}:")
                print(f"   - Usuario: {user_id}")
                print(f"   - Par: {pair}")
                print(f"   - Dirección: {direction}")
                print(f"   - Precio actual: {current_price}")
                print(f"   - Precio objetivo: {target_price}")
                print(f"   - Precio real: {actual_price}")
                print(f"   - Éxito: {prediction_success}")
                print(f"   - Porcentaje: {success_percentage}")
                print(f"   - Completada: {is_completed}")
                print(f"   - Creada: {created_at}")
                print(f"   - Expira: {expires_at}")
            
            cursor.close()
            
    except Exception as e:
        logger.error(f"Error verificando predicciones: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🔍 VERIFICANDO TODAS LAS PREDICCIONES")
    print("=" * 50)
    
    check_all_predictions()
    
    print("\n🏁 Verificación completada.") 