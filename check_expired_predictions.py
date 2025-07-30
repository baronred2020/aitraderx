#!/usr/bin/env python3
"""
Script para verificar predicciones expiradas que necesitan ser completadas
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'src'))

from config.database_config import db_config
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_expired_predictions():
    """Verificar predicciones expiradas que necesitan ser completadas"""
    try:
        # Usuario demo
        user_id = "0bb94f45-4299-4506-b8c4-9d12d438c79c"
        
        with db_config.get_connection() as connection:
            cursor = connection.cursor()
            
            # Buscar todas las predicciones del usuario
            query = """
                SELECT id, pair, direction, current_price, target_price, actual_price_at_expiry, 
                       prediction_success, success_percentage, is_completed, created_at, expires_at
                FROM user_predictions 
                WHERE user_id = %s 
                ORDER BY created_at DESC
            """
            
            cursor.execute(query, (user_id,))
            predictions = cursor.fetchall()
            
            print(f"🔍 Encontradas {len(predictions)} predicciones totales")
            
            expired_count = 0
            completed_count = 0
            pending_count = 0
            
            for pred in predictions:
                pred_id, pair, direction, current_price, target_price, actual_price, prediction_success, success_percentage, is_completed, created_at, expires_at = pred
                
                print(f"\n📊 Predicción {pred_id}:")
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
                
                if is_completed:
                    completed_count += 1
                    if actual_price is None:
                        print(f"   ❌ PROBLEMA: Completada pero sin precio real")
                    elif success_percentage is None:
                        print(f"   ❌ PROBLEMA: Completada pero sin porcentaje")
                else:
                    # Verificar si está expirada
                    from datetime import datetime
                    try:
                        # Manejar diferentes formatos de fecha
                        if expires_at and isinstance(expires_at, str):
                            # Remover 'Z' si existe y convertir
                            expires_clean = expires_at.replace('Z', '')
                            expires_dt = datetime.fromisoformat(expires_clean)
                            now = datetime.now()
                            
                            if now > expires_dt:
                                expired_count += 1
                                print(f"   ⏰ EXPIRADA: Necesita completarse")
                            else:
                                pending_count += 1
                                print(f"   ⏳ PENDIENTE: Aún no expira")
                        else:
                            pending_count += 1
                            print(f"   ⏳ PENDIENTE: Fecha de expiración no válida")
                    except Exception as e:
                        pending_count += 1
                        print(f"   ⏳ PENDIENTE: Error procesando fecha: {e}")
            
            cursor.close()
            
            print(f"\n📈 RESUMEN:")
            print(f"   - Total: {len(predictions)}")
            print(f"   - Completadas: {completed_count}")
            print(f"   - Expiradas sin completar: {expired_count}")
            print(f"   - Pendientes: {pending_count}")
            
            return {
                'total': len(predictions),
                'completed': completed_count,
                'expired': expired_count,
                'pending': pending_count
            }
            
    except Exception as e:
        logger.error(f"Error verificando predicciones: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🔍 VERIFICANDO PREDICCIONES EXPIRADAS")
    print("=" * 50)
    
    result = check_expired_predictions()
    
    if result:
        print(f"\n🏁 Verificación completada.")
        if result['expired'] > 0:
            print(f"⚠️  Hay {result['expired']} predicciones expiradas que necesitan completarse")
    else:
        print(f"\n❌ Error en la verificación") 