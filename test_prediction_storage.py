#!/usr/bin/env python3
"""
Script para verificar si las predicciones se están guardando correctamente
en la base de datos y si el contador está funcionando.
"""

import os
import sys
import asyncio
import json
from datetime import datetime, timedelta

# Agregar el directorio backend/src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'src'))

from services.prediction_service import PredictionService
from config.database_config import db_config

async def test_prediction_storage():
    """
    Probar el almacenamiento de predicciones y el contador
    """
    print("🔍 VERIFICACIÓN DE ALMACENAMIENTO DE PREDICCIONES")
    print("=" * 80)
    
    # 1. Verificar conexión a la base de datos
    print("📊 1. Verificando conexión a la base de datos...")
    if not db_config.test_connection():
        print("❌ No se puede conectar a la base de datos")
        return
    print("✅ Conexión a la base de datos exitosa")
    
    # 2. Verificar que la tabla user_predictions existe
    print("\n📋 2. Verificando tabla user_predictions...")
    try:
        with db_config.get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute("SHOW TABLES LIKE 'user_predictions'")
            if cursor.fetchone():
                print("✅ Tabla user_predictions existe")
                
                # Verificar estructura de la tabla
                cursor.execute("DESCRIBE user_predictions")
                columns = cursor.fetchall()
                print("📋 Estructura de la tabla:")
                for col in columns:
                    print(f"   - {col[0]}: {col[1]}")
            else:
                print("❌ Tabla user_predictions no existe")
                return
            cursor.close()
    except Exception as e:
        print(f"❌ Error verificando tabla: {e}")
        return
    
    # 3. Verificar predicciones existentes
    print("\n📈 3. Verificando predicciones existentes...")
    try:
        with db_config.get_connection() as connection:
            cursor = connection.cursor()
            
            # Contar total de predicciones
            cursor.execute("SELECT COUNT(*) FROM user_predictions")
            total_count = cursor.fetchone()[0]
            print(f"📊 Total de predicciones en la base de datos: {total_count}")
            
            # Contar predicciones de hoy
            today = datetime.now().date()
            cursor.execute("""
                SELECT COUNT(*) FROM user_predictions 
                WHERE DATE(created_at) = %s
            """, (today,))
            today_count = cursor.fetchone()[0]
            print(f"📊 Predicciones de hoy: {today_count}")
            
            # Mostrar las últimas 5 predicciones
            cursor.execute("""
                SELECT id, user_id, pair, direction, confidence, created_at, brain_type
                FROM user_predictions 
                ORDER BY created_at DESC 
                LIMIT 5
            """)
            recent_predictions = cursor.fetchall()
            
            if recent_predictions:
                print("📋 Últimas 5 predicciones:")
                for pred in recent_predictions:
                    print(f"   - ID: {pred[0]}, Usuario: {pred[1]}, Par: {pred[2]}, Dirección: {pred[3]}, Confianza: {pred[4]}%, Fecha: {pred[5]}, Brain: {pred[6]}")
            else:
                print("📋 No hay predicciones recientes")
            
            cursor.close()
    except Exception as e:
        print(f"❌ Error verificando predicciones existentes: {e}")
    
    # 4. Probar generación de una nueva predicción
    print("\n🧪 4. Probando generación de nueva predicción...")
    try:
        prediction_service = PredictionService()
        
        # Verificar límites antes de generar
        user_id = "4dabfd30-483d-4fa0-a8d0-bd151a46340f"  # Usuario válido que ya existe
        plan_type = "starter"
        style = "day_trading"
        
        print(f"🔍 Verificando límites para usuario: {user_id}")
        limits_before = await prediction_service.can_generate_prediction(user_id, style, plan_type)
        print(f"   - Puede generar: {limits_before.get('can_generate', False)}")
        print(f"   - Predicciones restantes: {limits_before.get('remaining_predictions', 0)}")
        print(f"   - Máximo por día: {limits_before.get('max_predictions_per_day', 0)}")
        
        # Generar predicción
        print(f"\n🎯 Generando predicción para EURUSD...")
        prediction_result = await prediction_service.generate_prediction(
            user_id=user_id,
            pair="EURUSD",
            brain_type="brain_max",
            style=style
        )
        
        if prediction_result:
            print("✅ Predicción generada exitosamente")
            print(f"   - ID: {prediction_result.get('id', 'N/A')}")
            print(f"   - Par: {prediction_result.get('pair')}")
            print(f"   - Dirección: {prediction_result.get('direction')}")
            print(f"   - Confianza: {prediction_result.get('confidence')}%")
            print(f"   - Precisión: {prediction_result.get('precision')}%")
            print(f"   - Win Rate: {prediction_result.get('win_rate')}%")
            print(f"   - Brain Type: {prediction_result.get('brain_type')}")
            print(f"   - Creada: {prediction_result.get('created_at')}")
            print(f"   - Expira: {prediction_result.get('expires_at')}")
        else:
            print("❌ No se pudo generar la predicción")
            return
        
        # Verificar límites después de generar
        print(f"\n🔍 Verificando límites después de generar...")
        limits_after = await prediction_service.can_generate_prediction(user_id, style, plan_type)
        print(f"   - Puede generar: {limits_after.get('can_generate', False)}")
        print(f"   - Predicciones restantes: {limits_after.get('remaining_predictions', 0)}")
        print(f"   - Máximo por día: {limits_after.get('max_predictions_per_day', 0)}")
        
        # Verificar si el contador se actualizó
        if limits_before.get('remaining_predictions', 0) > limits_after.get('remaining_predictions', 0):
            print("✅ Contador de predicciones se actualizó correctamente")
        else:
            print("⚠️  El contador no se actualizó como se esperaba")
        
    except Exception as e:
        print(f"❌ Error generando predicción: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. Verificar que la predicción se guardó en la base de datos
    print("\n💾 5. Verificando que la predicción se guardó en la base de datos...")
    try:
        with db_config.get_connection() as connection:
            cursor = connection.cursor()
            
            # Buscar la predicción recién creada
            cursor.execute("""
                SELECT id, user_id, pair, direction, confidence, `precision`, win_rate, brain_type, created_at
                FROM user_predictions 
                WHERE user_id = %s AND pair = 'EURUSD'
                ORDER BY created_at DESC 
                LIMIT 1
            """, (user_id,))
            
            saved_prediction = cursor.fetchone()
            if saved_prediction:
                print("✅ Predicción encontrada en la base de datos:")
                print(f"   - ID: {saved_prediction[0]}")
                print(f"   - Usuario: {saved_prediction[1]}")
                print(f"   - Par: {saved_prediction[2]}")
                print(f"   - Dirección: {saved_prediction[3]}")
                print(f"   - Confianza: {saved_prediction[4]}%")
                print(f"   - Precisión: {saved_prediction[5]}%")
                print(f"   - Win Rate: {saved_prediction[6]}%")
                print(f"   - Brain Type: {saved_prediction[7]}")
                print(f"   - Creada: {saved_prediction[8]}")
            else:
                print("❌ No se encontró la predicción en la base de datos")
            
            cursor.close()
    except Exception as e:
        print(f"❌ Error verificando predicción guardada: {e}")
    
    # 6. Probar obtención del historial
    print("\n📚 6. Probando obtención del historial...")
    try:
        history = await prediction_service.get_prediction_history(user_id, limit=10)
        print(f"📊 Historial obtenido: {len(history)} predicciones")
        
        if history:
            print("📋 Últimas predicciones del historial:")
            for i, pred in enumerate(history[:3]):
                print(f"   {i+1}. ID: {pred.get('id')}, Par: {pred.get('pair')}, Dirección: {pred.get('direction')}, Confianza: {pred.get('confidence')}%")
        else:
            print("📋 No hay predicciones en el historial")
            
    except Exception as e:
        print(f"❌ Error obteniendo historial: {e}")
    
    # 7. Verificar estadísticas del usuario
    print("\n📊 7. Verificando estadísticas del usuario...")
    try:
        stats = await prediction_service.get_user_stats(user_id)
        print("📈 Estadísticas del usuario:")
        print(f"   - Total predicciones: {stats.get('total_predictions', 0)}")
        print(f"   - Predicciones exitosas: {stats.get('successful_predictions', 0)}")
        print(f"   - Tasa de éxito: {stats.get('success_rate', 0):.2f}%")
        print(f"   - Porcentaje promedio de éxito: {stats.get('average_success_percentage', 0):.2f}%")
        print(f"   - Mejor par: {stats.get('best_pair', 'N/A')}")
        print(f"   - Predicciones hoy: {stats.get('total_predictions_today', 0)}")
        
    except Exception as e:
        print(f"❌ Error obteniendo estadísticas: {e}")
    
    print("\n" + "=" * 80)
    print("🏁 VERIFICACIÓN COMPLETADA")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_prediction_storage()) 