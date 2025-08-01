#!/usr/bin/env python3
"""
Script para probar el sistema de métricas reales basadas en predicciones completadas
"""

import os
import sys
import asyncio
from datetime import datetime, timedelta

# Agregar el directorio backend/src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'src'))

from services.prediction_service import PredictionService
from services.real_metrics_calculator import RealMetricsCalculator
from config.database_config import db_config

async def test_real_metrics():
    """
    Probar el sistema de métricas reales
    """
    print("🧪 PRUEBA DEL SISTEMA DE MÉTRICAS REALES")
    print("=" * 80)
    
    # 1. Verificar conexión a la base de datos
    print("📊 1. Verificando conexión a la base de datos...")
    if not db_config.test_connection():
        print("❌ No se puede conectar a la base de datos")
        return
    print("✅ Conexión a la base de datos exitosa")
    
    # 2. Inicializar servicios
    print("\n🔧 2. Inicializando servicios...")
    prediction_service = PredictionService()
    real_metrics_calculator = RealMetricsCalculator()
    print("✅ Servicios inicializados")
    
    # 3. Verificar predicciones existentes
    print("\n📈 3. Verificando predicciones existentes...")
    user_id = "4dabfd30-483d-4fa0-a8d0-bd151a46340f"  # Usuario válido
    
    try:
        with db_config.get_connection() as connection:
            cursor = connection.cursor()
            
            # Contar predicciones totales
            cursor.execute("SELECT COUNT(*) FROM user_predictions WHERE user_id = %s", (user_id,))
            total_predictions = cursor.fetchone()[0]
            
            # Contar predicciones completadas
            cursor.execute("SELECT COUNT(*) FROM user_predictions WHERE user_id = %s AND is_completed = 1", (user_id,))
            completed_predictions = cursor.fetchone()[0]
            
            # Contar predicciones expiradas no completadas
            cursor.execute("SELECT COUNT(*) FROM user_predictions WHERE user_id = %s AND expires_at < NOW() AND is_completed = 0", (user_id,))
            expired_not_completed = cursor.fetchone()[0]
            
            cursor.close()
            
            print(f"📊 Total de predicciones: {total_predictions}")
            print(f"📊 Predicciones completadas: {completed_predictions}")
            print(f"📊 Predicciones expiradas no completadas: {expired_not_completed}")
            
    except Exception as e:
        print(f"❌ Error verificando predicciones: {e}")
        return
    
    # 4. Completar predicciones expiradas con resultados reales
    print("\n🔄 4. Completando predicciones expiradas con resultados reales...")
    try:
        completion_result = await real_metrics_calculator.complete_expired_predictions(user_id)
        print(f"✅ Resultado de completar predicciones:")
        print(f"   - Total expiradas: {completion_result.get('total_expired', 0)}")
        print(f"   - Completadas: {completion_result.get('completed', 0)}")
        print(f"   - Fallidas: {completion_result.get('failed', 0)}")
        
    except Exception as e:
        print(f"❌ Error completando predicciones: {e}")
    
    # 5. Calcular métricas reales
    print("\n📊 5. Calculando métricas reales...")
    try:
        real_metrics = await real_metrics_calculator.calculate_real_metrics_for_user(user_id)
        
        if real_metrics:
            print("✅ Métricas reales calculadas:")
            print(f"   - Total predicciones: {real_metrics.get('total_predictions', 0)}")
            print(f"   - Predicciones exitosas: {real_metrics.get('successful_predictions', 0)}")
            print(f"   - Win Rate: {real_metrics.get('win_rate', 0):.2f}%")
            print(f"   - Precisión: {real_metrics.get('precision', 0):.2f}%")
            print(f"   - Confianza promedio: {real_metrics.get('average_confidence', 0):.2f}%")
            print(f"   - Porcentaje de éxito promedio: {real_metrics.get('average_success_percentage', 0):.2f}%")
            print(f"   - Mejor par: {real_metrics.get('best_pair', 'N/A')}")
            print(f"   - Mejor brain type: {real_metrics.get('best_brain_type', 'N/A')}")
            
            # Mostrar métricas por par
            metrics_by_pair = real_metrics.get('metrics_by_pair', {})
            if metrics_by_pair:
                print("\n📈 Métricas por par:")
                for pair, metrics in metrics_by_pair.items():
                    print(f"   {pair}:")
                    print(f"     - Total: {metrics.get('total_predictions', 0)}")
                    print(f"     - Exitosas: {metrics.get('successful_predictions', 0)}")
                    print(f"     - Win Rate: {metrics.get('win_rate', 0):.2f}%")
                    print(f"     - Precisión: {metrics.get('precision', 0):.2f}%")
            
            # Mostrar métricas por brain type
            metrics_by_brain = real_metrics.get('metrics_by_brain', {})
            if metrics_by_brain:
                print("\n🧠 Métricas por brain type:")
                for brain, metrics in metrics_by_brain.items():
                    print(f"   {brain}:")
                    print(f"     - Total: {metrics.get('total_predictions', 0)}")
                    print(f"     - Exitosas: {metrics.get('successful_predictions', 0)}")
                    print(f"     - Win Rate: {metrics.get('win_rate', 0):.2f}%")
                    print(f"     - Precisión: {metrics.get('precision', 0):.2f}%")
            
            # Mostrar rendimiento reciente
            recent_performance = real_metrics.get('recent_performance', [])
            if recent_performance:
                print("\n📋 Rendimiento reciente (últimas 5 predicciones):")
                for i, pred in enumerate(recent_performance[:5], 1):
                                    print(f"   {i}. ID: {pred.get('id')}, Par: {pred.get('pair')}, Dirección: {pred.get('direction')}")
                success_percentage = pred.get('success_percentage', 0)
                if success_percentage is None:
                    success_percentage = 0
                print(f"      Éxito: {pred.get('prediction_success')}, Porcentaje: {success_percentage:.2f}%")
                print(f"      Confianza: {pred.get('confidence', 0):.2f}%, Brain: {pred.get('brain_type')}")
        else:
            print("⚠️  No se pudieron calcular métricas reales")
            
    except Exception as e:
        print(f"❌ Error calculando métricas reales: {e}")
        import traceback
        traceback.print_exc()
    
    # 6. Probar métricas específicas por brain type
    print("\n🧠 6. Probando métricas específicas por brain type...")
    try:
        brain_metrics = await real_metrics_calculator.calculate_real_metrics_for_user(
            user_id, brain_type="brain_max"
        )
        
        if brain_metrics:
            print("✅ Métricas para brain_max:")
            print(f"   - Total predicciones: {brain_metrics.get('total_predictions', 0)}")
            print(f"   - Win Rate: {brain_metrics.get('win_rate', 0):.2f}%")
            print(f"   - Precisión: {brain_metrics.get('precision', 0):.2f}%")
        else:
            print("⚠️  No hay métricas para brain_max")
            
    except Exception as e:
        print(f"❌ Error calculando métricas por brain type: {e}")
    
    # 7. Probar métricas específicas por par
    print("\n💱 7. Probando métricas específicas por par...")
    try:
        pair_metrics = await real_metrics_calculator.calculate_real_metrics_for_user(
            user_id, pair="EURUSD"
        )
        
        if pair_metrics:
            print("✅ Métricas para EURUSD:")
            print(f"   - Total predicciones: {pair_metrics.get('total_predictions', 0)}")
            print(f"   - Win Rate: {pair_metrics.get('win_rate', 0):.2f}%")
            print(f"   - Precisión: {pair_metrics.get('precision', 0):.2f}%")
        else:
            print("⚠️  No hay métricas para EURUSD")
            
    except Exception as e:
        print(f"❌ Error calculando métricas por par: {e}")
    
    # 8. Comparar con métricas del servicio de predicciones
    print("\n📊 8. Comparando con métricas del servicio de predicciones...")
    try:
        service_stats = await prediction_service.get_user_stats(user_id)
        real_metrics = await prediction_service.get_real_metrics(user_id)
        
        print("📈 Comparación de métricas:")
        print(f"   Servicio tradicional:")
        print(f"     - Total: {service_stats.get('total_predictions', 0)}")
        print(f"     - Tasa de éxito: {service_stats.get('success_rate', 0):.2f}%")
        print(f"     - Porcentaje promedio: {service_stats.get('average_success_percentage', 0):.2f}%")
        
        print(f"   Métricas reales:")
        print(f"     - Total: {real_metrics.get('total_predictions', 0)}")
        print(f"     - Win Rate: {real_metrics.get('win_rate', 0):.2f}%")
        print(f"     - Precisión: {real_metrics.get('precision', 0):.2f}%")
        
    except Exception as e:
        print(f"❌ Error comparando métricas: {e}")
    
    print("\n" + "=" * 80)
    print("🏁 PRUEBA COMPLETADA")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_real_metrics()) 