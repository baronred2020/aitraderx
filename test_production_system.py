#!/usr/bin/env python3
"""
Script de prueba para el sistema de producción con datos reales
"""
import requests
import json
import time

def test_production_system():
    """Probar el sistema completo de producción con datos reales"""
    
    base_url = "http://localhost:8000"
    
    print("🧪 Iniciando pruebas del sistema de producción...\n")
    
    # 1. Probar límites iniciales
    print("1️⃣ Probando límites iniciales...")
    try:
        response = requests.get(f"{base_url}/api/v1/predictions/limits?style=day_trading")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Límites iniciales: {data.get('remaining_predictions')}/{data.get('max_predictions_per_day')}")
            print(f"📊 Plan: {data.get('plan_type')}")
            print(f"📊 Puede generar: {data.get('can_generate')}")
        else:
            print(f"❌ Error obteniendo límites: {response.status_code}")
            print(f"Respuesta: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # 2. Generar predicciones reales y verificar contador
    print("\n2️⃣ Generando predicciones reales y verificando contador...")
    
    for i in range(3):  # Generar 3 predicciones
        print(f"\n   Generando predicción {i+1}...")
        try:
            prediction_data = {
                "pair": "EURUSD",
                "brain_type": "brain_max",
                "style": "day_trading"
            }
            
            response = requests.post(f"{base_url}/api/v1/predictions/generate", json=prediction_data)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    prediction = data.get('prediction', {})
                    limits = data.get('limits', {})
                    print(f"   ✅ Predicción generada: {prediction.get('pair')} - {prediction.get('direction')}")
                    print(f"   📊 Precio actual: {prediction.get('current_price')}")
                    print(f"   📊 Precio objetivo: {prediction.get('target_price')}")
                    print(f"   📊 Confianza: {prediction.get('confidence'):.1f}%")
                    print(f"   📊 Límites actualizados: {limits.get('remaining_predictions')}/{limits.get('max_predictions_per_day')}")
                else:
                    print(f"   ❌ Error: {data.get('error')}")
            else:
                print(f"   ❌ Error HTTP: {response.status_code}")
                print(f"   Respuesta: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        time.sleep(1)  # Pausa entre predicciones
    
    # 3. Verificar historial real
    print("\n3️⃣ Verificando historial de predicciones reales...")
    try:
        response = requests.get(f"{base_url}/api/v1/predictions/history?limit=10")
        if response.status_code == 200:
            history = response.json()
            print(f"✅ Historial obtenido: {len(history)} predicciones")
            for i, pred in enumerate(history[:3]):  # Mostrar las primeras 3
                print(f"   {i+1}. {pred.get('pair')} - {pred.get('direction')} - {pred.get('confidence')}%")
                print(f"      Precio actual: {pred.get('current_price')} -> Objetivo: {pred.get('target_price')}")
                if pred.get('is_completed'):
                    print(f"      Éxito: {pred.get('prediction_success')} - Porcentaje: {pred.get('success_percentage')}%")
                    print(f"      Precio real: {pred.get('actual_price_at_expiry')}")
                else:
                    print(f"      Estado: En progreso")
        else:
            print(f"❌ Error obteniendo historial: {response.status_code}")
            print(f"Respuesta: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # 4. Verificar límites finales
    print("\n4️⃣ Verificando límites finales...")
    try:
        response = requests.get(f"{base_url}/api/v1/predictions/limits?style=day_trading")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Límites finales: {data.get('remaining_predictions')}/{data.get('max_predictions_per_day')}")
            
            # Verificar que se contaron las predicciones
            remaining = data.get('remaining_predictions', 0)
            max_pred = data.get('max_predictions_per_day', 5)
            used = max_pred - remaining
            
            print(f"📊 Predicciones usadas: {used}")
            print(f"📊 Predicciones restantes: {remaining}")
            
        else:
            print(f"❌ Error obteniendo límites finales: {response.status_code}")
            print(f"Respuesta: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # 5. Probar estadísticas del usuario
    print("\n5️⃣ Verificando estadísticas del usuario...")
    try:
        response = requests.get(f"{base_url}/api/v1/predictions/stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"✅ Estadísticas obtenidas:")
            print(f"   📊 Total de predicciones: {stats.get('total_predictions')}")
            print(f"   📊 Predicciones exitosas: {stats.get('successful_predictions')}")
            print(f"   📊 Tasa de éxito: {stats.get('success_rate')}%")
            print(f"   📊 Porcentaje promedio: {stats.get('average_success_percentage')}%")
            print(f"   📊 Mejor par: {stats.get('best_pair')}")
            print(f"   📊 Predicciones hoy: {stats.get('total_predictions_today')}")
        else:
            print(f"❌ Error obteniendo estadísticas: {response.status_code}")
            print(f"Respuesta: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n✅ Pruebas de producción completadas")

if __name__ == "__main__":
    test_production_system()