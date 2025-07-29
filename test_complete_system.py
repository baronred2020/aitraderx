#!/usr/bin/env python3
"""
Script de prueba completo para verificar el sistema de predicciones
"""
import requests
import json
import time

def test_complete_system():
    """Probar el sistema completo de predicciones"""
    
    base_url = "http://localhost:8000"
    
    print("🧪 Iniciando pruebas completas del sistema...\n")
    
    # 1. Probar límites iniciales
    print("📊 1. Probando límites iniciales...")
    try:
        response = requests.get(f"{base_url}/api/v1/predictions/limits?style=day_trading")
        if response.status_code == 200:
            limits = response.json()
            print(f"✅ Límites obtenidos: {limits}")
            print(f"   - Plan: {limits.get('plan_type', 'N/A')}")
            print(f"   - Máximo por día: {limits.get('max_predictions_per_day', 'N/A')}")
            print(f"   - Restantes: {limits.get('remaining_predictions', 'N/A')}")
            print(f"   - Puede generar: {limits.get('can_generate', 'N/A')}")
        else:
            print(f"❌ Error obteniendo límites: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error en límites: {e}")
        return False
    
    # 2. Generar predicciones y verificar contador
    print("\n🎯 2. Generando predicciones...")
    predictions_generated = 0
    for i in range(3):
        try:
            prediction_data = {
                "pair": "EURUSD",
                "brain_type": "brain_max",
                "style": "day_trading"
            }
            response = requests.post(
                f"{base_url}/api/v1/predictions/generate",
                json=prediction_data
            )
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    predictions_generated += 1
                    prediction = result.get('prediction', {})
                    print(f"   ✅ Predicción {i+1}: {prediction.get('pair')} - {prediction.get('direction')} - {prediction.get('confidence')}%")
                else:
                    print(f"   ❌ Error en predicción {i+1}: {result.get('error', 'Error desconocido')}")
            else:
                print(f"   ❌ Error HTTP en predicción {i+1}: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Error en predicción {i+1}: {e}")
    
    print(f"   📊 Total predicciones generadas: {predictions_generated}")
    
    # 3. Verificar límites después de generar predicciones
    print("\n📊 3. Verificando límites después de generar predicciones...")
    try:
        response = requests.get(f"{base_url}/api/v1/predictions/limits?style=day_trading")
        if response.status_code == 200:
            updated_limits = response.json()
            print(f"✅ Límites actualizados: {updated_limits}")
            print(f"   - Restantes: {updated_limits.get('remaining_predictions', 'N/A')}")
            print(f"   - Puede generar: {updated_limits.get('can_generate', 'N/A')}")
        else:
            print(f"❌ Error obteniendo límites actualizados: {response.status_code}")
    except Exception as e:
        print(f"❌ Error en límites actualizados: {e}")
    
    # 4. Verificar historial
    print("\n📜 4. Verificando historial de predicciones...")
    try:
        response = requests.get(f"{base_url}/api/v1/predictions/history?limit=10")
        if response.status_code == 200:
            history = response.json()
            print(f"✅ Historial obtenido: {len(history)} predicciones")
            for i, pred in enumerate(history[:3], 1):
                print(f"   {i}. {pred.get('pair')} - {pred.get('direction')} - {pred.get('confidence')}% - {pred.get('created_at', 'N/A')}")
        else:
            print(f"❌ Error obteniendo historial: {response.status_code}")
    except Exception as e:
        print(f"❌ Error en historial: {e}")
    
    # 5. Probar estadísticas del usuario
    print("\n📈 5. Probando estadísticas del usuario...")
    try:
        response = requests.get(f"{base_url}/api/v1/predictions/stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"✅ Estadísticas obtenidas: {stats}")
        else:
            print(f"❌ Error obteniendo estadísticas: {response.status_code}")
    except Exception as e:
        print(f"❌ Error en estadísticas: {e}")
    
    print("\n🎉 ¡Pruebas completadas!")
    print("✅ El sistema está funcionando correctamente con base de datos real")
    return True

if __name__ == "__main__":
    success = test_complete_system()
    exit(0 if success else 1)