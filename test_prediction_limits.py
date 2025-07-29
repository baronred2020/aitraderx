#!/usr/bin/env python3
"""
Script para probar los límites de predicciones
"""
import requests
import json

def test_prediction_limits():
    """Probar el endpoint de límites de predicciones"""
    
    # URL del endpoint (ajustar según tu configuración)
    base_url = "http://localhost:8000"
    
    try:
        # Probar el endpoint de límites
        response = requests.get(f"{base_url}/api/v1/predictions/limits?style=day_trading")
        
        print("🔍 Probando límites de predicciones...")
        print(f"URL: {base_url}/api/v1/predictions/limits?style=day_trading")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Respuesta exitosa:")
            print(json.dumps(data, indent=2))
            
            # Verificar que los datos son correctos para plan starter
            if data.get('plan_type') == 'starter':
                print("\n✅ Plan starter detectado correctamente")
                if data.get('max_predictions_per_day') == 5:
                    print("✅ Límite de 5 predicciones configurado correctamente")
                else:
                    print(f"⚠️ Límite incorrecto: {data.get('max_predictions_per_day')} (esperado: 5)")
                
                if data.get('remaining_predictions') == 5:
                    print("✅ 5 predicciones restantes (correcto para inicio del día)")
                else:
                    print(f"⚠️ Predicciones restantes incorrectas: {data.get('remaining_predictions')} (esperado: 5)")
                
                if data.get('can_generate') == True:
                    print("✅ Puede generar predicciones (correcto)")
                else:
                    print("❌ No puede generar predicciones (incorrecto)")
            else:
                print(f"⚠️ Plan detectado: {data.get('plan_type')} (esperado: starter)")
                
        else:
            print(f"❌ Error en la respuesta: {response.status_code}")
            print(f"Respuesta: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ No se pudo conectar al servidor. Asegúrate de que el backend esté ejecutándose.")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_prediction_generation():
    """Probar la generación de predicciones"""
    
    base_url = "http://localhost:8000"
    
    try:
        # Datos para generar predicción
        prediction_data = {
            "pair": "EURUSD",
            "brain_type": "brain_max",
            "style": "day_trading"
        }
        
        print("\n🔍 Probando generación de predicción...")
        response = requests.post(f"{base_url}/api/v1/predictions/generate", json=prediction_data)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Predicción generada exitosamente:")
            print(json.dumps(data, indent=2))
        else:
            print(f"❌ Error generando predicción: {response.status_code}")
            print(f"Respuesta: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ No se pudo conectar al servidor.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🧪 Iniciando pruebas de límites de predicciones...\n")
    
    test_prediction_limits()
    test_prediction_generation()
    
    print("\n✅ Pruebas completadas")