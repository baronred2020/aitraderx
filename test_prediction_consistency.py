#!/usr/bin/env python3
"""
Script de prueba para verificar la consistencia entre predicciones actuales y el historial
"""

import requests
import json
import time
from datetime import datetime, timedelta

# Configuración
BASE_URL = "http://localhost:8000"
USER_ID = "4dabfd30-483d-4fa0-a8d0-bd151a46340f"  # Usuario demo

def test_login():
    """Probar login para obtener token"""
    print("🔐 Probando login...")
    
    login_data = {
        "username": "demo_user",
        "password": "demo123"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/auth/login", json=login_data)
        print(f"🔍 Respuesta del login: {response.status_code} - {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            token = data.get('token')
            if token:
                print(f"✅ Login exitoso, token obtenido")
                return token
            else:
                print(f"❌ Token no encontrado en la respuesta")
                return None
        else:
            print(f"❌ Error en login: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        return None

def test_generate_prediction(token):
    """Generar una nueva predicción"""
    print("\n🎯 Generando nueva predicción...")
    
    headers = {"Authorization": f"Bearer {token}"}
    prediction_data = {
        "pair": "EURUSD",
        "brain_type": "brain_max",
        "style": "day_trading"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/v1/predictions/generate", 
                               json=prediction_data, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Predicción generada exitosamente")
            
            # La respuesta viene en formato {success: true, prediction: {...}, limits: {...}}
            if data.get('success') and data.get('prediction'):
                prediction = data['prediction']
                print(f"   - ID: {prediction.get('id')}")
                print(f"   - Par: {prediction.get('pair')}")
                print(f"   - Dirección: {prediction.get('direction')}")
                print(f"   - Precio actual: {prediction.get('current_price')}")
                print(f"   - Precio objetivo: {prediction.get('target_price')}")
                print(f"   - Confianza: {prediction.get('confidence')}%")
                print(f"   - Brain: {prediction.get('brain_type')}")
                print(f"   - Expira: {prediction.get('expires_at')}")
                return prediction
            else:
                print(f"❌ Respuesta no contiene predicción válida: {data}")
                return None
        else:
            print(f"❌ Error generando predicción: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        return None

def test_get_prediction_history(token):
    """Obtener historial de predicciones"""
    print("\n📊 Obteniendo historial de predicciones...")
    
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.get(f"{BASE_URL}/api/v1/predictions/history?limit=10", 
                              headers=headers)
        
        if response.status_code == 200:
            history = response.json()
            print(f"✅ Historial obtenido: {len(history)} predicciones")
            
            for i, pred in enumerate(history[:3]):  # Mostrar solo las 3 primeras
                print(f"\n   Predicción {i+1}:")
                print(f"   - ID: {pred.get('id')}")
                print(f"   - Par: {pred.get('pair')}")
                print(f"   - Dirección: {pred.get('direction')}")
                print(f"   - Precio actual: {pred.get('current_price')}")
                print(f"   - Precio objetivo: {pred.get('target_price')}")
                print(f"   - Precio real: {pred.get('actual_price_at_expiry')}")
                print(f"   - Éxito: {pred.get('prediction_success')}")
                print(f"   - Porcentaje éxito: {pred.get('success_percentage')}")
                print(f"   - Completada: {pred.get('is_completed')}")
                print(f"   - Creada: {pred.get('created_at')}")
                print(f"   - Expira: {pred.get('expires_at')}")
            
            return history
        else:
            print(f"❌ Error obteniendo historial: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        return None

def test_complete_expired_predictions(token):
    """Completar predicciones expiradas"""
    print("\n⏰ Completando predicciones expiradas...")
    
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.post(f"{BASE_URL}/api/v1/predictions/complete-expired", 
                               headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Completación exitosa")
            print(f"   - Completadas: {data.get('completed')}")
            print(f"   - Total: {data.get('total')}")
            print(f"   - Mensaje: {data.get('message')}")
            return data
        else:
            print(f"❌ Error completando predicciones: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        return None

def test_prediction_consistency(prediction, history):
    """Verificar consistencia entre predicción actual e historial"""
    print("\n🔍 Verificando consistencia...")
    
    if not prediction or not history:
        print("❌ No se puede verificar consistencia - datos faltantes")
        return False
    
    # Buscar la predicción en el historial
    pred_id = prediction.get('id')
    matching_history = None
    
    for hist_pred in history:
        if hist_pred.get('id') == pred_id:
            matching_history = hist_pred
            break
    
    if not matching_history:
        print(f"❌ Predicción {pred_id} no encontrada en el historial")
        return False
    
    # Verificar campos clave
    fields_to_check = ['pair', 'direction', 'current_price', 'target_price', 'confidence', 'brain_type']
    inconsistencies = []
    
    for field in fields_to_check:
        pred_value = prediction.get(field)
        hist_value = matching_history.get(field)
        
        if pred_value != hist_value:
            inconsistencies.append(f"{field}: {pred_value} vs {hist_value}")
    
    if inconsistencies:
        print(f"❌ Inconsistencias encontradas:")
        for inc in inconsistencies:
            print(f"   - {inc}")
        return False
    else:
        print("✅ Consistencia verificada - todos los campos coinciden")
        return True

def test_prediction_states(history):
    """Verificar estados de predicciones en el historial"""
    print("\n📈 Analizando estados de predicciones...")
    
    if not history:
        print("❌ No hay historial para analizar")
        return
    
    states = {
        'pending': 0,
        'completed_success': 0,
        'completed_failure': 0,
        'completed_unknown': 0
    }
    
    for pred in history:
        is_completed = pred.get('is_completed', False)
        prediction_success = pred.get('prediction_success')
        actual_price = pred.get('actual_price_at_expiry')
        
        if not is_completed or actual_price is None:
            states['pending'] += 1
        elif prediction_success is True:
            states['completed_success'] += 1
        elif prediction_success is False:
            states['completed_failure'] += 1
        else:
            states['completed_unknown'] += 1
    
    print(f"📊 Estados de predicciones:")
    print(f"   - Pendientes: {states['pending']}")
    print(f"   - Completadas exitosas: {states['completed_success']}")
    print(f"   - Completadas fallidas: {states['completed_failure']}")
    print(f"   - Completadas desconocidas: {states['completed_unknown']}")
    
    # Verificar lógica
    total = sum(states.values())
    if total > 0:
        print(f"\n🔍 Análisis de lógica:")
        
        # Verificar que no hay predicciones marcadas como incorrectas con precio pendiente
        problematic = []
        for pred in history:
            if (pred.get('prediction_success') is False and 
                pred.get('actual_price_at_expiry') is None):
                problematic.append(pred.get('id'))
        
        if problematic:
            print(f"   ❌ Predicciones problemáticas (incorrectas con precio pendiente): {problematic}")
        else:
            print(f"   ✅ No hay predicciones problemáticas")
        
        # Verificar que las predicciones completadas tienen precio real
        completed_without_price = []
        for pred in history:
            if (pred.get('is_completed') and 
                pred.get('actual_price_at_expiry') is None):
                completed_without_price.append(pred.get('id'))
        
        if completed_without_price:
            print(f"   ❌ Predicciones completadas sin precio real: {completed_without_price}")
        else:
            print(f"   ✅ Todas las predicciones completadas tienen precio real")

def main():
    """Función principal de prueba"""
    print("🧪 INICIANDO PRUEBAS DE CONSISTENCIA DE PREDICCIONES")
    print("=" * 60)
    
    # 1. Login
    token = test_login()
    if not token:
        print("❌ No se pudo obtener token - abortando pruebas")
        return
    
    # 2. Generar predicción
    prediction = test_generate_prediction(token)
    if not prediction:
        print("❌ No se pudo generar predicción - abortando pruebas")
        return
    
    # 3. Obtener historial inicial
    history_initial = test_get_prediction_history(token)
    
    # 4. Completar predicciones expiradas
    completion_result = test_complete_expired_predictions(token)
    
    # 5. Obtener historial actualizado
    history_updated = test_get_prediction_history(token)
    
    # 6. Verificar consistencia
    if history_initial:
        test_prediction_consistency(prediction, history_initial)
    
    # 7. Analizar estados
    if history_updated:
        test_prediction_states(history_updated)
    
    print("\n" + "=" * 60)
    print("🏁 PRUEBAS COMPLETADAS")

if __name__ == "__main__":
    main() 