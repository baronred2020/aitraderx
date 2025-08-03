#!/usr/bin/env python3
"""
Script de Prueba para el Sistema RL
==================================
Verifica que todos los endpoints de RL estén funcionando correctamente
"""

import requests
import json
import time
from datetime import datetime

# Configuración
BASE_URL = "http://localhost:8000"
USER_ID = "4dabfd30-483d-4fa0-a8d0-bd151a46340f"

def test_rl_endpoints():
    """Prueba todos los endpoints de RL"""
    
    print("🧪 Iniciando pruebas del Sistema RL...")
    print("=" * 50)
    
    # 1. Probar /api/rl/status
    print("\n1. Probando /api/rl/status...")
    try:
        response = requests.get(f"{BASE_URL}/api/rl/status")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status: {data.get('status', 'N/A')}")
            print(f"   Sesiones activas: {data.get('active_sessions', 0)}")
            print(f"   Estrategia: {data.get('current_strategy', 'N/A')}")
        else:
            print(f"❌ Error: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")

    # 2. Probar /api/rl/performance
    print("\n2. Probando /api/rl/performance...")
    try:
        response = requests.get(f"{BASE_URL}/api/rl/performance")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Win Rate: {data.get('win_rate', 0):.2%}")
            print(f"   Profit Factor: {data.get('profit_factor', 0):.2f}")
            print(f"   Sharpe Ratio: {data.get('sharpe_ratio', 0):.2f}")
        else:
            print(f"❌ Error: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")

    # 3. Probar /api/rl/active-signals
    print("\n3. Probando /api/rl/active-signals...")
    try:
        response = requests.get(f"{BASE_URL}/api/rl/active-signals")
        if response.status_code == 200:
            data = response.json()
            signals = data if isinstance(data, list) else data.get('signals', [])
            print(f"✅ Señales activas: {len(signals)}")
            for signal in signals[:3]:  # Mostrar solo las primeras 3
                print(f"   - {signal.get('pair', 'N/A')}: {signal.get('signal', 'N/A')}")
        else:
            print(f"❌ Error: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")

    # 4. Probar /api/rl/can-train/{user_id}
    print(f"\n4. Probando /api/rl/can-train/{USER_ID}...")
    try:
        response = requests.get(f"{BASE_URL}/api/rl/can-train/{USER_ID}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Puede entrenar: {data.get('can_train', False)}")
            print(f"   Razón: {data.get('reason', 'N/A')}")
        else:
            print(f"❌ Error: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")

    # 5. Probar /api/rl/validate-params
    print("\n5. Probando /api/rl/validate-params...")
    try:
        test_params = {
            "episodes": 500,
            "user_plan": "premium"
        }
        response = requests.post(
            f"{BASE_URL}/api/rl/validate-params",
            json=test_params
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Válido: {data.get('valid', False)}")
            print(f"   Razón: {data.get('reason', 'N/A')}")
            if data.get('limits'):
                limits = data['limits']
                print(f"   Límites: {limits.get('min', 0)} - {limits.get('max', 0)}")
        else:
            print(f"❌ Error: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")

    # 6. Probar /api/rl/start-training (solo si puede entrenar)
    print("\n6. Probando /api/rl/start-training...")
    try:
        # Primero verificar si puede entrenar
        can_train_response = requests.get(f"{BASE_URL}/api/rl/can-train/{USER_ID}")
        if can_train_response.status_code == 200:
            can_train_data = can_train_response.json()
            
            if can_train_data.get('can_train', False):
                training_params = {
                    "user_id": USER_ID,
                    "episodes": 100,  # Solo 100 episodios para prueba
                    "algorithm": "dqn",
                    "trading_pair": "EURUSD",
                    "timeframe": "1h"
                }
                
                response = requests.post(
                    f"{BASE_URL}/api/rl/start-training",
                    json=training_params
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('success', False):
                        session_id = data.get('session_id')
                        print(f"✅ Entrenamiento iniciado: {session_id}")
                        
                        # Probar progreso
                        print("\n7. Probando progreso de entrenamiento...")
                        time.sleep(2)  # Esperar un poco
                        
                        progress_response = requests.get(f"{BASE_URL}/api/rl/training-progress/{session_id}")
                        if progress_response.status_code == 200:
                            progress_data = progress_response.json()
                            print(f"✅ Progreso: {progress_data.get('progress', 0):.2%}")
                            print(f"   Episodio: {progress_data.get('current_episode', 0)}/{progress_data.get('total_episodes', 0)}")
                            
                            # Cancelar entrenamiento de prueba
                            print("\n8. Cancelando entrenamiento de prueba...")
                            cancel_response = requests.post(f"{BASE_URL}/api/rl/cancel-training/{session_id}?user_id={USER_ID}")
                            if cancel_response.status_code == 200:
                                cancel_data = cancel_response.json()
                                if cancel_data.get('success', False):
                                    print("✅ Entrenamiento cancelado correctamente")
                                else:
                                    print(f"❌ Error cancelando: {cancel_data.get('error', 'N/A')}")
                            else:
                                print(f"❌ Error cancelando: {cancel_response.status_code}")
                        else:
                            print(f"❌ Error obteniendo progreso: {progress_response.status_code}")
                    else:
                        print(f"❌ Error iniciando: {data.get('error', 'N/A')}")
                else:
                    print(f"❌ Error: {response.status_code}")
            else:
                print(f"⚠️ No puede entrenar: {can_train_data.get('reason', 'N/A')}")
        else:
            print(f"❌ Error verificando permisos: {can_train_response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")

    # 9. Probar /api/rl/training-history/{user_id}
    print(f"\n9. Probando /api/rl/training-history/{USER_ID}...")
    try:
        response = requests.get(f"{BASE_URL}/api/rl/training-history/{USER_ID}?limit=5")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Historial: {len(data)} sesiones")
            for session in data[:3]:  # Mostrar solo las primeras 3
                print(f"   - {session.get('session_id', 'N/A')}: {session.get('status', 'N/A')}")
        else:
            print(f"❌ Error: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n" + "=" * 50)
    print("🏁 Pruebas completadas!")

if __name__ == "__main__":
    test_rl_endpoints() 