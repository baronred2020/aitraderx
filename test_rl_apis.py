#!/usr/bin/env python3
"""
Script de prueba para verificar las APIs del RL Director
"""

import requests
import json
import time

# Configuración
BASE_URL = "http://localhost:8000"

def test_rl_apis():
    """Prueba todas las APIs del RL Director"""
    
    print("🧪 Probando APIs del RL Director...")
    print("=" * 50)
    
    # 1. Probar /api/rl/status
    print("\n1. Probando /api/rl/status...")
    try:
        response = requests.get(f"{BASE_URL}/api/rl/status")
        if response.status_code == 200:
            data = response.json()
            print("✅ Status API funcionando")
            print(f"   - Estado: {data.get('status')}")
            print(f"   - Estrategia: {data.get('current_strategy')}")
            print(f"   - Regimen: {data.get('market_regime')}")
            print(f"   - Nivel Riesgo: {data.get('risk_level')}")
            
            # Verificar coordinación de modelos
            if 'model_coordination' in data:
                print("   - Coordinación de Modelos:")
                for model, info in data['model_coordination'].items():
                    print(f"     * {model}: {info.get('weight')}% peso, {info.get('confidence')}% confianza")
        else:
            print(f"❌ Error en Status API: {response.status_code}")
    except Exception as e:
        print(f"❌ Error conectando a Status API: {e}")
    
    # 2. Probar /api/rl/performance
    print("\n2. Probando /api/rl/performance...")
    try:
        response = requests.get(f"{BASE_URL}/api/rl/performance")
        if response.status_code == 200:
            data = response.json()
            print("✅ Performance API funcionando")
            if 'performance_metrics' in data:
                metrics = data['performance_metrics']
                print(f"   - Profit Promedio: {metrics.get('avg_profit', 0)*100:.1f}%")
                print(f"   - Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
                print(f"   - Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
                
                if 'model_performance' in metrics:
                    print("   - Rendimiento por Modelo:")
                    for model, perf in metrics['model_performance'].items():
                        print(f"     * {model}: {perf:.1f}%")
        else:
            print(f"❌ Error en Performance API: {response.status_code}")
    except Exception as e:
        print(f"❌ Error conectando a Performance API: {e}")
    
    # 3. Probar /api/rl/active-signals
    print("\n3. Probando /api/rl/active-signals...")
    try:
        response = requests.get(f"{BASE_URL}/api/rl/active-signals")
        if response.status_code == 200:
            data = response.json()
            print("✅ Active Signals API funcionando")
            signals = data.get('signals', [])
            print(f"   - Señales activas: {len(signals)}")
            
            for i, signal in enumerate(signals):
                print(f"   - Señal {i+1}: {signal.get('action')} {signal.get('symbol')} ({signal.get('confidence')}% confianza)")
                print(f"     * Posición: {signal.get('position_size')}%")
                print(f"     * Stop Loss: {signal.get('stop_loss')}")
                print(f"     * Take Profit: {signal.get('take_profit')}")
                print(f"     * Modelos: {', '.join(signal.get('models_used', []))}")
        else:
            print(f"❌ Error en Active Signals API: {response.status_code}")
    except Exception as e:
        print(f"❌ Error conectando a Active Signals API: {e}")
    
    # 4. Probar /api/rl/train
    print("\n4. Probando /api/rl/train...")
    try:
        response = requests.post(f"{BASE_URL}/api/rl/train?episodes=100")
        if response.status_code == 200:
            print("✅ Train API funcionando")
            data = response.json()
            print(f"   - Mensaje: {data.get('message', 'Entrenamiento iniciado')}")
        else:
            print(f"❌ Error en Train API: {response.status_code}")
    except Exception as e:
        print(f"❌ Error conectando a Train API: {e}")
    
    # 5. Probar /api/rl/execute-signal (con señal de prueba)
    print("\n5. Probando /api/rl/execute-signal...")
    try:
        test_signal = {
            "symbol": "EURUSD",
            "action": "BUY",
            "confidence": 85,
            "position_size": 3,
            "stop_loss": 1.0850,
            "take_profit": 1.0950,
            "reasoning": "Prueba de API",
            "models_used": ["Brain Max", "Brain Ultra"],
            "timestamp": "2024-01-01T12:00:00"
        }
        
        response = requests.post(
            f"{BASE_URL}/api/rl/execute-signal",
            json=test_signal,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            print("✅ Execute Signal API funcionando")
            data = response.json()
            print(f"   - Order ID: {data.get('order_id')}")
            print(f"   - Status: {data.get('status')}")
        else:
            print(f"❌ Error en Execute Signal API: {response.status_code}")
    except Exception as e:
        print(f"❌ Error conectando a Execute Signal API: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 Prueba de APIs completada")

if __name__ == "__main__":
    test_rl_apis() 