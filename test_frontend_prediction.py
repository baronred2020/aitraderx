#!/usr/bin/env python3
"""
Script para probar la API de predicciones y verificar la consistencia
"""

import requests
import json
from datetime import datetime

def test_prediction_api():
    """Probar la API de predicciones"""
    print("🧪 **PRUEBA DE API DE PREDICCIONES**")
    print("=" * 50)
    
    # URL de la API
    base_url = "http://localhost:8000"
    
    # Probar diferentes estilos
    styles = ['day_trading', 'scalping', 'swing_trading', 'position_trading']
    
    for style in styles:
        print(f"\n📊 **Probando estilo: {style}**")
        
        try:
            # Hacer request a la API
            url = f"{base_url}/api/v1/brain-trader/predictions/brain_max"
            params = {
                'pair': 'EURUSD',
                'style': style,
                'limit': 1,
                'plan_type': 'starter'
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                predictions = response.json()
                
                if predictions:
                    prediction = predictions[0]
                    
                    pair = prediction['pair']
                    direction = prediction['direction']
                    confidence = prediction['confidence']
                    target_price = prediction['target_price']
                    reasoning = prediction['reasoning']
                    timeframe = prediction['timeframe']
                    
                    print(f"  🎯 Par: {pair}")
                    print(f"  📈 Dirección: {direction.upper()}")
                    print(f"  📊 Confianza: {confidence:.1f}%")
                    print(f"  💰 Precio Objetivo: ${target_price:.5f}")
                    print(f"  ⏰ Timeframe: {timeframe}")
                    print(f"  📝 Razón: {reasoning}")
                    
                    # Simular precio actual (normalmente vendría del frontend)
                    # Para esta prueba, usaremos un precio de ejemplo
                    current_price = 1.1761
                    
                    print(f"  💵 Precio Actual (simulado): ${current_price}")
                    
                    # Verificar consistencia
                    if direction == 'up':
                        if target_price > current_price:
                            print(f"  ✅ **CONSISTENTE:** Precio objetivo ({target_price:.5f}) > Precio actual ({current_price:.5f})")
                        else:
                            print(f"  ❌ **INCONSISTENTE:** Precio objetivo ({target_price:.5f}) <= Precio actual ({current_price:.5f})")
                    elif direction == 'down':
                        if target_price < current_price:
                            print(f"  ✅ **CONSISTENTE:** Precio objetivo ({target_price:.5f}) < Precio actual ({current_price:.5f})")
                        else:
                            print(f"  ❌ **INCONSISTENTE:** Precio objetivo ({target_price:.5f}) >= Precio actual ({current_price:.5f})")
                    else:  # sideways
                        print(f"  🔄 **LATERAL:** Precio objetivo ({target_price:.5f}) cerca del precio actual ({current_price:.5f})")
                    
                    # Calcular diferencia porcentual
                    diff_percent = ((target_price - current_price) / current_price) * 100
                    print(f"  📊 Diferencia: {diff_percent:+.3f}%")
                    
                else:
                    print("  ❌ No se recibieron predicciones")
            else:
                print(f"  ❌ Error en la API: {response.status_code}")
                print(f"  📄 Respuesta: {response.text}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("\n🎯 **RESUMEN DE LA PRUEBA**")
    print("=" * 50)
    print("✅ La API debería devolver predicciones consistentes:")
    print("   - Para predicciones 'UP': precio objetivo > precio actual")
    print("   - Para predicciones 'DOWN': precio objetivo < precio actual")
    print("   - Para predicciones 'SIDEWAYS': precio objetivo ≈ precio actual")

if __name__ == "__main__":
    test_prediction_api() 