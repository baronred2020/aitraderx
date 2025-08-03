#!/usr/bin/env python3
"""
Script de Prueba para Precios Reales en Señales RL
==================================================
Verifica que las señales incluyan precios reales de mercado
"""

import requests
import json
from datetime import datetime

# Configuración
BASE_URL = "http://localhost:8000"

def test_real_prices():
    """Prueba que las señales incluyan precios reales"""
    
    print("💰 Probando Precios Reales en Señales RL...")
    print("=" * 60)
    
    # Probar /api/rl/active-signals
    print("\n1. Probando señales con precios reales...")
    try:
        response = requests.get(f"{BASE_URL}/api/rl/active-signals")
        if response.status_code == 200:
            data = response.json()
            signals = data if isinstance(data, list) else data.get('signals', [])
            
            print(f"✅ Señales encontradas: {len(signals)}")
            
            for i, signal in enumerate(signals, 1):
                print(f"\n📊 Señal #{i}:")
                print(f"   Par: {signal.get('pair', 'N/A')}")
                print(f"   Señal: {signal.get('signal', 'N/A')}")
                print(f"   Precio Entrada: {signal.get('entry_price', 'N/A')}")
                print(f"   Stop Loss: {signal.get('stop_loss', 'N/A')}")
                print(f"   Take Profit: {signal.get('take_profit', 'N/A')}")
                print(f"   Confianza: {signal.get('confidence', 'N/A')}")
                print(f"   Posición: {signal.get('position_size', 'N/A')}")
                print(f"   Timestamp: {signal.get('timestamp', 'N/A')}")
                
                # Verificar que los precios son reales
                entry_price = signal.get('entry_price')
                stop_loss = signal.get('stop_loss')
                take_profit = signal.get('take_profit')
                
                if entry_price and stop_loss and take_profit:
                    print(f"   ✅ Precios válidos:")
                    print(f"      - Entrada: {entry_price:.5f}")
                    print(f"      - Stop Loss: {stop_loss:.5f}")
                    print(f"      - Take Profit: {take_profit:.5f}")
                    
                    # Verificar que los niveles tienen sentido
                    if signal.get('signal') == 'buy':
                        if stop_loss < entry_price < take_profit:
                            print(f"      ✅ Niveles correctos para BUY")
                        else:
                            print(f"      ❌ Niveles incorrectos para BUY")
                    else:  # sell
                        if take_profit < entry_price < stop_loss:
                            print(f"      ✅ Niveles correctos para SELL")
                        else:
                            print(f"      ❌ Niveles incorrectos para SELL")
                else:
                    print(f"   ❌ Faltan precios en la señal")
                    
        else:
            print(f"❌ Error: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n" + "=" * 60)
    print("🏁 Prueba de precios reales completada!")

if __name__ == "__main__":
    test_real_prices() 