#!/usr/bin/env python3
"""
Script de prueba para verificar el endpoint de market-data
"""

import requests
import json
import time

def test_market_data_endpoint():
    """Probar el endpoint de market-data"""
    print("🧪 Probando endpoint de market-data...")
    
    # URL del endpoint
    url = "http://localhost:8000/api/market-data"
    
    # Símbolos a probar
    symbols = "EURUSD,GBPUSD,USDJPY,AUDUSD,USDCAD"
    
    params = {
        'symbols': symbols
    }
    
    try:
        print(f"📡 Solicitando datos para: {symbols}")
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Respuesta exitosa!")
            print(f"📊 Datos recibidos: {len(data)} símbolos")
            
            # Mostrar datos de cada símbolo
            for symbol, info in data.items():
                if 'error' in info:
                    print(f"❌ {symbol}: Error - {info['error']}")
                else:
                    price = info.get('price', '0')
                    change = info.get('changePercent', '0')
                    print(f"💹 {symbol}: {price} ({change}%)")
            
            return True
        else:
            print(f"❌ Error HTTP {response.status_code}")
            print(f"📄 Respuesta: {response.text}")
            return False
            
    except requests.RequestException as e:
        print(f"❌ Error de conexión: {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Error al parsear JSON: {e}")
        return False

def main():
    """Función principal"""
    print("🚀 Iniciando prueba del endpoint de market-data...")
    print("=" * 60)
    
    # Esperar a que el servidor esté listo
    print("⏳ Esperando 3 segundos para que el servidor esté listo...")
    time.sleep(3)
    
    # Probar el endpoint
    if test_market_data_endpoint():
        print("\n✅ Todas las pruebas pasaron!")
        print("El endpoint de market-data está funcionando correctamente.")
    else:
        print("\n❌ Hay problemas con el endpoint.")
        print("Verifica que el servidor esté ejecutándose en localhost:8000")
    
    print("\n" + "=" * 60)
    print("🏁 Prueba completada")

if __name__ == "__main__":
    main() 