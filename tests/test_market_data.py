#!/usr/bin/env python3
"""
Script de prueba para verificar que el backend devuelve datos correctamente
incluso cuando el mercado está cerrado.
"""

import requests
import json
from datetime import datetime

def test_market_data():
    """Prueba el endpoint de datos de mercado"""
    print("=== PRUEBA DE DATOS DE MERCADO ===")
    
    # Probar diferentes símbolos
    symbols = ['EURUSD', 'GBPUSD', 'AAPL', 'BTCUSD']
    
    for symbol in symbols:
        print(f"\n--- Probando {symbol} ---")
        try:
            response = requests.get(f'http://localhost:8000/api/market-data?symbols={symbol}')
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Datos obtenidos para {symbol}:")
                print(f"   Precio: {data[symbol]['price']}")
                print(f"   Cambio: {data[symbol]['change']}")
                print(f"   % Cambio: {data[symbol]['changePercent']}%")
                print(f"   Estado: {data[symbol]['marketStatus']}")
            else:
                print(f"❌ Error {response.status_code} para {symbol}")
        except Exception as e:
            print(f"❌ Error de conexión para {symbol}: {e}")

def test_candles():
    """Prueba el endpoint de velas"""
    print("\n=== PRUEBA DE VELAS ===")
    
    # Probar diferentes símbolos y timeframes
    test_cases = [
        ('EURUSD', '15'),
        ('GBPUSD', '15'),
        ('AAPL', '15'),
        ('BTCUSD', '15')
    ]
    
    for symbol, interval in test_cases:
        print(f"\n--- Probando velas para {symbol} (intervalo {interval}) ---")
        try:
            response = requests.get(f'http://localhost:8000/api/candles?symbol={symbol}&interval={interval}&count=50')
            if response.status_code == 200:
                data = response.json()
                candle_count = len(data.get('values', []))
                print(f"✅ Velas obtenidas para {symbol}: {candle_count} velas")
                
                if candle_count > 0:
                    # Mostrar la primera y última vela
                    first_candle = data['values'][0]
                    last_candle = data['values'][-1]
                    print(f"   Primera vela: {first_candle['datetime']} - Cierre: {first_candle['close']}")
                    print(f"   Última vela: {last_candle['datetime']} - Cierre: {last_candle['close']}")
                else:
                    print("   ⚠️ No hay velas disponibles")
            else:
                print(f"❌ Error {response.status_code} para {symbol}")
        except Exception as e:
            print(f"❌ Error de conexión para {symbol}: {e}")

def test_market_status():
    """Prueba el endpoint de estado del mercado"""
    print("\n=== PRUEBA DE ESTADO DEL MERCADO ===")
    
    symbols = ['EURUSD', 'AAPL', 'BTCUSD']
    
    for symbol in symbols:
        print(f"\n--- Estado del mercado para {symbol} ---")
        try:
            response = requests.get(f'http://localhost:8000/api/market-status?symbol={symbol}')
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Estado obtenido para {symbol}:")
                print(f"   Abierto: {data['is_open']}")
                print(f"   Hora actual: {data['current_time']}")
                print(f"   Día: {data['weekday']}")
            else:
                print(f"❌ Error {response.status_code} para {symbol}")
        except Exception as e:
            print(f"❌ Error de conexión para {symbol}: {e}")

def main():
    """Función principal"""
    print(f"🚀 Iniciando pruebas del backend - {datetime.now()}")
    print("=" * 60)
    
    # Verificar que el backend esté corriendo
    try:
        response = requests.get('http://localhost:8000/docs')
        if response.status_code == 200:
            print("✅ Backend está corriendo")
        else:
            print("❌ Backend no responde correctamente")
            return
    except Exception as e:
        print(f"❌ No se puede conectar al backend: {e}")
        return
    
    # Ejecutar pruebas
    test_market_data()
    test_candles()
    test_market_status()
    
    print("\n" + "=" * 60)
    print("✅ Pruebas completadas")

if __name__ == "__main__":
    main() 