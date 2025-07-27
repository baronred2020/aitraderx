#!/usr/bin/env python3
"""
Script de prueba para verificar datos de mercado del backend
"""

import requests
import json
from datetime import datetime

def test_market_data():
    """Prueba el endpoint de datos de mercado"""
    print("🧪 Probando datos de mercado...")
    
    # Símbolos a probar
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "AAPL", "TSLA"]
    
    for symbol in symbols:
        try:
            print(f"\n📊 Probando {symbol}...")
            
            # Probar endpoint de precios
            url = f"http://localhost:8000/api/market-data?symbols={symbol}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if symbol in data and data[symbol]:
                    price_info = data[symbol]
                    print(f"✅ {symbol}:")
                    print(f"   Precio: {price_info.get('price', 'N/A')}")
                    print(f"   Cambio: {price_info.get('change', 'N/A')}")
                    print(f"   % Cambio: {price_info.get('changePercent', 'N/A')}")
                    print(f"   Alto: {price_info.get('high', 'N/A')}")
                    print(f"   Bajo: {price_info.get('low', 'N/A')}")
                    print(f"   Volumen: {price_info.get('volume', 'N/A')}")
                    print(f"   Estado: {price_info.get('marketStatus', 'N/A')}")
                else:
                    print(f"❌ {symbol}: No se encontraron datos")
            else:
                print(f"❌ {symbol}: Error HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ {symbol}: Error - {str(e)}")

def test_candles():
    """Prueba el endpoint de velas"""
    print("\n🕯️ Probando datos de velas...")
    
    symbols = ["EURUSD", "AAPL"]
    
    for symbol in symbols:
        try:
            print(f"\n📈 Probando velas para {symbol}...")
            
            # Probar endpoint de velas
            url = f"http://localhost:8000/api/candles?symbol={symbol}&interval=15&count=50"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'values' in data and data['values']:
                    candles = data['values']
                    print(f"✅ {symbol}: {len(candles)} velas obtenidas")
                    
                    # Mostrar las primeras 3 velas
                    for i, candle in enumerate(candles[:3]):
                        print(f"   Vela {i+1}:")
                        print(f"     Fecha: {candle.get('datetime', 'N/A')}")
                        print(f"     OHLC: {candle.get('open', 'N/A')}/{candle.get('high', 'N/A')}/{candle.get('low', 'N/A')}/{candle.get('close', 'N/A')}")
                        print(f"     Volumen: {candle.get('volume', 'N/A')}")
                else:
                    print(f"❌ {symbol}: No se encontraron velas")
            else:
                print(f"❌ {symbol}: Error HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ {symbol}: Error - {str(e)}")

def test_market_status():
    """Prueba el endpoint de estado del mercado"""
    print("\n🏪 Probando estado del mercado...")
    
    symbols = ["EURUSD", "AAPL"]
    
    for symbol in symbols:
        try:
            print(f"\n🔍 Estado de {symbol}...")
            
            url = f"http://localhost:8000/api/market-status?symbol={symbol}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ {symbol}:")
                print(f"   Abierto: {data.get('is_open', 'N/A')}")
                print(f"   Hora actual: {data.get('current_time', 'N/A')}")
                print(f"   Día: {data.get('weekday', 'N/A')}")
            else:
                print(f"❌ {symbol}: Error HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ {symbol}: Error - {str(e)}")

if __name__ == "__main__":
    print("🚀 Iniciando pruebas de datos de mercado...")
    print(f"⏰ Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        test_market_data()
        test_candles()
        test_market_status()
        
        print("\n✅ Todas las pruebas completadas")
        
    except Exception as e:
        print(f"\n❌ Error general: {str(e)}") 