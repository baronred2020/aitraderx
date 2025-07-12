#!/usr/bin/env python3
"""
Script de prueba para verificar que obtenemos datos reales de Yahoo Finance
"""

import requests
import json
from datetime import datetime

def test_yahoo_real_data():
    """Prueba que obtenemos datos reales de Yahoo Finance"""
    print("=== PRUEBA DE DATOS REALES DE YAHOO FINANCE ===")
    
    # Probar diferentes símbolos
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AAPL', 'MSFT', 'TSLA']
    
    for symbol in symbols:
        print(f"\n--- Probando {symbol} ---")
        try:
            response = requests.get(f'http://localhost:8000/api/market-data?symbols={symbol}')
            if response.status_code == 200:
                data = response.json()
                symbol_data = data.get(symbol, {})
                
                print(f"✅ Datos obtenidos para {symbol}:")
                print(f"   Precio: {symbol_data.get('price', 'N/A')}")
                print(f"   Cambio: {symbol_data.get('change', 'N/A')}")
                print(f"   Cambio %: {symbol_data.get('changePercent', 'N/A')}%")
                print(f"   Volumen: {symbol_data.get('volume', 'N/A')}")
                print(f"   Alto: {symbol_data.get('high', 'N/A')}")
                print(f"   Bajo: {symbol_data.get('low', 'N/A')}")
                print(f"   Abierto: {symbol_data.get('open', 'N/A')}")
                print(f"   Estado: {symbol_data.get('marketStatus', 'N/A')}")
                
                # Verificar que el precio es realista (no es fallback)
                price = float(symbol_data.get('price', '0'))
                if price > 0:
                    if symbol in ['EURUSD', 'GBPUSD', 'AUDUSD']:
                        if 0.5 <= price <= 2.0:
                            print(f"   ✅ Precio realista para {symbol}")
                        else:
                            print(f"   ⚠️ Precio sospechoso para {symbol}: {price}")
                    elif symbol == 'USDJPY':
                        if 100 <= price <= 200:
                            print(f"   ✅ Precio realista para {symbol}")
                        else:
                            print(f"   ⚠️ Precio sospechoso para {symbol}: {price}")
                    elif symbol in ['AAPL', 'MSFT', 'TSLA']:
                        if 50 <= price <= 1000:
                            print(f"   ✅ Precio realista para {symbol}")
                        else:
                            print(f"   ⚠️ Precio sospechoso para {symbol}: {price}")
                else:
                    print(f"   ❌ Precio inválido para {symbol}")
                    
            else:
                print(f"❌ Error HTTP: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def test_candles_real_data():
    """Prueba que obtenemos velas reales de Yahoo Finance"""
    print("\n=== PRUEBA DE VELAS REALES DE YAHOO FINANCE ===")
    
    symbols = ['EURUSD', 'AAPL']
    
    for symbol in symbols:
        print(f"\n--- Probando velas para {symbol} ---")
        try:
            response = requests.get(f'http://localhost:8000/api/candles?symbol={symbol}&interval=15&count=50')
            if response.status_code == 200:
                data = response.json()
                values = data.get('values', [])
                
                print(f"✅ Velas obtenidas para {symbol}: {len(values)} registros")
                
                if values:
                    # Mostrar las primeras 3 velas
                    for i, candle in enumerate(values[:3]):
                        print(f"   Vela {i+1}: {candle['datetime']} - O:{candle['open']} H:{candle['high']} L:{candle['low']} C:{candle['close']}")
                    
                    # Verificar que los precios son realistas
                    last_candle = values[-1]
                    close_price = float(last_candle['close'])
                    
                    if symbol == 'EURUSD':
                        if 0.5 <= close_price <= 2.0:
                            print(f"   ✅ Precio de cierre realista: {close_price}")
                        else:
                            print(f"   ⚠️ Precio de cierre sospechoso: {close_price}")
                    elif symbol == 'AAPL':
                        if 50 <= close_price <= 1000:
                            print(f"   ✅ Precio de cierre realista: {close_price}")
                        else:
                            print(f"   ⚠️ Precio de cierre sospechoso: {close_price}")
                else:
                    print(f"   ❌ No hay datos de velas para {symbol}")
                    
            else:
                print(f"❌ Error HTTP: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def test_market_status():
    """Prueba el estado del mercado"""
    print("\n=== PRUEBA DE ESTADO DEL MERCADO ===")
    
    symbols = ['EURUSD', 'AAPL']
    
    for symbol in symbols:
        print(f"\n--- Estado del mercado para {symbol} ---")
        try:
            response = requests.get(f'http://localhost:8000/api/market-status?symbol={symbol}')
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Estado: {data}")
            else:
                print(f"❌ Error HTTP: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    print(f"🚀 Iniciando pruebas de datos reales de Yahoo Finance")
    print(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_yahoo_real_data()
    test_candles_real_data()
    test_market_status()
    
    print(f"\n✅ Pruebas completadas") 