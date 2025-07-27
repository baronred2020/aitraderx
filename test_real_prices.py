#!/usr/bin/env python3
"""
Script de prueba para verificar que los precios reales están funcionando
"""

import requests
import json
from datetime import datetime

def test_real_prices():
    """Prueba que los precios reales están funcionando"""
    print("=== PRUEBA DE PRECIOS REALES ===")
    print(f"Fecha: {datetime.now()}")
    print("=" * 50)
    
    # 1. Probar datos de mercado reales
    print("\n1. 📊 Probando datos de mercado reales...")
    try:
        response = requests.get('http://localhost:8000/api/market-data?symbols=EURUSD,GBPUSD,AAPL')
        if response.status_code == 200:
            data = response.json()
            print("✅ Datos de mercado obtenidos:")
            for symbol, info in data.items():
                print(f"   {symbol}: ${info['price']} ({info['changePercent']}%)")
        else:
            print(f"❌ Error obteniendo datos de mercado: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # 2. Probar predicciones con precios reales
    print("\n2. 🧠 Probando predicciones con precios reales...")
    try:
        response = requests.get('http://localhost:8000/api/v1/brain-trader/predictions/brain_max?pair=EURUSD&limit=3')
        if response.status_code == 200:
            predictions = response.json()
            print("✅ Predicciones obtenidas:")
            for i, pred in enumerate(predictions, 1):
                print(f"   Predicción {i}: {pred['direction'].upper()} - Target: ${pred['target_price']:.5f} - Confianza: {pred['confidence']:.1f}%")
        else:
            print(f"❌ Error obteniendo predicciones: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # 3. Probar Mega Mind con precios reales
    print("\n3. 🧠 Probando Mega Mind con precios reales...")
    try:
        response = requests.get('http://localhost:8000/api/v1/mega-mind/predictions?pair=EURUSD&limit=3')
        if response.status_code == 200:
            predictions = response.json()
            print("✅ Predicciones Mega Mind obtenidas:")
            for i, pred in enumerate(predictions, 1):
                print(f"   Predicción {i}: {pred['direction'].upper()} - Target: ${pred['target_price']:.5f} - Confianza: {pred['confidence']:.1f}%")
        else:
            print(f"❌ Error obteniendo predicciones Mega Mind: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # 4. Comparar precios
    print("\n4. 🔍 Comparando precios...")
    try:
        # Obtener precio real de mercado
        market_response = requests.get('http://localhost:8000/api/market-data?symbols=EURUSD')
        market_data = market_response.json()
        real_price = float(market_data['EURUSD']['price'])
        
        # Obtener predicciones
        pred_response = requests.get('http://localhost:8000/api/v1/brain-trader/predictions/brain_max?pair=EURUSD&limit=1')
        predictions = pred_response.json()
        
        if predictions:
            pred_price = predictions[0]['target_price']
            difference = abs(pred_price - real_price)
            percentage_diff = (difference / real_price) * 100
            
            print(f"✅ Comparación de precios:")
            print(f"   Precio real EURUSD: ${real_price:.5f}")
            print(f"   Target predicción: ${pred_price:.5f}")
            print(f"   Diferencia: ${difference:.5f} ({percentage_diff:.2f}%)")
            
            if percentage_diff < 5:
                print("   ✅ Los precios están en rango realista")
            else:
                print("   ⚠️ Los precios podrían necesitar ajuste")
                
    except Exception as e:
        print(f"❌ Error comparando precios: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Prueba completada!")

if __name__ == "__main__":
    test_real_prices() 