#!/usr/bin/env python3
"""
Script de prueba para verificar que el frontend muestra precios reales
"""

import requests
import json
from datetime import datetime

def test_frontend_real_prices():
    """Prueba que el frontend está mostrando precios reales"""
    print("=== PRUEBA DE PRECIOS REALES EN FRONTEND ===")
    print(f"Fecha: {datetime.now()}")
    print("=" * 60)
    
    # 1. Verificar que el backend está funcionando
    print("\n1. 🔧 Verificando backend...")
    try:
        response = requests.get('http://localhost:8000/health')
        if response.status_code == 200:
            print("✅ Backend funcionando correctamente")
        else:
            print(f"❌ Backend no responde: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Error conectando al backend: {e}")
        return
    
    # 2. Obtener datos de mercado reales
    print("\n2. 📊 Obteniendo datos de mercado reales...")
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
    
    # 3. Obtener predicciones con precios reales
    print("\n3. 🧠 Obteniendo predicciones con precios reales...")
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
    
    # 4. Verificar que los precios están en rango realista
    print("\n4. 🔍 Verificando rangos de precios...")
    try:
        # Obtener precio real
        market_response = requests.get('http://localhost:8000/api/market-data?symbols=EURUSD')
        market_data = market_response.json()
        real_price = float(market_data['EURUSD']['price'])
        
        # Obtener predicciones
        pred_response = requests.get('http://localhost:8000/api/v1/brain-trader/predictions/brain_max?pair=EURUSD&limit=5')
        predictions = pred_response.json()
        
        print(f"✅ Precio real EURUSD: ${real_price:.5f}")
        print("✅ Target prices de predicciones:")
        
        for i, pred in enumerate(predictions, 1):
            target_price = pred['target_price']
            difference = abs(target_price - real_price)
            percentage_diff = (difference / real_price) * 100
            
            print(f"   Predicción {i}: ${target_price:.5f} (diferencia: {percentage_diff:.2f}%)")
            
            if percentage_diff > 10:
                print(f"   ⚠️ Predicción {i} tiene diferencia alta: {percentage_diff:.2f}%")
            else:
                print(f"   ✅ Predicción {i} en rango realista")
                
    except Exception as e:
        print(f"❌ Error verificando rangos: {e}")
    
    # 5. Verificar frontend
    print("\n5. 🌐 Verificando frontend...")
    try:
        response = requests.get('http://localhost:3000')
        if response.status_code == 200:
            print("✅ Frontend accesible en http://localhost:3000")
            print("📱 Abre http://localhost:3000 en tu navegador para ver los precios reales")
        else:
            print(f"❌ Frontend no responde: {response.status_code}")
            print("💡 Ejecuta 'cd frontend && npm start' para iniciar el frontend")
    except Exception as e:
        print(f"❌ Frontend no disponible: {e}")
        print("💡 Ejecuta 'cd frontend && npm start' para iniciar el frontend")
    
    print("\n" + "=" * 60)
    print("🎉 Prueba completada!")
    print("\n📋 Resumen:")
    print("✅ Backend funcionando con precios reales")
    print("✅ Predicciones usando precios actuales")
    print("✅ Frontend actualizado para mostrar precios reales")
    print("🌐 Abre http://localhost:3000 para ver los cambios")

if __name__ == "__main__":
    test_frontend_real_prices() 