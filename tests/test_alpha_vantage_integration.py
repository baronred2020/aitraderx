#!/usr/bin/env python3
"""
Script de prueba para verificar la integración de Alpha Vantage
==============================================================

Este script prueba la integración completa de Alpha Vantage en el sistema de forecasting.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from forex_forecasting_system import ForexForecastingSystem
from forex_real_data_system import ForexRealDataSystem
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_alpha_vantage_integration():
    """Probar la integración completa de Alpha Vantage"""
    
    print("\n🔍 PRUEBA DE INTEGRACIÓN ALPHA VANTAGE")
    print("=" * 60)
    
    # Configurar API keys
    api_keys = {
        'ALPHA_VANTAGE_API_KEY': 'WRAU1NL0NSYOJW60',
        'FRED_API_KEY': 'eef7174b7c21af9ee21506754f567190',
        'NEWS_API_KEY': 'your_news_api_key_here'
    }
    
    # 1. Probar sistema de datos reales
    print("\n📊 1. Probando sistema de datos reales...")
    data_system = ForexRealDataSystem(api_keys)
    
    # Probar datos económicos de Alpha Vantage
    print("\n   🔍 Obteniendo datos económicos de Alpha Vantage...")
    alpha_economic_data = data_system.get_alpha_vantage_economic_data()
    
    if alpha_economic_data:
        print(f"   ✅ Datos económicos Alpha Vantage obtenidos: {len(alpha_economic_data)} métricas")
        for key, value in list(alpha_economic_data.items())[:5]:  # Mostrar primeros 5
            print(f"      - {key}: {value}")
    else:
        print("   ⚠️ No se pudieron obtener datos económicos de Alpha Vantage")
    
    # Probar datos de Forex de Alpha Vantage
    print("\n   🔍 Obteniendo datos de Forex de Alpha Vantage...")
    symbols = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X']
    
    for symbol in symbols:
        print(f"\n      Probando {symbol}...")
        alpha_forex_data = data_system.get_alpha_vantage_forex_data(symbol)
        
        if alpha_forex_data:
            print(f"      ✅ Datos Alpha Vantage obtenidos para {symbol}")
            for key, value in alpha_forex_data.items():
                if isinstance(value, (int, float)):
                    print(f"         - {key}: {value}")
                else:
                    print(f"         - {key}: {str(value)[:50]}...")
        else:
            print(f"      ⚠️ No se pudieron obtener datos Alpha Vantage para {symbol}")
    
    # 2. Probar indicadores económicos combinados
    print("\n📈 2. Probando indicadores económicos combinados...")
    combined_indicators = data_system.get_economic_indicators()
    
    if combined_indicators:
        print(f"   ✅ Indicadores combinados obtenidos: {len(combined_indicators)} métricas")
        
        # Mostrar métricas de Alpha Vantage
        alpha_metrics = [k for k in combined_indicators.keys() if 'alpha_vantage' in k]
        if alpha_metrics:
            print(f"   📊 Métricas de Alpha Vantage: {len(alpha_metrics)}")
            for metric in alpha_metrics[:5]:  # Mostrar primeros 5
                print(f"      - {metric}: {combined_indicators[metric]}")
        
        # Mostrar métricas de FRED
        fred_metrics = [k for k in combined_indicators.keys() if 'alpha_vantage' not in k]
        if fred_metrics:
            print(f"   📊 Métricas de FRED: {len(fred_metrics)}")
            for metric in fred_metrics[:5]:  # Mostrar primeros 5
                print(f"      - {metric}: {combined_indicators[metric]}")
    else:
        print("   ⚠️ No se pudieron obtener indicadores combinados")
    
    # 3. Probar features comprehensivos
    print("\n🔧 3. Probando features comprehensivos...")
    symbol = 'EURUSD=X'
    features = data_system.create_comprehensive_features(symbol, period='1mo')
    
    if not features.empty:
        print(f"   ✅ Features comprehensivos creados: {len(features)} registros, {len(features.columns)} features")
        
        # Mostrar features de Alpha Vantage
        alpha_features = [col for col in features.columns if 'alpha_vantage' in col]
        if alpha_features:
            print(f"   📊 Features de Alpha Vantage: {len(alpha_features)}")
            for feature in alpha_features:
                value = features[feature].iloc[-1] if len(features) > 0 else 0
                print(f"      - {feature}: {value}")
        
        # Mostrar features económicos
        economic_features = [col for col in features.columns if 'economic_' in col]
        if economic_features:
            print(f"   📊 Features económicos: {len(economic_features)}")
            for feature in economic_features[:5]:  # Mostrar primeros 5
                value = features[feature].iloc[-1] if len(features) > 0 else 0
                print(f"      - {feature}: {value}")
    else:
        print("   ⚠️ No se pudieron crear features comprehensivos")
    
    # 4. Probar sistema de forecasting completo
    print("\n🔮 4. Probando sistema de forecasting completo...")
    forecasting_system = ForexForecastingSystem(api_keys)
    
    # Entrenar modelo con datos de Alpha Vantage
    print(f"\n   🎯 Entrenando modelo para {symbol}...")
    try:
        forecasting_system.train_forecasting_models(symbol, period='3mo')
        print("   ✅ Modelo entrenado exitosamente")
        
        # Realizar forecast
        print(f"\n   🔮 Realizando forecast para {symbol}...")
        forecast = forecasting_system.make_forecast(symbol, horizon_days=7)
        
        if forecast:
            print("   ✅ Forecast completado:")
            print(f"      - Precio actual: {forecast['current_price']:.5f}")
            print(f"      - Precio predicho: {forecast['predicted_price']:.5f}")
            print(f"      - Dirección: {forecast['predicted_direction']}")
            print(f"      - Confianza: {forecast['confidence']:.2f}")
            print(f"      - Eventos económicos: {forecast['economic_events']}")
            print(f"      - Sentimiento noticias: {forecast['news_sentiment']:.3f}")
        else:
            print("   ⚠️ No se pudo realizar forecast")
            
    except Exception as e:
        print(f"   ❌ Error en forecasting: {e}")
    
    print("\n🎉 Prueba de integración Alpha Vantage completada!")

def test_alpha_vantage_api_directly():
    """Probar la API de Alpha Vantage directamente"""
    
    print("\n🔍 PRUEBA DIRECTA DE ALPHA VANTAGE API")
    print("=" * 50)
    
    import requests
    
    api_key = 'WRAU1NL0NSYOJW60'
    
    # Probar diferentes endpoints
    tests = [
        {
            'name': 'Exchange Rate EUR/USD',
            'function': 'CURRENCY_EXCHANGE_RATE',
            'params': {'from_currency': 'EUR', 'to_currency': 'USD'}
        },
        {
            'name': 'Economic Indicators - GDP',
            'function': 'REAL_GDP',
            'params': {'interval': 'monthly'}
        },
        {
            'name': 'Economic Indicators - CPI',
            'function': 'CPI',
            'params': {'interval': 'monthly'}
        }
    ]
    
    for test in tests:
        print(f"\n🔍 Probando: {test['name']}")
        
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': test['function'],
                'apikey': api_key,
                **test['params']
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'Error Message' in data:
                    print(f"   ❌ Error API: {data['Error Message']}")
                elif 'Note' in data:
                    print(f"   ⚠️ Limitación API: {data['Note']}")
                else:
                    print(f"   ✅ Respuesta exitosa")
                    print(f"      - Keys en respuesta: {list(data.keys())}")
                    
                    # Mostrar algunos datos de ejemplo
                    if 'Realtime Currency Exchange Rate' in data:
                        exchange_data = data['Realtime Currency Exchange Rate']
                        print(f"      - Exchange Rate: {exchange_data.get('5. Exchange Rate', 'N/A')}")
                    elif 'data' in data and data['data']:
                        latest = data['data'][0]
                        print(f"      - Último valor: {latest.get('value', 'N/A')}")
                        print(f"      - Fecha: {latest.get('date', 'N/A')}")
            else:
                print(f"   ❌ Error HTTP: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    print("🚀 Iniciando pruebas de integración Alpha Vantage...")
    
    # 1. Probar API directamente
    test_alpha_vantage_api_directly()
    
    # 2. Probar integración completa
    test_alpha_vantage_integration()
    
    print("\n✅ Todas las pruebas completadas!") 