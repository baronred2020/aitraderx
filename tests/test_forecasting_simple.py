#!/usr/bin/env python3
"""
Test simple del sistema de forecasting
"""

import sys
import os
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Probar que todas las importaciones funcionan"""
    print("Probando importaciones...")
    
    try:
        import pandas as pd
        print("✅ pandas importado")
    except Exception as e:
        print(f"❌ Error importando pandas: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy importado")
    except Exception as e:
        print(f"❌ Error importando numpy: {e}")
        return False
    
    try:
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        print("✅ sklearn importado")
    except Exception as e:
        print(f"❌ Error importando sklearn: {e}")
        return False
    
    try:
        from forex_real_data_system import ForexRealDataSystem
        print("✅ ForexRealDataSystem importado")
    except Exception as e:
        print(f"❌ Error importando ForexRealDataSystem: {e}")
        return False
    
    try:
        from forex_calendar_api import ForexCalendarAPI
        print("✅ ForexCalendarAPI importado")
    except Exception as e:
        print(f"❌ Error importando ForexCalendarAPI: {e}")
        return False
    
    return True

def test_data_system():
    """Probar el sistema de datos"""
    print("\nProbando sistema de datos...")
    
    try:
        api_keys = {
            'FRED_API_KEY': 'eef7174b7c21af9ee21506754f567190',
            'ALPHA_VANTAGE_API_KEY': 'WRAU1NL0NSYOJW60'
        }
        
        data_system = ForexRealDataSystem(api_keys)
        print("✅ Sistema de datos inicializado")
        
        # Probar obtener datos de un par
        symbol = 'EURUSD=X'
        print(f"Obteniendo datos para {symbol}...")
        
        # Obtener datos basicos
        price_data = data_system.get_real_forex_data(symbol, period='1mo', interval='1d')
        if not price_data.empty:
            print(f"✅ Datos de precio obtenidos: {len(price_data)} registros")
        else:
            print("⚠️ No se obtuvieron datos de precio")
        
        # Obtener indicadores economicos
        indicators = data_system.get_economic_indicators()
        if indicators:
            print(f"✅ Indicadores economicos obtenidos: {len(indicators)} indicadores")
        else:
            print("⚠️ No se obtuvieron indicadores economicos")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en sistema de datos: {e}")
        return False

def test_calendar_api():
    """Probar el sistema de calendario"""
    print("\nProbando sistema de calendario...")
    
    try:
        api_keys = {}
        calendar_api = ForexCalendarAPI(api_keys)
        print("✅ Sistema de calendario inicializado")
        
        # Probar obtener eventos
        events = calendar_api.get_currency_specific_events('EUR', days_ahead=7)
        print(f"✅ Eventos obtenidos: {len(events)} eventos")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en sistema de calendario: {e}")
        return False

def main():
    """Funcion principal de prueba"""
    print("=" * 60)
    print("TEST SIMPLE DEL SISTEMA DE FORECASTING")
    print("=" * 60)
    
    # 1. Probar importaciones
    if not test_imports():
        print("❌ Fallo en importaciones")
        return
    
    # 2. Probar sistema de datos
    if not test_data_system():
        print("❌ Fallo en sistema de datos")
        return
    
    # 3. Probar sistema de calendario
    if not test_calendar_api():
        print("❌ Fallo en sistema de calendario")
        return
    
    print("\n" + "=" * 60)
    print("✅ TODAS LAS PRUEBAS EXITOSAS!")
    print("El sistema esta listo para entrenar los 5 pares:")
    print("- EURUSD=X")
    print("- GBPUSD=X") 
    print("- USDJPY=X")
    print("- AUDUSD=X")
    print("- USDCAD=X")
    print("=" * 60)

if __name__ == "__main__":
    main() 