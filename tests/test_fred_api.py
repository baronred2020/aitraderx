#!/usr/bin/env python3
"""
Test Script para FRED API
=========================

Script para probar la integración con FRED API y obtener
datos económicos reales para el sistema de forecasting.
"""

import requests
import json
from datetime import datetime, timedelta
import time
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FREDAPITester:
    """Clase para probar la API de FRED"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred"
    
    def test_series_observations(self, series_id: str, limit: int = 5):
        """Probar obtención de observaciones de una serie"""
        try:
            url = f"{self.base_url}/series/observations"
            params = {
                'series_id': series_id,
                'api_key': self.api_key,
                'file_type': 'json',
                'limit': limit,
                'sort_order': 'desc'
            }
            
            logger.info(f"🔍 Probando serie: {series_id}")
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                observations = data.get('observations', [])
                
                logger.info(f"✅ Serie {series_id}: {len(observations)} observaciones")
                
                if observations:
                    latest = observations[0]
                    logger.info(f"   Último valor: {latest.get('value')} ({latest.get('date')})")
                
                return True
            else:
                logger.error(f"❌ Error {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error probando {series_id}: {e}")
            return False
    
    def test_releases(self):
        """Probar obtención de releases"""
        try:
            url = f"{self.base_url}/releases"
            params = {
                'api_key': self.api_key,
                'file_type': 'json'
            }
            
            logger.info("🔍 Probando releases...")
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                releases = data.get('releases', [])
                logger.info(f"✅ Releases disponibles: {len(releases)}")
                
                # Mostrar algunos releases importantes
                for release in releases[:5]:
                    logger.info(f"   - {release.get('name')} (ID: {release.get('id')})")
                
                return True
            else:
                logger.error(f"❌ Error {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error probando releases: {e}")
            return False
    
    def test_series_search(self, search_text: str):
        """Probar búsqueda de series"""
        try:
            url = f"{self.base_url}/series/search"
            params = {
                'search_text': search_text,
                'api_key': self.api_key,
                'file_type': 'json',
                'limit': 5
            }
            
            logger.info(f"🔍 Buscando series: '{search_text}'")
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                series = data.get('seriess', [])
                logger.info(f"✅ Series encontradas: {len(series)}")
                
                for s in series[:3]:
                    logger.info(f"   - {s.get('title')} (ID: {s.get('id')})")
                
                return True
            else:
                logger.error(f"❌ Error {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error buscando series: {e}")
            return False

def main():
    """Función principal de prueba"""
    
    # API Key de FRED
    api_key = "eef7174b7c21af9ee21506754f567190"
    
    # Inicializar tester
    tester = FREDAPITester(api_key)
    
    print("\n🔍 Probando FRED API")
    print("=" * 50)
    
    # Series importantes para Forex
    important_series = [
        'FEDFUNDS',    # Federal Funds Rate
        'DGS10',       # 10-Year Treasury Rate
        'DGS2',        # 2-Year Treasury Rate
        'CPIAUCSL',    # Consumer Price Index
        'UNRATE',      # Unemployment Rate
        'GDP',         # Gross Domestic Product
        'PAYEMS',      # Nonfarm Payrolls
        'RSAFS',       # Retail Sales
        'INDPRO',      # Industrial Production
        'DTWEXBGS',    # Trade Weighted Dollar Index
        'UMCSENT',     # Consumer Sentiment
        'M2SL',        # M2 Money Supply
        'HOUST',       # Housing Starts
        'BOPGSTB',     # Trade Balance
        'NAPM'         # ISM Manufacturing PMI
    ]
    
    # 1. Probar series individuales
    print("\n📊 1. Probando series económicas importantes...")
    successful_series = 0
    
    for series_id in important_series:
        if tester.test_series_observations(series_id):
            successful_series += 1
        time.sleep(0.1)  # Rate limiting
    
    print(f"\n✅ Series exitosas: {successful_series}/{len(important_series)}")
    
    # 2. Probar releases
    print("\n📋 2. Probando releases...")
    tester.test_releases()
    
    # 3. Probar búsqueda de series
    print("\n🔍 3. Probando búsqueda de series...")
    search_terms = ['inflation', 'employment', 'gdp', 'interest rate']
    
    for term in search_terms:
        tester.test_series_search(term)
        time.sleep(0.1)
    
    # 4. Probar integración con el sistema de forecasting
    print("\n🔮 4. Probando integración con sistema de forecasting...")
    
    try:
        from forex_real_data_system import ForexRealDataSystem
        
        api_keys = {
            'FRED_API_KEY': api_key
        }
        
        system = ForexRealDataSystem(api_keys)
        indicators = system.get_economic_indicators()
        
        if indicators:
            print(f"✅ Indicadores económicos obtenidos: {len(indicators)}")
            print("📊 Métricas disponibles:")
            for key, value in list(indicators.items())[:10]:  # Mostrar primeros 10
                print(f"   - {key}: {value}")
        else:
            print("❌ No se pudieron obtener indicadores económicos")
            
    except Exception as e:
        print(f"❌ Error en integración: {e}")
    
    print("\n🎉 Prueba de FRED API completada!")

if __name__ == "__main__":
    main() 