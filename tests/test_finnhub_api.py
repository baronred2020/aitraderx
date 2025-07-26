#!/usr/bin/env python3
"""
Test Script para API de Finnhub
===============================
Verifica que la API key de Finnhub funcione correctamente
"""

import requests
import json
from datetime import datetime, timedelta
import time

class FinnhubAPITester:
    """Clase para probar la API de Finnhub"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://finnhub.io/api/v1"
        self.headers = {
            'X-Finnhub-Token': api_key
        }
    
    def test_connection(self):
        """Probar conexión básica"""
        try:
            url = f"{self.base_url}/quote"
            params = {'symbol': 'AAPL'}
            
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                print("✅ Conexión a Finnhub exitosa")
                return True
            else:
                print(f"❌ Error de conexión: {response.status_code}")
                print(f"Respuesta: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error de conexión: {e}")
            return False
    
    def test_forex_data(self):
        """Probar datos de Forex"""
        try:
            # Probar diferentes pares de Forex
            symbols = ['OANDA:EUR_USD', 'OANDA:GBP_USD', 'OANDA:USD_JPY']
            
            for symbol in symbols:
                print(f"\n📊 Probando {symbol}...")
                
                url = f"{self.base_url}/forex/candle"
                params = {
                    'symbol': symbol,
                    'resolution': '15',
                    'from': int(time.time()) - 86400,  # Últimas 24 horas
                    'to': int(time.time())
                }
                
                response = requests.get(url, headers=self.headers, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    if data['s'] == 'ok':
                        print(f"✅ {symbol}: {len(data['t'])} puntos de datos")
                        if data['t']:
                            print(f"   Último precio: {data['c'][-1]:.5f}")
                    else:
                        print(f"⚠️ {symbol}: Sin datos disponibles")
                else:
                    print(f"❌ {symbol}: Error {response.status_code}")
                    print(f"   Respuesta: {response.text}")
                    
        except Exception as e:
            print(f"❌ Error probando Forex: {e}")
    
    def test_news_data(self):
        """Probar datos de noticias"""
        try:
            print("\n📰 Probando datos de noticias...")
            
            url = f"{self.base_url}/news"
            params = {
                'category': 'forex',
                'minId': 0
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Noticias obtenidas: {len(data)} artículos")
                
                if data:
                    latest_news = data[0]
                    print(f"   Última noticia: {latest_news.get('headline', 'N/A')}")
                    print(f"   Fuente: {latest_news.get('source', 'N/A')}")
                    print(f"   Fecha: {latest_news.get('datetime', 'N/A')}")
            else:
                print(f"❌ Error obteniendo noticias: {response.status_code}")
                print(f"   Respuesta: {response.text}")
                
        except Exception as e:
            print(f"❌ Error probando noticias: {e}")
    
    def test_economic_calendar(self):
        """Probar calendario económico"""
        try:
            print("\n📅 Probando calendario económico...")
            
            url = f"{self.base_url}/calendar/economic"
            params = {
                'from': datetime.now().strftime('%Y-%m-%d'),
                'to': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Eventos económicos: {len(data)} eventos")
                
                if data:
                    upcoming_events = [e for e in data if e.get('impact') == 'High']
                    print(f"   Eventos de alto impacto: {len(upcoming_events)}")
                    
                    if upcoming_events:
                        next_event = upcoming_events[0]
                        print(f"   Próximo evento: {next_event.get('event', 'N/A')}")
                        print(f"   País: {next_event.get('country', 'N/A')}")
                        print(f"   Fecha: {next_event.get('time', 'N/A')}")
            else:
                print(f"❌ Error obteniendo calendario: {response.status_code}")
                print(f"   Respuesta: {response.text}")
                
        except Exception as e:
            print(f"❌ Error probando calendario: {e}")
    
    def run_full_test(self):
        """Ejecutar todas las pruebas"""
        print("🧪 Iniciando pruebas completas de Finnhub API")
        print("=" * 50)
        
        # Probar conexión
        if not self.test_connection():
            print("❌ No se puede continuar sin conexión")
            return False
        
        # Probar diferentes tipos de datos
        self.test_forex_data()
        self.test_news_data()
        self.test_economic_calendar()
        
        print("\n" + "=" * 50)
        print("✅ Pruebas completadas")
        return True

def main():
    """Función principal"""
    
    # Tu API key de Finnhub
    api_key = "d1nvqv1r01qtrauu8fggd1nvqv1r01qtrauu8fh0"
    
    print("🔧 Probando API de Finnhub")
    print(f"🔑 API Key: {api_key[:10]}...{api_key[-4:]}")
    
    # Crear tester
    tester = FinnhubAPITester(api_key)
    
    # Ejecutar pruebas
    success = tester.run_full_test()
    
    if success:
        print("\n🎉 API de Finnhub configurada correctamente!")
        print("📊 Ya puedes usar datos reales de Forex en tu sistema")
    else:
        print("\n❌ Hay problemas con la API de Finnhub")
        print("🔍 Revisa la API key o contacta soporte")

if __name__ == "__main__":
    main() 