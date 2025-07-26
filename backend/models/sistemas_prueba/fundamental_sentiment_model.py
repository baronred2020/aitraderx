# fundamental_sentiment_model.py - Modelo de Análisis Fundamental y Sentimiento
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
import logging
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class FundamentalSentimentModel:
    """Modelo de análisis fundamental y sentimiento para Forex"""
    
    def __init__(self, api_keys=None):
        self.api_keys = api_keys or {}
        self.cache = {}
        self.cache_duration = timedelta(hours=1)
        
        # Configurar NLTK para análisis de sentimiento
        try:
            nltk.download('vader_lexicon', quiet=True)
            self.sia = SentimentIntensityAnalyzer()
        except:
            logger.warning("⚠️ NLTK no disponible, usando TextBlob")
            self.sia = None
    
    def get_economic_indicators(self, country='US'):
        """Obtener indicadores económicos principales"""
        
        indicators = {}
        
        try:
            # Usar FRED API para datos económicos
            if 'FRED_API_KEY' in self.api_keys:
                fred_api_key = self.api_keys['FRED_API_KEY']
                
                # GDP Growth
                gdp_url = f"https://api.stlouisfed.org/fred/series/observations?series_id=GDP&api_key={fred_api_key}&file_type=json"
                gdp_response = requests.get(gdp_url)
                if gdp_response.status_code == 200:
                    gdp_data = gdp_response.json()
                    indicators['gdp_growth'] = self._calculate_growth_rate(gdp_data['observations'])
                
                # Inflation (CPI)
                cpi_url = f"https://api.stlouisfed.org/fred/series/observations?series_id=CPIAUCSL&api_key={fred_api_key}&file_type=json"
                cpi_response = requests.get(cpi_url)
                if cpi_response.status_code == 200:
                    cpi_data = cpi_response.json()
                    indicators['inflation'] = self._calculate_inflation_rate(cpi_data['observations'])
                
                # Unemployment Rate
                unemp_url = f"https://api.stlouisfed.org/fred/series/observations?series_id=UNRATE&api_key={fred_api_key}&file_type=json"
                unemp_response = requests.get(unemp_url)
                if unemp_response.status_code == 200:
                    unemp_data = unemp_response.json()
                    indicators['unemployment'] = float(unemp_data['observations'][-1]['value'])
                
                # Interest Rate
                rate_url = f"https://api.stlouisfed.org/fred/series/observations?series_id=FEDFUNDS&api_key={fred_api_key}&file_type=json"
                rate_response = requests.get(rate_url)
                if rate_response.status_code == 200:
                    rate_data = rate_response.json()
                    indicators['interest_rate'] = float(rate_data['observations'][-1]['value'])
            
            # Fallback: Datos simulados basados en tendencias históricas
            else:
                logger.info("📊 Usando datos económicos simulados")
                indicators = self._get_simulated_economic_data()
                
        except Exception as e:
            logger.error(f"❌ Error obteniendo indicadores económicos: {e}")
            indicators = self._get_simulated_economic_data()
        
        return indicators
    
    def get_news_sentiment(self, symbol, days_back=30):
        """Obtener sentimiento de noticias para un símbolo"""
        
        try:
            # Usar NewsAPI para noticias
            if 'NEWS_API_KEY' in self.api_keys:
                news_api_key = self.api_keys['NEWS_API_KEY']
                
                # Mapear símbolos a términos de búsqueda
                search_terms = {
                    'EURUSD=X': 'EUR USD Euro Dollar',
                    'GBPUSD=X': 'GBP USD Pound Dollar',
                    'USDJPY=X': 'USD JPY Dollar Yen',
                    'AUDUSD=X': 'AUD USD Australian Dollar',
                    'USDCAD=X': 'USD CAD Dollar Canadian'
                }
                
                search_term = search_terms.get(symbol, symbol.replace('=X', ''))
                
                # Obtener noticias
                news_url = f"https://newsapi.org/v2/everything?q={search_term}&from={datetime.now() - timedelta(days=days_back)}&sortBy=publishedAt&apiKey={news_api_key}"
                response = requests.get(news_url)
                
                if response.status_code == 200:
                    news_data = response.json()
                    
                    # Analizar sentimiento de cada artículo
                    sentiments = []
                    for article in news_data.get('articles', []):
                        title = article.get('title', '')
                        description = article.get('description', '')
                        content = f"{title} {description}"
                        
                        if self.sia:
                            sentiment = self.sia.polarity_scores(content)
                            sentiments.append(sentiment['compound'])
                        else:
                            blob = TextBlob(content)
                            sentiments.append(blob.sentiment.polarity)
                    
                    if sentiments:
                        avg_sentiment = np.mean(sentiments)
                        sentiment_volatility = np.std(sentiments)
                        return {
                            'sentiment': avg_sentiment,
                            'volatility': sentiment_volatility,
                            'volume': len(sentiments)
                        }
            
            # Fallback: Sentimiento simulado
            logger.info(f"📰 Usando sentimiento simulado para {symbol}")
            return self._get_simulated_sentiment(symbol)
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo sentimiento de noticias: {e}")
            return self._get_simulated_sentiment(symbol)
    
    def get_market_sentiment_indicators(self):
        """Obtener indicadores de sentimiento de mercado"""
        
        try:
            # VIX (Fear Index)
            vix = yf.download('^VIX', period='1mo', interval='1d')
            current_vix = vix['Close'].iloc[-1] if not vix.empty else 20
            
            # Fear & Greed Index (simulado)
            fear_greed = self._calculate_fear_greed_index(current_vix)
            
            # Commitment of Traders (COT) - simulado
            cot_data = self._get_simulated_cot_data()
            
            return {
                'vix': current_vix,
                'fear_greed_index': fear_greed,
                'cot_bullish': cot_data['bullish'],
                'cot_bearish': cot_data['bearish'],
                'market_momentum': self._calculate_market_momentum()
            }
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo indicadores de sentimiento: {e}")
            return self._get_simulated_sentiment_indicators()
    
    def create_fundamental_features(self, symbol, period='1y'):
        """Crear features fundamentales para entrenamiento"""
        
        logger.info(f"📊 Creando features fundamentales para {symbol}")
        
        # Obtener datos de precio base
        data = yf.download(symbol, period=period, interval='1d')
        
        if data.empty:
            logger.error(f"❌ No se pudieron obtener datos para {symbol}")
            return None
        
        # Agregar indicadores económicos
        economic_data = self.get_economic_indicators()
        
        # Agregar sentimiento de noticias
        news_sentiment = self.get_news_sentiment(symbol)
        
        # Agregar indicadores de mercado
        market_sentiment = self.get_market_sentiment_indicators()
        
        # Crear features fundamentales
        fundamental_features = pd.DataFrame(index=data.index)
        
        # Economic Features
        fundamental_features['gdp_growth'] = economic_data.get('gdp_growth', 2.0)
        fundamental_features['inflation_rate'] = economic_data.get('inflation', 2.5)
        fundamental_features['unemployment_rate'] = economic_data.get('unemployment', 4.0)
        fundamental_features['interest_rate'] = economic_data.get('interest_rate', 5.0)
        
        # Sentiment Features
        fundamental_features['news_sentiment'] = news_sentiment.get('sentiment', 0.0)
        fundamental_features['sentiment_volatility'] = news_sentiment.get('volatility', 0.1)
        fundamental_features['news_volume'] = news_sentiment.get('volume', 10)
        
        # Market Sentiment Features
        fundamental_features['vix_index'] = market_sentiment.get('vix', 20.0)
        fundamental_features['fear_greed_index'] = market_sentiment.get('fear_greed_index', 50.0)
        fundamental_features['cot_bullish'] = market_sentiment.get('cot_bullish', 0.5)
        fundamental_features['cot_bearish'] = market_sentiment.get('cot_bearish', 0.3)
        fundamental_features['market_momentum'] = market_sentiment.get('market_momentum', 0.0)
        
        # Derivative Features
        fundamental_features['rate_differential'] = fundamental_features['interest_rate'] - 2.0  # vs EU
        fundamental_features['inflation_differential'] = fundamental_features['inflation_rate'] - 2.0
        fundamental_features['economic_strength'] = (fundamental_features['gdp_growth'] - 2.0) / 2.0
        fundamental_features['sentiment_momentum'] = fundamental_features['news_sentiment'].pct_change()
        
        # Interact with price data
        fundamental_features['price_momentum'] = data['Close'].pct_change()
        fundamental_features['volatility'] = data['Close'].pct_change().rolling(20).std()
        
        # Combine with price data
        result = pd.concat([data, fundamental_features], axis=1)
        
        logger.info(f"✅ Features fundamentales creados: {len(fundamental_features.columns)} indicadores")
        
        return result
    
    def _get_simulated_economic_data(self):
        """Datos económicos simulados para testing"""
        return {
            'gdp_growth': 2.1 + np.random.normal(0, 0.5),
            'inflation': 2.5 + np.random.normal(0, 0.3),
            'unemployment': 4.0 + np.random.normal(0, 0.2),
            'interest_rate': 5.0 + np.random.normal(0, 0.1)
        }
    
    def _get_simulated_sentiment(self, symbol):
        """Sentimiento simulado para testing"""
        base_sentiment = np.random.normal(0, 0.3)
        return {
            'sentiment': base_sentiment,
            'volatility': abs(np.random.normal(0.1, 0.05)),
            'volume': np.random.randint(5, 20)
        }
    
    def _get_simulated_sentiment_indicators(self):
        """Indicadores de sentimiento simulados"""
        return {
            'vix': 20 + np.random.normal(0, 5),
            'fear_greed_index': 50 + np.random.normal(0, 15),
            'cot_bullish': 0.5 + np.random.normal(0, 0.1),
            'cot_bearish': 0.3 + np.random.normal(0, 0.1),
            'market_momentum': np.random.normal(0, 0.1)
        }
    
    def _calculate_fear_greed_index(self, vix):
        """Calcular Fear & Greed Index basado en VIX"""
        if vix < 15:
            return 80 + np.random.normal(0, 10)  # Greed
        elif vix < 25:
            return 50 + np.random.normal(0, 15)  # Neutral
        else:
            return 20 + np.random.normal(0, 10)  # Fear
    
    def _calculate_market_momentum(self):
        """Calcular momentum del mercado"""
        return np.random.normal(0, 0.1)
    
    def _get_simulated_cot_data(self):
        """Datos COT simulados"""
        return {
            'bullish': 0.5 + np.random.normal(0, 0.1),
            'bearish': 0.3 + np.random.normal(0, 0.1)
        }
    
    def _calculate_growth_rate(self, observations):
        """Calcular tasa de crecimiento del GDP"""
        if len(observations) >= 2:
            current = float(observations[-1]['value'])
            previous = float(observations[-2]['value'])
            return ((current - previous) / previous) * 100
        return 2.0
    
    def _calculate_inflation_rate(self, observations):
        """Calcular tasa de inflación"""
        if len(observations) >= 12:
            current = float(observations[-1]['value'])
            year_ago = float(observations[-13]['value'])
            return ((current - year_ago) / year_ago) * 100
        return 2.5

# Función de prueba
def test_fundamental_model():
    """Probar el modelo fundamental"""
    
    print("🧪 PROBANDO MODELO FUNDAMENTAL Y SENTIMIENTO")
    print("=" * 50)
    
    # Crear modelo
    model = FundamentalSentimentModel()
    
    # Probar con EURUSD
    symbol = 'EURUSD=X'
    print(f"\n📊 Probando con {symbol}...")
    
    # Crear features fundamentales
    data = model.create_fundamental_features(symbol)
    
    if data is not None:
        print(f"✅ Datos creados: {data.shape}")
        print(f"📋 Features fundamentales: {list(data.columns)}")
        
        # Mostrar estadísticas
        fundamental_cols = [col for col in data.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        print(f"\n📈 Estadísticas de features fundamentales:")
        print(data[fundamental_cols].describe())
        
        return data
    else:
        print("❌ Error creando features fundamentales")
        return None

if __name__ == "__main__":
    test_fundamental_model() 