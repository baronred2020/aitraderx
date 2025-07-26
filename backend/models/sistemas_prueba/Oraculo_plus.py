#!/usr/bin/env python3
"""
Sistema de Forecasting Forex - AITRADERX
=======================================

Sistema completo de forecasting que integra:
- Datos reales de Forex
- Calendario económico
- Análisis de sentimiento
- Indicadores económicos
- Machine Learning para predicciones
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# Importar sistemas anteriores
from forex_real_data_system import ForexRealDataSystem
from forex_calendar_api import ForexCalendarAPI, EconomicCalendarEvent

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForexForecastingSystem:
    """
    Sistema principal de forecasting para Forex
    """
    
    def __init__(self, api_keys: Dict[str, str] = None):
        self.api_keys = api_keys or {}
        
        # Inicializar sistemas de datos
        self.data_system = ForexRealDataSystem(api_keys)
        self.calendar_api = ForexCalendarAPI(api_keys)
        
        # Modelos de ML
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
        # Configuración de forecasting
        self.forecast_horizons = [1, 3, 7, 14, 30]  # Días
        self.prediction_features = [
            'price_momentum', 'volatility', 'news_sentiment',
            'economic_federal_funds_rate', 'economic_treasury_10y',
            'dxy_correlation', 'gold_correlation', 'sp500_correlation',
            'days_to_fomc_interest_rate_decision', 'days_to_non_farm_payrolls',
            'days_to_consumer_price_index_cpi', 'market_strength'
        ]
        
        logger.info("🚀 Sistema de Forecasting Forex inicializado")
    
    def prepare_forecasting_data(self, symbol: str, period: str = '6mo') -> pd.DataFrame:
        """
        Preparar datos para forecasting
        
        Args:
            symbol: Símbolo Forex
            period: Período de datos
        
        Returns:
            DataFrame con features para forecasting
        """
        logger.info(f"🔧 Preparando datos de forecasting para {symbol}")
        
        # Obtener datos comprehensivos
        features = self.data_system.create_comprehensive_features(symbol, period)
        
        if features.empty:
            logger.error(f"❌ No se pudieron obtener datos para {symbol}")
            return pd.DataFrame()
        
        # Crear targets para diferentes horizontes
        for horizon in self.forecast_horizons:
            # Target: cambio de precio en X días
            features[f'target_{horizon}d'] = features['price'].shift(-horizon).pct_change(horizon)
            
            # Target: dirección del precio (1 = sube, 0 = baja)
            features[f'direction_{horizon}d'] = (features[f'target_{horizon}d'] > 0).astype(int)
        
        # Limpiar datos
        features = features.dropna()
        
        # Filtrar solo features relevantes para predicción
        feature_columns = [col for col in features.columns if col not in ['target_1d', 'target_3d', 'target_7d', 'target_14d', 'target_30d', 'direction_1d', 'direction_3d', 'direction_7d', 'direction_14d', 'direction_30d']]
        
        # Seleccionar features más importantes
        selected_features = []
        for feature in self.prediction_features:
            if feature in features.columns:
                selected_features.append(feature)
        
        # Agregar features de precio básicos
        price_features = ['price', 'open', 'high', 'low', 'volume']
        for feature in price_features:
            if feature in features.columns:
                selected_features.append(feature)
        
        # Crear DataFrame final
        forecasting_data = features[selected_features + [col for col in features.columns if col.startswith('target_') or col.startswith('direction_')]]
        
        logger.info(f"✅ Datos preparados: {len(forecasting_data)} registros, {len(selected_features)} features")
        
        return forecasting_data
    
    def train_forecasting_models(self, symbol: str, period: str = '6mo'):
        """
        Entrenar modelos de forecasting para diferentes horizontes
        
        Args:
            symbol: Símbolo Forex
            period: Período de datos
        """
        logger.info(f"🎯 Entrenando modelos de forecasting para {symbol}")
        
        # Preparar datos
        data = self.prepare_forecasting_data(symbol, period)
        
        if data.empty:
            logger.error(f"❌ No hay datos para entrenar modelos de {symbol}")
            return
        
        # Separar features y targets
        feature_columns = [col for col in data.columns if not col.startswith('target_') and not col.startswith('direction_')]
        X = data[feature_columns]
        
        # Entrenar modelos para cada horizonte
        for horizon in self.forecast_horizons:
            logger.info(f"🎯 Entrenando modelo para horizonte {horizon}d")
            
            # Target de precio
            y_price = data[f'target_{horizon}d']
            
            # Target de dirección
            y_direction = data[f'direction_{horizon}d']
            
            # Dividir datos
            X_train, X_test, y_price_train, y_price_test, y_direction_train, y_direction_test = train_test_split(
                X, y_price, y_direction, test_size=0.2, random_state=42
            )
            
            # Escalar features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Modelo para predicción de precio
            price_model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            
            price_model.fit(X_train_scaled, y_price_train)
            
            # Modelo para predicción de dirección
            direction_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
            
            direction_model.fit(X_train_scaled, y_direction_train)
            
            # Evaluar modelos
            price_pred = price_model.predict(X_test_scaled)
            direction_pred = direction_model.predict(X_test_scaled)
            
            # Métricas
            price_rmse = np.sqrt(mean_squared_error(y_price_test, price_pred))
            price_mae = mean_absolute_error(y_price_test, price_pred)
            price_r2 = r2_score(y_price_test, price_pred)
            
            direction_accuracy = np.mean((direction_pred > 0.5) == y_direction_test)
            
            # Guardar modelos
            model_key = f"{symbol}_{horizon}d"
            self.models[model_key] = {
                'price_model': price_model,
                'direction_model': direction_model,
                'scaler': scaler,
                'feature_names': feature_columns,
                'metrics': {
                    'price_rmse': price_rmse,
                    'price_mae': price_mae,
                    'price_r2': price_r2,
                    'direction_accuracy': direction_accuracy
                }
            }
            
            # Feature importance
            self.feature_importance[model_key] = {
                'price_importance': dict(zip(feature_columns, price_model.feature_importances_)),
                'direction_importance': dict(zip(feature_columns, direction_model.feature_importances_))
            }
            
            logger.info(f"✅ Modelo {horizon}d entrenado:")
            logger.info(f"   - RMSE: {price_rmse:.4f}")
            logger.info(f"   - MAE: {price_mae:.4f}")
            logger.info(f"   - R²: {price_r2:.4f}")
            logger.info(f"   - Dirección Accuracy: {direction_accuracy:.4f}")
    
    def make_forecast(self, symbol: str, horizon_days: int = 7) -> Dict:
        """
        Realizar forecast para un símbolo
        
        Args:
            symbol: Símbolo Forex
            horizon_days: Horizonte de predicción en días
        
        Returns:
            Diccionario con predicciones
        """
        logger.info(f"🔮 Realizando forecast para {symbol} - {horizon_days}d")
        
        # Verificar si tenemos modelo entrenado
        model_key = f"{symbol}_{horizon_days}d"
        if model_key not in self.models:
            logger.warning(f"⚠️ Modelo no entrenado para {model_key}, entrenando...")
            self.train_forecasting_models(symbol)
        
        if model_key not in self.models:
            logger.error(f"❌ No se pudo entrenar modelo para {symbol}")
            return {}
        
        # Obtener datos más recientes
        recent_data = self.data_system.create_comprehensive_features(symbol, period='1mo')
        
        if recent_data.empty:
            logger.error(f"❌ No hay datos recientes para {symbol}")
            return {}
        
        # Preparar features para predicción
        feature_columns = self.models[model_key]['feature_names']
        available_features = [col for col in feature_columns if col in recent_data.columns]
        
        if len(available_features) < len(feature_columns) * 0.8:  # Al menos 80% de features
            logger.warning(f"⚠️ Features insuficientes para {symbol}")
            return {}
        
        # Usar datos más recientes
        latest_features = recent_data[available_features].iloc[-1:].fillna(0)
        
        # Escalar features
        scaler = self.models[model_key]['scaler']
        latest_features_scaled = scaler.transform(latest_features)
        
        # Realizar predicciones
        price_model = self.models[model_key]['price_model']
        direction_model = self.models[model_key]['direction_model']
        
        price_change_pred = price_model.predict(latest_features_scaled)[0]
        direction_prob = direction_model.predict(latest_features_scaled)[0]
        
        # Obtener precio actual
        current_price = recent_data['price'].iloc[-1]
        
        # Calcular precio predicho
        predicted_price = current_price * (1 + price_change_pred)
        
        # Determinar dirección
        direction = "UP" if direction_prob > 0.5 else "DOWN"
        confidence = max(direction_prob, 1 - direction_prob)
        
        # Obtener eventos económicos relevantes
        currency = symbol.replace('=X', '')[:3]  # EUR, GBP, etc.
        economic_events = self.calendar_api.get_currency_specific_events(currency, days_ahead=horizon_days)
        
        # Análisis de sentimiento
        news_items = self.data_system.get_news_sentiment(symbol, days_back=7)
        avg_sentiment = np.mean([item.sentiment for item in news_items]) if news_items else 0
        
        # Crear resultado
        forecast_result = {
            'symbol': symbol,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'price_change_pred': price_change_pred,
            'predicted_direction': direction,
            'confidence': confidence,
            'horizon_days': horizon_days,
            'economic_events': len(economic_events),
            'news_sentiment': avg_sentiment,
            'model_metrics': self.models[model_key]['metrics'],
            'feature_importance': self.feature_importance[model_key],
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Forecast completado para {symbol}:")
        logger.info(f"   - Precio actual: {current_price:.5f}")
        logger.info(f"   - Precio predicho: {predicted_price:.5f}")
        logger.info(f"   - Dirección: {direction} (confianza: {confidence:.2f})")
        logger.info(f"   - Eventos económicos: {len(economic_events)}")
        
        return forecast_result
    
    def get_multi_horizon_forecast(self, symbol: str) -> Dict:
        """
        Obtener forecast para múltiples horizontes
        
        Args:
            symbol: Símbolo Forex
        
        Returns:
            Diccionario con predicciones para todos los horizontes
        """
        logger.info(f"🔮 Realizando forecast multi-horizon para {symbol}")
        
        forecasts = {}
        
        for horizon in self.forecast_horizons:
            forecast = self.make_forecast(symbol, horizon)
            if forecast:
                forecasts[f'{horizon}d'] = forecast
        
        # Análisis de tendencia
        if forecasts:
            trend_analysis = self._analyze_trend(forecasts)
            forecasts['trend_analysis'] = trend_analysis
        
        logger.info(f"✅ Forecast multi-horizon completado: {len(forecasts)} horizontes")
        
        return forecasts
    
    def get_portfolio_forecast(self, symbols: List[str]) -> Dict:
        """
        Obtener forecast para un portfolio de símbolos
        
        Args:
            symbols: Lista de símbolos Forex
        
        Returns:
            Diccionario con predicciones para todo el portfolio
        """
        logger.info(f"🔮 Realizando forecast de portfolio para {len(symbols)} símbolos")
        
        portfolio_forecasts = {}
        
        for symbol in symbols:
            try:
                forecast = self.get_multi_horizon_forecast(symbol)
                if forecast:
                    portfolio_forecasts[symbol] = forecast
            except Exception as e:
                logger.error(f"❌ Error forecast para {symbol}: {e}")
        
        # Análisis de portfolio
        if portfolio_forecasts:
            portfolio_analysis = self._analyze_portfolio(portfolio_forecasts)
            portfolio_forecasts['portfolio_analysis'] = portfolio_analysis
        
        logger.info(f"✅ Forecast de portfolio completado: {len(portfolio_forecasts)} símbolos")
        
        return portfolio_forecasts
    
    def _analyze_trend(self, forecasts: Dict) -> Dict:
        """Analizar tendencia de las predicciones"""
        trend_analysis = {
            'overall_direction': 'NEUTRAL',
            'confidence_trend': 'STABLE',
            'price_trend': 'STABLE',
            'consistency_score': 0.0
        }
        
        directions = []
        confidences = []
        price_changes = []
        
        for horizon, forecast in forecasts.items():
            if isinstance(forecast, dict):
                directions.append(forecast.get('predicted_direction', 'NEUTRAL'))
                confidences.append(forecast.get('confidence', 0.5))
                price_changes.append(forecast.get('price_change_pred', 0))
        
        if directions:
            # Determinar dirección general
            up_count = directions.count('UP')
            down_count = directions.count('DOWN')
            
            if up_count > down_count:
                trend_analysis['overall_direction'] = 'UP'
            elif down_count > up_count:
                trend_analysis['overall_direction'] = 'DOWN'
            
            # Análisis de confianza
            avg_confidence = np.mean(confidences)
            if avg_confidence > 0.7:
                trend_analysis['confidence_trend'] = 'HIGH'
            elif avg_confidence < 0.5:
                trend_analysis['confidence_trend'] = 'LOW'
            
            # Análisis de precio
            avg_price_change = np.mean(price_changes)
            if avg_price_change > 0.01:
                trend_analysis['price_trend'] = 'BULLISH'
            elif avg_price_change < -0.01:
                trend_analysis['price_trend'] = 'BEARISH'
            
            # Consistencia
            trend_analysis['consistency_score'] = np.std(confidences)
        
        return trend_analysis
    
    def _analyze_portfolio(self, portfolio_forecasts: Dict) -> Dict:
        """Analizar portfolio completo"""
        portfolio_analysis = {
            'total_symbols': len(portfolio_forecasts),
            'bullish_symbols': 0,
            'bearish_symbols': 0,
            'high_confidence_symbols': 0,
            'avg_confidence': 0.0,
            'risk_score': 0.0
        }
        
        confidences = []
        directions = []
        
        for symbol, forecasts in portfolio_forecasts.items():
            if isinstance(forecasts, dict):
                # Obtener forecast de 7 días como referencia
                week_forecast = forecasts.get('7d', {})
                if week_forecast:
                    direction = week_forecast.get('predicted_direction', 'NEUTRAL')
                    confidence = week_forecast.get('confidence', 0.5)
                    
                    confidences.append(confidence)
                    directions.append(direction)
                    
                    if direction == 'UP':
                        portfolio_analysis['bullish_symbols'] += 1
                    elif direction == 'DOWN':
                        portfolio_analysis['bearish_symbols'] += 1
                    
                    if confidence > 0.7:
                        portfolio_analysis['high_confidence_symbols'] += 1
        
        if confidences:
            portfolio_analysis['avg_confidence'] = np.mean(confidences)
            portfolio_analysis['risk_score'] = 1 - np.mean(confidences)  # Menor confianza = mayor riesgo
        
        return portfolio_analysis
    
    def save_models(self, filepath: str):
        """Guardar modelos entrenados"""
        try:
            model_data = {
                'models': self.models,
                'feature_importance': self.feature_importance
            }
            joblib.dump(model_data, filepath)
            logger.info(f"✅ Modelos guardados en {filepath}")
        except Exception as e:
            logger.error(f"❌ Error guardando modelos: {e}")
    
    def load_models(self, filepath: str):
        """Cargar modelos entrenados"""
        try:
            model_data = joblib.load(filepath)
            self.models = model_data['models']
            self.feature_importance = model_data['feature_importance']
            logger.info(f"✅ Modelos cargados desde {filepath}")
        except Exception as e:
            logger.error(f"❌ Error cargando modelos: {e}")

def main():
    """Función principal para probar el sistema de forecasting"""
    
    # Configurar API keys
    api_keys = {
        'NEWS_API_KEY': 'your_news_api_key_here',
        'FRED_API_KEY': 'eef7174b7c21af9ee21506754f567190',
        'TRADING_ECONOMICS_API_KEY': 'your_trading_economics_api_key_here'
    }
    
    # Inicializar sistema
    forecasting_system = ForexForecastingSystem(api_keys)
    
    print("\n🔮 Probando Sistema de Forecasting Forex")
    print("=" * 60)
    
    # Símbolos para probar
    symbols = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X']
    
    # 1. Entrenar modelos
    print("\n🎯 1. Entrenando modelos...")
    for symbol in symbols:
        print(f"   Entrenando {symbol}...")
        forecasting_system.train_forecasting_models(symbol, period='6mo')
    
    # 2. Forecast individual
    print("\n🔮 2. Realizando forecast individual...")
    symbol = 'EURUSD=X'
    forecast = forecasting_system.make_forecast(symbol, horizon_days=7)
    
    if forecast:
        print(f"✅ Forecast para {symbol}:")
        print(f"   - Precio actual: {forecast['current_price']:.5f}")
        print(f"   - Precio predicho: {forecast['predicted_price']:.5f}")
        print(f"   - Dirección: {forecast['predicted_direction']}")
        print(f"   - Confianza: {forecast['confidence']:.2f}")
    
    # 3. Forecast multi-horizon
    print("\n🔮 3. Realizando forecast multi-horizon...")
    multi_forecast = forecasting_system.get_multi_horizon_forecast(symbol)
    
    if multi_forecast:
        print(f"✅ Forecast multi-horizon para {symbol}:")
        for horizon, forecast_data in multi_forecast.items():
            if isinstance(forecast_data, dict):
                print(f"   {horizon}: {forecast_data.get('predicted_direction', 'N/A')} "
                      f"(confianza: {forecast_data.get('confidence', 0):.2f})")
    
    # 4. Forecast de portfolio
    print("\n🔮 4. Realizando forecast de portfolio...")
    portfolio_forecast = forecasting_system.get_portfolio_forecast(symbols)
    
    if portfolio_forecast:
        print(f"✅ Forecast de portfolio completado:")
        for symbol, forecasts in portfolio_forecast.items():
            if isinstance(forecasts, dict) and '7d' in forecasts:
                week_forecast = forecasts['7d']
                print(f"   {symbol}: {week_forecast.get('predicted_direction', 'N/A')} "
                      f"(confianza: {week_forecast.get('confidence', 0):.2f})")
    
    # 5. Guardar modelos
    print("\n💾 5. Guardando modelos...")
    forecasting_system.save_models('forex_forecasting_models.pkl')
    
    print("\n🎉 Sistema de forecasting probado exitosamente!")

if __name__ == "__main__":
    main() 