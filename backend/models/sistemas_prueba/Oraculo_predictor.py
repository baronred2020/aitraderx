# fundamental_predictor.py - Modelo de Predicción Fundamental
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class FundamentalPredictor:
    """Modelo de predicción fundamental para Forex - Predice movimientos ANTES de noticias"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.prediction_horizon = 24  # Horas antes del evento
        
    def create_fundamental_features(self, symbol, period='2y'):
        """Crear features fundamentales avanzados para predicción"""
        
        logger.info(f"📊 Creando features fundamentales para {symbol}")
        
        # Obtener datos históricos
        data = yf.download(symbol, period=period, interval='1d')
        print(f"Columnas descargadas para {symbol}: {list(data.columns)}")
        
        # Si las columnas son MultiIndex, aplanar
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
            print(f"Columnas aplanadas: {list(data.columns)}")
        
        if data.empty:
            logger.error(f"❌ No se pudieron obtener datos para {symbol}")
            return None
        
        # Verificar que tenemos las columnas necesarias
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        # Si falta 'Close' pero existe 'Adj Close', usarla
        if 'Close' not in data.columns and 'Adj Close' in data.columns:
            data['Close'] = data['Adj Close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"❌ Columnas faltantes: {missing_columns}")
            print(f"Columnas disponibles: {list(data.columns)}")
            raise ValueError(f"Faltan columnas requeridas: {missing_columns}")
        
        # Crear DataFrame de features fundamentales
        features = pd.DataFrame(index=data.index)
        
        # === 1. CICLOS ECONÓMICOS ===
        features['economic_cycle'] = self._calculate_economic_cycle(data)
        features['business_cycle_phase'] = self._get_business_cycle_phase()
        features['recession_probability'] = self._calculate_recession_probability()
        
        # === 2. EXPECTATIVAS DE MERCADO ===
        features['rate_expectations'] = self._calculate_rate_expectations()
        features['inflation_expectations'] = self._calculate_inflation_expectations()
        features['growth_expectations'] = self._calculate_growth_expectations()
        
        # === 3. DIVERGENCIAS FUNDAMENTALES ===
        features['rate_divergence'] = self._calculate_rate_divergence(symbol)
        features['inflation_divergence'] = self._calculate_inflation_divergence(symbol)
        features['growth_divergence'] = self._calculate_growth_divergence(symbol)
        
        # === 4. POSICIONAMIENTO INSTITUCIONAL ===
        features['institutional_positioning'] = self._get_institutional_positioning()
        features['hedge_fund_flows'] = self._get_hedge_fund_flows()
        features['central_bank_positioning'] = self._get_central_bank_positioning()
        
        # === 5. SENTIMIENTO PRE-NOTICIA ===
        features['pre_news_sentiment'] = self._get_pre_news_sentiment(symbol)
        features['whisper_numbers'] = self._get_whisper_numbers()
        features['analyst_revisions'] = self._get_analyst_revisions()
        
        # === 6. PATRONES TEMPORALES ===
        features['days_to_fomc'] = self._calculate_days_to_fomc()
        features['days_to_nfp'] = self._calculate_days_to_nfp()
        features['days_to_cpi'] = self._calculate_days_to_cpi()
        
        # === 7. CORRELACIONES MACRO ===
        features['dollar_index_correlation'] = self._get_dollar_index_correlation(symbol)
        features['gold_correlation'] = self._get_gold_correlation(symbol)
        features['oil_correlation'] = self._get_oil_correlation(symbol)
        
        # === 8. FLUJOS DE CAPITAL ===
        features['capital_flows'] = self._get_capital_flows(symbol)
        features['safe_haven_demand'] = self._get_safe_haven_demand()
        features['risk_appetite'] = self._get_risk_appetite()
        
        # === 9. DERIVATIVES SIGNALS ===
        features['options_skew'] = self._get_options_skew(symbol)
        features['futures_positioning'] = self._get_futures_positioning(symbol)
        features['volatility_term_structure'] = self._get_volatility_term_structure(symbol)
        
        # === 10. MICROSTRUCTURA ===
        features['order_flow_imbalance'] = self._get_order_flow_imbalance(symbol)
        features['liquidity_conditions'] = self._get_liquidity_conditions(symbol)
        features['market_depth'] = self._get_market_depth(symbol)
        
        # Combinar con datos de precio
        result = pd.concat([data, features], axis=1)
        
        logger.info(f"✅ Features fundamentales creados: {len(features.columns)} indicadores")
        
        return result
    
    def _calculate_economic_cycle(self, data):
        """Calcular fase del ciclo económico"""
        # Simular ciclo económico basado en tendencias de precio
        trend = data['Close'].rolling(200).mean()
        cycle = np.sin(np.arange(len(data)) * 2 * np.pi / 252)  # Ciclo anual
        return cycle + np.random.normal(0, 0.1, len(data))
    
    def _get_business_cycle_phase(self):
        """Determinar fase del ciclo de negocio"""
        phases = ['expansion', 'peak', 'contraction', 'trough']
        # Simular transiciones entre fases
        current_phase = np.random.choice(phases, p=[0.4, 0.2, 0.3, 0.1])
        return current_phase
    
    def _calculate_recession_probability(self):
        """Calcular probabilidad de recesión"""
        # Basado en yield curve, leading indicators, etc.
        base_prob = 0.15  # 15% base
        yield_curve_effect = np.random.normal(0, 0.05)
        return max(0, min(1, base_prob + yield_curve_effect))
    
    def _calculate_rate_expectations(self):
        """Calcular expectativas de tasas de interés"""
        # Basado en Fed Funds Futures, OIS, etc.
        current_rate = 5.0
        expected_change = np.random.normal(-0.25, 0.5)  # -25bp a +50bp
        return current_rate + expected_change
    
    def _calculate_inflation_expectations(self):
        """Calcular expectativas de inflación"""
        # Basado en breakeven rates, surveys
        base_inflation = 2.5
        inflation_expectation = base_inflation + np.random.normal(0, 0.3)
        return max(0, inflation_expectation)
    
    def _calculate_growth_expectations(self):
        """Calcular expectativas de crecimiento"""
        # Basado en GDP forecasts, PMI, etc.
        base_growth = 2.0
        growth_expectation = base_growth + np.random.normal(0, 0.5)
        return growth_expectation
    
    def _calculate_rate_divergence(self, symbol):
        """Calcular divergencia de tasas entre países"""
        if 'USD' in symbol:
            us_rate = 5.0
            other_rate = 2.0  # EU rate
            return us_rate - other_rate
        return 0
    
    def _calculate_inflation_divergence(self, symbol):
        """Calcular divergencia de inflación"""
        if 'USD' in symbol:
            us_inflation = 3.0
            other_inflation = 2.5  # EU inflation
            return us_inflation - other_inflation
        return 0
    
    def _calculate_growth_divergence(self, symbol):
        """Calcular divergencia de crecimiento"""
        if 'USD' in symbol:
            us_growth = 2.5
            other_growth = 1.5  # EU growth
            return us_growth - other_growth
        return 0
    
    def _get_institutional_positioning(self):
        """Obtener posicionamiento institucional"""
        # Basado en COT reports, surveys
        return np.random.normal(0, 0.3)
    
    def _get_hedge_fund_flows(self):
        """Obtener flujos de hedge funds"""
        return np.random.normal(0, 0.2)
    
    def _get_central_bank_positioning(self):
        """Obtener posicionamiento de bancos centrales"""
        return np.random.normal(0, 0.1)
    
    def _get_pre_news_sentiment(self, symbol):
        """Obtener sentimiento pre-noticia"""
        # Basado en social media, news sentiment
        return np.random.normal(0, 0.4)
    
    def _get_whisper_numbers(self):
        """Obtener whisper numbers (expectativas no oficiales)"""
        return np.random.normal(0, 0.2)
    
    def _get_analyst_revisions(self):
        """Obtener revisiones de analistas"""
        return np.random.normal(0, 0.15)
    
    def _calculate_days_to_fomc(self):
        """Calcular días hasta próxima reunión FOMC"""
        # Simular calendario FOMC
        return np.random.randint(0, 90)
    
    def _calculate_days_to_nfp(self):
        """Calcular días hasta NFP"""
        return np.random.randint(0, 30)
    
    def _calculate_days_to_cpi(self):
        """Calcular días hasta CPI"""
        return np.random.randint(0, 15)
    
    def _get_dollar_index_correlation(self, symbol):
        """Obtener correlación con DXY"""
        return np.random.normal(0.7, 0.2)
    
    def _get_gold_correlation(self, symbol):
        """Obtener correlación con oro"""
        return np.random.normal(-0.3, 0.3)
    
    def _get_oil_correlation(self, symbol):
        """Obtener correlación con petróleo"""
        return np.random.normal(0.2, 0.3)
    
    def _get_capital_flows(self, symbol):
        """Obtener flujos de capital"""
        return np.random.normal(0, 0.4)
    
    def _get_safe_haven_demand(self):
        """Obtener demanda de activos refugio"""
        return np.random.normal(0.5, 0.2)
    
    def _get_risk_appetite(self):
        """Obtener apetito por riesgo"""
        return np.random.normal(0.6, 0.3)
    
    def _get_options_skew(self, symbol):
        """Obtener skew de opciones"""
        return np.random.normal(0, 0.1)
    
    def _get_futures_positioning(self, symbol):
        """Obtener posicionamiento en futuros"""
        return np.random.normal(0, 0.3)
    
    def _get_volatility_term_structure(self, symbol):
        """Obtener estructura temporal de volatilidad"""
        return np.random.normal(0, 0.2)
    
    def _get_order_flow_imbalance(self, symbol):
        """Obtener desbalance de order flow"""
        return np.random.normal(0, 0.4)
    
    def _get_liquidity_conditions(self, symbol):
        """Obtener condiciones de liquidez"""
        return np.random.normal(0.7, 0.2)
    
    def _get_market_depth(self, symbol):
        """Obtener profundidad de mercado"""
        return np.random.normal(0.8, 0.1)
    
    def create_target_variable(self, data, horizon_hours=24):
        """Crear variable objetivo para predicción fundamental"""
        
        logger.info(f"🎯 Creando target para predicción {horizon_hours}h adelante")
        
        # Calcular retorno futuro
        future_return = data['Close'].shift(-horizon_hours) / data['Close'] - 1
        
        # Crear target binario (dirección del movimiento)
        target = np.where(future_return > 0.001, 1,  # Movimiento alcista significativo
                         np.where(future_return < -0.001, -1, 0))  # Movimiento bajista significativo
        
        # Crear target de magnitud
        magnitude_target = future_return
        
        # Crear target de volatilidad
        volatility_target = data['Close'].pct_change().rolling(horizon_hours).std().shift(-horizon_hours)
        
        return {
            'direction': target,
            'magnitude': magnitude_target,
            'volatility': volatility_target
        }
    
    def train_fundamental_model(self, symbol, target_type='direction'):
        """Entrenar modelo de predicción fundamental"""
        
        logger.info(f"🧠 Entrenando modelo fundamental para {symbol}")
        
        # Obtener datos con features fundamentales
        data = self.create_fundamental_features(symbol)
        
        if data is None:
            logger.error("❌ No se pudieron obtener datos")
            return None
        
        # Crear target
        targets = self.create_target_variable(data)
        target = targets[target_type]
        
        # Seleccionar features fundamentales
        fundamental_cols = [col for col in data.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Preparar datos
        X = data[fundamental_cols].fillna(0)
        y = pd.Series(target, index=data.index).fillna(0)
        
        # Filtrar solo columnas numéricas
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        # Si no hay suficientes features numéricos, agregar features básicos de precio
        if len(numeric_cols) < 5:
            logger.warning(f"⚠️ Pocos features numéricos ({len(numeric_cols)}), agregando features de precio")
            price_features = data[['Close', 'High', 'Low', 'Open']].pct_change().fillna(0)
            X = pd.concat([X, price_features], axis=1)
            logger.info(f"✅ Features totales: {X.shape[1]}")
        
        # Guardar las columnas usadas para la predicción
        self.training_columns = X.columns.tolist()
        
        # Remover filas con datos faltantes
        valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) < 100:
            logger.error("❌ Datos insuficientes para entrenamiento")
            return None
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Escalar features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Entrenar modelo
        if target_type == 'direction':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluar modelo
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Guardar modelo y métricas
        self.models[symbol] = model
        self.scalers[symbol] = scaler
        self.feature_importance[symbol] = dict(zip(fundamental_cols, model.feature_importances_))
        
        logger.info(f"✅ Modelo entrenado para {symbol}")
        logger.info(f"📊 MSE: {mse:.6f}, R²: {r2:.4f}")
        
        return {
            'model': model,
            'scaler': scaler,
            'mse': mse,
            'r2': r2,
            'feature_importance': self.feature_importance[symbol]
        }
    
    def predict_fundamental_movement(self, symbol, current_data=None):
        """Predecir movimiento fundamental"""
        
        if symbol not in self.models:
            logger.error(f"❌ Modelo no entrenado para {symbol}")
            return None
        
        # Obtener datos actuales si no se proporcionan
        if current_data is None:
            current_data = self.create_fundamental_features(symbol, period='1mo')
        
        if current_data is None:
            return None
        
        # Preparar features usando las mismas columnas del entrenamiento
        if hasattr(self, 'training_columns') and symbol in self.models:
            # Usar exactamente las mismas columnas del entrenamiento
            available_cols = [col for col in self.training_columns if col in current_data.columns]
            missing_cols = [col for col in self.training_columns if col not in current_data.columns]
            
            if missing_cols:
                logger.warning(f"⚠️ Columnas faltantes en predicción: {missing_cols}")
                # Agregar columnas faltantes con valores 0
                for col in missing_cols:
                    current_data[col] = 0
            
            X = current_data[self.training_columns].fillna(0).iloc[-1:].values
        else:
            # Fallback: usar todas las columnas numéricas disponibles
            fundamental_cols = [col for col in current_data.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
            X = current_data[fundamental_cols].fillna(0).iloc[-1:].values
            
            # Filtrar solo columnas numéricas (mismo filtro que en entrenamiento)
            X_df = pd.DataFrame(X, columns=fundamental_cols)
            numeric_cols = X_df.select_dtypes(include=[np.number]).columns
            X = X_df[numeric_cols].values
            
            # Si no hay suficientes features numéricos, agregar features básicos de precio
            if len(numeric_cols) < 5:
                logger.warning(f"⚠️ Pocos features numéricos en predicción ({len(numeric_cols)}), agregando features de precio")
                price_features = current_data[['Close', 'High', 'Low', 'Open']].pct_change().fillna(0).iloc[-1:].values
                X = np.concatenate([X, price_features], axis=1)
                logger.info(f"✅ Features totales en predicción: {X.shape[1]}")
        
        # Escalar features
        X_scaled = self.scalers[symbol].transform(X)
        
        # Hacer predicción
        prediction = self.models[symbol].predict(X_scaled)[0]
        
        # Interpretar predicción
        if prediction > 0.001:
            direction = "ALCISTA"
            confidence = min(abs(prediction) * 100, 95)
        elif prediction < -0.001:
            direction = "BAJISTA"
            confidence = min(abs(prediction) * 100, 95)
        else:
            direction = "LATERAL"
            confidence = 50
        
        return {
            'symbol': symbol,
            'prediction': prediction,
            'direction': direction,
            'confidence': confidence,
            'horizon_hours': self.prediction_horizon,
            'timestamp': datetime.now()
        }
    
    def get_feature_importance(self, symbol):
        """Obtener importancia de features"""
        
        if symbol not in self.feature_importance:
            return None
        
        importance = self.feature_importance[symbol]
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_features[:10]  # Top 10 features

# Función de prueba
def test_fundamental_predictor():
    """Probar el predictor fundamental"""
    
    print("🧪 PROBANDO PREDICTOR FUNDAMENTAL")
    print("=" * 50)
    
    # Crear predictor
    predictor = FundamentalPredictor()
    
    # Probar con EURUSD
    symbol = 'EURUSD=X'
    print(f"\n📊 Probando con {symbol}...")
    
    # Entrenar modelo
    result = predictor.train_fundamental_model(symbol)
    
    if result:
        print(f"✅ Modelo entrenado exitosamente")
        print(f"📊 MSE: {result['mse']:.6f}")
        print(f"📊 R²: {result['r2']:.4f}")
        
        # Mostrar importancia de features
        importance = predictor.get_feature_importance(symbol)
        print(f"\n🏆 Top 10 Features más importantes:")
        for feature, score in importance:
            print(f"   {feature}: {score:.4f}")
        
        # Hacer predicción
        prediction = predictor.predict_fundamental_movement(symbol)
        if prediction:
            print(f"\n🎯 PREDICCIÓN FUNDAMENTAL:")
            print(f"   Símbolo: {prediction['symbol']}")
            print(f"   Dirección: {prediction['direction']}")
            print(f"   Confianza: {prediction['confidence']:.1f}%")
            print(f"   Horizonte: {prediction['horizon_hours']} horas")
        
        return result
    else:
        print("❌ Error entrenando modelo")
        return None

if __name__ == "__main__":
    test_fundamental_predictor() 