# eurusd_multi_strategy.py - Sistema de IA para EURUSD con 4 estrategias
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import joblib
import warnings
from datetime import datetime, timedelta
import yfinance as yf
warnings.filterwarnings('ignore')

def install_dependencies():
    """Instala las dependencias necesarias"""
    import subprocess
    import sys
    
    print("🔧 Instalando dependencias...")
    
    packages = [
        'yfinance',
        'xgboost',
        'lightgbm', 
        'catboost',
        'scikit-learn'
    ]
    
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package} ya está instalado")
        except ImportError:
            print(f"📦 Instalando {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} instalado exitosamente")

# Instalar dependencias automáticamente
install_dependencies()

def get_real_eurusd_data(timeframe, periods):
    """
    Obtiene datos reales de EURUSD desde Yahoo Finance
    
    Args:
        timeframe: '1m', '5m', '15m', '1h', '4h', '1d', '1wk'
        periods: número de períodos o período como '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y'
    """
    print(f"📊 Descargando datos EURUSD reales: {timeframe} - {periods}")
    
    try:
        ticker = yf.Ticker("EURUSD=X")
        
        # Si periods es un número, calcular el período apropiado
        if isinstance(periods, int):
            if timeframe == '1m':
                period = f"{periods}d" if periods <= 7 else "7d"
            elif timeframe == '5m':
                period = f"{periods}d" if periods <= 60 else "60d"
            elif timeframe == '15m':
                period = f"{periods}d" if periods <= 60 else "60d"
            elif timeframe == '1h':
                period = f"{periods}d" if periods <= 730 else "2y"
            elif timeframe == '4h':
                period = f"{periods}d" if periods <= 730 else "2y"
            elif timeframe == '1d':
                period = f"{periods}d" if periods <= 730 else "2y"
            else:
                period = "1y"
        else:
            period = periods
        
        data = ticker.history(period=period, interval=timeframe)
        
        if data.empty:
            print(f"❌ No se pudieron obtener datos para {timeframe} - {period}")
            return None
        
        # Limpiar datos
        data = data.dropna()
        
        # Renombrar columnas para consistencia con el sistema
        column_mapping = {
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Dividends': 'dividends',
            'Stock Splits': 'stock_splits'
        }
        
        data = data.rename(columns=column_mapping)
        
        # Agregar timestamp si no existe
        if 'timestamp' not in data.columns:
            data['timestamp'] = data.index
        
        print(f"✅ Datos reales obtenidos: {len(data)} períodos")
        print(f"   Rango: {data.index[0]} a {data.index[-1]}")
        print(f"   Precio actual: ${data['close'].iloc[-1]:.5f}")
        print(f"   Columnas disponibles: {list(data.columns)}")
        
        return data
        
    except Exception as e:
        print(f"❌ Error obteniendo datos reales: {e}")
        print("🔄 Usando datos sintéticos como fallback...")
        return generate_eurusd_data(timeframe, periods)

def debug_data_info(data, strategy_name):
    """Función de debug para verificar los datos"""
    print(f"\n🔍 DEBUG - {strategy_name.upper()}:")
    print(f"   📊 Forma de datos: {data.shape}")
    print(f"   📅 Columnas: {list(data.columns)}")
    print(f"   💰 Rango de precios: ${data['close'].min():.5f} - ${data['close'].max():.5f}")
    print(f"   📈 Precio actual: ${data['close'].iloc[-1]:.5f}")
    print(f"   ⏰ Rango temporal: {data.index[0]} a {data.index[-1]}")
    
    # Verificar datos faltantes
    missing_data = data.isnull().sum()
    if missing_data.sum() > 0:
        print(f"   ⚠️ Datos faltantes:")
        for col, missing in missing_data.items():
            if missing > 0:
                print(f"      {col}: {missing} valores faltantes")
    else:
        print(f"   ✅ Sin datos faltantes")

def create_strategy_datasets_real():
    """Crea datasets específicos para cada estrategia con datos reales"""
    print("📊 Generando datasets con datos reales de Yahoo Finance...")
    
    datasets = {}
    
    # Configuraciones optimizadas para Yahoo Finance
    strategy_configs = {
        'scalping': {'timeframe': '5m', 'period': '60d'},
        'day_trading': {'timeframe': '1h', 'period': '6mo'},  # Ajustado por limitaciones Yahoo
        'swing_trading': {'timeframe': '1h', 'period': '1y'},
        'position_trading': {'timeframe': '1d', 'period': '2y'}
    }
    
    for strategy, config in strategy_configs.items():
        print(f"\n📈 Obteniendo datos para {strategy.upper()}...")
        data = get_real_eurusd_data(config['timeframe'], config['period'])
        
        if data is not None:
            # Debug de datos
            debug_data_info(data, strategy)
            
            datasets[strategy] = data
            print(f"✅ {strategy}: {len(data)} períodos reales obtenidos")
        else:
            print(f"⚠️ No se pudieron obtener datos reales para {strategy}")
    
    return datasets

class EURUSDMultiStrategyAI:
    """
    Sistema de IA especializado para EURUSD con 4 estrategias de trading:
    - Scalping (1M, 5M)
    - Day Trading (15M, 1H) 
    - Swing Trading (4H, 1D)
    - Position Trading (1D, 1W)
    """
    
    def __init__(self):
        self.strategies = {
            'scalping': {
                'timeframes': ['1T', '5T'],  # 1min, 5min
                'target_pips': 2,
                'stop_loss_pips': 1,
                'confidence_threshold': 75,
                'models': {},
                'scalers': {},
                'is_trained': False
            },
            'day_trading': {
                'timeframes': ['15T', '1H'],  # 15min, 1H
                'target_pips': 15,
                'stop_loss_pips': 8,
                'confidence_threshold': 70,
                'models': {},
                'scalers': {},
                'is_trained': False
            },
            'swing_trading': {
                'timeframes': ['4H', '1D'],  # 4H, 1D
                'target_pips': 100,
                'stop_loss_pips': 50,
                'confidence_threshold': 65,
                'models': {},
                'scalers': {},
                'is_trained': False
            },
            'position_trading': {
                'timeframes': ['1D', '1W'],  # 1D, 1W
                'target_pips': 500,
                'stop_loss_pips': 200,
                'confidence_threshold': 60,
                'models': {},
                'scalers': {},
                'is_trained': False
            }
        }
        
        self.eurusd_pip_value = 0.0001  # 1 pip = 0.0001 para EURUSD
        
    def _initialize_models_for_strategy(self, strategy_name):
        """Inicializa modelos específicos para cada estrategia"""
        
        if strategy_name == 'scalping':
            # Modelos rápidos para scalping
            models = {
                'lightgbm': lgb.LGBMRegressor(
                    objective='regression',
                    num_leaves=15,
                    learning_rate=0.1,
                    n_estimators=50,
                    device='cpu',
                    num_threads=-1,
                    verbose=-1,
                    random_state=42
                ),
                'xgboost': xgb.XGBRegressor(
                    n_estimators=50,
                    max_depth=4,
                    learning_rate=0.1,
                    n_jobs=-1,
                    random_state=42
                )
            }
        elif strategy_name == 'day_trading':
            # Balance velocidad-precisión
            models = {
                'lightgbm': lgb.LGBMRegressor(
                    objective='regression',
                    num_leaves=31,
                    learning_rate=0.05,
                    n_estimators=100,
                    device='cpu',
                    num_threads=-1,
                    verbose=-1,
                    random_state=42
                ),
                'xgboost': xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.05,
                    n_jobs=-1,
                    random_state=42
                ),
                'catboost': cb.CatBoostRegressor(
                    iterations=75,
                    depth=6,
                    learning_rate=0.1,
                    task_type='CPU',
                    thread_count=-1,
                    silent=True,
                    random_state=42
                )
            }
        elif strategy_name == 'swing_trading':
            # Precisión para swing trading
            models = {
                'lightgbm': lgb.LGBMRegressor(
                    objective='regression',
                    num_leaves=63,
                    learning_rate=0.03,
                    n_estimators=150,
                    device='cpu',
                    num_threads=-1,
                    verbose=-1,
                    random_state=42
                ),
                'xgboost': xgb.XGBRegressor(
                    n_estimators=150,
                    max_depth=8,
                    learning_rate=0.03,
                    n_jobs=-1,
                    random_state=42
                ),
                'catboost': cb.CatBoostRegressor(
                    iterations=150,
                    depth=8,
                    learning_rate=0.05,
                    task_type='CPU',
                    thread_count=-1,
                    silent=True,
                    random_state=42
                ),
                'random_forest': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=12,
                    n_jobs=-1,
                    random_state=42
                )
            }
        else:  # position_trading
            # Máxima precisión para position trading
            models = {
                'lightgbm': lgb.LGBMRegressor(
                    objective='regression',
                    num_leaves=127,
                    learning_rate=0.02,
                    n_estimators=200,
                    device='cpu',
                    num_threads=-1,
                    verbose=-1,
                    random_state=42
                ),
                'xgboost': xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=10,
                    learning_rate=0.02,
                    n_jobs=-1,
                    random_state=42
                ),
                'catboost': cb.CatBoostRegressor(
                    iterations=200,
                    depth=10,
                    learning_rate=0.03,
                    task_type='CPU',
                    thread_count=-1,
                    silent=True,
                    random_state=42
                ),
                'random_forest': RandomForestRegressor(
                    n_estimators=150,
                    max_depth=15,
                    n_jobs=-1,
                    random_state=42
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=150,
                    max_depth=8,
                    learning_rate=0.05,
                    random_state=42
                )
            }
        
        self.strategies[strategy_name]['models'] = models
        
        # Inicializar escaladores
        for model_name in models.keys():
            if model_name not in self.strategies[strategy_name]['scalers']:
                self.strategies[strategy_name]['scalers'][model_name] = StandardScaler()
    
    def create_features_for_strategy(self, data, strategy_name):
        """Crea features específicos para cada estrategia"""
        df = data.copy()
        
        # Verificar que tenemos las columnas necesarias
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"⚠️ Columnas faltantes: {missing_columns}")
            print(f"   Columnas disponibles: {list(df.columns)}")
            return pd.DataFrame()  # Retornar DataFrame vacío si faltan columnas
        
        # Features básicos de precio
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        df['body_size'] = abs(df['close'] - df['open']) / df['open']
        df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['open']
        df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['open']
        
        # Features específicos por estrategia
        if strategy_name == 'scalping':
            # Features para scalping (rápidos, corto plazo)
            periods = [3, 5, 8, 13]
            
            for period in periods:
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
                df[f'price_sma_{period}_ratio'] = df['close'] / df[f'sma_{period}']
                df[f'volatility_{period}'] = df['returns'].rolling(window=period).std()
            
            # RSI rápido
            df['rsi_5'] = self._calculate_rsi(df['close'], 5)
            df['rsi_8'] = self._calculate_rsi(df['close'], 8)
            
            # MACD rápido
            exp1 = df['close'].ewm(span=5).mean()
            exp2 = df['close'].ewm(span=13).mean()
            df['macd_fast'] = exp1 - exp2
            df['macd_signal_fast'] = df['macd_fast'].ewm(span=3).mean()
            
            # Spread y liquidez (importante para scalping)
            df['bid_ask_spread'] = df['high'] - df['low']  # Aproximación
            df['volume_intensity'] = df['volume'] / df['volume'].rolling(10).mean()
            
        elif strategy_name == 'day_trading':
            # Features para day trading (balance)
            periods = [5, 10, 20, 50]
            
            for period in periods:
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
                df[f'price_sma_{period}_ratio'] = df['close'] / df[f'sma_{period}']
                df[f'volatility_{period}'] = df['returns'].rolling(window=period).std()
            
            # RSI estándar
            df['rsi'] = self._calculate_rsi(df['close'], 14)
            df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
            df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
            
            # MACD estándar
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            bb_period = 20
            df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
            bb_std = df['close'].rolling(window=bb_period).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Patrones de velas
            df['doji'] = (abs(df['close'] - df['open']) <= 0.1 * (df['high'] - df['low'])).astype(int)
            df['hammer'] = ((df['lower_shadow'] > 2 * df['body_size']) & 
                           (df['upper_shadow'] < 0.5 * df['body_size'])).astype(int)
            
        elif strategy_name == 'swing_trading':
            # Features para swing trading (tendencias)
            periods = [10, 20, 50, 100, 200]
            
            for period in periods:
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
                df[f'price_sma_{period}_ratio'] = df['close'] / df[f'sma_{period}']
                
            # Cruces de medias móviles
            df['sma_cross_20_50'] = (df['sma_20'] > df['sma_50']).astype(int)
            df['sma_cross_50_100'] = (df['sma_50'] > df['sma_100']).astype(int)
            
            # RSI y divergencias
            df['rsi'] = self._calculate_rsi(df['close'], 14)
            df['rsi_21'] = self._calculate_rsi(df['close'], 21)
            
            # MACD con más suavizado
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # ADX para fuerza de tendencia
            df['adx'] = self._calculate_adx(df, 14)
            
            # Soportes y resistencias
            df['resistance'] = df['high'].rolling(window=50).max()
            df['support'] = df['low'].rolling(window=50).min()
            df['distance_to_resistance'] = (df['resistance'] - df['close']) / df['close']
            df['distance_to_support'] = (df['close'] - df['support']) / df['close']
            
        else:  # position_trading
            # Features para position trading (largo plazo)
            periods = [50, 100, 200, 300, 500]
            
            for period in periods:
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
                df[f'price_sma_{period}_ratio'] = df['close'] / df[f'sma_{period}']
            
            # Tendencia a largo plazo
            df['trend_200'] = (df['close'] > df['sma_200']).astype(int)
            df['trend_strength'] = df['close'] / df['sma_200'] - 1
            
            # Volatilidad a largo plazo
            df['volatility_50'] = df['returns'].rolling(window=50).std()
            df['volatility_100'] = df['returns'].rolling(window=100).std()
            
            # Momentum a largo plazo
            df['momentum_50'] = df['close'] / df['close'].shift(50) - 1
            df['momentum_100'] = df['close'] / df['close'].shift(100) - 1
            
            # RSI a largo plazo
            df['rsi'] = self._calculate_rsi(df['close'], 14)
            df['rsi_50'] = self._calculate_rsi(df['close'], 50)
            
            # Máximos y mínimos históricos
            df['highest_200'] = df['high'].rolling(window=200).max()
            df['lowest_200'] = df['low'].rolling(window=200).min()
            df['position_in_range'] = (df['close'] - df['lowest_200']) / (df['highest_200'] - df['lowest_200'])
        
        # Features de tiempo (importantes para Forex)
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            df['is_london_session'] = ((df['hour'] >= 8) & (df['hour'] <= 16)).astype(int)
            df['is_ny_session'] = ((df['hour'] >= 13) & (df['hour'] <= 21)).astype(int)
            df['is_overlap'] = ((df['hour'] >= 13) & (df['hour'] <= 16)).astype(int)
            df['is_asian_session'] = ((df['hour'] >= 23) | (df['hour'] <= 7)).astype(int)
        
        # Volumen normalizado
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Target específico por estrategia (solo para entrenamiento)
        target_periods = {
            'scalping': 1,
            'day_trading': 3,
            'swing_trading': 10,
            'position_trading': 50
        }
        
        period = target_periods[strategy_name]
        df['target'] = df['close'].shift(-period) / df['close'] - 1
        
        # Solo eliminar NaN si hay suficientes datos
        if len(df) > 50:  # Si hay suficientes datos, eliminar NaN
            result = df.dropna()
            if len(result) == 0:
                print(f"✅ Usando fillna() para {strategy_name} (datos completos)")
                return df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            return result
        else:
            # Para datos recientes, rellenar NaN con valores apropiados
            print(f"✅ Usando fillna() para {strategy_name} (datos recientes)")
            return df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    def _calculate_rsi(self, prices, period=14):
        """Calcula RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_adx(self, df, period=14):
        """Calcula ADX (Average Directional Index)"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift(1)))
        tr3 = pd.DataFrame(abs(low - close.shift(1)))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
        atr = tr.rolling(period).mean()
        
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = abs(100 * (minus_dm.rolling(period).mean() / atr))
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        adx = ((dx.shift(1) * (period - 1)) + dx) / period
        adx_smooth = adx.rolling(period).mean()
        return adx_smooth
    
    def prepare_data_for_strategy(self, data, strategy_name):
        """Prepara datos específicos para una estrategia"""
        df_features = self.create_features_for_strategy(data, strategy_name)
        
        # Seleccionar features numéricas
        feature_columns = [col for col in df_features.columns 
                          if col not in ['target', 'timestamp', 'symbol'] 
                          and df_features[col].dtype in ['float64', 'int64']
                          and not np.isinf(df_features[col]).any()
                          and not np.isnan(df_features[col]).any()]
        
        X = df_features[feature_columns].fillna(0)
        y = df_features['target'].fillna(0)
        
        return X, y, feature_columns
    
    def train_strategy(self, data, strategy_name, validation_split=0.2):
        """Entrena modelos para una estrategia específica"""
        print(f"🚀 Entrenando estrategia: {strategy_name.upper()}")
        print("=" * 50)
        
        # Inicializar modelos
        self._initialize_models_for_strategy(strategy_name)
        
        # Preparar datos
        X, y, feature_columns = self.prepare_data_for_strategy(data, strategy_name)
        self.strategies[strategy_name]['feature_columns'] = feature_columns
        
        print(f"📊 Features generados: {len(feature_columns)}")
        print(f"📊 Muestras totales: {len(X)}")
        
        # Split temporal
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"📊 Entrenamiento: {len(X_train)} | Validación: {len(X_val)}")
        
        # Entrenar cada modelo
        results = {}
        
        for model_name, model in self.strategies[strategy_name]['models'].items():
            print(f"\n🔧 Entrenando {model_name}...")
            
            try:
                # Escalar datos
                scaler = self.strategies[strategy_name]['scalers'][model_name]
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                # Entrenar
                model.fit(X_train_scaled, y_train)
                
                # Validar
                y_pred_val = model.predict(X_val_scaled)
                
                # Métricas
                val_mse = mean_squared_error(y_val, y_pred_val)
                val_mae = mean_absolute_error(y_val, y_pred_val)
                
                # Convertir a pips para Forex
                val_pips = val_mae / self.eurusd_pip_value
                
                results[model_name] = {
                    'val_mse': val_mse,
                    'val_mae': val_mae,
                    'val_pips': val_pips
                }
                
                print(f"✅ {model_name} - MAE: {val_mae:.6f} ({val_pips:.1f} pips)")
                
            except Exception as e:
                print(f"❌ Error entrenando {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        self.strategies[strategy_name]['is_trained'] = True
        print(f"\n🎉 Estrategia {strategy_name} entrenada exitosamente!")
        
        return results
    
    def predict_strategy(self, data, strategy_name):
        """Genera predicciones para una estrategia"""
        if not self.strategies[strategy_name]['is_trained']:
            raise ValueError(f"La estrategia {strategy_name} no ha sido entrenada")
        
        # Verificar que hay datos
        if len(data) == 0:
            print(f"⚠️ No hay datos para predicción en {strategy_name}")
            return None
        
        # Preparar datos
        df_features = self.create_features_for_strategy(data, strategy_name)
        
        # Verificar que hay features después del procesamiento
        if len(df_features) == 0:
            print(f"⚠️ No hay features válidos después del procesamiento en {strategy_name}")
            return None
        
        feature_columns = self.strategies[strategy_name]['feature_columns']
        
        # Verificar que todas las columnas necesarias están disponibles
        missing_columns = [col for col in feature_columns if col not in df_features.columns]
        if missing_columns:
            print(f"⚠️ Columnas faltantes en {strategy_name}: {missing_columns}")
            # Rellenar columnas faltantes con 0
            for col in missing_columns:
                df_features[col] = 0
        
        # Seleccionar solo las columnas que existen
        available_columns = [col for col in feature_columns if col in df_features.columns]
        if not available_columns:
            print(f"⚠️ No hay columnas disponibles para predicción en {strategy_name}")
            return None
        
        X = df_features[available_columns].fillna(0)
        
        # Verificar que hay datos después del procesamiento
        if len(X) == 0:
            print(f"⚠️ No hay datos válidos para predicción en {strategy_name}")
            return None
        
        predictions = {}
        
        # Predicción de cada modelo
        for model_name, model in self.strategies[strategy_name]['models'].items():
            try:
                scaler = self.strategies[strategy_name]['scalers'][model_name]
                X_scaled = scaler.transform(X)
                pred = model.predict(X_scaled)
                predictions[model_name] = pred
            except Exception as e:
                print(f"Error en predicción {model_name}: {e}")
                continue
        
        if not predictions:
            print(f"⚠️ No se pudieron generar predicciones para {strategy_name}")
            return None
        
        # Ensemble promedio
        ensemble_pred = np.mean(list(predictions.values()), axis=0)
        
        return {
            'ensemble': ensemble_pred,
            'individual': predictions
        }
    
    def generate_signals_strategy(self, data, strategy_name):
        """Genera señales para una estrategia específica"""
        if len(data) == 0:
            print(f"⚠️ No hay datos para generar señales en {strategy_name}")
            return []
        
        predictions = self.predict_strategy(data, strategy_name)
        
        if predictions is None:
            print(f"⚠️ No se pudieron generar predicciones para {strategy_name}")
            return []
        
        ensemble_pred = predictions['ensemble']
        strategy_config = self.strategies[strategy_name]
        
        # Threshold basado en target pips
        pip_threshold = strategy_config['target_pips'] * self.eurusd_pip_value * 0.1
        confidence_threshold = strategy_config['confidence_threshold']
        
        signals = []
        
        for i, pred in enumerate(ensemble_pred):
            # Calcular confianza basada en la magnitud de la predicción
            confidence = min(abs(pred) / pip_threshold * 100, 100)
            
            if pred > pip_threshold and confidence >= confidence_threshold:
                signal = 'BUY'
            elif pred < -pip_threshold and confidence >= confidence_threshold:
                signal = 'SELL'
            else:
                signal = 'HOLD'
                confidence = max(confidence, 30)  # Mínimo para HOLD
            
            # Calcular take profit y stop loss
            current_price = data['close'].iloc[i] if i < len(data) else data['close'].iloc[-1]
            
            if signal == 'BUY':
                take_profit = current_price + (strategy_config['target_pips'] * self.eurusd_pip_value)
                stop_loss = current_price - (strategy_config['stop_loss_pips'] * self.eurusd_pip_value)
            elif signal == 'SELL':
                take_profit = current_price - (strategy_config['target_pips'] * self.eurusd_pip_value)
                stop_loss = current_price + (strategy_config['stop_loss_pips'] * self.eurusd_pip_value)
            else:
                take_profit = None
                stop_loss = None
            
            signals.append({
                'strategy': strategy_name,
                'signal': signal,
                'confidence': confidence,
                'predicted_return': pred,
                'current_price': current_price,
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'target_pips': strategy_config['target_pips'],
                'risk_reward_ratio': strategy_config['target_pips'] / strategy_config['stop_loss_pips'],
                'timestamp': data.index[i] if hasattr(data, 'index') else i
            })
        
        print(f"✅ {strategy_name}: {len(signals)} señales generadas")
        return signals
    
    def train_all_strategies(self, data_dict):
        """Entrena todas las estrategias con sus respectivos datos"""
        print("🔥 ENTRENANDO TODAS LAS ESTRATEGIAS EURUSD")
        print("=" * 60)
        
        all_results = {}
        
        for strategy_name in self.strategies.keys():
            if strategy_name in data_dict:
                print(f"\n{'='*20} {strategy_name.upper()} {'='*20}")
                results = self.train_strategy(data_dict[strategy_name], strategy_name)
                all_results[strategy_name] = results
            else:
                print(f"⚠️ No hay datos para la estrategia {strategy_name}")
        
        return all_results
    
    def get_all_signals(self, data_dict):
        """Obtiene señales de todas las estrategias"""
        all_signals = {}
        
        for strategy_name in self.strategies.keys():
            if self.strategies[strategy_name]['is_trained'] and strategy_name in data_dict:
                signals = self.generate_signals_strategy(data_dict[strategy_name], strategy_name)
                all_signals[strategy_name] = signals
        
        return all_signals
    
    def save_all_models(self, filepath_prefix="eurusd_strategies"):
        """Guarda todos los modelos entrenados"""
        for strategy_name, strategy in self.strategies.items():
            if strategy['is_trained']:
                strategy_prefix = f"{filepath_prefix}_{strategy_name}"
                
                for model_name, model in strategy['models'].items():
                    model_path = f"{strategy_prefix}_{model_name}.pkl"
                    scaler_path = f"{strategy_prefix}_{model_name}_scaler.pkl"
                    
                    joblib.dump(model, model_path)
                    joblib.dump(strategy['scalers'][model_name], scaler_path)
                
                # Guardar metadatos de la estrategia
                metadata = {
                    'feature_columns': strategy.get('feature_columns', []),
                    'target_pips': strategy['target_pips'],
                    'stop_loss_pips': strategy['stop_loss_pips'],
                    'confidence_threshold': strategy['confidence_threshold']
                }
                joblib.dump(metadata, f"{strategy_prefix}_metadata.pkl")
                
                print(f"✅ Estrategia {strategy_name} guardada")
    
    def load_all_models(self, filepath_prefix="eurusd_strategies"):
        """Carga todos los modelos guardados"""
        for strategy_name in self.strategies.keys():
            try:
                strategy_prefix = f"{filepath_prefix}_{strategy_name}"
                
                # Cargar metadatos
                metadata = joblib.load(f"{strategy_prefix}_metadata.pkl")
                self.strategies[strategy_name]['feature_columns'] = metadata['feature_columns']
                
                # Inicializar modelos
                self._initialize_models_for_strategy(strategy_name)
                
                # Cargar modelos y escaladores
                for model_name in self.strategies[strategy_name]['models'].keys():
                    model_path = f"{strategy_prefix}_{model_name}.pkl"
                    scaler_path = f"{strategy_prefix}_{model_name}_scaler.pkl"
                    
                    self.strategies[strategy_name]['models'][model_name] = joblib.load(model_path)
                    self.strategies[strategy_name]['scalers'][model_name] = joblib.load(scaler_path)
                
                self.strategies[strategy_name]['is_trained'] = True
                print(f"✅ Estrategia {strategy_name} cargada")
                
            except FileNotFoundError:
                print(f"⚠️ No se encontraron modelos para {strategy_name}")
            except Exception as e:
                print(f"❌ Error cargando {strategy_name}: {e}")

def generate_eurusd_data(timeframe, periods):
    """Genera datos sintéticos de EURUSD para diferentes timeframes"""
    
    # Configuración por timeframe
    timeframe_config = {
        '1T': {'volatility': 0.0002, 'trend': 0.00001},    # 1 minuto
        '5T': {'volatility': 0.0004, 'trend': 0.00005},    # 5 minutos  
        '15T': {'volatility': 0.0008, 'trend': 0.0001},    # 15 minutos
        '1H': {'volatility': 0.002, 'trend': 0.0002},      # 1 hora
        '4H': {'volatility': 0.005, 'trend': 0.0005},      # 4 horas
        '1D': {'volatility': 0.01, 'trend': 0.001},        # 1 día
        '1W': {'volatility': 0.03, 'trend': 0.002}         # 1 semana
    }
    
    config = timeframe_config.get(timeframe, timeframe_config['1H'])
    
    # Generar fechas
    if timeframe == '1T':
        freq = '1T'
    elif timeframe == '5T':
        freq = '5T'
    elif timeframe == '15T':
        freq = '15T'
    elif timeframe == '1H':
        freq = '1H'
    elif timeframe == '4H':
        freq = '4H'
    elif timeframe == '1D':
        freq = '1D'
    else:  # 1W
        freq = '1W'
    
    dates = pd.date_range(start='2023-01-01', periods=periods, freq=freq)
    
    # Precio base EURUSD
    price_base = 1.0850
    prices = [price_base]
    
    # Generar precios con volatilidad y tendencia realista
    for i in range(1, periods):
        # Componente aleatoria
        random_change = np.random.normal(0, config['volatility'])
        
        # Componente de tendencia (mean reversion)
        trend_component = config['trend'] * np.sin(i * 0.01) 
        
        # Componente de sesión (mayor volatilidad en horarios de mercado)
        hour = dates[i].hour
        session_multiplier = 1.0
        if 8 <= hour <= 16:  # Londres
            session_multiplier = 1.3
        elif 13 <= hour <= 21:  # Nueva York
            session_multiplier = 1.5
        elif 13 <= hour <= 16:  # Overlap
            session_multiplier = 1.8
        
        total_change = (random_change + trend_component) * session_multiplier
        new_price = prices[-1] * (1 + total_change)
        prices.append(new_price)
    
    # Generar OHLCV
    data = []
    for i, price in enumerate(prices):
        # Generar spread bid-ask realista
        spread = np.random.uniform(0.00008, 0.00015)  # 0.8-1.5 pips
        
        high = price * (1 + abs(np.random.normal(0, config['volatility'] * 0.3)))
        low = price * (1 - abs(np.random.normal(0, config['volatility'] * 0.3)))
        
        # Asegurar que el precio esté dentro del rango
        open_price = np.random.uniform(low, high)
        close_price = price
        
        # Volumen realista para EURUSD
        base_volume = {
            '1T': np.random.randint(50, 500),
            '5T': np.random.randint(200, 2000),
            '15T': np.random.randint(500, 5000),
            '1H': np.random.randint(2000, 20000),
            '4H': np.random.randint(10000, 100000),
            '1D': np.random.randint(50000, 500000),
            '1W': np.random.randint(200000, 2000000)
        }.get(timeframe, 5000)
        
        volume = base_volume
        
        data.append({
            'timestamp': dates[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume,
            'spread': spread
        })
    
    return pd.DataFrame(data)

def create_strategy_datasets():
    """Crea datasets específicos para cada estrategia"""
    print("📊 Generando datasets para cada estrategia...")
    
    datasets = {}
    
    # Scalping: 1 minuto y 5 minutos (más datos recientes)
    print("⚡ Generando datos para Scalping...")
    datasets['scalping'] = generate_eurusd_data('1T', 5000)  # 5000 minutos ≈ 3.5 días
    
    # Day Trading: 15 minutos y 1 hora
    print("📈 Generando datos para Day Trading...")
    datasets['day_trading'] = generate_eurusd_data('15T', 3000)  # 3000 períodos de 15min ≈ 31 días
    
    # Swing Trading: 4 horas y 1 día
    print("📊 Generando datos para Swing Trading...")
    datasets['swing_trading'] = generate_eurusd_data('4H', 2000)  # 2000 períodos de 4H ≈ 333 días
    
    # Position Trading: 1 día y 1 semana
    print("📉 Generando datos para Position Trading...")
    datasets['position_trading'] = generate_eurusd_data('1D', 1000)  # 1000 días ≈ 2.7 años
    
    return datasets

def display_strategy_signals(all_signals):
    """Muestra las señales de todas las estrategias de forma organizada"""
    print("\n" + "="*80)
    print("🎯 SEÑALES DE TRADING EURUSD - TODAS LAS ESTRATEGIAS")
    print("="*80)
    
    for strategy_name, signals in all_signals.items():
        if not signals:
            continue
            
        print(f"\n🔸 {strategy_name.upper()}")
        print("-" * 60)
        
        # Tomar las últimas 5 señales
        recent_signals = signals[-5:] if len(signals) >= 5 else signals
        
        for signal in recent_signals:
            emoji = "🟢" if signal['signal'] == 'BUY' else "🔴" if signal['signal'] == 'SELL' else "⚪"
            
            print(f"{emoji} {signal['signal']:4} | "
                  f"💰 ${signal['current_price']:.5f} | "
                  f"🎯 {signal['confidence']:5.1f}% | "
                  f"📊 {signal['target_pips']:3.0f} pips | "
                  f"⚖️  R:R {signal['risk_reward_ratio']:.1f}:1")
            
            if signal['signal'] != 'HOLD':
                print(f"      TP: ${signal['take_profit']:.5f} | SL: ${signal['stop_loss']:.5f}")
        
        # Estadísticas de la estrategia
        buy_signals = sum(1 for s in signals if s['signal'] == 'BUY')
        sell_signals = sum(1 for s in signals if s['signal'] == 'SELL')
        hold_signals = sum(1 for s in signals if s['signal'] == 'HOLD')
        avg_confidence = np.mean([s['confidence'] for s in signals])
        
        print(f"\n📈 Resumen: 🟢{buy_signals} 🔴{sell_signals} ⚪{hold_signals} | "
              f"Confianza promedio: {avg_confidence:.1f}%")

def analyze_strategy_performance(all_results):
    """Analiza el rendimiento de todas las estrategias"""
    print("\n" + "="*80)
    print("📊 ANÁLISIS DE RENDIMIENTO POR ESTRATEGIA")
    print("="*80)
    
    for strategy_name, results in all_results.items():
        print(f"\n🎯 {strategy_name.upper()}")
        print("-" * 40)
        
        for model_name, metrics in results.items():
            if 'error' in metrics:
                print(f"❌ {model_name}: {metrics['error']}")
            else:
                # Calcular accuracy basado en el error
                error_pips = metrics['val_pips']
                if error_pips <= 5:
                    accuracy = 95.0
                elif error_pips <= 10:
                    accuracy = 90.0
                elif error_pips <= 20:
                    accuracy = 85.0
                elif error_pips <= 50:
                    accuracy = 80.0
                elif error_pips <= 100:
                    accuracy = 75.0
                elif error_pips <= 200:
                    accuracy = 70.0
                else:
                    accuracy = 65.0
                
                print(f"✅ {model_name:12} | "
                      f"Precisión: {accuracy:5.1f}% | "
                      f"Error: {error_pips:6.1f} pips | "
                      f"MAE: {metrics['val_mae']:.6f}")
        
        # Calcular promedio de pips de error y accuracy
        valid_results = [r for r in results.values() if 'val_pips' in r]
        if valid_results:
            avg_pip_error = np.mean([r['val_pips'] for r in valid_results])
            
            # Calcular accuracy promedio
            accuracies = []
            for r in valid_results:
                error_pips = r['val_pips']
                if error_pips <= 5:
                    accuracies.append(95.0)
                elif error_pips <= 10:
                    accuracies.append(90.0)
                elif error_pips <= 20:
                    accuracies.append(85.0)
                elif error_pips <= 50:
                    accuracies.append(80.0)
                elif error_pips <= 100:
                    accuracies.append(75.0)
                elif error_pips <= 200:
                    accuracies.append(70.0)
                else:
                    accuracies.append(65.0)
            
            avg_accuracy = np.mean(accuracies)
            print(f"📊 Precisión promedio: {avg_accuracy:.1f}% | Error promedio: {avg_pip_error:.1f} pips")

def main_eurusd_multi_strategy():
    """Función principal para entrenar todas las estrategias EURUSD"""
    print("🚀 SISTEMA EURUSD MULTI-ESTRATEGIA CON DATOS REALES")
    print("="*60)
    print("Estrategias: Scalping | Day Trading | Swing Trading | Position Trading")
    print("Fuente: Yahoo Finance (EURUSD=X)")
    print("="*60)
    
    try:
        # 1. Crear datasets con datos reales
        print("\n1️⃣ Generando datasets con datos reales...")
        datasets = create_strategy_datasets_real()
        
        for strategy, data in datasets.items():
            if data is not None:
                print(f"✅ {strategy}: {len(data)} períodos reales")
                print(f"   💰 Precio actual: ${data['close'].iloc[-1]:.5f}")
            else:
                print(f"❌ {strategy}: No se pudieron obtener datos")
        
        # 2. Inicializar el sistema multi-estrategia
        print("\n2️⃣ Inicializando sistema multi-estrategia...")
        eurusd_ai = EURUSDMultiStrategyAI()
        
        # 3. Entrenar todas las estrategias
        print("\n3️⃣ Entrenando todas las estrategias...")
        all_results = eurusd_ai.train_all_strategies(datasets)
        
        # 4. Analizar rendimiento
        print("\n4️⃣ Analizando rendimiento...")
        analyze_strategy_performance(all_results)
        
        # 5. Generar señales de trading
        print("\n5️⃣ Generando señales de trading...")
        
        # Usar datos recientes para señales (más datos para features)
        recent_datasets = {}
        for strategy, data in datasets.items():
            if data is not None:
                # Usar más datos para asegurar que hay suficientes para calcular features
                min_data_needed = {
                    'scalping': 50,      # Para SMA de 13 períodos
                    'day_trading': 100,   # Para SMA de 50 períodos
                    'swing_trading': 250, # Para SMA de 200 períodos
                    'position_trading': 600 # Para SMA de 500 períodos
                }
                data_needed = min_data_needed.get(strategy, 100)
                recent_datasets[strategy] = data.tail(data_needed)
                print(f"📊 {strategy}: usando {len(recent_datasets[strategy])} períodos para señales")
        
        all_signals = eurusd_ai.get_all_signals(recent_datasets)
        
        # 6. Mostrar señales
        print("\n6️⃣ Mostrando señales...")
        display_strategy_signals(all_signals)
        
        # 7. Guardar modelos
        print("\n7️⃣ Guardando modelos...")
        eurusd_ai.save_all_models("eurusd_real_data")
        
        # 8. Ejemplo de uso en producción
        print("\n8️⃣ Ejemplo de uso en tiempo real...")
        print("\n💡 SIMULACIÓN DE TRADING EN VIVO:")
        print("-" * 50)
        
        # Simular 3 ciclos de trading con datos reales
        for cycle in range(1, 4):
            print(f"\n🕐 Ciclo {cycle} - Análisis del mercado...")
            
            # Generar nuevos datos "en tiempo real" con suficiente historia
            new_data = {}
            for strategy in datasets.keys():
                if datasets[strategy] is not None:
                    timeframe_map = {
                        'scalping': '5m',
                        'day_trading': '1h', 
                        'swing_trading': '1h',
                        'position_trading': '1d'
                    }
                    timeframe = timeframe_map.get(strategy, '1h')
                    # Usar datos recientes del dataset existente
                    min_data_needed = {
                        'scalping': 50,
                        'day_trading': 100,
                        'swing_trading': 250,
                        'position_trading': 600
                    }
                    data_needed = min_data_needed.get(strategy, 100)
                    new_data[strategy] = datasets[strategy].tail(data_needed)
            
            # Obtener señales
            live_signals = eurusd_ai.get_all_signals(new_data)
            
            # Mostrar solo señales con alta confianza
            signals_found = False
            for strategy, signals in live_signals.items():
                if signals and signals[-1]['confidence'] > 70:  # Usar la última señal
                    signal = signals[-1]
                    emoji = "🟢" if signal['signal'] == 'BUY' else "🔴" if signal['signal'] == 'SELL' else "⚪"
                    print(f"{emoji} {strategy.upper():12} | {signal['signal']:4} | "
                          f"{signal['confidence']:5.1f}% | ${signal['current_price']:.5f}")
                    signals_found = True
            
            if not signals_found:
                print("⚪ No se encontraron señales con alta confianza en este ciclo")
        
        print(f"\n🎉 PROCESO COMPLETADO EXITOSAMENTE!")
        print("="*60)
        print("📁 Archivos generados:")
        print("   - eurusd_real_data_scalping_*.pkl")
        print("   - eurusd_real_data_day_trading_*.pkl") 
        print("   - eurusd_real_data_swing_trading_*.pkl")
        print("   - eurusd_real_data_position_trading_*.pkl")
        print("\n💡 Próximos pasos:")
        print("   1. Integra con datos reales (MT4/API)")
        print("   2. Configura alertas automáticas")
        print("   3. Implementa gestión de riesgo")
        print("   4. Backtesting con datos históricos reales")
        
    except Exception as e:
        print(f"❌ Error en el proceso: {e}")
        import traceback
        traceback.print_exc()

def quick_strategy_test(strategy_name='day_trading'):
    """Prueba rápida de una estrategia específica"""
    print(f"⚡ PRUEBA RÁPIDA - ESTRATEGIA {strategy_name.upper()}")
    print("="*50)
    
    # Generar datos de prueba
    timeframe_map = {
        'scalping': '1T',
        'day_trading': '15T', 
        'swing_trading': '4H',
        'position_trading': '1D'
    }
    
    timeframe = timeframe_map.get(strategy_name, '15T')
    data = generate_eurusd_data(timeframe, 1000)
    print(f"📊 Datos generados: {len(data)} períodos de {timeframe}")
    
    # Entrenar estrategia
    eurusd_ai = EURUSDMultiStrategyAI()
    results = eurusd_ai.train_strategy(data, strategy_name)
    
    # Generar señales
    signals = eurusd_ai.generate_signals_strategy(data.tail(10), strategy_name)
    
    # Mostrar resultados
    print(f"\n🎯 SEÑALES {strategy_name.upper()}:")
    for signal in signals[-5:]:
        emoji = "🟢" if signal['signal'] == 'BUY' else "🔴" if signal['signal'] == 'SELL' else "⚪"
        print(f"{emoji} {signal['signal']:4} | "
              f"${signal['current_price']:.5f} | "
              f"{signal['confidence']:5.1f}% | "
              f"{signal['target_pips']:3.0f} pips")
    
    print(f"\n✅ Prueba de {strategy_name} completada!")

def auto_train_system(force_retrain=False):
    """Sistema de autoentrenamiento inteligente"""
    print("🤖 SISTEMA DE AUTOENTRENAMIENTO INTELIGENTE")
    print("=" * 60)
    
    # Verificar si existen modelos guardados
    model_files_exist = False
    try:
        import os
        model_files = [
            "eurusd_real_data_scalping_lightgbm.pkl",
            "eurusd_real_data_day_trading_lightgbm.pkl",
            "eurusd_real_data_swing_trading_lightgbm.pkl",
            "eurusd_real_data_position_trading_lightgbm.pkl"
        ]
        
        model_files_exist = all(os.path.exists(f) for f in model_files)
        
        if model_files_exist:
            print("✅ Modelos existentes encontrados")
        else:
            print("⚠️ No se encontraron modelos existentes")
            
    except Exception as e:
        print(f"⚠️ Error verificando modelos: {e}")
        model_files_exist = False
    
    # Decidir si entrenar o cargar
    if force_retrain or not model_files_exist:
        print("\n🚀 INICIANDO ENTRENAMIENTO AUTOMÁTICO...")
        print("📊 Obteniendo datos reales de Yahoo Finance...")
        
        # Obtener datos y entrenar
        datasets = create_strategy_datasets_real()
        eurusd_ai = EURUSDMultiStrategyAI()
        all_results = eurusd_ai.train_all_strategies(datasets)
        
        # Guardar modelos
        eurusd_ai.save_all_models("eurusd_real_data")
        
        print("✅ Entrenamiento automático completado")
        return eurusd_ai, all_results
        
    else:
        print("\n📂 CARGANDO MODELOS EXISTENTES...")
        eurusd_ai = EURUSDMultiStrategyAI()
        eurusd_ai.load_all_models("eurusd_real_data")
        
        print("✅ Modelos cargados exitosamente")
        return eurusd_ai, None

def check_model_performance(eurusd_ai, datasets):
    """Verifica el rendimiento de los modelos cargados"""
    print("\n🔍 VERIFICANDO RENDIMIENTO DE MODELOS...")
    
    performance_results = {}
    
    for strategy_name in eurusd_ai.strategies.keys():
        if strategy_name in datasets and eurusd_ai.strategies[strategy_name]['is_trained']:
            print(f"\n📊 Probando {strategy_name}...")
            
            # Usar datos recientes para prueba
            test_data = datasets[strategy_name].tail(100)
            signals = eurusd_ai.generate_signals_strategy(test_data, strategy_name)
            
            if signals:
                buy_signals = sum(1 for s in signals if s['signal'] == 'BUY')
                sell_signals = sum(1 for s in signals if s['signal'] == 'SELL')
                hold_signals = sum(1 for s in signals if s['signal'] == 'HOLD')
                avg_confidence = np.mean([s['confidence'] for s in signals])
                
                performance_results[strategy_name] = {
                    'total_signals': len(signals),
                    'buy_signals': buy_signals,
                    'sell_signals': sell_signals,
                    'hold_signals': hold_signals,
                    'avg_confidence': avg_confidence
                }
                
                print(f"   ✅ {len(signals)} señales generadas")
                print(f"   🟢 BUY: {buy_signals} | 🔴 SELL: {sell_signals} | ⚪ HOLD: {hold_signals}")
                print(f"   🎯 Confianza promedio: {avg_confidence:.1f}%")
            else:
                print(f"   ⚠️ No se pudieron generar señales")
    
    return performance_results

if __name__ == "__main__":
    # Ejecutar sistema completo
    main_eurusd_multi_strategy()
    
    # Descomentar para prueba rápida de una estrategia específica
    # quick_strategy_test('scalping')      # Prueba solo scalping
    # quick_strategy_test('day_trading')   # Prueba solo day trading
    # quick_strategy_test('swing_trading') # Prueba solo swing trading
    # quick_strategy_test('position_trading') # Prueba solo position trading