#!/usr/bin/env python3
"""
Sistema Universal de IA para Trading Forex
==========================================
Sistema profesional que maneja múltiples símbolos forex con configuraciones
automáticas específicas por par y estrategias optimizadas.

Autor: Ingeniero Experto en IA y Trading
Fecha: 2024
"""

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
import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
warnings.filterwarnings('ignore')

# Configurar logging profesional
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('universal_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SymbolConfig:
    """Configuración específica por símbolo forex"""
    symbol: str
    pip_value: float
    typical_spread: float
    volatility_factor: float
    session_sensitivity: str
    base_price: float
    min_data_points: int = 100
    max_retries: int = 5
    
    def __post_init__(self):
        """Validar configuración después de la inicialización"""
        if self.pip_value <= 0:
            raise ValueError(f"pip_value debe ser positivo para {self.symbol}")
        if self.typical_spread <= 0:
            raise ValueError(f"typical_spread debe ser positivo para {self.symbol}")

class UniversalSymbolConfigs:
    """Configuraciones universales para todos los símbolos forex"""
    
    # Configuraciones específicas por símbolo
    SYMBOL_CONFIGS = {
        'EURUSD': {
            'pip_value': 0.0001,
            'typical_spread': 1.5,
            'volatility_factor': 1.0,
            'session_sensitivity': 'HIGH',
            'base_price': 1.0850,
            'min_data_points': 100
        },
        'GBPUSD': {
            'pip_value': 0.0001,
            'typical_spread': 2.5,  # Más volátil
            'volatility_factor': 1.3,
            'session_sensitivity': 'HIGH',
            'base_price': 1.2650,
            'min_data_points': 120
        },
        'USDJPY': {
            'pip_value': 0.01,  # Diferente para JPY
            'typical_spread': 2.0,
            'volatility_factor': 1.2,
            'session_sensitivity': 'MEDIUM',
            'base_price': 150.50,
            'min_data_points': 150
        },
        'AUDUSD': {
            'pip_value': 0.0001,
            'typical_spread': 2.0,
            'volatility_factor': 1.1,
            'session_sensitivity': 'MEDIUM',
            'base_price': 0.6550,
            'min_data_points': 100
        },
        'USDCAD': {
            'pip_value': 0.0001,
            'typical_spread': 2.0,
            'volatility_factor': 1.0,
            'session_sensitivity': 'MEDIUM',
            'base_price': 1.3550,
            'min_data_points': 100
        }
    }
    
    @classmethod
    def get_config(cls, symbol: str) -> SymbolConfig:
        """Obtener configuración para un símbolo específico"""
        if symbol not in cls.SYMBOL_CONFIGS:
            raise ValueError(f"Símbolo {symbol} no está configurado")
        
        config = cls.SYMBOL_CONFIGS[symbol]
        return SymbolConfig(symbol=symbol, **config)
    
    @classmethod
    def get_all_symbols(cls) -> List[str]:
        """Obtener lista de todos los símbolos configurados"""
        return list(cls.SYMBOL_CONFIGS.keys())
    
    @classmethod
    def validate_symbol(cls, symbol: str) -> bool:
        """Validar si un símbolo está configurado"""
        return symbol in cls.SYMBOL_CONFIGS

class UniversalTradingAI:
    """
    Sistema Universal de IA para Trading Forex
    =========================================
    
    Características profesionales:
    - Soporte para múltiples símbolos forex
    - Configuraciones automáticas por par
    - 4 estrategias de trading optimizadas
    - Gestión de riesgo específica por símbolo
    - Logging profesional
    - Manejo robusto de errores
    """
    
    def __init__(self, symbol: str = 'EURUSD'):
        """
        Inicializar sistema universal para un símbolo específico
        
        Args:
            symbol: Símbolo forex (ej: 'EURUSD', 'GBPUSD')
        """
        # Validar símbolo
        if not UniversalSymbolConfigs.validate_symbol(symbol):
            raise ValueError(f"Símbolo {symbol} no está soportado. Símbolos válidos: {UniversalSymbolConfigs.get_all_symbols()}")
        
        self.symbol = symbol
        self.symbol_config = UniversalSymbolConfigs.get_config(symbol)
        
        logger.info(f"🚀 Inicializando UniversalTradingAI para {symbol}")
        logger.info(f"📊 Configuración: pip_value={self.symbol_config.pip_value}, spread={self.symbol_config.typical_spread}")
        
        # Estrategias de trading universales
        self.strategies = {
            'scalping': {
                'timeframes': ['1T', '5T'],
                'target_pips': self._calculate_target_pips(2),
                'stop_loss_pips': self._calculate_stop_loss_pips(1),
                'confidence_threshold': 75,
                'models': {},
                'scalers': {},
                'is_trained': False,
                'min_data_points': 1000
            },
            'day_trading': {
                'timeframes': ['15T', '1H'],
                'target_pips': self._calculate_target_pips(15),
                'stop_loss_pips': self._calculate_stop_loss_pips(8),
                'confidence_threshold': 70,
                'models': {},
                'scalers': {},
                'is_trained': False,
                'min_data_points': 500
            },
            'swing_trading': {
                'timeframes': ['4H', '1D'],
                'target_pips': self._calculate_target_pips(100),
                'stop_loss_pips': self._calculate_stop_loss_pips(50),
                'confidence_threshold': 65,
                'models': {},
                'scalers': {},
                'is_trained': False,
                'min_data_points': 200
            },
            'position_trading': {
                'timeframes': ['1D', '1W'],
                'target_pips': self._calculate_target_pips(500),
                'stop_loss_pips': self._calculate_stop_loss_pips(200),
                'confidence_threshold': 60,
                'models': {},
                'scalers': {},
                'is_trained': False,
                'min_data_points': 100
            }
        }
        
        # Directorio para modelos
        self.models_dir = f"models/trained_models/Brain_Ultra/{self.symbol.replace('=X', '')}"
        os.makedirs(self.models_dir, exist_ok=True)
        
        logger.info(f"✅ Sistema inicializado para {symbol}")
        logger.info(f"📁 Directorio de modelos: {self.models_dir}")
    
    def _calculate_target_pips(self, base_pips: int) -> int:
        """Calcular target pips ajustado por volatilidad del símbolo"""
        return int(base_pips * self.symbol_config.volatility_factor)
    
    def _calculate_stop_loss_pips(self, base_pips: int) -> int:
        """Calcular stop loss pips ajustado por volatilidad del símbolo"""
        return int(base_pips * self.symbol_config.volatility_factor)
    
    def get_real_market_data(self, timeframe: str, periods: str) -> Optional[pd.DataFrame]:
        """
        Obtener datos reales de mercado desde Yahoo Finance
        
        Args:
            timeframe: Intervalo de tiempo ('1m', '5m', '15m', '1h', '4h', '1d')
            periods: Período de datos ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y')
        
        Returns:
            DataFrame con datos OHLCV o None si hay error
        """
        logger.info(f"📊 Descargando datos reales de {self.symbol}: {timeframe} - {periods}")
        
        try:
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(period=periods, interval=timeframe)
            
            if data.empty:
                logger.error(f"❌ No se pudieron obtener datos para {self.symbol} - {timeframe} - {periods}")
                return None
            
            # Limpiar y procesar datos
            data = data.dropna()
            
            # Renombrar columnas para consistencia
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
            
            logger.info(f"✅ Datos obtenidos: {len(data)} períodos")
            logger.info(f"   📅 Rango: {data.index[0]} a {data.index[-1]}")
            logger.info(f"   💰 Precio actual: ${data['close'].iloc[-1]:.5f}")
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo datos de {self.symbol}: {e}")
            return None
    
    def create_strategy_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Crear datasets específicos para cada estrategia
        
        Returns:
            Diccionario con datasets por estrategia
        """
        logger.info(f"📊 Generando datasets para {self.symbol}...")
        
        datasets = {}
        
        # Configuraciones optimizadas por estrategia
        strategy_configs = {
            'scalping': {'timeframe': '5m', 'period': '60d'},
            'day_trading': {'timeframe': '1h', 'period': '6mo'},
            'swing_trading': {'timeframe': '1h', 'period': '1y'},
            'position_trading': {'timeframe': '1d', 'period': '2y'}
        }
        
        for strategy, config in strategy_configs.items():
            logger.info(f"📈 Obteniendo datos para {strategy.upper()}...")
            
            data = self.get_real_market_data(config['timeframe'], config['period'])
            
            if data is not None and len(data) >= self.strategies[strategy]['min_data_points']:
                datasets[strategy] = data
                logger.info(f"✅ {strategy}: {len(data)} períodos obtenidos")
            else:
                logger.warning(f"⚠️ Datos insuficientes para {strategy}")
        
        return datasets
    
    def _initialize_models_for_strategy(self, strategy_name: str) -> None:
        """Inicializar modelos específicos para cada estrategia"""
        
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
    
    def create_features_for_strategy(self, data: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
        """Crear features específicos para cada estrategia"""
        df = data.copy()
        
        # Verificar columnas necesarias
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"⚠️ Columnas faltantes: {missing_columns}")
            return pd.DataFrame()
        
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
        
        # Features de tiempo (importantes para Forex)
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            df['is_london_session'] = ((df['hour'] >= 8) & (df['hour'] <= 16)).astype(int)
            df['is_ny_session'] = ((df['hour'] >= 13) & (df['hour'] <= 21)).astype(int)
            df['is_overlap'] = ((df['hour'] >= 13) & (df['hour'] <= 16)).astype(int)
        
        # Target específico por estrategia
        target_periods = {
            'scalping': 1,
            'day_trading': 3,
            'swing_trading': 10,
            'position_trading': 50
        }
        
        period = target_periods[strategy_name]
        df['target'] = df['close'].shift(-period) / df['close'] - 1
        
        # Limpiar datos
        if len(df) > 50:
            result = df.dropna()
            if len(result) == 0:
                return df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            return result
        else:
            return df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcular RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def prepare_data_for_strategy(self, data: pd.DataFrame, strategy_name: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Preparar datos específicos para una estrategia"""
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
    
    def train_strategy(self, data: pd.DataFrame, strategy_name: str, validation_split: float = 0.2) -> Dict[str, Any]:
        """Entrenar modelos para una estrategia específica"""
        logger.info(f"🚀 Entrenando estrategia: {strategy_name.upper()} para {self.symbol}")
        
        # Inicializar modelos
        self._initialize_models_for_strategy(strategy_name)
        
        # Preparar datos
        X, y, feature_columns = self.prepare_data_for_strategy(data, strategy_name)
        self.strategies[strategy_name]['feature_columns'] = feature_columns
        
        logger.info(f"📊 Features generados: {len(feature_columns)}")
        logger.info(f"📊 Muestras totales: {len(X)}")
        
        # Split temporal
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        logger.info(f"📊 Entrenamiento: {len(X_train)} | Validación: {len(X_val)}")
        
        # Entrenar cada modelo
        results = {}
        
        for model_name, model in self.strategies[strategy_name]['models'].items():
            logger.info(f"🔧 Entrenando {model_name}...")
            
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
                val_pips = val_mae / self.symbol_config.pip_value
                
                results[model_name] = {
                    'val_mse': val_mse,
                    'val_mae': val_mae,
                    'val_pips': val_pips
                }
                
                logger.info(f"✅ {model_name} - MAE: {val_mae:.6f} ({val_pips:.1f} pips)")
                
            except Exception as e:
                logger.error(f"❌ Error entrenando {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        self.strategies[strategy_name]['is_trained'] = True
        logger.info(f"🎉 Estrategia {strategy_name} entrenada exitosamente!")
        
        return results
    
    def train_all_strategies(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """Entrenar todas las estrategias"""
        logger.info(f"🔥 ENTRENANDO TODAS LAS ESTRATEGIAS PARA {self.symbol}")
        
        all_results = {}
        
        for strategy_name in self.strategies.keys():
            if strategy_name in data_dict:
                logger.info(f"\n{'='*20} {strategy_name.upper()} {'='*20}")
                results = self.train_strategy(data_dict[strategy_name], strategy_name)
                all_results[strategy_name] = results
            else:
                logger.warning(f"⚠️ No hay datos para la estrategia {strategy_name}")
        
        return all_results
    
    def save_all_models(self, filepath_prefix: str = None) -> None:
        """Guardar todos los modelos entrenados"""
        if filepath_prefix is None:
            filepath_prefix = f"{self.symbol.replace('=X', '')}_real_data"
        
        logger.info(f"💾 Guardando modelos para {self.symbol}...")
        
        for strategy_name, strategy in self.strategies.items():
            if strategy['is_trained']:
                strategy_prefix = f"{filepath_prefix}_{strategy_name}"
                
                for model_name, model in strategy['models'].items():
                    model_path = os.path.join(self.models_dir, f"{strategy_prefix}_{model_name}.pkl")
                    scaler_path = os.path.join(self.models_dir, f"{strategy_prefix}_{model_name}_scaler.pkl")
                    
                    joblib.dump(model, model_path)
                    joblib.dump(strategy['scalers'][model_name], scaler_path)
                
                # Guardar metadatos de la estrategia
                metadata = {
                    'feature_columns': strategy.get('feature_columns', []),
                    'target_pips': strategy['target_pips'],
                    'stop_loss_pips': strategy['stop_loss_pips'],
                    'confidence_threshold': strategy['confidence_threshold'],
                    'symbol': self.symbol,
                    'symbol_config': {
                        'pip_value': self.symbol_config.pip_value,
                        'typical_spread': self.symbol_config.typical_spread,
                        'volatility_factor': self.symbol_config.volatility_factor
                    }
                }
                metadata_path = os.path.join(self.models_dir, f"{strategy_prefix}_metadata.pkl")
                joblib.dump(metadata, metadata_path)
                
                logger.info(f"✅ Estrategia {strategy_name} guardada en {self.models_dir}")
    
    def load_all_models(self, filepath_prefix: str = None) -> bool:
        """Cargar todos los modelos guardados"""
        if filepath_prefix is None:
            filepath_prefix = f"{self.symbol.replace('=X', '')}_real_data"
        
        logger.info(f"📂 Cargando modelos para {self.symbol}...")
        
        success_count = 0
        total_strategies = len(self.strategies)
        
        for strategy_name in self.strategies.keys():
            try:
                strategy_prefix = f"{filepath_prefix}_{strategy_name}"
                
                # Cargar metadatos
                metadata_path = os.path.join(self.models_dir, f"{strategy_prefix}_metadata.pkl")
                metadata = joblib.load(metadata_path)
                self.strategies[strategy_name]['feature_columns'] = metadata['feature_columns']
                
                # Inicializar modelos
                self._initialize_models_for_strategy(strategy_name)
                
                # Cargar modelos y escaladores
                for model_name in self.strategies[strategy_name]['models'].keys():
                    model_path = os.path.join(self.models_dir, f"{strategy_prefix}_{model_name}.pkl")
                    scaler_path = os.path.join(self.models_dir, f"{strategy_prefix}_{model_name}_scaler.pkl")
                    
                    self.strategies[strategy_name]['models'][model_name] = joblib.load(model_path)
                    self.strategies[strategy_name]['scalers'][model_name] = joblib.load(scaler_path)
                
                self.strategies[strategy_name]['is_trained'] = True
                success_count += 1
                logger.info(f"✅ Estrategia {strategy_name} cargada desde {self.models_dir}")
                
            except FileNotFoundError:
                logger.warning(f"⚠️ No se encontraron modelos para {strategy_name} en {self.models_dir}")
            except Exception as e:
                logger.error(f"❌ Error cargando {strategy_name}: {e}")
        
        if success_count == total_strategies:
            logger.info(f"🎉 Todos los modelos cargados exitosamente para {self.symbol}")
            return True
        else:
            logger.warning(f"⚠️ Solo {success_count}/{total_strategies} estrategias cargadas")
            return False

def main_universal_training():
    """Función principal para entrenar todos los símbolos"""
    logger.info("🚀 SISTEMA UNIVERSAL DE TRADING FOREX")
    logger.info("=" * 60)
    
    # Símbolos a entrenar
    symbols = UniversalSymbolConfigs.get_all_symbols()
    logger.info(f"📊 Símbolos configurados: {symbols}")
    
    for symbol in symbols:
        try:
            logger.info(f"\n{'='*20} ENTRENANDO {symbol} {'='*20}")
            
            # Crear instancia para el símbolo
            ai_system = UniversalTradingAI(symbol)
            
            # Obtener datos
            datasets = ai_system.create_strategy_datasets()
            
            if datasets:
                # Entrenar todas las estrategias
                results = ai_system.train_all_strategies(datasets)
                
                # Guardar modelos
                ai_system.save_all_models()
                
                logger.info(f"✅ {symbol} entrenado exitosamente!")
            else:
                logger.error(f"❌ No se pudieron obtener datos para {symbol}")
                
        except Exception as e:
            logger.error(f"❌ Error entrenando {symbol}: {e}")
            continue
    
    logger.info("🎉 Entrenamiento universal completado!")

if __name__ == "__main__":
    main_universal_training() 