# ====================================================================
# UNIVERSAL TRADING BRAIN - SISTEMA MULTI-CURRENCY
# Adaptable a cualquier par de divisas automáticamente
# ====================================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
import yfinance as yf
import ta
from datetime import datetime, timedelta
import warnings
import json
import pickle
from typing import Dict, List, Tuple, Optional
import gym
from gym import spaces
from collections import deque
import random
from dataclasses import dataclass
import logging

warnings.filterwarnings('ignore')

# ====================================================================
# CONFIGURACIONES UNIVERSALES POR TIPO DE PAR
# ====================================================================

class UniversalPairConfig:
    """Configuraciones automáticas según el tipo de par"""
    
    # Configuraciones base por tipo de par
    PAIR_CONFIGS = {
        # MAJORS (G7 currencies)
        'MAJORS': {
            'pairs': ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD'],
            'typical_spread': 1.5,
            'pip_value': 0.0001,
            'daily_volatility': 100,  # pips
            'session_sensitivity': 'HIGH',
            'news_sensitivity': 'HIGH',
            'scalping_viable': True,
            'leverage_available': 500
        },
        
        # MINORS (Cross pairs)
        'MINORS': {
            'pairs': ['EURGBP', 'EURJPY', 'GBPJPY', 'EURCHF', 'AUDCAD', 'CADJPY'],
            'typical_spread': 2.5,
            'pip_value': 0.0001,
            'daily_volatility': 120,
            'session_sensitivity': 'MEDIUM',
            'news_sensitivity': 'MEDIUM',
            'scalping_viable': True,
            'leverage_available': 200
        },
        
        # EXOTICS
        'EXOTICS': {
            'pairs': ['USDTRY', 'USDZAR', 'USDMXN', 'EURPLN', 'GBPTRY'],
            'typical_spread': 15,
            'pip_value': 0.0001,
            'daily_volatility': 300,
            'session_sensitivity': 'LOW',
            'news_sensitivity': 'EXTREME',
            'scalping_viable': False,
            'leverage_available': 50
        },
        
        # COMMODITIES
        'COMMODITIES': {
            'pairs': ['XAUUSD', 'XAGUSD', 'USOIL', 'BRENT'],
            'typical_spread': 3,
            'pip_value': 0.01,  # Different for gold
            'daily_volatility': 2000,  # cents for gold
            'session_sensitivity': 'MEDIUM',
            'news_sensitivity': 'HIGH',
            'scalping_viable': True,
            'leverage_available': 100
        },
        
        # CRYPTOCURRENCIES
        'CRYPTO': {
            'pairs': ['BTCUSD', 'ETHUSD', 'ADAUSD', 'DOTUSD'],
            'typical_spread': 10,
            'pip_value': 1,
            'daily_volatility': 5000,  # Much higher volatility
            'session_sensitivity': 'NONE',  # 24/7 market
            'news_sensitivity': 'EXTREME',
            'scalping_viable': True,
            'leverage_available': 100
        }
    }
    
    @classmethod
    def detect_pair_type(cls, symbol):
        """Detecta automáticamente el tipo de par"""
        symbol = symbol.upper().replace('=X', '').replace('-USD', 'USD')
        
        for pair_type, config in cls.PAIR_CONFIGS.items():
            if symbol in config['pairs']:
                return pair_type
        
        # Auto-detection based on symbol pattern
        if 'USD' in symbol and len(symbol) == 6:
            if symbol in ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY']:
                return 'MAJORS'
            else:
                return 'MINORS'
        elif 'XAU' in symbol or 'XAG' in symbol or 'OIL' in symbol:
            return 'COMMODITIES'
        elif 'BTC' in symbol or 'ETH' in symbol or any(crypto in symbol for crypto in ['ADA', 'DOT', 'LTC']):
            return 'CRYPTO'
        else:
            return 'EXOTICS'
    
    @classmethod
    def get_pair_config(cls, symbol):
        """Obtiene configuración específica para el par"""
        pair_type = cls.detect_pair_type(symbol)
        base_config = cls.PAIR_CONFIGS[pair_type].copy()
        base_config['pair_type'] = pair_type
        base_config['symbol'] = symbol
        
        # Ajustes específicos por par
        specific_adjustments = cls._get_specific_adjustments(symbol)
        base_config.update(specific_adjustments)
        
        return base_config
    
    @classmethod
    def _get_specific_adjustments(cls, symbol):
        """Ajustes específicos por par individual"""
        
        specific_configs = {
            'GBPJPY': {'daily_volatility': 180, 'session_sensitivity': 'EXTREME'},
            'XAUUSD': {'daily_volatility': 3000, 'typical_spread': 5},
            'BTCUSD': {'daily_volatility': 8000, 'news_sensitivity': 'EXTREME'},
            'USDJPY': {'daily_volatility': 80, 'pip_value': 0.01},  # JPY pairs
            'EURJPY': {'daily_volatility': 120, 'pip_value': 0.01},
            'GBPJPY': {'daily_volatility': 180, 'pip_value': 0.01},
        }
        
        return specific_configs.get(symbol, {})

@dataclass
class UniversalTradingConfig:
    """Configuraciones de trading adaptables por par"""
    
    @classmethod
    def get_style_config(cls, symbol, style='DAY_TRADING'):
        """Genera configuración de estilo adaptada al par"""
        
        pair_config = UniversalPairConfig.get_pair_config(symbol)
        
        # Base configurations por estilo
        base_configs = {
            'SCALPING': {
                'timeframe': '1Min',
                'sequence_length': 30,
                'prediction_horizon': 1,
                'd_model': 128,
                'num_heads': 4,
                'num_layers': 3,
                'learning_rate': 1e-3,
                'batch_size': 64,
                'min_confidence': 0.8
            },
            'DAY_TRADING': {
                'timeframe': '5Min',
                'sequence_length': 96,
                'prediction_horizon': 6,
                'd_model': 256,
                'num_heads': 8,
                'num_layers': 6,
                'learning_rate': 5e-4,
                'batch_size': 32,
                'min_confidence': 0.7
            },
            'SWING_TRADING': {
                'timeframe': '1H',
                'sequence_length': 168,
                'prediction_horizon': 24,
                'd_model': 512,
                'num_heads': 16,
                'num_layers': 8,
                'learning_rate': 1e-4,
                'batch_size': 16,
                'min_confidence': 0.75
            },
            'POSITION_TRADING': {
                'timeframe': '1D',
                'sequence_length': 252,
                'prediction_horizon': 30,
                'd_model': 512,
                'num_heads': 16,
                'num_layers': 10,
                'learning_rate': 5e-5,
                'batch_size': 8,
                'min_confidence': 0.8
            }
        }
        
        config = base_configs[style].copy()
        
        # Adaptar según características del par
        config.update({
            'symbol': symbol,
            'pair_type': pair_config['pair_type'],
            'pip_value': pair_config['pip_value'],
            'typical_spread': pair_config['typical_spread'],
            'daily_volatility': pair_config['daily_volatility'],
            'leverage_available': pair_config['leverage_available']
        })
        
        # Ajustes específicos por tipo de par
        config.update(cls._adjust_for_pair_type(config, pair_config, style))
        
        # Verificar viabilidad de scalping
        if style == 'SCALPING' and not pair_config['scalping_viable']:
            print(f"⚠️ Scalping no recomendado para {symbol} - spread muy alto")
            config['min_confidence'] = 0.9  # Aumentar umbral
        
        return config
    
    @classmethod
    def _adjust_for_pair_type(cls, config, pair_config, style):
        """Ajusta configuración según tipo de par"""
        
        adjustments = {}
        pair_type = pair_config['pair_type']
        
        if pair_type == 'MAJORS':
            # Configuración estándar, ya optimizada
            pass
            
        elif pair_type == 'MINORS':
            # Spreads más altos, targets más grandes
            adjustments.update({
                'target_pips': config.get('target_pips', 10) * 1.5,
                'stop_loss_pips': config.get('stop_loss_pips', 5) * 1.5,
                'min_confidence': config['min_confidence'] + 0.05
            })
            
        elif pair_type == 'EXOTICS':
            # Alta volatilidad, targets mucho más grandes
            adjustments.update({
                'target_pips': config.get('target_pips', 10) * 5,
                'stop_loss_pips': config.get('stop_loss_pips', 5) * 5,
                'min_confidence': 0.85,  # Muy conservador
                'batch_size': max(config['batch_size'] // 2, 8),  # Menos datos
                'learning_rate': config['learning_rate'] * 0.5  # Más conservador
            })
            
        elif pair_type == 'COMMODITIES':
            # Volatilidad alta, sesiones específicas
            adjustments.update({
                'target_pips': config.get('target_pips', 10) * 3,
                'stop_loss_pips': config.get('stop_loss_pips', 5) * 3,
                'sequence_length': min(config['sequence_length'], 120),  # Menos historia
                'news_weight': 0.3  # Mayor peso a noticias
            })
            
        elif pair_type == 'CRYPTO':
            # Extrema volatilidad, mercado 24/7
            adjustments.update({
                'target_pips': config.get('target_pips', 10) * 10,
                'stop_loss_pips': config.get('stop_loss_pips', 5) * 10,
                'learning_rate': config['learning_rate'] * 2,  # Aprender más rápido
                'sequence_length': config['sequence_length'] // 2,  # Memoria más corta
                'session_features': False  # No hay sesiones tradicionales
            })
        
        # Calcular targets específicos según volatilidad
        daily_vol = pair_config['daily_volatility']
        
        if style == 'SCALPING':
            adjustments['target_pips'] = max(daily_vol * 0.02, pair_config['typical_spread'] * 2)
            adjustments['stop_loss_pips'] = adjustments['target_pips'] * 0.7
            adjustments['trades_per_day'] = min(50, 2000 // daily_vol)  # Menos trades si más volátil
            
        elif style == 'DAY_TRADING':
            adjustments['target_pips'] = daily_vol * 0.15
            adjustments['stop_loss_pips'] = adjustments['target_pips'] * 0.6
            adjustments['trades_per_day'] = min(10, 500 // daily_vol)
            
        elif style == 'SWING_TRADING':
            adjustments['target_pips'] = daily_vol * 0.5
            adjustments['stop_loss_pips'] = adjustments['target_pips'] * 0.5
            adjustments['trades_per_week'] = max(1, min(8, 200 // daily_vol))
            
        elif style == 'POSITION_TRADING':
            adjustments['target_pips'] = daily_vol * 2
            adjustments['stop_loss_pips'] = adjustments['target_pips'] * 0.5
            adjustments['trades_per_month'] = max(1, min(5, 50 // daily_vol))
        
        return adjustments

# ====================================================================
# DATA COLLECTOR UNIVERSAL
# ====================================================================

class UniversalDataCollector:
    """Recolector de datos adaptable a cualquier par"""
    
    def __init__(self, symbol):
        self.symbol = symbol.upper()
        self.pair_config = UniversalPairConfig.get_pair_config(symbol)
        self.scaler = RobustScaler()
        
        # Determinar símbolo para yfinance
        self.yf_symbol = self._get_yfinance_symbol()
        
        print(f"📊 Configurando collector para {self.symbol}")
        print(f"   • Tipo: {self.pair_config['pair_type']}")
        print(f"   • Símbolo YF: {self.yf_symbol}")
        print(f"   • Volatilidad diaria: {self.pair_config['daily_volatility']}")
    
    def _get_yfinance_symbol(self):
        """Mapea símbolo a formato yfinance"""
        
        # Mapeo específico para yfinance
        symbol_map = {
            # Majors
            'EURUSD': 'EURUSD=X',
            'GBPUSD': 'GBPUSD=X', 
            'USDJPY': 'USDJPY=X',
            'USDCHF': 'USDCHF=X',
            'AUDUSD': 'AUDUSD=X',
            'USDCAD': 'USDCAD=X',
            'NZDUSD': 'NZDUSD=X',
            
            # Minors
            'EURGBP': 'EURGBP=X',
            'EURJPY': 'EURJPY=X',
            'GBPJPY': 'GBPJPY=X',
            'EURCHF': 'EURCHF=X',
            'AUDCAD': 'AUDCAD=X',
            'CADJPY': 'CADJPY=X',
            
            # Exotics
            'USDTRY': 'USDTRY=X',
            'USDZAR': 'USDZAR=X',
            'USDMXN': 'USDMXN=X',
            'EURPLN': 'EURPLN=X',
            'GBPTRY': 'GBPTRY=X',
            
            # Commodities
            'XAUUSD': 'GC=F',  # Gold futures
            'XAGUSD': 'SI=F',  # Silver futures
            'USOIL': 'CL=F',   # Crude oil futures
            'BRENT': 'BZ=F',   # Brent oil futures
            
            # Crypto
            'BTCUSD': 'BTC-USD',
            'ETHUSD': 'ETH-USD',
            'ADAUSD': 'ADA-USD',
            'DOTUSD': 'DOT-USD'
        }
        
        return symbol_map.get(self.symbol, f"{self.symbol}=X")
    
    def collect_data(self, period='5y', interval='5m'):
        """Recolecta datos adaptado al tipo de par"""
        
        print(f"📊 Recolectando datos {self.symbol} - Periodo: {period}, Intervalo: {interval}")
        
        # OPTIMIZACIÓN: Priorizar intervalos pequeños y periodos largos
        attempts = [
            ('5y', '1h'), ('2y', '1h'), ('1y', '1h'),
            ('60d', '5m'), ('30d', '5m'), ('7d', '1m'),
            ('5y', '1d'), ('2y', '1d'), ('1y', '1d'),
        ]
        for attempt_period, attempt_interval in attempts:
            try:
                print(f"🔄 Intentando {attempt_period}, {attempt_interval}...")
                ticker = yf.Ticker(self.yf_symbol)
                data = ticker.history(period=attempt_period, interval=attempt_interval)
                if not data.empty and len(data) > 50:
                    print(f"✅ Obtenidos {len(data)} registros de {self.yf_symbol}")
                    data = self._add_universal_features(data)
                    data = self._add_pair_specific_features(data)
                    data = data.dropna()
                    if len(data) > 25:
                        print(f"✅ Datos procesados: {len(data)} registros con {len(data.columns)} features")
                        return data
                    else:
                        print(f"⚠️ Muy pocos datos después de limpieza: {len(data)}")
                        continue
            except Exception as e:
                print(f"⚠️ Error con {attempt_period}, {attempt_interval}: {e}")
                continue
        print("⚠️ No se pudieron obtener datos de yfinance, usando datos sintéticos")
        return self._generate_synthetic_data(period, interval)
    
    def _generate_synthetic_data(self, period='5y', interval='5m'):
        """Genera datos sintéticos realistas para el par"""
        
        print(f"🔄 Generando datos sintéticos para {self.symbol} - {period}, {interval}")
        
        # 🔧 FIX: Generar más datos para compensar pérdidas en feature engineering
        # Determinar número de puntos según intervalo
        freq_map = {'1m': 1, '5m': 5, '15m': 15, '1h': 60, '1d': 1440}
        interval_minutes = freq_map.get(interval, 5)
        
        # Calcular fechas - Generar 3x más datos para compensar pérdidas
        if period.endswith('y'):
            years = int(period[:-1])
            total_minutes = years * 365 * 24 * 60
        elif period.endswith('d'):
            days = int(period[:-1])
            total_minutes = days * 24 * 60
        elif period.endswith('mo'):
            months = int(period[:-2])
            total_minutes = months * 30 * 24 * 60
        else:
            total_minutes = 365 * 24 * 60  # Default 1 year
        
        # 🔧 FIX: Generar 15,000 puntos mínimo para compensar pérdidas en feature engineering
        n_points = max(15000, total_minutes // interval_minutes * 3)  # 3x más datos
        
        dates = pd.date_range(
            start=datetime.now() - timedelta(minutes=total_minutes),
            periods=n_points,
            freq=f'{interval_minutes}min'
        )
        
        # Parámetros según tipo de par - más realistas
        daily_vol = self.pair_config['daily_volatility']
        
        if self.pair_config['pair_type'] == 'MAJORS':
            base_price = 1.10 if 'EUR' in self.symbol else 1.30
            vol_factor = 0.0001
            spread_factor = 0.0001  # Spread típico de majors
        elif self.pair_config['pair_type'] == 'COMMODITIES':
            base_price = 2000 if 'XAU' in self.symbol else 80
            vol_factor = 0.01
            spread_factor = 0.001
        elif self.pair_config['pair_type'] == 'CRYPTO':
            base_price = 50000 if 'BTC' in self.symbol else 3000
            vol_factor = 0.02
            spread_factor = 0.002
        else:
            base_price = 1.20
            vol_factor = 0.0001
            spread_factor = 0.0002
        
        # Generar precios con características más realistas
        # Usar GARCH-like process para volatilidad clustering
        volatility = np.ones(n_points) * daily_vol * vol_factor
        for i in range(1, n_points):
            # Volatilidad clustering (GARCH-like)
            volatility[i] = 0.95 * volatility[i-1] + 0.05 * np.random.normal(0, daily_vol * vol_factor)
            volatility[i] = max(volatility[i], daily_vol * vol_factor * 0.1)  # Mínimo
        
        returns = np.random.normal(0, volatility / np.sqrt(1440/interval_minutes), n_points)
        
        # Agregar tendencias y ciclos más realistas
        trend = np.linspace(0, 0.05, n_points) * np.random.choice([-1, 1])  # Trend más suave
        daily_cycle = 0.0005 * np.sin(2 * np.pi * np.arange(n_points) / (1440/interval_minutes))
        weekly_cycle = 0.0002 * np.sin(2 * np.pi * np.arange(n_points) / (1440/interval_minutes * 7))
        
        # Precio close con características más realistas
        price_changes = returns + trend/n_points + daily_cycle + weekly_cycle
        close_prices = base_price * np.exp(np.cumsum(price_changes))
        
        # OHLV sintético más realista
        data = pd.DataFrame(index=dates)
        data['Close'] = close_prices
        
        # Generar Open, High, Low más realistas
        for i in range(len(data)):
            if i == 0:
                data.loc[data.index[i], 'Open'] = data.loc[data.index[i], 'Close']
            else:
                # Open basado en close anterior + gap
                gap = np.random.normal(0, volatility[i] * 0.5)
                data.loc[data.index[i], 'Open'] = data.loc[data.index[i-1], 'Close'] + gap
            
            close_price = data.loc[data.index[i], 'Close']
            open_price = data.loc[data.index[i], 'Open']
            
            # High y Low basados en Open y Close
            price_range = abs(close_price - open_price) + volatility[i] * np.random.exponential(1)
            if close_price > open_price:  # Bullish candle
                data.loc[data.index[i], 'High'] = close_price + price_range * 0.3
                data.loc[data.index[i], 'Low'] = open_price - price_range * 0.2
            else:  # Bearish candle
                data.loc[data.index[i], 'High'] = open_price + price_range * 0.2
                data.loc[data.index[i], 'Low'] = close_price - price_range * 0.3
        
        # Volume más realista
        base_volume = 1000000 if self.pair_config['pair_type'] == 'MAJORS' else 100000
        data['Volume'] = np.random.lognormal(np.log(base_volume), 0.5, n_points)
        
        # Agregar sesiones de trading (para forex)
        if self.pair_config['pair_type'] in ['MAJORS', 'MINORS', 'EXOTICS']:
            # Reducir volumen en fines de semana
            for i, date in enumerate(data.index):
                if date.weekday() >= 5:  # Weekend
                    data.loc[date, 'Volume'] *= 0.1
        
        # 🔧 FIX: Asegurar que no hay NaN antes de feature engineering
        data = data.fillna(method='ffill').fillna(method='bfill').fillna(0)
        print(f"✅ Generados {len(data)} puntos sintéticos realistas para {self.symbol}")
        
        return data
    
    def _add_universal_features(self, data):
        """Agrega features universales para cualquier par"""
        
        # Technical indicators básicos
        data['rsi'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
        data['macd'] = ta.trend.MACD(data['Close']).macd()
        data['macd_signal'] = ta.trend.MACD(data['Close']).macd_signal()
        data['bb_upper'] = ta.volatility.BollingerBands(data['Close']).bollinger_hband()
        data['bb_lower'] = ta.volatility.BollingerBands(data['Close']).bollinger_lband()
        data['atr'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
        
        # Momentum features
        data['momentum_1'] = data['Close'].pct_change(1)
        data['momentum_5'] = data['Close'].pct_change(5)
        data['momentum_20'] = data['Close'].pct_change(20)
        
        # Volatility features
        data['volatility_5'] = data['Close'].pct_change().rolling(5).std()
        data['volatility_20'] = data['Close'].pct_change().rolling(20).std()
        
        # Volume features (si hay volumen)
        if 'Volume' in data.columns and data['Volume'].sum() > 0:
            data['volume_sma'] = data['Volume'].rolling(20).mean()
            data['volume_ratio'] = data['Volume'] / data['volume_sma']
        else:
            data['volume_ratio'] = 1.0  # Neutral para pares sin volumen
        
        # Support/Resistance levels
        data['resistance'] = data['High'].rolling(20).max()
        data['support'] = data['Low'].rolling(20).min()
        data['resistance_distance'] = (data['resistance'] - data['Close']) / data['Close']
        data['support_distance'] = (data['Close'] - data['support']) / data['Close']
        
        # 🔧 FIX: Validar y limpiar NaN/Inf en todos los features
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if data[col].isnull().any() or np.isinf(data[col]).any():
                print(f"⚠️ NaN/Inf detectado en {col} para {self.symbol}, limpiando...")
                # Reemplazar NaN con forward fill, luego backward fill, luego 0
                data[col] = data[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
                # Reemplazar infinitos con valores seguros
                data[col] = data[col].replace([np.inf, -np.inf], 0)
        
        return data
    
    def _add_pair_specific_features(self, data):
        """Agrega features específicos según el tipo de par"""
        
        pair_type = self.pair_config['pair_type']
        
        if pair_type in ['MAJORS', 'MINORS']:
            data = self._add_forex_features(data)
        elif pair_type == 'COMMODITIES':
            data = self._add_commodity_features(data)
        elif pair_type == 'CRYPTO':
            data = self._add_crypto_features(data)
        else:  # EXOTICS
            data = self._add_exotic_features(data)
        
        return data
    
    def _add_forex_features(self, data):
        """Features específicos para Forex"""
        
        # Session features (solo si no es crypto)
        data['hour'] = data.index.hour
        data['day_of_week'] = data.index.dayofweek
        
        # Trading sessions
        data['asian_session'] = ((data['hour'] >= 0) & (data['hour'] <= 8)).astype(int)
        data['london_session'] = ((data['hour'] >= 8) & (data['hour'] <= 16)).astype(int)
        data['ny_session'] = ((data['hour'] >= 13) & (data['hour'] <= 21)).astype(int)
        data['overlap_london_ny'] = ((data['hour'] >= 13) & (data['hour'] <= 16)).astype(int)
        
        # Pivot points
        data['pivot'] = (data['High'] + data['Low'] + data['Close']) / 3
        data['r1'] = 2 * data['pivot'] - data['Low']
        data['s1'] = 2 * data['pivot'] - data['High']
        data['pivot_distance'] = (data['Close'] - data['pivot']) / data['pivot']
        
        # Currency strength simulation
        np.random.seed(42)
        data['currency_strength'] = np.sin(2 * np.pi * np.arange(len(data)) / 1440) + \
                                   np.random.normal(0, 0.1, len(data))
        
        # 🔧 FIX: Validar y limpiar NaN/Inf en features forex
        forex_features = ['pivot', 'r1', 's1', 'pivot_distance', 'currency_strength']
        for col in forex_features:
            if col in data.columns:
                if data[col].isnull().any() or np.isinf(data[col]).any():
                    print(f"⚠️ NaN/Inf detectado en {col} para {self.symbol}, limpiando...")
                    data[col] = data[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
                    data[col] = data[col].replace([np.inf, -np.inf], 0)
        
        return data
    
    def _add_commodity_features(self, data):
        """Features específicos para commodities"""
        
        # No hay sesiones forex tradicionales, pero sí horarios de mercado
        data['hour'] = data.index.hour
        data['market_hours'] = ((data['hour'] >= 9) & (data['hour'] <= 16)).astype(int)
        
        # Volatility clustering (común en commodities)
        data['vol_cluster'] = data['atr'].rolling(10).mean() / data['atr'].rolling(50).mean()
        
        # Seasonality (importante para commodities)
        data['month'] = data.index.month
        data['seasonal_factor'] = np.sin(2 * np.pi * data['month'] / 12)
        
        # Economic sensitivity
        data['economic_factor'] = np.random.normal(0, 0.05, len(data))  # Simulated
        
        # 🔧 FIX: Validar y limpiar NaN/Inf en features de commodities
        commodity_features = ['vol_cluster', 'seasonal_factor', 'economic_factor']
        for col in commodity_features:
            if col in data.columns:
                if data[col].isnull().any() or np.isinf(data[col]).any():
                    print(f"⚠️ NaN/Inf detectado en {col} para {self.symbol}, limpiando...")
                    data[col] = data[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
                    data[col] = data[col].replace([np.inf, -np.inf], 0)
        
        return data
    
    def _add_crypto_features(self, data):
        """Features específicos para crypto"""
        
        # No hay sesiones (24/7), pero sí patrones de actividad
        data['hour'] = data.index.hour
        data['weekend'] = (data.index.dayofweek >= 5).astype(int)
        
        # Momentum ROC (Rate of Change)
        data['momentum_roc'] = data['Close'].pct_change(10)
        
        # Volatility regime detection
        data['vol_regime'] = data['volatility_20'] / data['volatility_20'].rolling(100).mean()
        
        # Social sentiment simulation (importante para crypto)
        np.random.seed(42)
        data['social_sentiment'] = np.sin(2 * np.pi * np.arange(len(data)) / 288) + \
                                  np.random.normal(0, 0.2, len(data))
        
        # 🔧 FIX: Validar y limpiar NaN/Inf en features de crypto
        crypto_features = ['momentum_roc', 'vol_regime', 'social_sentiment']
        for col in crypto_features:
            if col in data.columns:
                if data[col].isnull().any() or np.isinf(data[col]).any():
                    print(f"⚠️ NaN/Inf detectado en {col} para {self.symbol}, limpiando...")
                    data[col] = data[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
                    data[col] = data[col].replace([np.inf, -np.inf], 0)
        
        return data
    
    def _add_exotic_features(self, data):
        """Features específicos para exóticos"""
        
        # Sessions importantes pero menos definidas
        data['hour'] = data.index.hour
        data['local_session'] = ((data['hour'] >= 8) & (data['hour'] <= 18)).astype(int)
        
        # High impact news events (más frecuentes en exóticos)
        data['news_impact'] = np.random.exponential(0.1, len(data))  # Occasional spikes
        
        # Political risk factor
        data['political_risk'] = np.random.gamma(1, 0.05, len(data))
        
        # Carry trade factor (importante para exóticos)
        data['carry_factor'] = np.random.normal(0.02, 0.01, len(data))
        
        # 🔧 FIX: Validar y limpiar NaN/Inf en features de exóticos
        exotic_features = ['news_impact', 'political_risk', 'carry_factor']
        for col in exotic_features:
            if col in data.columns:
                if data[col].isnull().any() or np.isinf(data[col]).any():
                    print(f"⚠️ NaN/Inf detectado en {col} para {self.symbol}, limpiando...")
                    data[col] = data[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
                    data[col] = data[col].replace([np.inf, -np.inf], 0)
        
        return data
    
    def prepare_features(self, data, style='DAY_TRADING'):
        """Prepara features específicos para el estilo de trading"""
        
        config = UniversalTradingConfig.get_style_config(self.symbol, style)
        
        # Seleccionar features según el estilo y tipo de par
        feature_cols = self._select_features_for_style(data, style)
        
        # Filtrar features disponibles
        available_features = [col for col in feature_cols if col in data.columns]
        feature_data = data[available_features].copy()
        
        # 🔧 FIX: Validar que no hay NaN o Inf en los datos originales
        if feature_data.isnull().any().any():
            print(f"⚠️ NaN detectado en datos originales para {self.symbol}, limpiando...")
            feature_data = feature_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # 🔧 FIX: Reemplazar valores infinitos
        feature_data = feature_data.replace([np.inf, -np.inf], 0)
        
        # 🔧 FIX: Validar que los datos son numéricos
        for col in feature_data.columns:
            if feature_data[col].dtype == 'object':
                feature_data[col] = pd.to_numeric(feature_data[col], errors='coerce').fillna(0)
        
        # Normalizar features con validación robusta
        try:
            scaled_features = self.scaler.fit_transform(feature_data)
            
            # 🔧 FIX: Validar que la normalización no produjo NaN o Inf
            if np.isnan(scaled_features).any() or np.isinf(scaled_features).any():
                print(f"⚠️ NaN/Inf después de normalización para {self.symbol}, usando StandardScaler...")
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
                scaled_features = self.scaler.fit_transform(feature_data)
                
                # 🔧 FIX: Validación final después de StandardScaler
                if np.isnan(scaled_features).any() or np.isinf(scaled_features).any():
                    print(f"⚠️ NaN/Inf persistente para {self.symbol}, usando valores de fallback...")
                    scaled_features = np.nan_to_num(scaled_features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            feature_data = pd.DataFrame(scaled_features, columns=available_features, index=data.index)
            
        except Exception as e:
            print(f"❌ Error en normalización para {self.symbol}: {e}")
            # Fallback: usar datos sin escalar pero limpiados
            feature_data = feature_data.fillna(0).replace([np.inf, -np.inf], 0)
        
        # 🔧 FIX: Asegurar mínimo de datos después de procesamiento
        if len(feature_data) < 100:  # Mínimo absoluto para entrenamiento
            print(f"⚠️ Muy pocos datos después de procesamiento: {len(feature_data)}")
            print("🔄 Regenerando datos sintéticos con más puntos...")
            # Regenerar con más datos
            synthetic_data = self._generate_synthetic_data(period='10y', interval='5m')
            # Aplicar feature engineering a los datos sintéticos
            synthetic_data = self._add_universal_features(synthetic_data)
            synthetic_data = self._add_pair_specific_features(synthetic_data)
            # Seleccionar features y procesar
            synthetic_features = self._select_features_for_style(synthetic_data, style)
            available_synthetic = [col for col in synthetic_features if col in synthetic_data.columns]
            feature_data = synthetic_data[available_synthetic].copy()
            # Aplicar limpieza final
            feature_data = feature_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
            feature_data = feature_data.replace([np.inf, -np.inf], 0)
            # Normalizar
            try:
                scaled_features = self.scaler.fit_transform(feature_data)
                if np.isnan(scaled_features).any() or np.isinf(scaled_features).any():
                    scaled_features = np.nan_to_num(scaled_features, nan=0.0, posinf=1.0, neginf=-1.0)
                feature_data = pd.DataFrame(scaled_features, columns=available_synthetic, index=synthetic_data.index)
            except:
                feature_data = feature_data.fillna(0).replace([np.inf, -np.inf], 0)
        
        print(f"✅ Features preparados: {len(feature_data.columns)} features para {style}")
        return feature_data
    
    def _select_features_for_style(self, data, style):
        """Selecciona features óptimos según estilo y tipo de par"""
        
        pair_type = self.pair_config['pair_type']
        
        # Features base siempre incluidos
        base_features = ['Close', 'High', 'Low', 'rsi', 'macd', 'atr', 
                        'momentum_1', 'momentum_5', 'volatility_5']
        
        # Features por estilo
        if style == 'SCALPING':
            style_features = ['volume_ratio', 'resistance_distance', 'support_distance']
            if pair_type in ['MAJORS', 'MINORS']:
                style_features.extend(['asian_session', 'london_session', 'ny_session', 'pivot_distance'])
        
        elif style == 'DAY_TRADING':
            style_features = ['bb_upper', 'bb_lower', 'momentum_20', 'macd_signal']
            if pair_type in ['MAJORS', 'MINORS']:
                style_features.extend(['pivot', 'overlap_london_ny', 'currency_strength'])
            elif pair_type == 'COMMODITIES':
                style_features.extend(['seasonal_factor', 'vol_cluster', 'market_hours'])
            elif pair_type == 'CRYPTO':
                style_features.extend(['momentum_roc', 'vol_regime', 'social_sentiment'])
        
        elif style == 'SWING_TRADING':
            style_features = ['volatility_20', 'resistance', 'support']
            if pair_type in ['MAJORS', 'MINORS']:
                style_features.extend(['r1', 's1', 'currency_strength'])
            elif pair_type == 'COMMODITIES':
                style_features.extend(['seasonal_factor', 'economic_factor'])
            elif pair_type == 'CRYPTO':
                style_features.extend(['momentum_strength', 'weekend'])
            elif pair_type == 'EXOTICS':
                style_features.extend(['political_risk', 'carry_factor'])
        
        else:  # POSITION_TRADING
            style_features = ['momentum_20', 'volatility_20']
            if pair_type in ['MAJORS', 'MINORS']:
                style_features.extend(['currency_strength'])
            elif pair_type == 'COMMODITIES':
                style_features.extend(['seasonal_factor', 'economic_factor'])
            elif pair_type == 'CRYPTO':
                style_features.extend(['social_sentiment', 'vol_regime'])
            elif pair_type == 'EXOTICS':
                style_features.extend(['political_risk', 'carry_factor', 'news_impact'])
        
        return base_features + style_features

# ====================================================================
# ARQUITECTURAS DE REDES NEURONALES (REUTILIZADAS)
# ====================================================================

# [Copiamos las clases de redes neuronales del script anterior]
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TradingTransformer(nn.Module):
    def __init__(self, input_dim, d_model=256, num_heads=8, num_layers=6, num_classes=3):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )
        self.volatility_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1)
        )
    
    def forward(self, x):
        # x shape: [batch_size, sequence_length, input_dim]
        batch_size, seq_len, input_dim = x.shape
        
        # 🔧 FIX: Validar dimensiones antes de procesar
        if input_dim != self.input_projection.in_features:
            raise ValueError(f"Input dimension mismatch: expected {self.input_projection.in_features}, got {input_dim}")
        
        # Project to d_model
        x = self.input_projection(x)
        
        # Add positional encoding with proper dimension handling
        x = self.pos_encoding(x)
        
        # Transformer encoding
        transformer_output = self.transformer(x)
        
        # Use last timestep
        last_hidden = transformer_output[:, -1, :]
        
        # Multiple predictions
        direction = self.direction_head(last_hidden)
        confidence = self.confidence_head(last_hidden)
        volatility = self.volatility_head(last_hidden)
        
        return {
            'direction': direction,
            'confidence': confidence,
            'volatility': volatility
        }

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=64, d_conv=4, expand=2):
        super().__init__()
        d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv//2, groups=d_inner)
        self.x_proj = nn.Linear(d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(d_inner, d_state, bias=True)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.activation = nn.SiLU()
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # 🔧 FIX: Validar dimensiones antes de procesar
        if seq_len < 4:  # Mínimo para convolución
            print(f"⚠️ Secuencia muy corta para Mamba: {seq_len}")
            # Rellenar con ceros si es necesario
            if seq_len < 4:
                padding = torch.zeros(batch_size, 4 - seq_len, d_model, device=x.device)
                x = torch.cat([padding, x], dim=1)
                seq_len = 4
        
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)
        
        # 🔧 FIX: Ajustar padding de convolución para evitar dimension mismatch
        # Usar 'same' padding para mantener la longitud de secuencia
        x_inner_transposed = x_inner.transpose(-1, -2)  # (batch, d_inner, seq_len)
        
        # Calcular padding necesario para mantener la longitud
        kernel_size = self.conv1d.kernel_size[0]
        padding_needed = kernel_size // 2
        
        # Aplicar padding manual
        if padding_needed > 0:
            x_inner_padded = torch.nn.functional.pad(x_inner_transposed, (padding_needed, padding_needed), mode='replicate')
        else:
            x_inner_padded = x_inner_transposed
        
        x_conv = self.conv1d(x_inner_padded)
        x_conv = x_conv.transpose(-1, -2)  # Volver a (batch, seq_len, d_inner)
        
        # 🔧 FIX: Asegurar que la salida tiene la longitud correcta
        if x_conv.shape[1] != seq_len:
            if x_conv.shape[1] > seq_len:
                x_conv = x_conv[:, :seq_len, :]
            else:
                padding = torch.zeros(batch_size, seq_len - x_conv.shape[1], x_conv.shape[2], device=x_conv.device)
                x_conv = torch.cat([x_conv, padding], dim=1)
        
        x_conv = self.activation(x_conv)
        output = x_conv  # Simplified
        output = output * self.activation(z)
        output = self.out_proj(output)
        
        return output

class TradingMamba(nn.Module):
    def __init__(self, input_dim, d_model=512, num_layers=8, num_classes=3):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.mamba_layers = nn.ModuleList([MambaBlock(d_model) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        self.regime_head = nn.Sequential(
            nn.Linear(d_model, 128), nn.ReLU(),
            nn.Linear(128, 4)  # TRENDING_UP, TRENDING_DOWN, RANGING, HIGH_VOL
        )
    
    def forward(self, x):
        # x shape: [batch_size, sequence_length, input_dim]
        batch_size, seq_len, input_dim = x.shape
        
        # 🔧 FIX: Validar dimensiones antes de procesar
        if input_dim != self.input_projection.in_features:
            raise ValueError(f"Input dimension mismatch: expected {self.input_projection.in_features}, got {input_dim}")
        
        x = self.input_projection(x)
        for mamba_layer in self.mamba_layers:
            x = x + mamba_layer(x)
        x = self.norm(x)
        last_hidden = x[:, -1, :]
        
        return {
            'direction': self.direction_head(last_hidden),
            'regime': self.regime_head(last_hidden)
        }

# ====================================================================
# AMBIENTE DE TRADING UNIVERSAL
# ====================================================================

class UniversalTradingEnvironment(gym.Env):
    """Ambiente de trading adaptable a cualquier par"""
    
    def __init__(self, data, symbol, style, initial_balance=100000):
        super().__init__()
        
        self.data = data
        self.symbol = symbol
        self.style = style
        self.config = UniversalTradingConfig.get_style_config(symbol, style)
        self.pair_config = UniversalPairConfig.get_pair_config(symbol)
        self.initial_balance = initial_balance
        
        # Adaptar espacios según el tipo de par
        self.action_space = spaces.Box(
            low=np.array([-1, 0, 0]),  # [direction, size, hold_time]
            high=np.array([1, 1, 1]),
            dtype=np.float32
        )
        
        # 🔧 FIX: Correct obs_size calculation to match actual observation
        # The actual observation uses market_features[-200:] (200) + portfolio_state (15) = 215
        obs_size = 200 + 15  # Fixed size: 200 market features + 15 portfolio state
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        
        # Configurar costos de transacción específicos
        self.spread = self.pair_config['typical_spread'] * self.pair_config['pip_value']
        self.commission = self._calculate_commission()
        
        print(f"🏛️ Ambiente configurado para {symbol} ({self.pair_config['pair_type']})")
        print(f"   • Spread: {self.pair_config['typical_spread']} pips")
        print(f"   • Comisión: ${self.commission:.2f} por lote")
        
        self.reset()
    
    def _calculate_commission(self):
        """Calcula comisión específica por tipo de par"""
        
        pair_type = self.pair_config['pair_type']
        
        commission_rates = {
            'MAJORS': 5.0,      # $5 per standard lot
            'MINORS': 8.0,      # $8 per standard lot
            'EXOTICS': 15.0,    # $15 per standard lot
            'COMMODITIES': 10.0, # $10 per lot
            'CRYPTO': 0.1       # 0.1% of notional
        }
        
        return commission_rates.get(pair_type, 5.0)
    
    def reset(self):
        self.current_step = self.config['sequence_length']
        self.balance = self.initial_balance
        self.position = 0
        self.position_size = 0
        self.entry_price = 0
        self.unrealized_pnl = 0
        self.trade_history = []
        self.max_drawdown = 0
        self.peak_balance = self.initial_balance
        self.consecutive_losses = 0
        
        return self._get_observation()
    
    def _get_observation(self):
        """Obtiene observación adaptada al par"""
        
        # Market features
        market_data = self.data.iloc[self.current_step - self.config['sequence_length']:self.current_step]
        market_features = market_data.values.flatten()
        
        # 🔧 FIX: Validar que market_features no contenga NaN o Inf
        if np.isnan(market_features).any() or np.isinf(market_features).any():
            print(f"⚠️ NaN/Inf en market_features para {self.symbol}, usando valores de fallback")
            market_features = np.nan_to_num(market_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Portfolio state
        current_price = self.data.iloc[self.current_step]['Close']
        if self.position != 0:
            price_diff = (current_price - self.entry_price) / self.entry_price
            self.unrealized_pnl = price_diff * self.position * self.position_size
        else:
            self.unrealized_pnl = 0
        
        # Adaptar portfolio state según tipo de par
        portfolio_state = np.array([
            self.balance / self.initial_balance,
            self.position,
            self.position_size / self.initial_balance,
            self.unrealized_pnl / self.initial_balance,
            (self.balance + self.unrealized_pnl) / self.peak_balance - 1,
            len(self.trade_history) / 1000,
            current_price / current_price if current_price != 0 else 1,  # Normalized
            self.entry_price / current_price if self.entry_price != 0 else 1,
            self._get_win_rate(),
            self._get_sharpe_ratio(),
            self.consecutive_losses / 10,  # Risk management
            self._get_volatility_regime(),
            self._get_time_factor(),
            self._get_spread_factor(),
            self._get_pair_specific_factor()
        ])
        
        # 🔧 FIX: Validar que portfolio_state no contenga NaN o Inf
        if np.isnan(portfolio_state).any() or np.isinf(portfolio_state).any():
            print(f"⚠️ NaN/Inf en portfolio_state para {self.symbol}, usando valores de fallback")
            portfolio_state = np.nan_to_num(portfolio_state, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Combine observations
        observation = np.concatenate([
            market_features[-200:],  # Last 200 market features
            portfolio_state
        ])
        
        # 🔧 FIX: Validación final del observation completo
        if np.isnan(observation).any() or np.isinf(observation).any():
            print(f"⚠️ NaN/Inf en observation final para {self.symbol}, limpiando...")
            observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return observation.astype(np.float32)
    
    def _get_volatility_regime(self):
        """Detecta régimen de volatilidad"""
        try:
            recent_vol = self.data.iloc[max(0, self.current_step-20):self.current_step]['Close'].pct_change().std()
            historical_vol = self.data.iloc[max(0, self.current_step-100):self.current_step]['Close'].pct_change().std()
            
            # 🔧 FIX: Validar que las volatilidades no sean NaN o Inf
            if np.isnan(recent_vol) or np.isinf(recent_vol):
                recent_vol = 0.01  # Valor de fallback
            if np.isnan(historical_vol) or np.isinf(historical_vol):
                historical_vol = 0.01  # Valor de fallback
            
            if historical_vol == 0:
                return 1.0
            
            vol_ratio = recent_vol / historical_vol
            
            # 🔧 FIX: Validar que el ratio no sea NaN o Inf
            if np.isnan(vol_ratio) or np.isinf(vol_ratio):
                return 1.0  # Valor de fallback
            
            return min(vol_ratio, 3.0)  # Cap at 3x
            
        except Exception as e:
            print(f"⚠️ Error en _get_volatility_regime: {e}")
            return 1.0  # Valor de fallback
    
    def _get_time_factor(self):
        """Factor temporal específico del par"""
        
        if self.pair_config['pair_type'] == 'CRYPTO':
            return 1.0  # 24/7 market
        
        # For forex/commodities, consider session timing
        current_hour = self.data.index[self.current_step].hour
        
        if self.pair_config['pair_type'] in ['MAJORS', 'MINORS']:
            # Peak forex hours
            if 8 <= current_hour <= 16 or 13 <= current_hour <= 21:
                return 1.0  # Peak hours
            else:
                return 0.5  # Off hours
        
        return 0.8  # Default for other types
    
    def _get_spread_factor(self):
        """Factor de spread actual vs típico"""
        
        # Simulated current spread (would be real-time in production)
        volatility_factor = self._get_volatility_regime()
        current_spread = self.pair_config['typical_spread'] * volatility_factor
        
        return min(current_spread / self.pair_config['typical_spread'], 3.0)
    
    def _get_pair_specific_factor(self):
        """Factor específico del tipo de par"""
        
        pair_type = self.pair_config['pair_type']
        
        if pair_type == 'MAJORS':
            return 1.0  # Baseline
        elif pair_type == 'MINORS':
            return 0.9  # Slightly less liquid
        elif pair_type == 'EXOTICS':
            return 0.6  # Much less liquid
        elif pair_type == 'COMMODITIES':
            return 0.8  # Session dependent
        elif pair_type == 'CRYPTO':
            return 1.2  # High volatility opportunity
        
        return 1.0
    
    def step(self, action):
        direction = action[0]
        size = abs(action[1])
        
        current_price = self.data.iloc[self.current_step]['Close']
        
        # Calculate reward
        reward = self._calculate_reward(direction, size, current_price)
        
        # Execute trade logic
        self._execute_trade(direction, size, current_price)
        
        self.current_step += 1
        
        # Check if episode is done
        done = (self.current_step >= len(self.data) - 1) or \
               (self.balance <= self.initial_balance * 0.3) or \
               (self.consecutive_losses >= 10)  # Risk management
        
        info = {
            'balance': self.balance,
            'position': self.position,
            'unrealized_pnl': self.unrealized_pnl,
            'num_trades': len(self.trade_history),
            'win_rate': self._get_win_rate(),
            'sharpe_ratio': self._get_sharpe_ratio(),
            'pair_type': self.pair_config['pair_type'],
            'spread_factor': self._get_spread_factor()
        }
        
        return self._get_observation(), reward, done, info
    
    def _execute_trade(self, direction, size, current_price):
        """Ejecuta trade con costos específicos del par"""
        
        if direction > 0.1:  # BUY
            if self.position <= 0:
                self._close_position(current_price)
                self._open_position(1, size, current_price)
        elif direction < -0.1:  # SELL
            if self.position >= 0:
                self._close_position(current_price)
                self._open_position(-1, size, current_price)
    
    def _open_position(self, direction, size, price):
        """Abre posición con costos específicos"""
        
        # Calculate position size based on pair type and risk
        max_position_value = self.balance * 0.1  # 10% max risk per trade
        
        # Adjust for leverage
        available_leverage = self.pair_config['leverage_available']
        leveraged_size = min(size * max_position_value * available_leverage, 
                           self.balance * available_leverage * 0.3)  # Max 30% of leveraged capital
        
        # Apply spread cost
        spread_cost = leveraged_size * self.spread / price
        commission_cost = self.commission * (leveraged_size / 100000)  # Per standard lot
        
        total_cost = spread_cost + commission_cost
        
        if total_cost < self.balance * 0.02:  # Max 2% cost per trade
            self.position = direction
            self.position_size = leveraged_size
            self.entry_price = price
            self.balance -= total_cost
    
    def _close_position(self, price):
        """Cierra posición con costos de salida"""
        
        if self.position != 0:
            # Calculate P&L
            price_diff = (price - self.entry_price) / self.entry_price
            gross_pnl = price_diff * self.position * self.position_size
            
            # Exit costs
            spread_cost = self.position_size * self.spread / price
            commission_cost = self.commission * (self.position_size / 100000)
            
            net_pnl = gross_pnl - spread_cost - commission_cost
            self.balance += net_pnl
            
            # Track consecutive losses
            if net_pnl < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
            
            self.trade_history.append({
                'entry_price': self.entry_price,
                'exit_price': price,
                'position': self.position,
                'size': self.position_size,
                'gross_pnl': gross_pnl,
                'net_pnl': net_pnl,
                'step': self.current_step,
                'costs': spread_cost + commission_cost
            })
            
            # Update peak and drawdown
            if self.balance > self.peak_balance:
                self.peak_balance = self.balance
            
            current_drawdown = (self.peak_balance - self.balance) / self.peak_balance
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        self.position = 0
        self.position_size = 0
        self.entry_price = 0
    
    def _calculate_reward(self, direction, size, current_price):
        """Calcula reward adaptado al tipo de par"""
        
        # Base reward: P&L change
        prev_unrealized = self.unrealized_pnl
        if self.position != 0:
            price_diff = (current_price - self.entry_price) / self.entry_price
            new_unrealized = price_diff * self.position * self.position_size
        else:
            new_unrealized = 0
        
        pnl_change = new_unrealized - prev_unrealized
        pnl_reward = pnl_change / self.initial_balance * 100
        
        # Risk penalties adaptadas al par
        risk_penalty = 0
        max_dd_threshold = {
            'MAJORS': 0.08,      # 8% max DD
            'MINORS': 0.10,      # 10% max DD
            'EXOTICS': 0.15,     # 15% max DD (más volátil)
            'COMMODITIES': 0.12, # 12% max DD
            'CRYPTO': 0.20       # 20% max DD (muy volátil)
        }
        
        threshold = max_dd_threshold.get(self.pair_config['pair_type'], 0.10)
        if self.max_drawdown > threshold:
            risk_penalty = -(self.max_drawdown - threshold) * 50
        
        # Consecutive loss penalty
        if self.consecutive_losses >= 5:
            risk_penalty -= self.consecutive_losses * 2
        
        # Pair-specific bonuses
        pair_bonus = 0
        if self.pair_config['pair_type'] == 'MAJORS' and len(self.trade_history) >= 10:
            # Bonus for consistency in majors
            win_rate = self._get_win_rate()
            if win_rate > 0.6:
                pair_bonus = (win_rate - 0.6) * 10
        
        elif self.pair_config['pair_type'] == 'CRYPTO' and pnl_change > 0:
            # Bonus for capturing crypto volatility
            vol_regime = self._get_volatility_regime()
            if vol_regime > 1.5:
                pair_bonus = min(vol_regime, 3.0) * 2
        
        total_reward = pnl_reward + risk_penalty + pair_bonus
        return total_reward
    
    def _get_win_rate(self):
        """Calcula win rate de trades cerrados"""
        try:
            if len(self.trade_history) == 0:
                return 0.5
            
            # 🔧 FIX: Usar 'net_pnl' en lugar de 'pnl' para consistencia con _close_position
            winning_trades = sum(1 for trade in self.trade_history if trade['net_pnl'] > 0)
            win_rate = winning_trades / len(self.trade_history)
            
            # 🔧 FIX: Validar que win_rate no sea NaN o Inf
            if np.isnan(win_rate) or np.isinf(win_rate):
                return 0.5  # Valor de fallback
            
            return win_rate
            
        except Exception as e:
            print(f"⚠️ Error en _get_win_rate: {e}")
            return 0.5  # Valor de fallback
    
    def _get_sharpe_ratio(self):
        """Calcula Sharpe ratio simplificado"""
        try:
            if len(self.trade_history) < 2:
                return 0.0
            
            # 🔧 FIX: Usar 'net_pnl' en lugar de 'pnl' para consistencia con _close_position
            pnls = [trade['net_pnl'] for trade in self.trade_history]
            mean_pnl = np.mean(pnls)
            std_pnl = np.std(pnls)
            
            # 🔧 FIX: Validar que los cálculos no sean NaN o Inf
            if np.isnan(mean_pnl) or np.isinf(mean_pnl):
                mean_pnl = 0.0
            if np.isnan(std_pnl) or np.isinf(std_pnl):
                std_pnl = 1.0
            
            if std_pnl == 0:
                return 0.0
            
            sharpe = mean_pnl / std_pnl
            
            # 🔧 FIX: Validar que Sharpe no sea NaN o Inf
            if np.isnan(sharpe) or np.isinf(sharpe):
                return 0.0  # Valor de fallback
            
            return sharpe
            
        except Exception as e:
            print(f"⚠️ Error en _get_sharpe_ratio: {e}")
            return 0.0  # Valor de fallback

# ====================================================================
# PPO AGENT (REUTILIZADO DEL SCRIPT ANTERIOR)
# ====================================================================

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh()
        ).to(self.device)
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        ).to(self.device)
        
        # 🔧 FIX: Inicialización de pesos más estable
        self._initialize_weights()
        
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )
        self.memory = []
    
    def _initialize_weights(self):
        """Inicializa los pesos de forma más estable"""
        for module in self.actor.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)  # Gain más pequeño
                nn.init.zeros_(module.bias)
        
        for module in self.critic.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)  # Gain más pequeño
                nn.init.zeros_(module.bias)
    
    def get_action(self, state):
        # 🔧 FIX: Validar dimensión del estado antes de procesar
        if len(state) != self.actor[0].in_features:
            print(f"⚠️ Dimensión del estado ({len(state)}) no coincide con PPO ({self.actor[0].in_features})")
            # Ajustar dimensión del estado
            if len(state) > self.actor[0].in_features:
                state = state[:self.actor[0].in_features]
            else:
                # Rellenar con ceros si es muy corto
                padding = np.zeros(self.actor[0].in_features - len(state))
                state = np.concatenate([state, padding])
        
        # 🔧 FIX: Validar que el estado no contenga NaN o Inf
        if np.isnan(state).any() or np.isinf(state).any():
            print("⚠️ Estado contiene NaN o Inf, usando estado de fallback")
            state = np.zeros(len(state))
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        try:
            with torch.no_grad():
                action_mean = self.actor(state)
                
                # 🔧 FIX: Validar que action_mean no contenga NaN o Inf
                if torch.isnan(action_mean).any() or torch.isinf(action_mean).any():
                    print("⚠️ Actor produjo NaN o Inf en get_action, usando valores de fallback")
                    action_mean = torch.zeros_like(action_mean)
                
                # 🔧 FIX: Clamp action_mean para evitar valores extremos
                action_mean = torch.clamp(action_mean, -10, 10)
                
                action_std = 0.1
                action = torch.normal(action_mean, action_std)
                action = torch.clamp(action, -1, 1)
                
                value = self.critic(state)
                
                # 🔧 FIX: Validar que value no sea NaN o Inf
                if torch.isnan(value) or torch.isinf(value):
                    print("⚠️ Critic produjo NaN o Inf, usando valor de fallback")
                    value = torch.tensor(0.0).to(self.device)
                
                return action.cpu().numpy()[0], value.item()
                
        except Exception as e:
            print(f"⚠️ Error en get_action: {e}, usando valores de fallback")
            # Retornar valores de fallback seguros
            fallback_action = np.zeros(self.actor[-1].out_features)
            return fallback_action, 0.0
    
    def store_transition(self, state, action, reward, next_state, done, log_prob, value):
        # 🔧 FIX: Validar que los datos no contengan NaN o Inf antes de almacenar
        if (np.isnan(state).any() or np.isinf(state).any() or 
            np.isnan(action).any() or np.isinf(action).any() or
            np.isnan(reward) or np.isinf(reward) or
            np.isnan(log_prob) or np.isinf(log_prob) or
            np.isnan(value) or np.isinf(value)):
            print("⚠️ Datos inválidos detectados, saltando almacenamiento")
            return
        
        self.memory.append({
            'state': state, 'action': action, 'reward': reward,
            'next_state': next_state, 'done': done, 'log_prob': log_prob, 'value': value
        })
    
    def update(self, gamma=0.99, lambda_=0.95, clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01):
        if len(self.memory) < 64:
            return
        
        states = torch.FloatTensor([m['state'] for m in self.memory]).to(self.device)
        actions = torch.FloatTensor([m['action'] for m in self.memory]).to(self.device)
        rewards = torch.FloatTensor([m['reward'] for m in self.memory]).to(self.device)
        dones = torch.BoolTensor([m['done'] for m in self.memory]).to(self.device)
        old_log_probs = torch.FloatTensor([m['log_prob'] for m in self.memory]).to(self.device)
        old_values = torch.FloatTensor([m['value'] for m in self.memory]).to(self.device)
        
        # 🔧 FIX: Validar que los datos no contengan NaN o Inf
        if torch.isnan(states).any() or torch.isinf(states).any():
            print("⚠️ Estados contienen NaN o Inf, saltando actualización")
            self.memory.clear()
            return
        
        advantages = self._calculate_advantages(rewards, old_values, dones, gamma, lambda_)
        returns = advantages + old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(10):
            try:
                action_mean = self.actor(states)
                
                # 🔧 FIX: Validar que action_mean no contenga NaN o Inf
                if torch.isnan(action_mean).any() or torch.isinf(action_mean).any():
                    print("⚠️ Actor produjo NaN o Inf, reinicializando pesos")
                    self._reinitialize_networks()
                    break
                
                # 🔧 FIX: Clamp action_mean para evitar valores extremos
                action_mean = torch.clamp(action_mean, -10, 10)
                
                action_std = 0.1
                dist = torch.distributions.Normal(action_mean, action_std)
                new_log_probs = dist.log_prob(actions).sum(axis=-1)
                
                # 🔧 FIX: Validar log_probs
                if torch.isnan(new_log_probs).any() or torch.isinf(new_log_probs).any():
                    print("⚠️ Log probs contienen NaN o Inf, saltando iteración")
                    continue
                
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                current_values = self.critic(states).squeeze()
                value_loss = nn.MSELoss()(current_values, returns)
                entropy_loss = -dist.entropy().mean()
                
                total_loss = actor_loss + value_coef * value_loss + entropy_coef * entropy_loss
                
                # 🔧 FIX: Validar que la pérdida no sea NaN o Inf
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print("⚠️ Pérdida total es NaN o Inf, saltando iteración")
                    continue
                
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # 🔧 FIX: Clamp gradients más agresivamente
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()), 0.1
                )
                
                self.optimizer.step()
                
            except Exception as e:
                print(f"⚠️ Error en iteración PPO: {e}")
                break
        
        self.memory.clear()
    
    def _reinitialize_networks(self):
        """Reinicializa las redes si producen valores inválidos"""
        print("🔄 Reinicializando redes PPO...")
        
        # Reinicializar actor
        for layer in self.actor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        # Reinicializar critic
        for layer in self.critic:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        print("✅ Redes PPO reinicializadas")
    
    def _calculate_advantages(self, rewards, values, dones, gamma, lambda_):
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        # 🔧 FIX: Convert boolean tensor to float to avoid subtraction error
        dones_float = dones.float()
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * (1 - dones_float[t]) - values[t]
            gae = delta + gamma * lambda_ * (1 - dones_float[t]) * gae
            advantages[t] = gae
        
        return advantages

# ====================================================================
# UNIVERSAL TRADING BRAIN
# ====================================================================

class UniversalTradingBrain:
    """Sistema universal adaptable a cualquier par"""
    
    def __init__(self, symbol, style='DAY_TRADING'):
        self.symbol = symbol.upper()
        self.style = style
        self.config = UniversalTradingConfig.get_style_config(symbol, style)
        self.pair_config = UniversalPairConfig.get_pair_config(symbol)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🧠 Inicializando Universal Trading Brain")
        print(f"   • Par: {self.symbol} ({self.pair_config['pair_type']})")
        print(f"   • Estilo: {style}")
        print(f"   • Dispositivo: {self.device}")
        print(f"   • Target pips: {self.config.get('target_pips', 'N/A')}")
        print(f"   • Spread típico: {self.config['typical_spread']} pips")
        
        # Components
        self.data_collector = UniversalDataCollector(symbol)
        self.models = {}
        self.ppo_agent = None
        self.performance_tracker = {
            'training_losses': [],
            'validation_metrics': [],
            'trading_performance': [],
            'model_updates': 0,
            'pair_info': self.pair_config
        }
        
        self._setup_logging()
    
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f'universal_brain_{self.symbol}_{self.style.lower()}.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_complete_training(self, supervised_epochs=None, rl_episodes=None):
        """Ejecuta entrenamiento completo adaptado al par"""
        
        # Auto-configure epochs based on pair type
        if supervised_epochs is None:
            epoch_config = {
                'MAJORS': 80,
                'MINORS': 60,
                'EXOTICS': 40,      # Less data available
                'COMMODITIES': 70,
                'CRYPTO': 100       # More volatile, needs more training
            }
            supervised_epochs = epoch_config.get(self.pair_config['pair_type'], 60)
        
        if rl_episodes is None:
            episode_config = {
                'MAJORS': 800,
                'MINORS': 600,
                'EXOTICS': 400,     # Less liquid
                'COMMODITIES': 700,
                'CRYPTO': 1000      # High volatility needs more episodes
            }
            rl_episodes = episode_config.get(self.pair_config['pair_type'], 600)
        
        print(f"🚀 ENTRENAMIENTO UNIVERSAL - {self.symbol}")
        print(f"📊 Épocas supervisadas: {supervised_epochs}")
        print(f"🤖 Episodios RL: {rl_episodes}")
        print("=" * 60)
        
        start_time = datetime.now()
        
        try:
            # Phase 1: Data Collection
            print(f"\n📊 FASE 1: RECOLECCIÓN DE DATOS - {self.symbol}")
            print("-" * 40)
            self.collect_and_prepare_data()
            
            # Phase 2: Model Building
            print(f"\n🏗️ FASE 2: CONSTRUCCIÓN DE MODELOS - {self.symbol}")
            print("-" * 40)
            self.build_models()
            
            # Phase 3: Supervised Training
            print(f"\n🎓 FASE 3: ENTRENAMIENTO SUPERVISADO - {self.symbol}")
            print("-" * 40)
            self.train_supervised_models(epochs=supervised_epochs)
            
            # Phase 4: RL Setup
            print(f"\n🤖 FASE 4: CONFIGURACIÓN RL - {self.symbol}")
            print("-" * 40)
            self.setup_rl_agent()
            
            # Phase 5: RL Training
            print(f"\n🏆 FASE 5: ENTRENAMIENTO RL - {self.symbol}")
            print("-" * 40)
            self.train_rl_agent(episodes=rl_episodes)
            
            # Phase 6: Evaluation
            print(f"\n📊 FASE 6: EVALUACIÓN - {self.symbol}")
            print("-" * 40)
            metrics = self.evaluate_system()
            
            # Phase 7: Save
            print(f"\n💾 FASE 7: GUARDADO - {self.symbol}")
            print("-" * 40)
            self.save_system()
            
            end_time = datetime.now()
            training_time = end_time - start_time
            
            print(f"\n" + "=" * 60)
            print(f"🎉 ENTRENAMIENTO COMPLETADO - {self.symbol}!")
            print(f"⏱️ Tiempo total: {training_time}")
            print(f"📈 Score: {metrics['overall_score']:.3f}")
            print(f"📊 Tipo de par: {self.pair_config['pair_type']}")
            print("=" * 60)
            
            self.logger.info(f"Training completed for {self.symbol} - Score: {metrics['overall_score']:.3f}")
            return metrics
            
        except Exception as e:
            print(f"❌ Error durante entrenamiento de {self.symbol}: {e}")
            self.logger.error(f"Training failed for {self.symbol}: {e}")
            raise
    
    def collect_and_prepare_data(self):
        """Recolecta datos adaptados al par"""
        
        # Determine data parameters based on pair type and style
        data_config = self._get_data_config()
        
        self.raw_data = self.data_collector.collect_data(
            period=data_config['period'],
            interval=data_config['interval']
        )
        
        self.features_data = self.data_collector.prepare_features(self.raw_data, self.style)
        self.targets = self._create_targets()
        self.train_data, self.val_data, self.test_data = self._split_data()
        
        print(f"✅ Datos {self.symbol} preparados:")
        print(f"   • Train: {len(self.train_data)} muestras")
        print(f"   • Validation: {len(self.val_data)} muestras") 
        print(f"   • Test: {len(self.test_data)} muestras")
        print(f"   • Features: {len(self.features_data.columns)}")
    
    def _get_data_config(self):
        """Configuración de datos según par y estilo"""
        
        pair_type = self.pair_config['pair_type']
        
        configs = {
            'SCALPING': {
                'MAJORS': {'period': '30d', 'interval': '1m'},
                'MINORS': {'period': '30d', 'interval': '1m'},
                'EXOTICS': {'period': '15d', 'interval': '5m'},  # Less granular
                'COMMODITIES': {'period': '30d', 'interval': '1m'},
                'CRYPTO': {'period': '30d', 'interval': '1m'}
            },
            'DAY_TRADING': {
                'MAJORS': {'period': '2y', 'interval': '5m'},
                'MINORS': {'period': '1y', 'interval': '5m'},
                'EXOTICS': {'period': '6mo', 'interval': '15m'},
                'COMMODITIES': {'period': '1y', 'interval': '5m'},
                'CRYPTO': {'period': '2y', 'interval': '5m'}
            },
            'SWING_TRADING': {
                'MAJORS': {'period': '5y', 'interval': '1h'},
                'MINORS': {'period': '3y', 'interval': '1h'},
                'EXOTICS': {'period': '2y', 'interval': '4h'},
                'COMMODITIES': {'period': '3y', 'interval': '1h'},
                'CRYPTO': {'period': '5y', 'interval': '1h'}
            },
            'POSITION_TRADING': {
                'MAJORS': {'period': '10y', 'interval': '1d'},
                'MINORS': {'period': '8y', 'interval': '1d'},
                'EXOTICS': {'period': '5y', 'interval': '1d'},
                'COMMODITIES': {'period': '8y', 'interval': '1d'},
                'CRYPTO': {'period': '7y', 'interval': '1d'}
            }
        }
        
        return configs[self.style][pair_type]
    
    def build_models(self):
        """Construye modelos adaptados al par"""
        
        input_dim = len(self.features_data.columns)
        
        # Transformer always included
        self.models['transformer'] = TradingTransformer(
            input_dim=input_dim,
            d_model=self.config['d_model'],
            num_heads=self.config['num_heads'],
            num_layers=self.config['num_layers']
        ).to(self.device)
        
        # Mamba for longer timeframes and stable pairs
        use_mamba = (
            self.style in ['SWING_TRADING', 'POSITION_TRADING'] or
            self.pair_config['pair_type'] in ['MAJORS', 'COMMODITIES']
        )
        
        if use_mamba:
            self.models['mamba'] = TradingMamba(
                input_dim=input_dim,
                d_model=self.config['d_model'],
                num_layers=max(4, self.config['num_layers'] - 2)  # Slightly smaller
            ).to(self.device)
        
        models_info = f"Transformer: ✓, Mamba: {'✓' if use_mamba else '✗'}"
        print(f"✅ Modelos construidos para {self.symbol} - {models_info}")
    
    def _create_targets(self):
        """Crea targets adaptados al tipo de par y balancea si es necesario"""
        horizon = self.config['prediction_horizon']
        if len(self.features_data) < horizon + 10:
            print(f"⚠️ Datos insuficientes para crear targets: {len(self.features_data)} < {horizon + 10}")
            return pd.Series(dtype=int)
        future_returns = self.features_data['Close'].pct_change(horizon).shift(-horizon)
        base_volatility = self.features_data['Close'].pct_change().std()
        if pd.isna(base_volatility) or base_volatility < 1e-6:
            base_volatility = 0.001
        volatility_multipliers = {
            'MAJORS': 0.5, 'MINORS': 0.6, 'EXOTICS': 1.5, 'COMMODITIES': 1.0, 'CRYPTO': 2.0
        }
        multiplier = volatility_multipliers.get(self.pair_config['pair_type'], 1.0)
        threshold = max(base_volatility * multiplier, 0.0005)
        # Balanceo automático de targets
        for _ in range(5):
            targets = pd.Series(index=self.features_data.index, dtype=int)
            targets[future_returns > threshold] = 2
            targets[future_returns < -threshold] = 0
            targets[(future_returns >= -threshold) & (future_returns <= threshold)] = 1
            counts = targets.value_counts(normalize=True)
            print(f"📊 Distribución targets: {dict(counts)}")
            # Si alguna clase es > 60%, reduce threshold para balancear
            if any(counts > 0.6):
                threshold *= 0.7
            # Si todas las clases tienen al menos 20%, está balanceado
            elif all(counts.get(i,0) > 0.2 for i in [0,1,2]):
                break
            else:
                threshold *= 0.85
        return targets.dropna()
    
    def create_sequences(self, features, targets, sequence_length):
        """Crea secuencias para entrenamiento"""
        sequences = []
        sequence_targets = []
        
        # Verificar que tenemos suficientes datos
        if len(features) < sequence_length + 1:
            print(f"⚠️ Datos insuficientes para secuencias: {len(features)} < {sequence_length + 1}")
            return np.array([]), np.array([])
        
        # 🔧 FIX: Asegurar que targets tiene el mismo índice que features
        if len(targets) != len(features):
            print(f"⚠️ Mismatch entre features ({len(features)}) y targets ({len(targets)})")
            # Alinear targets con features
            targets = targets.reindex(features.index, method='ffill')
        
        for i in range(sequence_length, len(features)):
            # 🔧 FIX: Verificar que el índice existe
            if i >= len(targets):
                print(f"⚠️ Índice {i} fuera de rango para targets (len={len(targets)})")
                break
                
            seq = features.iloc[i-sequence_length:i].values
            target = targets.iloc[i]
            
            # Verificar que la secuencia no tiene NaN
            if not pd.isna(target) and not np.isnan(seq).any():
                # 🔧 FIX: Asegurar que la secuencia tiene exactamente sequence_length
                if seq.shape[0] == sequence_length:
                    sequences.append(seq)
                    sequence_targets.append(target)
        
        if len(sequences) == 0:
            print(f"⚠️ No se pudieron crear secuencias válidas para {self.symbol}")
            return np.array([]), np.array([])
        
        print(f"✅ Creadas {len(sequences)} secuencias válidas para {self.symbol}")
        return np.array(sequences), np.array(sequence_targets)
    
    def train_supervised_models(self, epochs=100):
        """Entrena modelos supervisados con early stopping y batch size adaptativo"""
        print(f"🚀 Entrenamiento supervisado {self.symbol} - {epochs} épocas")
        
        sequence_length = self.config['sequence_length']
        
        # Prepare training data
        X_train, y_train = self.create_sequences(
            self.train_data['features'], self.train_data['targets'], sequence_length
        )
        X_val, y_val = self.create_sequences(
            self.val_data['features'], self.val_data['targets'], sequence_length
        )
        
        print(f"📊 {self.symbol} - Train: {len(X_train)}, Val: {len(X_val)}")
        
        # 🔧 FIX: Validar que las secuencias tienen la longitud correcta
        if len(X_train) > 0 and X_train.shape[1] != sequence_length:
            print(f"⚠️ Ajustando secuencias de entrenamiento: {X_train.shape[1]} -> {sequence_length}")
            # Recortar o rellenar según sea necesario
            if X_train.shape[1] > sequence_length:
                X_train = X_train[:, -sequence_length:, :]
            else:
                # Rellenar con ceros si es más corto
                padding = np.zeros((X_train.shape[0], sequence_length - X_train.shape[1], X_train.shape[2]))
                X_train = np.concatenate([padding, X_train], axis=1)
        
        if len(X_val) > 0 and X_val.shape[1] != sequence_length:
            print(f"⚠️ Ajustando secuencias de validación: {X_val.shape[1]} -> {sequence_length}")
            if X_val.shape[1] > sequence_length:
                X_val = X_val[:, -sequence_length:, :]
            else:
                padding = np.zeros((X_val.shape[0], sequence_length - X_val.shape[1], X_val.shape[2]))
                X_val = np.concatenate([padding, X_val], axis=1)
        
        # Verificar que tenemos datos válidos
        if len(X_train) == 0 or len(X_val) == 0:
            print(f"❌ No hay datos suficientes para entrenar {self.symbol}")
            return {'error': 'No hay datos suficientes'}
        
        # Verificar que tenemos al menos batch_size muestras
        min_samples = self.config['batch_size'] * 2
        if len(X_train) < min_samples:
            print(f"⚠️ Muy pocas muestras de entrenamiento: {len(X_train)} < {min_samples}")
            print("🔄 Reduciendo batch_size...")
            self.config['batch_size'] = max(8, len(X_train) // 2)
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.LongTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.LongTensor(y_val).to(self.device)
        
        # DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        
        # Train each model
        for model_name, model in self.models.items():
            print(f"\n🔄 Entrenando {model_name.upper()} para {self.symbol}...")
            
            # Adjust learning rate for pair type
            lr_multipliers = {
                'MAJORS': 1.0,
                'MINORS': 0.8,
                'EXOTICS': 0.5,     # More conservative
                'COMMODITIES': 0.9,
                'CRYPTO': 1.5       # Faster learning for high volatility
            }
            
            adjusted_lr = self.config['learning_rate'] * lr_multipliers.get(self.pair_config['pair_type'], 1.0)
            
            optimizer = optim.Adam(model.parameters(), lr=adjusted_lr)
            criterion = nn.CrossEntropyLoss()
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)
            
            best_val_loss = float('inf')
            patience_counter = 0
            patience = 12 if self.pair_config['pair_type'] == 'EXOTICS' else 15
            
            for epoch in range(epochs):
                # Training
                model.train()
                train_loss = 0
                train_correct = 0
                train_total = 0
                
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    
                    # 🔧 FIX: Validar dimensiones del batch
                    if batch_X.shape[1] != self.config['sequence_length']:
                        print(f"⚠️ Saltando batch con secuencia incorrecta: {batch_X.shape[1]} != {self.config['sequence_length']}")
                        continue
                    
                    try:
                        outputs = model(batch_X)
                        predictions = outputs['direction'] if isinstance(outputs, dict) else outputs
                        
                        loss = criterion(predictions, batch_y)
                        loss.backward()
                        
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        
                        train_loss += loss.item()
                        _, predicted = torch.max(predictions.data, 1)
                        train_total += batch_y.size(0)
                        train_correct += (predicted == batch_y).sum().item()
                        
                    except RuntimeError as e:
                        if "size" in str(e) and "match" in str(e):
                            print(f"⚠️ Error de dimensiones en batch: {e}")
                            print(f"   Batch shape: {batch_X.shape}")
                            continue
                        else:
                            raise e
                
                # Validation
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val)
                    val_predictions = val_outputs['direction'] if isinstance(val_outputs, dict) else val_outputs
                    val_loss = criterion(val_predictions, y_val).item()
                    _, val_predicted = torch.max(val_predictions.data, 1)
                    val_accuracy = (val_predicted == y_val).sum().item() / len(y_val)
                
                train_accuracy = train_correct / train_total
                avg_train_loss = train_loss / len(train_loader)
                
                scheduler.step(val_loss)
                
                # Progress reporting
                if epoch % 15 == 0:
                    print(f"  Época {epoch:3d} | Loss: {avg_train_loss:.4f} | "
                          f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.3f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), f'best_{model_name}_{self.symbol}_{self.style.lower()}.pth')
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"  ⏰ Early stopping en época {epoch}")
                        break
                
                # Track performance
                self.performance_tracker['training_losses'].append({
                    'epoch': epoch, 'model': model_name, 'symbol': self.symbol,
                    'train_loss': avg_train_loss, 'val_loss': val_loss,
                    'train_acc': train_accuracy, 'val_acc': val_accuracy
                })
            
            # Load best model
            model.load_state_dict(torch.load(f'best_{model_name}_{self.symbol}_{self.style.lower()}.pth'))
            print(f"  ✅ {model_name.upper()} completado - Best Val Loss: {best_val_loss:.4f}")
        
        print(f"\n🎉 Entrenamiento supervisado completado para {self.symbol}!")
    
    def setup_rl_agent(self):
        """Configura agente RL adaptado al par"""
        
        print(f"🤖 Configurando PPO para {self.symbol}...")
        
        # Create trading environment
        self.trading_env = UniversalTradingEnvironment(
            data=self.features_data,
            symbol=self.symbol,
            style=self.style,
            initial_balance=100000
        )
        
        # 🔧 FIX: Calcular dimensión del estado mejorado de forma más robusta
        base_state_dim = self.trading_env.observation_space.shape[0]
        
        # Calcular predicciones adicionales (4 por modelo)
        num_models = len(self.models) if hasattr(self, 'models') and len(self.models) > 0 else 0
        enhanced_state_dim = base_state_dim + (num_models * 4)  # 4 predicciones por modelo
        
        action_dim = self.trading_env.action_space.shape[0]
        
        # Adjust learning rate for pair type - 🔧 FIX: Learning rates más conservadores
        lr_adjustments = {
            'MAJORS': 5e-5,       # Más conservador
            'MINORS': 3e-5,       # Más conservador
            'EXOTICS': 2e-5,      # Muy conservador
            'COMMODITIES': 5e-5,  # Más conservador
            'CRYPTO': 8e-5        # Conservador
        }
        
        rl_lr = lr_adjustments.get(self.pair_config['pair_type'], 5e-5)
        
        # 🔧 FIX: Guardar dimensiones para validación posterior
        self.base_state_dim = base_state_dim
        self.enhanced_state_dim = enhanced_state_dim
        self.action_dim = action_dim
        
        self.ppo_agent = PPOAgent(state_dim=enhanced_state_dim, action_dim=action_dim, lr=rl_lr)
        
        print(f"✅ PPO configurado para {self.symbol}")
        print(f"   • Estado base: {base_state_dim}, Mejorado: {enhanced_state_dim}, Acciones: {action_dim}")
        print(f"   • Learning rate: {rl_lr}")
    
    def train_rl_agent(self, episodes=1000):
        """Entrena agente RL adaptado al par"""
        
        print(f"🤖 Entrenamiento RL {self.symbol} - {episodes} episodios")
        
        # 🔧 FIX: Verificar que los modelos están disponibles
        if not hasattr(self, 'models') or len(self.models) == 0:
            print(f"⚠️ No hay modelos supervisados disponibles para {self.symbol}, usando solo estado base")
            self.use_enhanced_state = False
        else:
            self.use_enhanced_state = True
        
        episode_rewards = []
        episode_lengths = []
        
        # Adaptive training parameters
        update_frequency = max(5, min(20, 100 // self.pair_config['daily_volatility'] * 10))
        
        try:
            for episode in range(episodes):
                state = self.trading_env.reset()
                episode_reward = 0
                episode_length = 0
                episode_info = {'balance': 100000, 'num_trades': 0, 'win_rate': 0, 'sharpe_ratio': 0}  # Default info
                
                while True:
                    try:
                        # Get enhanced state with supervised predictions
                        enhanced_state = self._get_enhanced_state(state)
                        
                        # 🔧 FIX: Debugging de dimensiones
                        if episode == 0 and episode_length == 0:
                            print(f"🔍 Debug dimensiones - Estado original: {len(state)}, Mejorado: {len(enhanced_state)}")
                        
                        # Get action from PPO
                        action, value = self.ppo_agent.get_action(enhanced_state)
                        log_prob = self._calculate_log_prob(action)
                        
                        # Execute action
                        next_state, reward, done, info = self.trading_env.step(action)
                        
                        # Store transition
                        self.ppo_agent.store_transition(
                            enhanced_state, action, reward, next_state, done, log_prob, value
                        )
                        
                        episode_reward += reward
                        episode_length += 1
                        state = next_state
                        episode_info = info  # Update episode info
                        
                        if done:
                            break
                            
                    except Exception as e:
                        print(f"❌ Error en episodio {episode}, paso {episode_length}: {e}")
                        # Intentar continuar con estado simplificado
                        try:
                            # Usar solo el estado base sin predicciones
                            simple_state = state[:self.base_state_dim] if hasattr(self, 'base_state_dim') else state
                            action, value = self.ppo_agent.get_action(simple_state)
                            log_prob = self._calculate_log_prob(action)
                            
                            next_state, reward, done, info = self.trading_env.step(action)
                            
                            self.ppo_agent.store_transition(
                                simple_state, action, reward, next_state, done, log_prob, value
                            )
                            
                            episode_reward += reward
                            episode_length += 1
                            state = next_state
                            episode_info = info
                            
                            if done:
                                break
                        except Exception as e2:
                            print(f"❌ Error de recuperación en episodio {episode}: {e2}")
                            break
                
                # Update PPO
                if episode % update_frequency == 0:
                    try:
                        self.ppo_agent.update()
                    except Exception as e:
                        print(f"⚠️ Error en actualización PPO episodio {episode}: {e}")
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                # Progress reporting
                if episode % 100 == 0:
                    avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
                    
                    print(f"  Ep {episode:4d} | Reward: {episode_reward:7.1f} | "
                          f"Avg: {avg_reward:7.1f} | Balance: ${episode_info.get('balance', 0):,.0f} | "
                          f"Trades: {episode_info.get('num_trades', 0)}")
                    
                    self.performance_tracker['trading_performance'].append({
                        'episode': episode, 'symbol': self.symbol,
                        'reward': episode_reward, 'avg_reward': avg_reward,
                        'balance': episode_info.get('balance', 0), 'num_trades': episode_info.get('num_trades', 0),
                        'win_rate': episode_info.get('win_rate', 0), 'sharpe_ratio': episode_info.get('sharpe_ratio', 0)
                    })
            
            print(f"🎉 Entrenamiento RL completado para {self.symbol}!")
            
        except Exception as e:
            print(f"❌ Error durante entrenamiento de {self.symbol}: {e}")
            print(f"⚠️ Continuando con configuración simplificada...")
            
            # Intentar entrenamiento simplificado sin predicciones supervisadas
            try:
                self.use_enhanced_state = False
                print(f"🔄 Reintentando entrenamiento RL con estado base para {self.symbol}")
                
                for episode in range(min(episodes, 100)):  # Menos episodios para recuperación
                    state = self.trading_env.reset()
                    episode_reward = 0
                    episode_length = 0
                    
                    while True:
                        try:
                            # Usar solo estado base
                            simple_state = state[:self.base_state_dim] if hasattr(self, 'base_state_dim') else state
                            action, value = self.ppo_agent.get_action(simple_state)
                            log_prob = self._calculate_log_prob(action)
                            
                            next_state, reward, done, info = self.trading_env.step(action)
                            
                            self.ppo_agent.store_transition(
                                simple_state, action, reward, next_state, done, log_prob, value
                            )
                            
                            episode_reward += reward
                            episode_length += 1
                            state = next_state
                            
                            if done:
                                break
                                
                        except Exception as e2:
                            print(f"❌ Error en recuperación episodio {episode}: {e2}")
                            break
                    
                    if episode % 50 == 0:
                        print(f"  Recuperación Ep {episode:4d} | Reward: {episode_reward:7.1f}")
                
                print(f"✅ Entrenamiento RL de recuperación completado para {self.symbol}")
                
            except Exception as e2:
                print(f"❌ Error fatal durante entrenamiento de {self.symbol}: {e2}")
                raise
    
    def _get_enhanced_state(self, state):
        """Combina estado con predicciones supervisadas"""
        
        # 🔧 FIX: Validar que el estado base no contenga NaN o Inf
        if np.isnan(state).any() or np.isinf(state).any():
            print(f"⚠️ NaN/Inf en estado base para {self.symbol}, limpiando...")
            state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 🔧 FIX: Usar solo estado base si no hay modelos disponibles
        if hasattr(self, 'use_enhanced_state') and not self.use_enhanced_state:
            return state
        
        try:
            # Get current features from state
            current_features = state[:len(self.features_data.columns)]
            
            # Get supervised predictions
            supervised_predictions = self._get_supervised_predictions(current_features)
            
            # 🔧 FIX: Validar dimensiones antes de concatenar
            if len(supervised_predictions) == 0:
                print(f"⚠️ Predicciones vacías para {self.symbol}, usando fallback")
                return state
            
            # 🔧 FIX: Validar que las predicciones supervisadas no contengan NaN o Inf
            if np.isnan(supervised_predictions).any() or np.isinf(supervised_predictions).any():
                print(f"⚠️ NaN/Inf en predicciones supervisadas para {self.symbol}, usando valores de fallback")
                supervised_predictions = np.nan_to_num(supervised_predictions, nan=0.5, posinf=1.0, neginf=0.0)
            
            # Combine
            enhanced_state = np.concatenate([state, supervised_predictions])
            
            # 🔧 FIX: Validar que el estado mejorado tiene la dimensión esperada
            expected_dim = len(state) + len(supervised_predictions)
            if len(enhanced_state) != expected_dim:
                print(f"⚠️ Dimensión incorrecta del estado mejorado: {len(enhanced_state)} vs {expected_dim}")
                # Usar solo el estado original si hay problemas
                return state
            
            # 🔧 FIX: Validar que el estado mejorado coincide con la dimensión esperada del PPO
            if hasattr(self, 'enhanced_state_dim') and len(enhanced_state) != self.enhanced_state_dim:
                print(f"⚠️ Dimensión del estado mejorado ({len(enhanced_state)}) no coincide con PPO ({self.enhanced_state_dim})")
                # Ajustar el estado para que coincida con la dimensión esperada
                if len(enhanced_state) > self.enhanced_state_dim:
                    # Truncar si es muy largo
                    enhanced_state = enhanced_state[:self.enhanced_state_dim]
                else:
                    # Rellenar con ceros si es muy corto
                    padding = np.zeros(self.enhanced_state_dim - len(enhanced_state))
                    enhanced_state = np.concatenate([enhanced_state, padding])
            
            # 🔧 FIX: Validación final del estado mejorado
            if np.isnan(enhanced_state).any() or np.isinf(enhanced_state).any():
                print(f"⚠️ NaN/Inf en estado mejorado final para {self.symbol}, limpiando...")
                enhanced_state = np.nan_to_num(enhanced_state, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return enhanced_state
            
        except Exception as e:
            print(f"❌ Error en _get_enhanced_state para {self.symbol}: {e}")
            # Fallback: devolver estado original sin predicciones
            return state
    
    def _get_supervised_predictions(self, features):
        """Obtiene predicciones de modelos supervisados"""
        
        # 🔧 FIX: Verificar que hay modelos disponibles
        if not hasattr(self, 'models') or len(self.models) == 0:
            return np.array([])
        
        # 🔧 FIX: Validar que features no contenga NaN o Inf
        if np.isnan(features).any() or np.isinf(features).any():
            print(f"⚠️ NaN/Inf en features para predicciones supervisadas de {self.symbol}, usando valores de fallback")
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        try:
            predictions = []
            sequence_length = self.config['sequence_length']
            
            # Use recent features if available, otherwise repeat current
            if hasattr(self, 'recent_features') and len(self.recent_features) >= sequence_length:
                input_sequence = self.recent_features[-sequence_length:]
            else:
                input_sequence = np.tile(features, (sequence_length, 1))
            
            # 🔧 FIX: Validar que input_sequence no contenga NaN o Inf
            if np.isnan(input_sequence).any() or np.isinf(input_sequence).any():
                print(f"⚠️ NaN/Inf en input_sequence para {self.symbol}, usando valores de fallback")
                input_sequence = np.nan_to_num(input_sequence, nan=0.0, posinf=1.0, neginf=-1.0)
            
            input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0).to(self.device)
            
            # Get predictions from each model
            for model_name, model in self.models.items():
                try:
                    model.eval()
                    with torch.no_grad():
                        output = model(input_tensor)
                        
                        if isinstance(output, dict):
                            direction_probs = torch.softmax(output['direction'], dim=-1)[0]
                            
                            # 🔧 FIX: Manejar diferentes tipos de salida de modelos
                            if 'confidence' in output:
                                # Transformer output
                                confidence = output['confidence'][0].item()
                            elif 'regime' in output:
                                # Mamba output - usar máxima probabilidad de régimen como confianza
                                regime_probs = torch.softmax(output['regime'], dim=-1)[0]
                                confidence = regime_probs.max().item()
                            else:
                                # Fallback: usar máxima probabilidad de dirección
                                confidence = direction_probs.max().item()
                            
                            # 🔧 FIX: Validar que las predicciones no sean NaN o Inf
                            pred_values = [
                                direction_probs[0].item(),  # DOWN
                                direction_probs[1].item(),  # SIDEWAYS
                                direction_probs[2].item(),  # UP
                                confidence
                            ]
                            
                            if any(np.isnan(pred_values)) or any(np.isinf(pred_values)):
                                print(f"⚠️ NaN/Inf en predicciones de {model_name} para {self.symbol}, usando valores de fallback")
                                pred_values = [0.33, 0.34, 0.33, 0.5]
                            
                            predictions.extend(pred_values)
                        else:
                            # Fallback para salidas no-diccionario
                            if hasattr(output, 'shape') and len(output.shape) >= 2:
                                direction_probs = torch.softmax(output, dim=-1)[0]
                                confidence = direction_probs.max().item()
                            else:
                                # Último recurso
                                confidence = 0.5
                                direction_probs = torch.tensor([0.33, 0.34, 0.33])
                            
                            # 🔧 FIX: Validar predicciones fallback
                            pred_values = [
                                direction_probs[0].item(),
                                direction_probs[1].item(), 
                                direction_probs[2].item(),
                                confidence
                            ]
                            
                            if any(np.isnan(pred_values)) or any(np.isinf(pred_values)):
                                print(f"⚠️ NaN/Inf en predicciones fallback de {model_name} para {self.symbol}")
                                pred_values = [0.33, 0.34, 0.33, 0.5]
                            
                            predictions.extend(pred_values)
                            
                except Exception as e:
                    print(f"⚠️ Error en modelo {model_name}: {e}")
                    # Fallback para este modelo
                    predictions.extend([0.33, 0.34, 0.33, 0.5])
            
            predictions_array = np.array(predictions)
            
            # 🔧 FIX: Validación final de todas las predicciones
            if np.isnan(predictions_array).any() or np.isinf(predictions_array).any():
                print(f"⚠️ NaN/Inf en predictions_array final para {self.symbol}, usando valores de fallback")
                predictions_array = np.nan_to_num(predictions_array, nan=0.5, posinf=1.0, neginf=0.0)
            
            return predictions_array
            
        except Exception as e:
            print(f"❌ Error en _get_supervised_predictions: {e}")
            # Fallback completo
            num_models = len(self.models) if hasattr(self, 'models') and len(self.models) > 0 else 0
            return np.array([0.33, 0.34, 0.33, 0.5] * num_models)
    
    def _calculate_log_prob(self, action):
        """Calcula log probability simplificado"""
        return -0.5 * np.sum(action**2)
    
    def evaluate_system(self):
        """Evalúa sistema adaptado al par"""
        
        print(f"📊 Evaluando sistema {self.symbol}...")
        
        supervised_metrics = self._evaluate_supervised_models()
        rl_metrics = self._evaluate_rl_agent()
        
        # Pair-specific scoring
        pair_weight = {
            'MAJORS': 1.0,
            'MINORS': 0.9,
            'EXOTICS': 0.7,     # Lower expectations
            'COMMODITIES': 0.8,
            'CRYPTO': 1.1       # Bonus for handling volatility
        }.get(self.pair_config['pair_type'], 1.0)
        
        overall_score = (
            supervised_metrics.get('accuracy', 0) * 0.4 +
            max(0, min(1, (rl_metrics.get('sharpe_ratio', 0) + 1) / 3)) * 0.6
        ) * pair_weight
        
        metrics = {
            'supervised': supervised_metrics,
            'reinforcement_learning': rl_metrics,
            'overall_score': overall_score,
            'pair_info': self.pair_config
        }
        
        print(f"📈 Resultados {self.symbol}:")
        print(f"  • Precisión: {supervised_metrics.get('accuracy', 0):.3f}")
        print(f"  • Sharpe RL: {rl_metrics.get('sharpe_ratio', 0):.3f}")
        print(f"  • Score final: {overall_score:.3f}")
        
        return metrics
    
    def _evaluate_supervised_models(self):
        """Evalúa modelos supervisados"""
        
        sequence_length = self.config['sequence_length']
        X_test, y_test = self.create_sequences(
            self.test_data['features'], self.test_data['targets'], sequence_length
        )
        
        if len(X_test) == 0:
            return {'accuracy': 0}
        
        X_test = torch.FloatTensor(X_test).to(self.device)
        y_test = torch.LongTensor(y_test).to(self.device)
        
        metrics = {}
        
        for model_name, model in self.models.items():
            model.eval()
            with torch.no_grad():
                outputs = model(X_test)
                predictions = outputs['direction'] if isinstance(outputs, dict) else outputs
                
                _, predicted = torch.max(predictions.data, 1)
                accuracy = (predicted == y_test).sum().item() / len(y_test)
                metrics[f'{model_name}_accuracy'] = accuracy
        
        metrics['accuracy'] = np.mean([v for k, v in metrics.items() if 'accuracy' in k])
        return metrics
    
    def _evaluate_rl_agent(self):
        """Evalúa agente RL"""
        
        if self.ppo_agent is None:
            return {'sharpe_ratio': 0, 'total_return': 0}
        
        # Run evaluation episode
        state = self.trading_env.reset()
        total_reward = 0
        
        while True:
            enhanced_state = self._get_enhanced_state(state)
            action, _ = self.ppo_agent.get_action(enhanced_state)
            next_state, reward, done, info = self.trading_env.step(action)
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        return {
            'total_return': (info['balance'] - 100000) / 100000,
            'sharpe_ratio': info.get('sharpe_ratio', 0),
            'win_rate': info.get('win_rate', 0),
            'num_trades': info.get('num_trades', 0)
        }
    
    def save_system(self, filepath=None):
        """Guarda sistema específico del par"""
        
        if filepath is None:
            filepath = f'universal_brain_{self.symbol}_{self.style.lower()}'
        
        print(f"💾 Guardando sistema {self.symbol}...")
        
        # Save models
        for model_name, model in self.models.items():
            torch.save(model.state_dict(), f'{filepath}_{model_name}.pth')
        
        # Save PPO agent
        if self.ppo_agent is not None:
            torch.save({
                'actor': self.ppo_agent.actor.state_dict(),
                'critic': self.ppo_agent.critic.state_dict()
            }, f'{filepath}_ppo.pth')
        
        # Save metadata
        metadata = {
            'symbol': self.symbol,
            'style': self.style,
            'pair_config': self.pair_config,
            'config': self.config,
            'performance_tracker': self.performance_tracker
        }
        
        with open(f'{filepath}_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save scaler
        with open(f'{filepath}_scaler.pkl', 'wb') as f:
            pickle.dump(self.data_collector.scaler, f)
        
        print(f"✅ Sistema {self.symbol} guardado")
    
    def _split_data(self):
        """Split temporal de datos"""
        # Align features and targets
        common_index = self.features_data.index.intersection(self.targets.index)
        features_aligned = self.features_data.loc[common_index]
        targets_aligned = self.targets.loc[common_index]
        min_required = self.config['sequence_length'] * 10
        min_required_synthetic = self.config['sequence_length'] * 5
        if len(features_aligned) < min_required:
            print(f"⚠️ Datos insuficientes: {len(features_aligned)} < {min_required}")
            print("🔄 Generando más datos sintéticos...")
            self.data_collector = UniversalDataCollector(self.symbol)
            data = self.data_collector.collect_data(period='10y', interval='5m')
            features_data = self.data_collector.prepare_features(data, self.style)
            targets = self._create_targets()
            common_index = features_data.index.intersection(targets.index)
            features_aligned = features_data.loc[common_index]
            targets_aligned = targets.loc[common_index]
            if len(features_aligned) < min_required_synthetic:
                print(f"⚠️ Datos aún insuficientes después de regeneración: {len(features_aligned)}")
                print("🔄 Ajustando configuración para datos sintéticos...")
                # Ajustar sequence_length al mínimo posible
                min_seq = max(5, len(features_aligned) // 10)
                if min_seq < 5:
                    min_seq = 5
                print(f"⚠️ Forzando sequence_length a {min_seq} para continuar con pocos datos.")
                self.config['sequence_length'] = min_seq
                # No lanzar excepción, continuar con lo que haya
        # Temporal split
        total_len = len(features_aligned)
        train_end = int(total_len * 0.7)
        val_end = int(total_len * 0.85)
        train_size = train_end
        val_size = val_end - train_end
        test_size = total_len - val_end
        min_split_size = self.config['sequence_length'] * 3
        if train_size < min_split_size or val_size < min_split_size or test_size < min_split_size:
            print(f"⚠️ Split muy pequeño - Train: {train_size}, Val: {val_size}, Test: {test_size}")
            train_end = min_split_size
            val_end = min_split_size * 2
            if val_end >= total_len:
                val_end = total_len - min_split_size
        return {
            'features': features_aligned.iloc[:train_end],
            'targets': targets_aligned.iloc[:train_end]
        }, {
            'features': features_aligned.iloc[train_end:val_end],
            'targets': targets_aligned.iloc[train_end:val_end]
        }, {
            'features': features_aligned.iloc[val_end:],
            'targets': targets_aligned.iloc[val_end:]
        }

# ====================================================================
# MULTI-PAIR TRAINING SYSTEM
# ====================================================================

def train_multiple_pairs(pairs, styles=['DAY_TRADING'], max_concurrent=2):
    """Entrena múltiples pares de forma eficiente"""
    
    print("🌍 SISTEMA UNIVERSAL MULTI-PAR")
    print("=" * 60)
    print(f"📊 Pares: {pairs}")
    print(f"🎯 Estilos: {styles}")
    print("=" * 60)
    
    results = {}
    
    for style in styles:
        print(f"\n🎯 ENTRENANDO ESTILO: {style}")
        print("-" * 40)
        
        style_results = {}
        
        for i, pair in enumerate(pairs):
            print(f"\n📈 Par {i+1}/{len(pairs)}: {pair}")
            
            try:
                # Detect pair type and check viability
                pair_config = UniversalPairConfig.get_pair_config(pair)
                
                # Skip scalping for high-spread pairs
                if style == 'SCALPING' and not pair_config['scalping_viable']:
                    print(f"⚠️ Saltando {pair} - Scalping no viable")
                    continue
                
                # Create and train brain
                brain = UniversalTradingBrain(symbol=pair, style=style)
                
                # Adaptive training parameters
                epochs = min(60, max(20, 1000 // pair_config['daily_volatility']))
                episodes = min(800, max(200, 5000 // pair_config['daily_volatility']))
                
                metrics = brain.run_complete_training(
                    supervised_epochs=epochs,
                    rl_episodes=episodes
                )
                
                style_results[pair] = {
                    'metrics': metrics,
                    'brain': brain,
                    'pair_config': pair_config
                }
                
                print(f"✅ {pair} completado - Score: {metrics['overall_score']:.3f}")
                
            except Exception as e:
                print(f"❌ Error en {pair}: {e}")
                style_results[pair] = {'error': str(e)}
                continue
        
        results[style] = style_results
    
    # Summary report
    print("\n" + "=" * 60)
    print("📊 RESUMEN FINAL MULTI-PAR")
    print("=" * 60)
    
    for style, style_results in results.items():
        print(f"\n🎯 {style}:")
        
        successful_pairs = [p for p, r in style_results.items() if 'metrics' in r]
        failed_pairs = [p for p, r in style_results.items() if 'error' in r]
        
        if successful_pairs:
            scores = [style_results[p]['metrics']['overall_score'] for p in successful_pairs]
            best_pair = max(successful_pairs, key=lambda p: style_results[p]['metrics']['overall_score'])
            
            print(f"  ✅ Exitosos: {len(successful_pairs)}")
            print(f"  📈 Score promedio: {np.mean(scores):.3f}")
            print(f"  🏆 Mejor par: {best_pair} ({style_results[best_pair]['metrics']['overall_score']:.3f})")
        
        if failed_pairs:
            print(f"  ❌ Fallidos: {failed_pairs}")
    
    return results

# ====================================================================
# SCRIPT PRINCIPAL UNIVERSAL
# ====================================================================

def main():
    """Función principal para sistema universal"""
    
    print("🌍 UNIVERSAL TRADING BRAIN - MULTI-CURRENCY")
    print("=" * 60)
    print("🎯 Sistema adaptable a cualquier par automáticamente")
    print("🧠 Arquitectura: Transformer + Mamba + PPO")
    print("📊 Soporte: Forex, Commodities, Crypto, Exóticos")
    print("=" * 60)
    
    # Configuración para diferentes escenarios
    scenarios = {
        'DEMO_SINGLE': {
            'pairs': ['EURUSD'],
            'styles': ['DAY_TRADING'],
            'description': 'Demo rápido con EURUSD'
        },
        
        'FOREX_MAJORS': {
            'pairs': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'],
            'styles': ['SCALPING', 'DAY_TRADING'],
            'description': 'Pares principales Forex'
        },
        
        'MULTI_ASSET': {
            'pairs': ['EURUSD', 'XAUUSD', 'BTCUSD', 'GBPJPY'],
            'styles': ['DAY_TRADING', 'SWING_TRADING'],
            'description': 'Diversificación multi-asset'
        },
        
        'CRYPTO_FOCUS': {
            'pairs': ['BTCUSD', 'ETHUSD'],
            'styles': ['SCALPING', 'DAY_TRADING', 'SWING_TRADING'],
            'description': 'Especialización en criptomonedas'
        },
        
        'COMPLETE_SYSTEM': {
            'pairs': ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD', 'BTCUSD', 'USDTRY'],
            'styles': ['DAY_TRADING', 'SWING_TRADING'],
            'description': 'Sistema completo multi-par'
        }
    }
    
    # Seleccionar escenario (cambiar aquí para diferentes tests)
    selected_scenario = 'DEMO_SINGLE'  # Cambiar según necesidades
    
    if selected_scenario not in scenarios:
        print(f"❌ Escenario '{selected_scenario}' no encontrado")
        return
    
    scenario = scenarios[selected_scenario]
    print(f"🎯 Ejecutando: {scenario['description']}")
    print(f"📊 Pares: {scenario['pairs']}")
    print(f"🎨 Estilos: {scenario['styles']}")
    print("-" * 60)
    
    # Entrenar sistema
    try:
        results = train_multiple_pairs(
            pairs=scenario['pairs'],
            styles=scenario['styles'],
            max_concurrent=2
        )
        
        # Análisis de resultados
        analyze_results(results)
        
        # Guardar reporte final
        save_final_report(results, selected_scenario)
        
        print("\n🎉 SISTEMA UNIVERSAL COMPLETADO EXITOSAMENTE!")
        return results
        
    except Exception as e:
        print(f"❌ Error en sistema universal: {e}")
        raise

def analyze_results(results):
    """Analiza resultados del entrenamiento multi-par"""
    
    print("\n" + "=" * 60)
    print("📊 ANÁLISIS DETALLADO DE RESULTADOS")
    print("=" * 60)
    
    all_scores = []
    pair_performance = {}
    style_performance = {}
    
    for style, style_results in results.items():
        style_scores = []
        
        for pair, result in style_results.items():
            if 'metrics' in result:
                score = result['metrics']['overall_score']
                pair_type = result['pair_config']['pair_type']
                
                all_scores.append(score)
                style_scores.append(score)
                
                if pair not in pair_performance:
                    pair_performance[pair] = {'scores': [], 'type': pair_type}
                pair_performance[pair]['scores'].append(score)
        
        if style_scores:
            style_performance[style] = {
                'avg_score': np.mean(style_scores),
                'best_score': max(style_scores),
                'count': len(style_scores)
            }
    
    # Análisis por estilo
    print("\n📈 PERFORMANCE POR ESTILO:")
    for style, perf in style_performance.items():
        print(f"  {style:15s}: Avg {perf['avg_score']:.3f} | "
              f"Best {perf['best_score']:.3f} | "
              f"Pares: {perf['count']}")
    
    # Análisis por par
    print("\n💱 PERFORMANCE POR PAR:")
    for pair, perf in pair_performance.items():
        avg_score = np.mean(perf['scores'])
        best_score = max(perf['scores'])
        print(f"  {pair:8s} ({perf['type']:11s}): Avg {avg_score:.3f} | Best {best_score:.3f}")
    
    # Análisis por tipo de par
    print("\n🏦 PERFORMANCE POR TIPO:")
    type_scores = {}
    for pair, perf in pair_performance.items():
        pair_type = perf['type']
        if pair_type not in type_scores:
            type_scores[pair_type] = []
        type_scores[pair_type].extend(perf['scores'])
    
    for pair_type, scores in type_scores.items():
        avg_score = np.mean(scores)
        print(f"  {pair_type:12s}: Avg {avg_score:.3f} | Count: {len(scores)}")
    
    # Estadísticas generales
    if all_scores:
        print(f"\n📊 ESTADÍSTICAS GENERALES:")
        print(f"  • Score promedio: {np.mean(all_scores):.3f}")
        print(f"  • Score máximo: {max(all_scores):.3f}")
        print(f"  • Score mínimo: {min(all_scores):.3f}")
        print(f"  • Desviación estándar: {np.std(all_scores):.3f}")
        print(f"  • Modelos exitosos: {len(all_scores)}")

def save_final_report(results, scenario_name):
    """Guarda reporte final en JSON"""
    
    report = {
        'scenario': scenario_name,
        'timestamp': datetime.now().isoformat(),
        'summary': {},
        'detailed_results': {}
    }
    
    # Procesar resultados para JSON
    for style, style_results in results.items():
        report['detailed_results'][style] = {}
        
        for pair, result in style_results.items():
            if 'metrics' in result:
                # Extract serializable data
                report['detailed_results'][style][pair] = {
                    'overall_score': result['metrics']['overall_score'],
                    'supervised_accuracy': result['metrics']['supervised'].get('accuracy', 0),
                    'rl_sharpe': result['metrics']['reinforcement_learning'].get('sharpe_ratio', 0),
                    'rl_return': result['metrics']['reinforcement_learning'].get('total_return', 0),
                    'pair_type': result['pair_config']['pair_type'],
                    'daily_volatility': result['pair_config']['daily_volatility']
                }
            else:
                report['detailed_results'][style][pair] = {
                    'error': result.get('error', 'Unknown error')
                }
    
    # Summary statistics
    all_scores = []
    for style_results in results.values():
        for result in style_results.values():
            if 'metrics' in result:
                all_scores.append(result['metrics']['overall_score'])
    
    if all_scores:
        report['summary'] = {
            'total_models': len(all_scores),
            'avg_score': float(np.mean(all_scores)),
            'max_score': float(max(all_scores)),
            'min_score': float(min(all_scores)),
            'std_score': float(np.std(all_scores))
        }
    
    # Save report
    filename = f'universal_trading_report_{scenario_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Reporte guardado: {filename}")

# ====================================================================
# UTILIDADES ADICIONALES
# ====================================================================

def quick_test_pair(symbol, style='DAY_TRADING'):
    """Test rápido de un solo par"""
    
    print(f"🚀 TEST RÁPIDO: {symbol} - {style}")
    print("-" * 40)
    
    try:
        # Detect pair configuration
        pair_config = UniversalPairConfig.get_pair_config(symbol)
        print(f"📊 Tipo detectado: {pair_config['pair_type']}")
        print(f"💱 Volatilidad diaria: {pair_config['daily_volatility']}")
        print(f"📈 Spread típico: {pair_config['typical_spread']} pips")
        
        # Create brain
        brain = UniversalTradingBrain(symbol=symbol, style=style)
        
        # Quick training (reduced parameters)
        metrics = brain.run_complete_training(
            supervised_epochs=20,
            rl_episodes=100
        )
        
        print(f"\n✅ TEST COMPLETADO:")
        print(f"   • Score final: {metrics['overall_score']:.3f}")
        print(f"   • Precisión: {metrics['supervised'].get('accuracy', 0):.3f}")
        print(f"   • Sharpe RL: {metrics['reinforcement_learning'].get('sharpe_ratio', 0):.3f}")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Error en test: {e}")
        return None

def list_supported_pairs():
    """Lista todos los pares soportados por tipo"""
    
    print("📋 PARES SOPORTADOS POR EL SISTEMA UNIVERSAL")
    print("=" * 50)
    
    for pair_type, config in UniversalPairConfig.PAIR_CONFIGS.items():
        print(f"\n🏦 {pair_type}:")
        print(f"   • Pares: {', '.join(config['pairs'])}")
        print(f"   • Spread típico: {config['typical_spread']} pips")
        print(f"   • Volatilidad: {config['daily_volatility']} pips/día")
        print(f"   • Scalping viable: {'✅' if config['scalping_viable'] else '❌'}")
        print(f"   • Leverage disponible: 1:{config['leverage_available']}")

def compare_pair_configs(symbols):
    """Compara configuraciones de múltiples pares"""
    
    print(f"⚖️ COMPARACIÓN DE CONFIGURACIONES")
    print("=" * 60)
    
    configs = []
    for symbol in symbols:
        config = UniversalPairConfig.get_pair_config(symbol)
        config['symbol'] = symbol
        configs.append(config)
    
    # Create comparison table
    print(f"{'Símbolo':<10} {'Tipo':<12} {'Volatilidad':<11} {'Spread':<7} {'Leverage':<8}")
    print("-" * 60)
    
    for config in configs:
        print(f"{config['symbol']:<10} "
              f"{config['pair_type']:<12} "
              f"{config['daily_volatility']:<11} "
              f"{config['typical_spread']:<7} "
              f"1:{config['leverage_available']:<6}")

# ====================================================================
# EJEMPLO DE USO ESPECÍFICO
# ====================================================================

def demo_single_pair():
    """Demo con un solo par para pruebas rápidas"""
    
    print("🎯 DEMO: ENTRENAMIENTO DE UN SOLO PAR")
    print("=" * 40)
    
    # Configuración del demo
    symbol = 'EURUSD'  # Cambiar aquí por cualquier par
    style = 'DAY_TRADING'  # O cualquier otro estilo
    
    return quick_test_pair(symbol, style)

def demo_crypto_trading():
    """Demo especializado en criptomonedas"""
    
    crypto_pairs = ['BTCUSD', 'ETHUSD']
    crypto_styles = ['DAY_TRADING', 'SWING_TRADING']
    
    print("₿ DEMO: TRADING DE CRIPTOMONEDAS")
    print("=" * 40)
    
    return train_multiple_pairs(crypto_pairs, crypto_styles)

def demo_forex_majors():
    """Demo con pares principales de Forex"""
    
    major_pairs = ['EURUSD', 'GBPUSD', 'USDJPY']
    forex_styles = ['SCALPING', 'DAY_TRADING']
    
    print("💱 DEMO: FOREX MAJORS")
    print("=" * 40)
    
    return train_multiple_pairs(major_pairs, forex_styles)

# ====================================================================
# CONFIGURACIÓN PARA DIFERENTES ENTORNOS
# ====================================================================

def configure_for_kaggle():
    """Configuración optimizada para Kaggle"""
    
    import os
    
    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        print("🏃‍♂️ Configuración para Kaggle detectada")
        
        # Optimizaciones para Kaggle
        torch.set_num_threads(2)
        
        # Configuración de memoria conservadora
        return {
            'max_pairs': 2,
            'max_epochs': 30,
            'max_episodes': 200,
            'reduced_features': True
        }
    
    return {
        'max_pairs': 10,
        'max_epochs': 100,
        'max_episodes': 1000,
        'reduced_features': False
    }

def configure_for_colab():
    """Configuración optimizada para Google Colab"""
    
    try:
        import google.colab
        print("📱 Configuración para Google Colab detectada")
        
        return {
            'max_pairs': 4,
            'max_epochs': 60,
            'max_episodes': 500,
            'use_gpu': True
        }
    except ImportError:
        return configure_for_kaggle()

# ====================================================================
# EJECUCIÓN PRINCIPAL
# ====================================================================

def test_fixes():
    """Test simple para verificar que los fixes funcionan"""
    print("🧪 TESTING FIXES...")
    
    try:
        # Test data collection
        collector = UniversalDataCollector('EURUSD')
        data = collector.collect_data(period='1y', interval='5m')
        print(f"✅ Data collection: {len(data)} points")
        
        # Test feature preparation
        features = collector.prepare_features(data, 'DAY_TRADING')
        print(f"✅ Feature preparation: {len(features)} features")
        
        # Test brain initialization
        brain = UniversalTradingBrain('EURUSD', 'DAY_TRADING')
        print("✅ Brain initialization")
        
        # Test data preparation
        brain.collect_and_prepare_data()
        print(f"✅ Data preparation: Train={len(brain.train_data['features'])}, Val={len(brain.val_data['features'])}")
        
        print("✅ ALL TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        return False

if __name__ == "__main__":
    # Test fixes first
    if test_fixes():
        print("\n🚀 Starting main execution...")
        main()
    else:
        print("\n❌ Fixes failed, stopping execution")