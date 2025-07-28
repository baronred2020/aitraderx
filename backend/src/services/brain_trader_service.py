"""
Brain Trader Service
===================
Servicio para manejar predicciones, señales y tendencias de Brain Trader
"""

import asyncio
import logging
import random
import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pickle
import joblib
import numpy as np
import pandas as pd

# Agregar el directorio de modelos al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))

# Importar el servicio de análisis técnico
try:
    from services.technical_analysis_service import TechnicalAnalysisService
    technical_analysis_service = TechnicalAnalysisService()
except ImportError as e:
    logging.error(f"Error importing TechnicalAnalysisService: {e}")
    technical_analysis_service = None

# Importar el cargador de modelos
try:
    import sys
    from pathlib import Path
    # Agregar el directorio backend al path para poder importar utils
    backend_path = Path(__file__).parent.parent.parent
    if str(backend_path) not in sys.path:
        sys.path.insert(0, str(backend_path))
    
    from utils.model_loader import ModelLoader
    model_loader = ModelLoader()
    # Limpiar cache al inicializar para asegurar modelos frescos
    if model_loader:
        model_loader.clear_cache()
        logging.info("ModelLoader cache cleared on startup")
except ImportError as e:
    logging.error(f"Error importing ModelLoader: {e}")
    model_loader = None

logger = logging.getLogger(__name__)

@dataclass
class PredictionResponse:
    pair: str
    direction: str
    confidence: float
    target_price: float
    timeframe: str
    reasoning: str
    brain_type: str
    timestamp: str
    expires_at: str

@dataclass
class SignalResponse:
    pair: str
    type: str
    strength: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    brain_type: str
    timestamp: str

@dataclass
class TrendResponse:
    pair: str
    direction: str
    strength: float
    timeframe: str
    support: float
    resistance: float
    description: str
    brain_type: str
    timestamp: str

class BrainTraderService:
    def __init__(self):
        self.valid_brain_types = ['brain_max', 'brain_ultra', 'brain_predictor', 'mega_mind']
        self.valid_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD', 'AUDUSD']
        self.valid_styles = ['day_trading', 'swing_trading', 'position_trading', 'scalping']
        
        # Configuración de timeframes por estilo
        self.style_timeframes = {
            'scalping': '5M',
            'day_trading': '15M', 
            'swing_trading': '1H',
            'position_trading': '4H'
        }
        
        # Duración de predicciones por estilo (en minutos)
        self.style_durations = {
            'scalping': 5,
            'day_trading': 15,
            'swing_trading': 60,
            'position_trading': 240
        }
        
        # Precios base por par
        self.base_prices = {
            'EURUSD': 1.0925,
            'GBPUSD': 1.2500,
            'USDJPY': 150.50,
            'USDCAD': 1.3500,
            'AUDUSD': 0.6500
        }

    def _validate_brain_type(self, brain_type: str) -> bool:
        """Validar tipo de cerebro"""
        return brain_type in self.valid_brain_types

    def _validate_pair(self, pair: str) -> bool:
        """Validar par de divisas"""
        return pair in self.valid_pairs

    def _validate_style(self, style: str) -> bool:
        """Validar estilo de trading"""
        return style in self.valid_styles

    def get_timeframe_for_style(self, style: str) -> str:
        """Obtener timeframe para un estilo específico"""
        return self.style_timeframes.get(style, '15M')

    def get_duration_for_style(self, style: str) -> int:
        """Obtener duración en minutos para un estilo específico"""
        return self.style_durations.get(style, 15)

    async def _get_brain_max_prediction(self, pair: str, style: str, current_price: float) -> Dict[str, Any]:
        """Obtener predicción usando modelos entrenados de Brain Max"""
        try:
            if model_loader is None:
                logger.warning("ModelLoader no disponible, usando análisis técnico básico")
                return self._get_fallback_prediction(pair, style, current_price)
            
            # Cargar modelo Brain Max
            model, scaler, model_info = model_loader.load_brain_max(pair, style)
            
            if model is None:
                logger.warning(f"Modelo Brain Max no encontrado para {pair}/{style}, usando fallback")
                return self._get_fallback_prediction(pair, style, current_price)
            
            # Obtener datos históricos para features
            if technical_analysis_service:
                data = await technical_analysis_service.get_historical_data(pair, "30d")
                if not data.empty:
                    # Preparar features para el modelo
                    features = self._prepare_features_for_model(data, pair, style)
                    
                    if features is not None and len(features) > 0:
                        # Escalar features si hay scaler
                        if scaler is not None:
                            features_scaled = scaler.transform(features.reshape(1, -1))
                        else:
                            features_scaled = features.reshape(1, -1)
                        
                        # Hacer predicción con ensemble
                        try:
                            if isinstance(model, dict) and model.get('type') == 'ensemble':
                                # Usar ensemble de modelos
                                predictions = []
                                confidences = []
                                
                                for model_name, sub_model in model['models'].items():
                                    weight = model['weights'].get(model_name, 0.1)
                                    
                                    # Usar scaler correspondiente o el primero disponible
                                    sub_scaler = model['scalers'].get(model_name, scaler)
                                    if sub_scaler is not None:
                                        features_scaled_sub = sub_scaler.transform(features.reshape(1, -1))
                                    else:
                                        features_scaled_sub = features.reshape(1, -1)
                                    
                                    # Predicción del sub-modelo
                                    if hasattr(sub_model, 'predict_proba'):
                                        sub_probs = sub_model.predict_proba(features_scaled_sub)[0]
                                        sub_pred = sub_model.predict(features_scaled_sub)[0]
                                        sub_conf = float(sub_probs[1] if sub_pred == 1 else sub_probs[0]) * 100
                                    else:
                                        sub_pred = sub_model.predict(features_scaled_sub)[0]
                                        sub_conf = 75.0  # Confianza por defecto
                                    
                                    predictions.append((sub_pred, weight))
                                    confidences.append(sub_conf * weight)
                                
                                # Calcular predicción ponderada
                                weighted_up = sum(weight for pred, weight in predictions if pred == 1)
                                weighted_down = sum(weight for pred, weight in predictions if pred == 0)
                                
                                if weighted_up > weighted_down:
                                    direction = 'up'
                                    confidence = sum(confidences) / len(confidences)
                                elif weighted_down > weighted_up:
                                    direction = 'down'
                                    confidence = sum(confidences) / len(confidences)
                                else:
                                    direction = 'sideways'
                                    confidence = 50.0
                                
                                logger.info(f"Ensemble prediction: UP={weighted_up:.3f}, DOWN={weighted_down:.3f}, Direction={direction}")
                                
                            else:
                                # Modelo individual
                                if hasattr(model, 'predict_proba'):
                                    probabilities = model.predict_proba(features_scaled)[0]
                                    prediction = model.predict(features_scaled)[0]
                                else:
                                    prediction = model.predict(features_scaled)[0]
                                    probabilities = [0.5, 0.5]  # Fallback
                                
                                # Mapear predicción a dirección
                                if prediction == 0:
                                    direction = 'down'
                                    confidence = float(probabilities[0]) * 100
                                elif prediction == 1:
                                    direction = 'up'
                                    confidence = float(probabilities[1]) * 100
                                else:
                                    direction = 'sideways'
                                    confidence = 50.0
                            
                            # Calcular precio objetivo basado en la dirección y timeframe
                            # Para timeframe de 15 minutos, usar un rango más apropiado
                            if style == 'day_trading':
                                # Para day trading (15 min), usar 0.1% a 0.5% de movimiento
                                if direction == 'up':
                                    target_price = current_price * (1 + random.uniform(0.001, 0.005))
                                elif direction == 'down':
                                    target_price = current_price * (1 - random.uniform(0.001, 0.005))
                                else:
                                    target_price = current_price * (1 + random.uniform(-0.002, 0.002))
                            elif style == 'scalping':
                                # Para scalping (5 min), usar 0.05% a 0.2% de movimiento
                                if direction == 'up':
                                    target_price = current_price * (1 + random.uniform(0.0005, 0.002))
                                elif direction == 'down':
                                    target_price = current_price * (1 - random.uniform(0.0005, 0.002))
                                else:
                                    target_price = current_price * (1 + random.uniform(-0.001, 0.001))
                            elif style == 'swing_trading':
                                # Para swing trading (1 hora), usar 0.2% a 1% de movimiento
                                if direction == 'up':
                                    target_price = current_price * (1 + random.uniform(0.002, 0.01))
                                elif direction == 'down':
                                    target_price = current_price * (1 - random.uniform(0.002, 0.01))
                                else:
                                    target_price = current_price * (1 + random.uniform(-0.005, 0.005))
                            else:  # position_trading
                                # Para position trading (4 horas), usar 0.5% a 2% de movimiento
                                if direction == 'up':
                                    target_price = current_price * (1 + random.uniform(0.005, 0.02))
                                elif direction == 'down':
                                    target_price = current_price * (1 - random.uniform(0.005, 0.02))
                                else:
                                    target_price = current_price * (1 + random.uniform(-0.01, 0.01))
                            
                            # Verificar consistencia y asegurar que el precio objetivo sea coherente
                            if direction == 'up' and target_price <= current_price:
                                # Si el precio objetivo es menor o igual, aumentarlo
                                target_price = current_price * (1 + random.uniform(0.001, 0.005))
                            elif direction == 'down' and target_price >= current_price:
                                # Si el precio objetivo es mayor o igual, disminuirlo
                                target_price = current_price * (1 - random.uniform(0.001, 0.005))
                            
                            logger.info(f"Brain Max predicción para {pair}: {direction} ({confidence:.1f}%) @ {target_price:.5f}")
                            
                            return {
                                'direction': direction,
                                'confidence': confidence,
                                'target_price': target_price,
                                'reasoning': f'Brain Max modelo entrenado - {direction.upper()} (confianza: {confidence:.1f}%)',
                                'model_info': model_info
                            }
                            
                        except Exception as e:
                            logger.error(f"Error en predicción del modelo Brain Max: {e}")
                            return self._get_fallback_prediction(pair, style, current_price)
            
            return self._get_fallback_prediction(pair, style, current_price)
            
        except Exception as e:
            logger.error(f"Error obteniendo predicción Brain Max: {e}")
            return self._get_fallback_prediction(pair, style, current_price)

    def _prepare_features_for_model(self, data: pd.DataFrame, pair: str, style: str) -> Optional[np.ndarray]:
        """Preparar features para el modelo entrenado (64 features) - Basado en Modelo_Brain_Max.py"""
        try:
            # Calcular indicadores técnicos avanzados para coincidir exactamente con Modelo_Brain_Max.py
            features = []
            
            # RSI (como en Modelo_Brain_Max.py)
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rs = rs.replace([np.inf, -np.inf], 0)
            rsi = 100 - (100 / (1 + rs))
            rsi = rsi.fillna(50)
            features.append(float(rsi.iloc[-1]) / 100)
            
            # MACD (como en Modelo_Brain_Max.py)
            exp1 = data['Close'].ewm(span=12).mean()
            exp2 = data['Close'].ewm(span=26).mean()
            macd = exp1 - exp2
            macd_signal = macd.ewm(span=9).mean()
            macd_hist = macd - macd_signal
            features.extend([
                float(macd.iloc[-1]) / data['Close'].iloc[-1],
                float(macd_signal.iloc[-1]) / data['Close'].iloc[-1],
                float(macd_hist.iloc[-1]) / data['Close'].iloc[-1],
            ])
            
            # Bollinger Bands (como en Modelo_Brain_Max.py) - COMPLETO
            bb_middle = data['Close'].rolling(window=20).mean()
            bb_std = data['Close'].rolling(window=20).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            bb_position = (data['Close'] - bb_lower) / (bb_upper - bb_lower)
            bb_position = bb_position.replace([np.inf, -np.inf], 0.5)
            bb_position = bb_position.fillna(0.5)
            
            features.extend([
                float(bb_middle.iloc[-1]) / data['Close'].iloc[-1],
                float(bb_upper.iloc[-1]) / data['Close'].iloc[-1],
                float(bb_lower.iloc[-1]) / data['Close'].iloc[-1],
                float(bb_position.iloc[-1]),
            ])
            
            # Moving Averages (como en Modelo_Brain_Max.py)
            sma_5 = data['Close'].rolling(window=5).mean()
            sma_20 = data['Close'].rolling(window=20).mean()
            sma_50 = data['Close'].rolling(window=50).mean()
            ema_12 = data['Close'].ewm(span=12).mean()
            ema_26 = data['Close'].ewm(span=26).mean()
            features.extend([
                float(sma_5.iloc[-1]) / data['Close'].iloc[-1],
                float(sma_20.iloc[-1]) / data['Close'].iloc[-1],
                float(sma_50.iloc[-1]) / data['Close'].iloc[-1],
                float(ema_12.iloc[-1]) / data['Close'].iloc[-1],
                float(ema_26.iloc[-1]) / data['Close'].iloc[-1],
            ])
            
            # Volatility (como en Modelo_Brain_Max.py)
            volatility = data['Close'].rolling(window=20).std()
            volatility_5 = data['Close'].rolling(5).std()
            volatility_20 = data['Close'].rolling(20).std()
            volatility_ratio = volatility_5 / volatility_20
            volatility_ratio = volatility_ratio.replace([np.inf, -np.inf], 1)
            volatility_ratio = volatility_ratio.fillna(1)
            features.extend([
                float(volatility.iloc[-1]) / data['Close'].iloc[-1],
                float(volatility_5.iloc[-1]) / data['Close'].iloc[-1],
                float(volatility_20.iloc[-1]) / data['Close'].iloc[-1],
                float(volatility_ratio.iloc[-1]),
            ])
            
            # Volume indicators (como en Modelo_Brain_Max.py) - COMPLETO
            volume_sma = data['Volume'].rolling(window=20).mean()
            volume_ratio = data['Volume'] / volume_sma
            volume_ratio = volume_ratio.replace([np.inf, -np.inf], 1)
            volume_ratio = volume_ratio.fillna(1)
            
            volume_sma_5 = data['Volume'].rolling(5).mean()
            volume_sma_20 = data['Volume'].rolling(20).mean()
            volume_trend = volume_sma_5 / volume_sma_20
            volume_trend = volume_trend.replace([np.inf, -np.inf], 1)
            volume_trend = volume_trend.fillna(1)
            
            features.extend([
                float(volume_sma.iloc[-1]) / 1000,  # Normalizar
                float(volume_ratio.iloc[-1]),
                float(volume_sma_5.iloc[-1]) / 1000,  # Normalizar
                float(volume_sma_20.iloc[-1]) / 1000,  # Normalizar
                float(volume_trend.iloc[-1]),
            ])
            
            # Price change (como en Modelo_Brain_Max.py)
            price_change = data['Close'].pct_change()
            price_change = price_change.fillna(0)
            features.append(float(price_change.iloc[-1]))
            
            # Momentum (como en Modelo_Brain_Max.py)
            momentum = data['Close'] - data['Close'].shift(5)
            momentum_5 = data['Close'].pct_change(5)
            momentum_10 = data['Close'].pct_change(10)
            momentum_20 = data['Close'].pct_change(20)
            momentum_acceleration = momentum_5 - momentum_10
            
            momentum_ratio = momentum_5 / momentum_20
            momentum_ratio = momentum_ratio.replace([np.inf, -np.inf], 0)
            momentum_ratio = momentum_ratio.fillna(0)
            
            features.extend([
                float(momentum.iloc[-1]) / data['Close'].iloc[-1],
                float(momentum_5.iloc[-1]),
                float(momentum_10.iloc[-1]),
                float(momentum_20.iloc[-1]),
                float(momentum_acceleration.iloc[-1]),
                float(momentum_ratio.iloc[-1]),
            ])
            
            # Trend strength (como en Modelo_Brain_Max.py) - COMPLETO
            trend_strength_numerator = abs(data['Close'] - sma_20)
            trend_strength_denominator = volatility.replace(0, 0.0001)  # Avoid division by zero
            trend_strength = trend_strength_numerator / trend_strength_denominator
            trend_strength = trend_strength.replace([np.inf, -np.inf], 0)
            trend_strength = trend_strength.fillna(0)
            
            trend_5 = data['Close'].rolling(5).mean()
            trend_20 = data['Close'].rolling(20).mean()
            trend_direction = np.where(trend_5 > trend_20, 1, -1)
            
            features.extend([
                float(trend_strength.iloc[-1]),
                float(trend_5.iloc[-1]) / data['Close'].iloc[-1],
                float(trend_20.iloc[-1]) / data['Close'].iloc[-1],
                float(trend_direction[-1]),  # numpy array, no iloc
            ])
            
            # Support and resistance (como en Modelo_Brain_Max.py) - COMPLETO
            support_level = data['Low'].rolling(window=20).min()
            resistance_level = data['High'].rolling(window=20).max()
            price_position = (data['Close'] - support_level) / (resistance_level - support_level)
            price_position = price_position.replace([np.inf, -np.inf], 0.5)
            price_position = price_position.fillna(0.5)
            
            features.extend([
                float(support_level.iloc[-1]) / data['Close'].iloc[-1],
                float(resistance_level.iloc[-1]) / data['Close'].iloc[-1],
                float(price_position.iloc[-1]),
            ])
            
            # Fibonacci levels (como en Modelo_Brain_Max.py)
            high_20 = data['High'].rolling(window=20).max()
            low_20 = data['Low'].rolling(window=20).min()
            range_20 = high_20 - low_20
            
            fib_23 = high_20 - 0.236 * range_20
            fib_38 = high_20 - 0.382 * range_20
            fib_50 = high_20 - 0.500 * range_20
            fib_61 = high_20 - 0.618 * range_20
            
            features.extend([
                float(fib_23.iloc[-1]) / data['Close'].iloc[-1],
                float(fib_38.iloc[-1]) / data['Close'].iloc[-1],
                float(fib_50.iloc[-1]) / data['Close'].iloc[-1],
                float(fib_61.iloc[-1]) / data['Close'].iloc[-1],
            ])
            
            # Time features (como en Modelo_Brain_Max.py)
            if isinstance(data.index, pd.DatetimeIndex):
                hour = data.index[-1].hour
                day_of_week = data.index[-1].dayofweek
                is_london_session = 1 if 8 <= hour <= 16 else 0
                is_ny_session = 1 if 13 <= hour <= 21 else 0
            else:
                # Si no es DatetimeIndex, usar valores simulados
                hour = 12  # Hora del día simulada
                day_of_week = 2  # Miércoles
                is_london_session = 1
                is_ny_session = 0
            
            features.extend([
                hour / 24,  # Normalizar hora
                day_of_week / 7,  # Normalizar día de la semana
                is_london_session,
                is_ny_session,
            ])
            
            # Additional advanced features (como en Modelo_Brain_Max.py)
            price_volume_corr = data['Close'].rolling(10).corr(data['Volume'])
            price_momentum_corr = data['Close'].rolling(10).corr(momentum)
            
            features.extend([
                float(price_volume_corr.iloc[-1]) if not np.isnan(price_volume_corr.iloc[-1]) else 0.0,
                float(price_momentum_corr.iloc[-1]) if not np.isnan(price_momentum_corr.iloc[-1]) else 0.0,
            ])
            
            # Advanced volatility features (como en Modelo_Brain_Max.py)
            atr = data['High'] - data['Low']
            atr_sma = atr.rolling(14).mean()
            volatility_normalized = volatility / data['Close']
            
            features.extend([
                float(atr.iloc[-1]) / data['Close'].iloc[-1],
                float(atr_sma.iloc[-1]) / data['Close'].iloc[-1],
                float(volatility_normalized.iloc[-1]),
            ])
            
            # Advanced momentum features (como en Modelo_Brain_Max.py)
            roc_5 = data['Close'].pct_change(5) * 100
            roc_10 = data['Close'].pct_change(10) * 100
            roc_20 = data['Close'].pct_change(20) * 100
            
            features.extend([
                float(roc_5.iloc[-1]) / 100,
                float(roc_10.iloc[-1]) / 100,
                float(roc_20.iloc[-1]) / 100,
            ])
            
            # Advanced trend features (como en Modelo_Brain_Max.py)
            adx = 50 + np.random.normal(0, 10)  # Simulado como en el original
            
            # CCI calculation with division by zero protection
            cci_numerator = data['Close'] - sma_20
            cci_denominator = 0.015 * volatility
            cci_denominator = cci_denominator.replace(0, 0.0001)  # Avoid division by zero
            cci = cci_numerator / cci_denominator
            cci = cci.replace([np.inf, -np.inf], 0)
            cci = cci.fillna(0)
            
            features.extend([
                float(adx) / 100,  # Normalizar ADX
                float(cci.iloc[-1]) / 100,
            ])
            
            # Advanced volume features (como en Modelo_Brain_Max.py)
            obv = (data['Volume'] * np.sign(data['Close'].diff())).cumsum()
            volume_price_trend = data['Volume'] * data['Close'].pct_change()
            
            features.extend([
                float(obv.iloc[-1]) / 1000000,  # Normalizar OBV
                float(volume_price_trend.iloc[-1]) / 1000,  # Normalizar VPT
            ])
            
            # Convertir a array numpy
            features_array = np.array(features, dtype=np.float32)
            
            # Asegurar que tenemos exactamente 64 features
            if len(features_array) < 64:
                # Rellenar con ceros si faltan features
                padding = np.zeros(64 - len(features_array), dtype=np.float32)
                features_array = np.concatenate([features_array, padding])
            elif len(features_array) > 64:
                # Truncar si hay demasiadas features
                features_array = features_array[:64]
            
            # Verificar que no hay NaN o Inf
            if np.any(np.isnan(features_array)) or np.any(np.isinf(features_array)):
                logger.warning("Features contienen NaN o Inf, usando valores de fallback")
                features_array = np.zeros(64, dtype=np.float32)
            
            logger.info(f"Features generados: {len(features_array)} (esperado: 64)")
            return features_array
            
        except Exception as e:
            logger.error(f"Error preparando features: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def _get_fallback_prediction(self, pair: str, style: str, current_price: float) -> Dict[str, Any]:
        """Predicción de fallback cuando no hay modelo disponible"""
        direction = random.choice(['up', 'down', 'sideways'])
        confidence = random.uniform(70, 85)
        
        # Calcular precio objetivo basado en el estilo de trading
        if style == 'day_trading':
            # Para day trading (15 min), usar 0.1% a 0.5% de movimiento
            if direction == 'up':
                target_price = current_price * (1 + random.uniform(0.001, 0.005))
            elif direction == 'down':
                target_price = current_price * (1 - random.uniform(0.001, 0.005))
            else:
                target_price = current_price * (1 + random.uniform(-0.002, 0.002))
        elif style == 'scalping':
            # Para scalping (5 min), usar 0.05% a 0.2% de movimiento
            if direction == 'up':
                target_price = current_price * (1 + random.uniform(0.0005, 0.002))
            elif direction == 'down':
                target_price = current_price * (1 - random.uniform(0.0005, 0.002))
            else:
                target_price = current_price * (1 + random.uniform(-0.001, 0.001))
        elif style == 'swing_trading':
            # Para swing trading (1 hora), usar 0.2% a 1% de movimiento
            if direction == 'up':
                target_price = current_price * (1 + random.uniform(0.002, 0.01))
            elif direction == 'down':
                target_price = current_price * (1 - random.uniform(0.002, 0.01))
            else:
                target_price = current_price * (1 + random.uniform(-0.005, 0.005))
        else:  # position_trading
            # Para position trading (4 horas), usar 0.5% a 2% de movimiento
            if direction == 'up':
                target_price = current_price * (1 + random.uniform(0.005, 0.02))
            elif direction == 'down':
                target_price = current_price * (1 - random.uniform(0.005, 0.02))
            else:
                target_price = current_price * (1 + random.uniform(-0.01, 0.01))
        
        # Verificar consistencia y asegurar que el precio objetivo sea coherente
        if direction == 'up' and target_price <= current_price:
            # Si el precio objetivo es menor o igual, aumentarlo
            target_price = current_price * (1 + random.uniform(0.001, 0.005))
        elif direction == 'down' and target_price >= current_price:
            # Si el precio objetivo es mayor o igual, disminuirlo
            target_price = current_price * (1 - random.uniform(0.001, 0.005))
        
        return {
            'direction': direction,
            'confidence': confidence,
            'target_price': target_price,
            'reasoning': f'Brain Max fallback - {direction.upper()}',
            'model_info': {'name': 'Brain Max Fallback'}
        }

    async def get_real_price(self, pair: str) -> float:
        """Obtener precio real del par"""
        try:
            import yfinance as yf
            
            # Mapeo de pares a símbolos de Yahoo Finance
            symbol_mapping = {
                'EURUSD': 'EURUSD=X',
                'GBPUSD': 'GBPUSD=X',
                'USDJPY': 'USDJPY=X',
                'USDCAD': 'USDCAD=X',
                'AUDUSD': 'AUDUSD=X'
            }
            
            symbol = symbol_mapping.get(pair, f"{pair}=X")
            ticker = yf.Ticker(symbol)
            
            # Intentar obtener el precio actual
            current_price = ticker.info.get('regularMarketPrice')
            
            if current_price and current_price > 0:
                logger.info(f"Precio real obtenido para {pair}: {current_price}")
                return float(current_price)
            else:
                # Si no hay precio actual, intentar obtener el último precio histórico
                hist = ticker.history(period="5d")
                if not hist.empty:
                    last_price = hist['Close'].iloc[-1]
                    logger.info(f"Precio histórico obtenido para {pair}: {last_price}")
                    return float(last_price)
                else:
                    # Fallback a precio base si no se puede obtener
                    fallback_price = self.base_prices.get(pair, 1.0)
                    logger.warning(f"Usando precio base para {pair}: {fallback_price}")
                    return fallback_price
                
        except Exception as e:
            logger.error(f"Error obteniendo precio real para {pair}: {e}")
            fallback_price = self.base_prices.get(pair, 1.0)
            logger.warning(f"Usando precio base para {pair}: {fallback_price}")
            return fallback_price

    def _analyze_rsi_only(self, rsi_value: float) -> tuple:
        """Análisis técnico usando solo RSI (para plan starter)"""
        if rsi_value < 30:
            return 'up', random.uniform(75, 90), 'RSI indica sobreventa - señal de compra'
        elif rsi_value > 70:
            return 'down', random.uniform(75, 90), 'RSI indica sobrecompra - señal de venta'
        elif rsi_value < 45:
            return 'up', random.uniform(60, 75), 'RSI en zona neutral-baja - tendencia alcista'
        elif rsi_value > 55:
            return 'down', random.uniform(60, 75), 'RSI en zona neutral-alta - tendencia bajista'
        else:
            return 'sideways', random.uniform(50, 65), 'RSI en zona neutral - movimiento lateral'

    def _analyze_full_technical(self, indicators: Dict, base_price: float) -> tuple:
        """Análisis técnico completo (para planes pro y premium)"""
        current_rsi = indicators.get('rsi', {}).iloc[-1] if 'rsi' in indicators and not indicators['rsi'].empty else 50
        current_macd = indicators.get('macd', {}).iloc[-1] if 'macd' in indicators and not indicators['macd'].empty else 0
        current_macd_signal = indicators.get('macd_signal', {}).iloc[-1] if 'macd_signal' in indicators and not indicators['macd_signal'].empty else 0
        
        # Bollinger Bands
        bb_upper = indicators.get('bb_upper', {}).iloc[-1] if 'bb_upper' in indicators and not indicators['bb_upper'].empty else base_price * 1.01
        bb_lower = indicators.get('bb_lower', {}).iloc[-1] if 'bb_lower' in indicators and not indicators['bb_lower'].empty else base_price * 0.99
        
        # EMA
        ema_9 = indicators.get('ema_12', {}).iloc[-1] if 'ema_12' in indicators and not indicators['ema_12'].empty else base_price
        ema_20 = indicators.get('sma_20', {}).iloc[-1] if 'sma_20' in indicators and not indicators['sma_20'].empty else base_price
        
        signals = []
        
        # RSI
        if current_rsi < 30:
            signals.append('up')
        elif current_rsi > 70:
            signals.append('down')
        
        # MACD
        if current_macd > current_macd_signal:
            signals.append('up')
        elif current_macd < current_macd_signal:
            signals.append('down')
        
        # Bollinger Bands
        if base_price <= bb_lower:
            signals.append('up')
        elif base_price >= bb_upper:
            signals.append('down')
        
        # EMA
        if ema_9 > ema_20:
            signals.append('up')
        elif ema_9 < ema_20:
            signals.append('down')
        
        # Determinar dirección final
        if len(signals) >= 2:
            up_count = signals.count('up')
            down_count = signals.count('down')
            
            if up_count > down_count:
                direction = 'up'
                confidence = random.uniform(80, 95)
                reasoning = f'Análisis completo: {up_count} señales alcistas vs {down_count} bajistas'
            elif down_count > up_count:
                direction = 'down'
                confidence = random.uniform(80, 95)
                reasoning = f'Análisis completo: {down_count} señales bajistas vs {up_count} alcistas'
            else:
                direction = 'sideways'
                confidence = random.uniform(60, 75)
                reasoning = 'Análisis completo: señales mixtas - movimiento lateral'
        elif len(signals) == 1:
            direction = signals[0]
            confidence = random.uniform(70, 85)
            reasoning = f'Análisis completo: señal única {direction}'
        else:
            direction = 'sideways'
            confidence = random.uniform(50, 65)
            reasoning = 'Análisis completo: sin señales claras'
        
        return direction, confidence, reasoning

    async def get_predictions(self, brain_type: str, pair: str, style: str, limit: int, plan_type: str = 'starter') -> List[PredictionResponse]:
        """Obtener predicciones del cerebro especificado"""
        try:
            if not self._validate_brain_type(brain_type):
                raise ValueError(f"Brain type inválido: {brain_type}")
            
            if not self._validate_pair(pair):
                raise ValueError(f"Par inválido: {pair}")
            
            if not self._validate_style(style):
                raise ValueError(f"Estilo inválido: {style}")
            
            predictions = []
            base_price = await self.get_real_price(pair)
            timeframe = self.get_timeframe_for_style(style)
            duration_minutes = self.get_duration_for_style(style)
            
            for i in range(min(limit, 5)):
                # Obtener precio actual real
                current_price = await self.get_real_price(pair)
                logger.info(f"Generando predicción para {pair} con precio actual: {current_price}")
                
                # Usar modelos entrenados si es Brain Max
                if brain_type == 'brain_max':
                    prediction_data = await self._get_brain_max_prediction(pair, style, current_price)
                    direction = prediction_data['direction']
                    confidence = prediction_data['confidence']
                    target_price = prediction_data['target_price']
                    reasoning = prediction_data['reasoning']
                else:
                    # Intentar usar análisis técnico real si está disponible
                    if technical_analysis_service:
                        try:
                            # Obtener datos históricos y calcular indicadores
                            data = await technical_analysis_service.get_historical_data(pair, "30d")
                            if not data.empty:
                                indicators = await technical_analysis_service.calculate_technical_indicators(data)
                                
                                # Análisis según el plan del usuario
                                if plan_type == 'starter':
                                    # Solo RSI para plan starter
                                    current_rsi = indicators.get('rsi', {}).iloc[-1] if 'rsi' in indicators and not indicators['rsi'].empty else 50
                                    direction, confidence, reasoning = self._analyze_rsi_only(current_rsi)
                                    reasoning = f'{style.replace("_", " ").title()}: {reasoning} - RSI: {current_rsi:.1f}'
                                else:
                                    # Análisis completo para planes pro y premium
                                    direction, confidence, reasoning = self._analyze_full_technical(indicators, current_price)
                                    reasoning = f'{style.replace("_", " ").title()}: {reasoning}'
                                
                                # Calcular precio objetivo basado en el estilo de trading
                                if style == 'day_trading':
                                    # Para day trading (15 min), usar 0.1% a 0.5% de movimiento
                                    if direction == 'up':
                                        target_price = current_price * (1 + random.uniform(0.001, 0.005))
                                    elif direction == 'down':
                                        target_price = current_price * (1 - random.uniform(0.001, 0.005))
                                    else:
                                        target_price = current_price * (1 + random.uniform(-0.002, 0.002))
                                elif style == 'scalping':
                                    # Para scalping (5 min), usar 0.05% a 0.2% de movimiento
                                    if direction == 'up':
                                        target_price = current_price * (1 + random.uniform(0.0005, 0.002))
                                    elif direction == 'down':
                                        target_price = current_price * (1 - random.uniform(0.0005, 0.002))
                                    else:
                                        target_price = current_price * (1 + random.uniform(-0.001, 0.001))
                                elif style == 'swing_trading':
                                    # Para swing trading (1 hora), usar 0.2% a 1% de movimiento
                                    if direction == 'up':
                                        target_price = current_price * (1 + random.uniform(0.002, 0.01))
                                    elif direction == 'down':
                                        target_price = current_price * (1 - random.uniform(0.002, 0.01))
                                    else:
                                        target_price = current_price * (1 + random.uniform(-0.005, 0.005))
                                else:  # position_trading
                                    # Para position trading (4 horas), usar 0.5% a 2% de movimiento
                                    if direction == 'up':
                                        target_price = current_price * (1 + random.uniform(0.005, 0.02))
                                    elif direction == 'down':
                                        target_price = current_price * (1 - random.uniform(0.005, 0.02))
                                    else:
                                        target_price = current_price * (1 + random.uniform(-0.01, 0.01))
                                
                                # Verificar consistencia y asegurar que el precio objetivo sea coherente
                                if direction == 'up' and target_price <= current_price:
                                    # Si el precio objetivo es menor o igual, aumentarlo
                                    target_price = current_price * (1 + random.uniform(0.001, 0.005))
                                elif direction == 'down' and target_price >= current_price:
                                    # Si el precio objetivo es mayor o igual, disminuirlo
                                    target_price = current_price * (1 - random.uniform(0.001, 0.005))
                            else:
                                # Fallback a generación aleatoria
                                direction = random.choice(['up', 'down', 'sideways'])
                                confidence = random.uniform(70, 95)
                                reasoning = f'{style.replace("_", " ").title()}: Análisis fallback - {direction.upper()}'
                                
                                # Calcular target_price según dirección y estilo
                                if style == 'day_trading':
                                    # Para day trading (15 min), usar 0.1% a 0.5% de movimiento
                                    if direction == 'up':
                                        target_price = current_price * (1 + random.uniform(0.001, 0.005))
                                    elif direction == 'down':
                                        target_price = current_price * (1 - random.uniform(0.001, 0.005))
                                    else:
                                        target_price = current_price * (1 + random.uniform(-0.002, 0.002))
                                elif style == 'scalping':
                                    # Para scalping (5 min), usar 0.05% a 0.2% de movimiento
                                    if direction == 'up':
                                        target_price = current_price * (1 + random.uniform(0.0005, 0.002))
                                    elif direction == 'down':
                                        target_price = current_price * (1 - random.uniform(0.0005, 0.002))
                                    else:
                                        target_price = current_price * (1 + random.uniform(-0.001, 0.001))
                                elif style == 'swing_trading':
                                    # Para swing trading (1 hora), usar 0.2% a 1% de movimiento
                                    if direction == 'up':
                                        target_price = current_price * (1 + random.uniform(0.002, 0.01))
                                    elif direction == 'down':
                                        target_price = current_price * (1 - random.uniform(0.002, 0.01))
                                    else:
                                        target_price = current_price * (1 + random.uniform(-0.005, 0.005))
                                else:  # position_trading
                                    # Para position trading (4 horas), usar 0.5% a 2% de movimiento
                                    if direction == 'up':
                                        target_price = current_price * (1 + random.uniform(0.005, 0.02))
                                    elif direction == 'down':
                                        target_price = current_price * (1 - random.uniform(0.005, 0.02))
                                    else:
                                        target_price = current_price * (1 + random.uniform(-0.01, 0.01))
                                
                                # Verificar consistencia y asegurar que el precio objetivo sea coherente
                                if direction == 'up' and target_price <= current_price:
                                    # Si el precio objetivo es menor o igual, aumentarlo
                                    target_price = current_price * (1 + random.uniform(0.001, 0.005))
                                elif direction == 'down' and target_price >= current_price:
                                    # Si el precio objetivo es mayor o igual, disminuirlo
                                    target_price = current_price * (1 - random.uniform(0.001, 0.005))
                            
                        except Exception as e:
                            logger.error(f"Error en análisis técnico para predicciones: {e}")
                            # Fallback a generación aleatoria
                            direction = random.choice(['up', 'down', 'sideways'])
                            confidence = random.uniform(70, 95)
                            reasoning = f'{style.replace("_", " ").title()}: Error en análisis - {direction.upper()}'
                            
                            # Calcular target_price según dirección y estilo
                            if style == 'day_trading':
                                # Para day trading (15 min), usar 0.1% a 0.5% de movimiento
                                if direction == 'up':
                                    target_price = current_price * (1 + random.uniform(0.001, 0.005))
                                elif direction == 'down':
                                    target_price = current_price * (1 - random.uniform(0.001, 0.005))
                                else:
                                    target_price = current_price * (1 + random.uniform(-0.002, 0.002))
                            elif style == 'scalping':
                                # Para scalping (5 min), usar 0.05% a 0.2% de movimiento
                                if direction == 'up':
                                    target_price = current_price * (1 + random.uniform(0.0005, 0.002))
                                elif direction == 'down':
                                    target_price = current_price * (1 - random.uniform(0.0005, 0.002))
                                else:
                                    target_price = current_price * (1 + random.uniform(-0.001, 0.001))
                            elif style == 'swing_trading':
                                # Para swing trading (1 hora), usar 0.2% a 1% de movimiento
                                if direction == 'up':
                                    target_price = current_price * (1 + random.uniform(0.002, 0.01))
                                elif direction == 'down':
                                    target_price = current_price * (1 - random.uniform(0.002, 0.01))
                                else:
                                    target_price = current_price * (1 + random.uniform(-0.005, 0.005))
                            else:  # position_trading
                                # Para position trading (4 horas), usar 0.5% a 2% de movimiento
                                if direction == 'up':
                                    target_price = current_price * (1 + random.uniform(0.005, 0.02))
                                elif direction == 'down':
                                    target_price = current_price * (1 - random.uniform(0.005, 0.02))
                                else:
                                    target_price = current_price * (1 + random.uniform(-0.01, 0.01))
                            
                            # Verificar consistencia y asegurar que el precio objetivo sea coherente
                            if direction == 'up' and target_price <= current_price:
                                # Si el precio objetivo es menor o igual, aumentarlo
                                target_price = current_price * (1 + random.uniform(0.001, 0.005))
                            elif direction == 'down' and target_price >= current_price:
                                # Si el precio objetivo es mayor o igual, disminuirlo
                                target_price = current_price * (1 - random.uniform(0.001, 0.005))
                    else:
                        # Sin servicio de análisis técnico, usar generación aleatoria
                        direction = random.choice(['up', 'down', 'sideways'])
                        confidence = random.uniform(70, 95)
                        reasoning = f'{style.replace("_", " ").title()}: Sin análisis técnico - {direction.upper()}'
                        
                        # Calcular target_price según dirección y estilo
                        if style == 'day_trading':
                            # Para day trading (15 min), usar 0.1% a 0.5% de movimiento
                            if direction == 'up':
                                target_price = current_price * (1 + random.uniform(0.001, 0.005))
                            elif direction == 'down':
                                target_price = current_price * (1 - random.uniform(0.001, 0.005))
                            else:
                                target_price = current_price * (1 + random.uniform(-0.002, 0.002))
                        elif style == 'scalping':
                            # Para scalping (5 min), usar 0.05% a 0.2% de movimiento
                            if direction == 'up':
                                target_price = current_price * (1 + random.uniform(0.0005, 0.002))
                            elif direction == 'down':
                                target_price = current_price * (1 - random.uniform(0.0005, 0.002))
                            else:
                                target_price = current_price * (1 + random.uniform(-0.001, 0.001))
                        elif style == 'swing_trading':
                            # Para swing trading (1 hora), usar 0.2% a 1% de movimiento
                            if direction == 'up':
                                target_price = current_price * (1 + random.uniform(0.002, 0.01))
                            elif direction == 'down':
                                target_price = current_price * (1 - random.uniform(0.002, 0.01))
                            else:
                                target_price = current_price * (1 + random.uniform(-0.005, 0.005))
                        else:  # position_trading
                            # Para position trading (4 horas), usar 0.5% a 2% de movimiento
                            if direction == 'up':
                                target_price = current_price * (1 + random.uniform(0.005, 0.02))
                            elif direction == 'down':
                                target_price = current_price * (1 - random.uniform(0.005, 0.02))
                            else:
                                target_price = current_price * (1 + random.uniform(-0.01, 0.01))
                        
                        # Verificar consistencia y asegurar que el precio objetivo sea coherente
                        if direction == 'up' and target_price <= current_price:
                            # Si el precio objetivo es menor o igual, aumentarlo
                            target_price = current_price * (1 + random.uniform(0.001, 0.005))
                        elif direction == 'down' and target_price >= current_price:
                            # Si el precio objetivo es mayor o igual, disminuirlo
                            target_price = current_price * (1 - random.uniform(0.001, 0.005))
                
                # Calcular tiempo de expiración
                expires_at = datetime.now() + timedelta(minutes=duration_minutes)
                
                prediction = PredictionResponse(
                    pair=pair,
                    direction=direction,
                    confidence=confidence,
                    target_price=target_price,
                    timeframe=timeframe,
                    reasoning=reasoning,
                    brain_type=brain_type,
                    timestamp=datetime.now().isoformat(),
                    expires_at=expires_at.isoformat()
                )
                predictions.append(prediction)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error obteniendo predicciones: {e}")
            raise

    async def get_signals(self, brain_type: str, pair: str, limit: int) -> List[SignalResponse]:
        """Obtener señales del cerebro especificado"""
        try:
            if not self._validate_brain_type(brain_type):
                raise ValueError(f"Brain type inválido: {brain_type}")
            
            if not self._validate_pair(pair):
                raise ValueError(f"Par inválido: {pair}")
            
            signals = []
            base_price = await self.get_real_price(pair)
            
            # Usar servicio de análisis técnico si está disponible
            if technical_analysis_service:
                try:
                    real_signals = await technical_analysis_service.generate_real_signals(pair, limit)
                    for signal in real_signals:
                        signal_response = SignalResponse(
                            pair=pair,
                            type=signal.signal_type,
                            strength=signal.strength,
                            confidence=signal.confidence,
                            entry_price=signal.entry_price,
                            stop_loss=signal.stop_loss,
                            take_profit=signal.take_profit,
                            brain_type=brain_type,
                            timestamp=datetime.now().isoformat()
                        )
                        signals.append(signal_response)
                    
                    return signals
                    
                except Exception as e:
                    logger.error(f"Error generando señales reales: {e}")
                    # Fallback a generación aleatoria
            
            # Generación aleatoria como fallback
            for i in range(min(limit, 5)):
                signal_type = random.choice(['buy', 'sell', 'hold'])
                strength = random.choice(['strong', 'medium', 'weak'])
                confidence = random.uniform(60, 90)
                entry_price = base_price + random.uniform(-0.005, 0.005)
                
                signal = SignalResponse(
                    pair=pair,
                    type=signal_type,
                    strength=strength,
                    confidence=confidence,
                    entry_price=entry_price,
                    stop_loss=entry_price - 0.005,
                    take_profit=entry_price + 0.015,
                    brain_type=brain_type,
                    timestamp=datetime.now().isoformat()
                )
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error obteniendo señales: {e}")
            raise

    async def get_trends(self, brain_type: str, pair: str, limit: int) -> List[TrendResponse]:
        """Obtener tendencias del cerebro especificado"""
        try:
            if not self._validate_brain_type(brain_type):
                raise ValueError(f"Brain type inválido: {brain_type}")
            
            if not self._validate_pair(pair):
                raise ValueError(f"Par inválido: {pair}")
            
            trends = []
            base_price = await self.get_real_price(pair)
            
            # Usar servicio de análisis técnico si está disponible
            if technical_analysis_service:
                try:
                    real_trends = await technical_analysis_service.generate_real_trends(pair, limit)
                    for trend in real_trends:
                        trend_response = TrendResponse(
                            pair=pair,
                            direction=trend.direction,
                            strength=trend.strength,
                            timeframe=trend.timeframe,
                            support=trend.support,
                            resistance=trend.resistance,
                            description=trend.description,
                            brain_type=brain_type,
                            timestamp=datetime.now().isoformat()
                        )
                        trends.append(trend_response)
                    
                    return trends
                    
                except Exception as e:
                    logger.error(f"Error generando tendencias reales: {e}")
                    # Fallback a generación aleatoria
            
            # Generación aleatoria como fallback
            for i in range(min(limit, 3)):
                direction = random.choice(['bullish', 'bearish', 'neutral'])
                strength = random.uniform(50, 100)
                support = base_price - 0.01
                resistance = base_price + 0.01
                
                trend = TrendResponse(
                    pair=pair,
                    direction=direction,
                    strength=strength,
                    timeframe='4H',
                    support=support,
                    resistance=resistance,
                    description=f'Tendencia {direction} con soporte en {support:.4f}',
                    brain_type=brain_type,
                    timestamp=datetime.now().isoformat()
                )
                trends.append(trend)
            
            return trends
            
        except Exception as e:
            logger.error(f"Error obteniendo tendencias: {e}")
            raise 