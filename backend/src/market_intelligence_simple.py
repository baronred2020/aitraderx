"""
Market Intelligence Module (Simplified)
======================================
Sistema inteligente para análisis dinámico del mercado (versión simplificada)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class MarketIntelligenceSimple:
    """
    Sistema de inteligencia de mercado simplificado que calcula dinámicamente:
    - Régimen de mercado (trending, ranging, volatile)
    - Nivel de riesgo (low, moderate, high)
    - Estrategia óptima
    - Pesos de modelos adaptativos
    """
    
    def __init__(self):
        self.symbol = "EURUSD=X"  # Yahoo Finance symbol for EURUSD
        self.cache_duration = 300  # 5 minutes cache
        self.last_update = None
        self.cached_data = None
        
    def get_market_data(self, symbol: str = "EURUSD=X", period: str = "30d", interval: str = "1h") -> pd.DataFrame:
        """Obtiene datos reales del mercado"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No se pudieron obtener datos para {symbol}")
                return pd.DataFrame()
                
            return data
        except Exception as e:
            logger.error(f"Error obteniendo datos de mercado: {e}")
            return pd.DataFrame()
    
    def calculate_volatility(self, data: pd.DataFrame, window: int = 20) -> float:
        """Calcula la volatilidad del mercado"""
        if data.empty:
            return 0.0
            
        try:
            # Calcular retornos logarítmicos
            returns = np.log(data['Close'] / data['Close'].shift(1))
            
            # Volatilidad anualizada
            volatility = returns.rolling(window=window).std().iloc[-1] * np.sqrt(252)
            
            return float(volatility)
        except Exception as e:
            logger.error(f"Error calculando volatilidad: {e}")
            return 0.0
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calcula RSI simplificado"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1])
        except:
            return 50.0
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[float, float, float]:
        """Calcula Bollinger Bands simplificado"""
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            return float(upper_band.iloc[-1]), float(sma.iloc[-1]), float(lower_band.iloc[-1])
        except:
            return 1.1, 1.085, 1.07
    
    def detect_market_regime(self, data: pd.DataFrame) -> str:
        """Detecta el régimen actual del mercado"""
        if data.empty:
            return "unknown"
            
        try:
            # Calcular indicadores técnicos
            close_prices = data['Close']
            
            # RSI
            current_rsi = self.calculate_rsi(close_prices)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close_prices)
            bb_width = (bb_upper - bb_lower) / bb_middle
            
            # Volatilidad
            volatility = self.calculate_volatility(data)
            
            # Tendencia simple
            short_ma = close_prices.rolling(window=10).mean().iloc[-1]
            long_ma = close_prices.rolling(window=30).mean().iloc[-1]
            trend_strength = abs(short_ma - long_ma) / long_ma
            
            # Lógica de detección de régimen
            if trend_strength > 0.02 and volatility > 0.15:
                return "trending"
            elif bb_width < 0.05 and volatility < 0.15:
                return "ranging"
            elif volatility > 0.25:
                return "volatile"
            elif current_rsi > 70 or current_rsi < 30:
                return "overbought_oversold"
            else:
                return "neutral"
                
        except Exception as e:
            logger.error(f"Error detectando régimen de mercado: {e}")
            return "unknown"
    
    def calculate_risk_level(self, data: pd.DataFrame) -> str:
        """Calcula el nivel de riesgo actual"""
        if data.empty:
            return "moderate"
            
        try:
            # Calcular métricas de riesgo
            volatility = self.calculate_volatility(data)
            
            # Drawdown máximo reciente
            close_prices = data['Close']
            rolling_max = close_prices.expanding().max()
            drawdown = (close_prices - rolling_max) / rolling_max
            max_drawdown = abs(drawdown.min())
            
            # Volatilidad de volumen
            volume_volatility = data['Volume'].pct_change().std()
            
            # Determinar nivel de riesgo
            risk_score = 0
            
            if volatility > 0.25:
                risk_score += 3
            elif volatility > 0.15:
                risk_score += 2
            elif volatility > 0.10:
                risk_score += 1
                
            if max_drawdown > 0.05:
                risk_score += 2
            elif max_drawdown > 0.03:
                risk_score += 1
                
            if volume_volatility > 0.5:
                risk_score += 1
                
            # Clasificar riesgo
            if risk_score >= 5:
                return "high"
            elif risk_score >= 3:
                return "moderate"
            else:
                return "low"
                
        except Exception as e:
            logger.error(f"Error calculando nivel de riesgo: {e}")
            return "moderate"
    
    def determine_optimal_strategy(self, market_regime: str, risk_level: str) -> str:
        """Determina la estrategia óptima basada en condiciones de mercado"""
        strategy_map = {
            "trending": {
                "low": "momentum_trending",
                "moderate": "trend_following",
                "high": "conservative_trending"
            },
            "ranging": {
                "low": "mean_reversion",
                "moderate": "range_trading",
                "high": "conservative_ranging"
            },
            "volatile": {
                "low": "volatility_breakout",
                "moderate": "adaptive_volatility",
                "high": "defensive_volatility"
            },
            "overbought_oversold": {
                "low": "contrarian",
                "moderate": "momentum_reversal",
                "high": "conservative_reversal"
            },
            "neutral": {
                "low": "balanced",
                "moderate": "adaptive_ensemble",
                "high": "defensive_ensemble"
            }
        }
        
        return strategy_map.get(market_regime, {}).get(risk_level, "ensemble_optimization")
    
    def calculate_adaptive_weights(self, data: pd.DataFrame, market_regime: str) -> Dict[str, float]:
        """Calcula pesos adaptativos para los modelos basándose en condiciones de mercado"""
        try:
            # Obtener métricas de rendimiento reciente (simuladas)
            recent_performance = self.get_recent_model_performance(data)
            
            # Pesos base por régimen de mercado
            base_weights = {
                "trending": {
                    "brain_max": 0.40,      # Mejor para tendencias
                    "brain_ultra": 0.30,
                    "brain_predictor": 0.20,
                    "megamind": 0.10
                },
                "ranging": {
                    "brain_max": 0.25,
                    "brain_ultra": 0.35,    # Mejor para rangos
                    "brain_predictor": 0.25,
                    "megamind": 0.15
                },
                "volatile": {
                    "brain_max": 0.20,
                    "brain_ultra": 0.25,
                    "brain_predictor": 0.30, # Mejor para volatilidad
                    "megamind": 0.25        # Más conservador
                },
                "overbought_oversold": {
                    "brain_max": 0.30,
                    "brain_ultra": 0.25,
                    "brain_predictor": 0.25,
                    "megamind": 0.20
                },
                "neutral": {
                    "brain_max": 0.35,
                    "brain_ultra": 0.30,
                    "brain_predictor": 0.25,
                    "megamind": 0.10
                }
            }
            
            # Obtener pesos base
            weights = base_weights.get(market_regime, base_weights["neutral"])
            
            # Ajustar basándose en rendimiento reciente
            for model, performance in recent_performance.items():
                if performance > 0.7:  # Buen rendimiento
                    weights[model] *= 1.1
                elif performance < 0.5:  # Mal rendimiento
                    weights[model] *= 0.9
            
            # Normalizar pesos
            total_weight = sum(weights.values())
            normalized_weights = {k: v/total_weight for k, v in weights.items()}
            
            return normalized_weights
            
        except Exception as e:
            logger.error(f"Error calculando pesos adaptativos: {e}")
            return {
                "brain_max": 0.35,
                "brain_ultra": 0.30,
                "brain_predictor": 0.25,
                "megamind": 0.10
            }
    
    def get_recent_model_performance(self, data: pd.DataFrame) -> Dict[str, float]:
        """Simula el rendimiento reciente de los modelos (en producción sería real)"""
        try:
            # En producción, esto vendría de la base de datos de rendimiento real
            # Por ahora, simulamos basándonos en condiciones de mercado
            
            volatility = self.calculate_volatility(data)
            rsi = self.calculate_rsi(data['Close'])
            
            # Simular rendimiento basado en condiciones
            base_performance = {
                "brain_max": 0.68,
                "brain_ultra": 0.71,
                "brain_predictor": 0.65,
                "megamind": 0.73
            }
            
            # Ajustar basándose en volatilidad
            if volatility > 0.20:
                base_performance["megamind"] += 0.05  # Mejor en alta volatilidad
                base_performance["brain_predictor"] += 0.03
            elif volatility < 0.10:
                base_performance["brain_max"] += 0.05  # Mejor en baja volatilidad
                base_performance["brain_ultra"] += 0.03
            
            # Ajustar basándose en RSI
            if rsi > 70 or rsi < 30:
                base_performance["brain_predictor"] += 0.02  # Mejor en extremos
            
            return base_performance
            
        except Exception as e:
            logger.error(f"Error obteniendo rendimiento de modelos: {e}")
            return {
                "brain_max": 0.68,
                "brain_ultra": 0.71,
                "brain_predictor": 0.65,
                "megamind": 0.73
            }
    
    def calculate_dynamic_risk_levels(self, entry_price: float, signal: str, data: pd.DataFrame) -> Tuple[float, float]:
        """Calcula niveles de stop loss y take profit dinámicamente"""
        try:
            # Calcular volatilidad actual
            volatility = self.calculate_volatility(data)
            
            # Calcular ATR simplificado (Average True Range)
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift())
            low_close = np.abs(data['Low'] - data['Close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean().iloc[-1]
            
            # Multiplicadores basados en volatilidad
            if volatility > 0.25:  # Alta volatilidad
                sl_multiplier = 2.5
                tp_multiplier = 3.0
            elif volatility > 0.15:  # Volatilidad media
                sl_multiplier = 2.0
                tp_multiplier = 2.5
            else:  # Baja volatilidad
                sl_multiplier = 1.5
                tp_multiplier = 2.0
            
            # Calcular niveles
            if signal.lower() == "buy":
                stop_loss = entry_price - (atr * sl_multiplier)
                take_profit = entry_price + (atr * tp_multiplier)
            else:  # sell
                stop_loss = entry_price + (atr * sl_multiplier)
                take_profit = entry_price - (atr * tp_multiplier)
            
            return float(stop_loss), float(take_profit)
            
        except Exception as e:
            logger.error(f"Error calculando niveles de riesgo dinámicos: {e}")
            # Fallback a niveles fijos
            if signal.lower() == "buy":
                return entry_price * 0.995, entry_price * 1.015
            else:
                return entry_price * 1.005, entry_price * 0.985
    
    def get_market_intelligence(self) -> Dict:
        """Obtiene toda la inteligencia de mercado en un solo call"""
        try:
            # Verificar cache
            if (self.last_update and 
                (datetime.now() - self.last_update).seconds < self.cache_duration and 
                self.cached_data):
                return self.cached_data
            
            # Obtener datos actuales
            data = self.get_market_data()
            
            if data.empty:
                return self.get_fallback_intelligence()
            
            # Calcular métricas
            market_regime = self.detect_market_regime(data)
            risk_level = self.calculate_risk_level(data)
            optimal_strategy = self.determine_optimal_strategy(market_regime, risk_level)
            adaptive_weights = self.calculate_adaptive_weights(data, market_regime)
            
            # Crear respuesta
            intelligence = {
                "market_regime": market_regime,
                "risk_level": risk_level,
                "current_strategy": optimal_strategy,
                "model_coordination": {
                    "brain_max_weight": adaptive_weights["brain_max"],
                    "brain_ultra_weight": adaptive_weights["brain_ultra"],
                    "brain_predictor_weight": adaptive_weights["brain_predictor"],
                    "megamind_weight": adaptive_weights["megamind"]
                },
                "market_metrics": {
                    "volatility": self.calculate_volatility(data),
                    "current_price": float(data['Close'].iloc[-1]),
                    "volume": float(data['Volume'].iloc[-1]),
                    "timestamp": datetime.now().isoformat()
                },
                "last_updated": datetime.now().isoformat()
            }
            
            # Actualizar cache
            self.cached_data = intelligence
            self.last_update = datetime.now()
            
            return intelligence
            
        except Exception as e:
            logger.error(f"Error obteniendo inteligencia de mercado: {e}")
            return self.get_fallback_intelligence()
    
    def get_fallback_intelligence(self) -> Dict:
        """Inteligencia de fallback cuando no hay datos"""
        return {
            "market_regime": "neutral",
            "risk_level": "moderate",
            "current_strategy": "ensemble_optimization",
            "model_coordination": {
                "brain_max_weight": 0.35,
                "brain_ultra_weight": 0.30,
                "brain_predictor_weight": 0.25,
                "megamind_weight": 0.10
            },
            "market_metrics": {
                "volatility": 0.15,
                "current_price": 1.0850,
                "volume": 1000000,
                "timestamp": datetime.now().isoformat()
            },
            "last_updated": datetime.now().isoformat()
        }

# Instancia global
market_intelligence = MarketIntelligenceSimple() 