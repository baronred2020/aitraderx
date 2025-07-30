"""
Technical Analysis Service
=========================
Servicio para calcular indicadores técnicos y generar señales reales
"""

import asyncio
import logging
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TechnicalSignal:
    signal_type: str
    strength: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    reasoning: str

@dataclass
class TrendAnalysis:
    direction: str
    strength: float
    timeframe: str
    support: float
    resistance: float
    description: str

class TechnicalAnalysisService:
    def __init__(self):
        self.symbol_mapping = {
            'EURUSD': 'EURUSD=X',
            'GBPUSD': 'GBPUSD=X',
            'USDJPY': 'USDJPY=X',
            'USDCAD': 'USDCAD=X',
            'AUDUSD': 'AUDUSD=X'
        }

    async def get_historical_data(self, pair: str, period: str = "30d") -> pd.DataFrame:
        """Obtener datos históricos usando yfinance"""
        try:
            symbol = self.symbol_mapping.get(pair, f"{pair}=X")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            return data
        except Exception as e:
            logger.error(f"Error obteniendo datos históricos para {pair}: {e}")
            return pd.DataFrame()

    async def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calcular indicadores técnicos usando pandas"""
        indicators = {}
        
        if data.empty:
            return indicators
            
        try:
            # RSI
            indicators['rsi'] = self._calculate_rsi(data['Close'])
            
            # MACD
            macd, macd_signal, macd_hist = self._calculate_macd(data['Close'])
            indicators['macd'] = macd
            indicators['macd_signal'] = macd_signal
            indicators['macd_histogram'] = macd_hist
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(data['Close'])
            indicators['bb_upper'] = bb_upper
            indicators['bb_middle'] = bb_middle
            indicators['bb_lower'] = bb_lower
            
            # Moving Averages
            indicators['sma_20'] = self._calculate_sma(data['Close'], 20)
            indicators['sma_50'] = self._calculate_sma(data['Close'], 50)
            indicators['ema_12'] = self._calculate_ema(data['Close'], 12)
            indicators['ema_26'] = self._calculate_ema(data['Close'], 26)
            
            # Stochastic
            stoch_k, stoch_d = self._calculate_stochastic(data)
            indicators['stoch_k'] = stoch_k
            indicators['stoch_d'] = stoch_d
            
            # ADX
            indicators['adx'] = self._calculate_adx(data)
            
        except Exception as e:
            logger.error(f"Error calculando indicadores técnicos: {e}")
            
        return indicators

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcular RSI usando pandas"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calcular MACD usando pandas"""
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        macd = ema_fast - ema_slow
        macd_signal = self._calculate_ema(macd, signal)
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calcular Bollinger Bands usando pandas"""
        sma = self._calculate_sma(prices, period)
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band

    def _calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
        """Calcular Simple Moving Average"""
        return prices.rolling(window=period).mean()

    def _calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calcular Exponential Moving Average"""
        return prices.ewm(span=period).mean()

    def _calculate_stochastic(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calcular Stochastic Oscillator"""
        low_min = data['Low'].rolling(window=k_period).min()
        high_max = data['High'].rolling(window=k_period).max()
        k_percent = 100 * ((data['Close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent

    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calcular Average Directional Index"""
        high_diff = data['High'].diff()
        low_diff = data['Low'].diff()
        
        plus_dm = high_diff.where((high_diff > 0) & (high_diff > low_diff.abs()), 0)
        minus_dm = -low_diff.where((low_diff < 0) & (low_diff.abs() > high_diff), 0)
        
        tr = pd.concat([
            data['High'] - data['Low'],
            (data['High'] - data['Close'].shift()).abs(),
            (data['Low'] - data['Close'].shift()).abs()
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
        adx = dx.rolling(window=period).mean()
        
        return adx

    async def generate_real_signals(self, pair: str, limit: int = 10) -> List[TechnicalSignal]:
        """Generar señales reales basadas en análisis técnico"""
        try:
            data = await self.get_historical_data(pair, "30d")
            if data.empty:
                return []
                
            indicators = await self.calculate_technical_indicators(data)
            if not indicators:
                return []
                
            signals = []
            current_price = data['Close'].iloc[-1]
            
            # Obtener valores actuales de los indicadores
            rsi = indicators['rsi'].iloc[-1] if 'rsi' in indicators and not indicators['rsi'].empty else 50
            macd = indicators['macd'].iloc[-1] if 'macd' in indicators and not indicators['macd'].empty else 0
            macd_signal = indicators['macd_signal'].iloc[-1] if 'macd_signal' in indicators and not indicators['macd_signal'].empty else 0
            bb_upper = indicators['bb_upper'].iloc[-1] if 'bb_upper' in indicators and not indicators['bb_upper'].empty else current_price * 1.02
            bb_lower = indicators['bb_lower'].iloc[-1] if 'bb_lower' in indicators and not indicators['bb_lower'].empty else current_price * 0.98
            sma_20 = indicators['sma_20'].iloc[-1] if 'sma_20' in indicators and not indicators['sma_20'].empty else current_price
            sma_50 = indicators['sma_50'].iloc[-1] if 'sma_50' in indicators and not indicators['sma_50'].empty else current_price
            stoch_k = indicators['stoch_k'].iloc[-1] if 'stoch_k' in indicators and not indicators['stoch_k'].empty else 50
            adx = indicators['adx'].iloc[-1] if 'adx' in indicators and not indicators['adx'].empty else 25
            
            for i in range(min(limit, 5)):
                signal_type, strength, confidence, reasoning = self._analyze_technical_signals(
                    current_price, rsi, macd, macd_signal, bb_upper, bb_lower, sma_20, sma_50, stoch_k, adx
                )
                
                entry_price = current_price
                stop_loss = self._calculate_stop_loss(signal_type, current_price, bb_lower, bb_upper)
                take_profit = self._calculate_take_profit(signal_type, current_price, bb_lower, bb_upper)
                
                signal = TechnicalSignal(
                    signal_type=signal_type,
                    strength=strength,
                    confidence=confidence,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reasoning=reasoning
                )
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generando señales reales: {e}")
            return []

    async def generate_real_trends(self, pair: str, limit: int = 10) -> List[TrendAnalysis]:
        """Generar tendencias reales basadas en análisis técnico"""
        try:
            data = await self.get_historical_data(pair, "30d")
            if data.empty:
                return []
                
            indicators = await self.calculate_technical_indicators(data)
            if not indicators:
                return []
                
            trends = []
            current_price = data['Close'].iloc[-1]
            
            # Obtener valores actuales de los indicadores
            sma_20 = indicators['sma_20'].iloc[-1] if 'sma_20' in indicators and not indicators['sma_20'].empty else current_price
            sma_50 = indicators['sma_50'].iloc[-1] if 'sma_50' in indicators and not indicators['sma_50'].empty else current_price
            ema_12 = indicators['ema_12'].iloc[-1] if 'ema_12' in indicators and not indicators['ema_12'].empty else current_price
            ema_26 = indicators['ema_26'].iloc[-1] if 'ema_26' in indicators and not indicators['ema_26'].empty else current_price
            adx = indicators['adx'].iloc[-1] if 'adx' in indicators and not indicators['adx'].empty else 25
            
            # INTEGRAR INFORMACIÓN DEL BRAIN MAX PARA TENDENCIAS MÁS PRECISAS
            brain_max_trend_info = await self._get_brain_max_trend_analysis(data, indicators)
            
            # Generar solo 1 tendencia con análisis mejorado
            direction, strength, description = self._analyze_trend_direction_enhanced(
                current_price, sma_20, sma_50, ema_12, ema_26, adx, brain_max_trend_info
            )
            
            support, resistance = self._calculate_support_resistance(data, indicators)
            
            trend = TrendAnalysis(
                direction=direction,
                strength=strength,
                timeframe='1D',
                support=support,
                resistance=resistance,
                description=description
            )
            trends.append(trend)
            
            return trends
            
        except Exception as e:
            logger.error(f"Error generando tendencias reales: {e}")
            return []

    async def _get_brain_max_trend_analysis(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Obtener análisis de tendencia basado en las features del Brain Max"""
        try:
            # Calcular features avanzadas como en Brain Max
            brain_max_features = {}
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rs = rs.replace([np.inf, -np.inf], 0)
            rsi = 100 - (100 / (1 + rs))
            rsi = rsi.fillna(50)
            brain_max_features['rsi'] = float(rsi.iloc[-1])
            
            # MACD
            exp1 = data['Close'].ewm(span=12).mean()
            exp2 = data['Close'].ewm(span=26).mean()
            macd = exp1 - exp2
            macd_signal = macd.ewm(span=9).mean()
            macd_hist = macd - macd_signal
            brain_max_features['macd'] = float(macd.iloc[-1])
            brain_max_features['macd_signal'] = float(macd_signal.iloc[-1])
            brain_max_features['macd_hist'] = float(macd_hist.iloc[-1])
            
            # Bollinger Bands
            bb_middle = data['Close'].rolling(window=20).mean()
            bb_std = data['Close'].rolling(window=20).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            bb_position = (data['Close'] - bb_lower) / (bb_upper - bb_lower)
            bb_position = bb_position.replace([np.inf, -np.inf], 0.5)
            bb_position = bb_position.fillna(0.5)
            
            brain_max_features['bb_position'] = float(bb_position.iloc[-1])
            brain_max_features['bb_upper'] = float(bb_upper.iloc[-1])
            brain_max_features['bb_lower'] = float(bb_lower.iloc[-1])
            
            # Moving Averages
            sma_5 = data['Close'].rolling(window=5).mean()
            sma_20 = data['Close'].rolling(window=20).mean()
            sma_50 = data['Close'].rolling(window=50).mean()
            ema_12 = data['Close'].ewm(span=12).mean()
            ema_26 = data['Close'].ewm(span=26).mean()
            
            brain_max_features['sma_5'] = float(sma_5.iloc[-1])
            brain_max_features['sma_20'] = float(sma_20.iloc[-1])
            brain_max_features['sma_50'] = float(sma_50.iloc[-1])
            brain_max_features['ema_12'] = float(ema_12.iloc[-1])
            brain_max_features['ema_26'] = float(ema_26.iloc[-1])
            
            # Volatility
            volatility = data['Close'].rolling(window=20).std()
            volatility_5 = data['Close'].rolling(5).std()
            volatility_20 = data['Close'].rolling(20).std()
            volatility_ratio = volatility_5 / volatility_20
            volatility_ratio = volatility_ratio.replace([np.inf, -np.inf], 1)
            volatility_ratio = volatility_ratio.fillna(1)
            
            brain_max_features['volatility'] = float(volatility.iloc[-1])
            brain_max_features['volatility_ratio'] = float(volatility_ratio.iloc[-1])
            
            # Momentum
            momentum = data['Close'] - data['Close'].shift(5)
            momentum_5 = data['Close'].pct_change(5)
            momentum_10 = data['Close'].pct_change(10)
            momentum_20 = data['Close'].pct_change(20)
            momentum_acceleration = momentum_5 - momentum_10
            
            brain_max_features['momentum'] = float(momentum.iloc[-1])
            brain_max_features['momentum_5'] = float(momentum_5.iloc[-1])
            brain_max_features['momentum_acceleration'] = float(momentum_acceleration.iloc[-1])
            
            # Trend strength (como en Brain Max)
            trend_strength = abs(data['Close'].iloc[-1] - sma_20.iloc[-1]) / volatility.iloc[-1]
            trend_strength = trend_strength if not np.isinf(trend_strength) else 0
            brain_max_features['trend_strength'] = float(trend_strength)
            
            # Trend direction (como en Brain Max)
            trend_direction = 1 if sma_20.iloc[-1] > sma_50.iloc[-1] else -1
            brain_max_features['trend_direction'] = trend_direction
            
            # Support and resistance
            support_level = data['Low'].rolling(window=20).min()
            resistance_level = data['High'].rolling(window=20).max()
            price_position = (data['Close'].iloc[-1] - support_level.iloc[-1]) / (resistance_level.iloc[-1] - support_level.iloc[-1])
            price_position = price_position if not np.isnan(price_position) else 0.5
            
            brain_max_features['support_level'] = float(support_level.iloc[-1])
            brain_max_features['resistance_level'] = float(resistance_level.iloc[-1])
            brain_max_features['price_position'] = float(price_position)
            
            return brain_max_features
            
        except Exception as e:
            logger.error(f"Error obteniendo análisis Brain Max: {e}")
            return {}

    def _analyze_technical_signals(self, price: float, rsi: float, macd: float, macd_signal: float, 
                                 bb_upper: float, bb_lower: float, sma_20: float, sma_50: float, 
                                 stoch_k: float, adx: float) -> Tuple[str, str, float, str]:
        """Analizar señales técnicas y determinar tipo, fuerza, confianza y razonamiento"""
        
        # Determinar tipo de señal
        signal_type = 'hold'
        strength = 'weak'
        confidence = 50.0
        reasoning_parts = []
        
        # Análisis RSI
        if rsi < 30:
            signal_type = 'buy'
            strength = 'strong' if rsi < 20 else 'medium'
            confidence += 20
            reasoning_parts.append("RSI sobrevendido")
        elif rsi > 70:
            signal_type = 'sell'
            strength = 'strong' if rsi > 80 else 'medium'
            confidence += 20
            reasoning_parts.append("RSI sobrecomprado")
        
        # Análisis MACD
        if macd > macd_signal:
            if signal_type == 'buy':
                confidence += 15
                reasoning_parts.append("MACD alcista")
            elif signal_type == 'hold':
                signal_type = 'buy'
                strength = 'medium'
                confidence += 10
                reasoning_parts.append("MACD alcista")
        elif macd < macd_signal:
            if signal_type == 'sell':
                confidence += 15
                reasoning_parts.append("MACD bajista")
            elif signal_type == 'hold':
                signal_type = 'sell'
                strength = 'medium'
                confidence += 10
                reasoning_parts.append("MACD bajista")
        
        # Análisis Bollinger Bands
        if price < bb_lower:
            if signal_type == 'buy':
                confidence += 10
                reasoning_parts.append("Precio bajo BB")
            elif signal_type == 'hold':
                signal_type = 'buy'
                strength = 'medium'
                confidence += 15
                reasoning_parts.append("Precio bajo BB")
        elif price > bb_upper:
            if signal_type == 'sell':
                confidence += 10
                reasoning_parts.append("Precio sobre BB")
            elif signal_type == 'hold':
                signal_type = 'sell'
                strength = 'medium'
                confidence += 15
                reasoning_parts.append("Precio sobre BB")
        
        # Análisis Moving Averages
        if sma_20 > sma_50:
            if signal_type == 'buy':
                confidence += 5
                reasoning_parts.append("MA alcista")
        elif sma_20 < sma_50:
            if signal_type == 'sell':
                confidence += 5
                reasoning_parts.append("MA bajista")
        
        # Análisis Stochastic
        if stoch_k < 20:
            if signal_type == 'buy':
                confidence += 5
                reasoning_parts.append("Stoch sobrevendido")
        elif stoch_k > 80:
            if signal_type == 'sell':
                confidence += 5
                reasoning_parts.append("Stoch sobrecomprado")
        
        # Análisis ADX (fuerza de la tendencia)
        if adx > 25:
            confidence += 10
            reasoning_parts.append("Tendencia fuerte")
        
        # Limitar confianza
        confidence = min(confidence, 95.0)
        
        reasoning = " + ".join(reasoning_parts) if reasoning_parts else "Análisis neutro"
        
        return signal_type, strength, confidence, reasoning

    def _analyze_trend_direction(self, price: float, sma_20: float, sma_50: float, 
                                ema_12: float, ema_26: float, adx: float) -> Tuple[str, float, str]:
        """Analizar dirección de la tendencia"""
        
        direction = 'neutral'
        strength = 50.0
        description_parts = []
        
        # Análisis de Moving Averages
        if sma_20 > sma_50 and ema_12 > ema_26:
            direction = 'bullish'
            strength += 20
            description_parts.append("MA alcista")
        elif sma_20 < sma_50 and ema_12 < ema_26:
            direction = 'bearish'
            strength += 20
            description_parts.append("MA bajista")
        
        # Análisis de precio vs MA
        if price > sma_20:
            if direction == 'bullish':
                strength += 10
                description_parts.append("Precio sobre MA20")
        elif price < sma_20:
            if direction == 'bearish':
                strength += 10
                description_parts.append("Precio bajo MA20")
        
        # Análisis ADX
        if adx > 25:
            strength += 15
            description_parts.append("Tendencia fuerte")
        elif adx < 20:
            strength -= 10
            description_parts.append("Tendencia débil")
        
        # Limitar fuerza
        strength = max(min(strength, 100.0), 0.0)
        
        description = " + ".join(description_parts) if description_parts else "Tendencia neutra"
        
        return direction, strength, description

    def _analyze_trend_direction_enhanced(self, price: float, sma_20: float, sma_50: float, 
                                        ema_12: float, ema_26: float, adx: float, 
                                        brain_max_info: Dict[str, Any]) -> Tuple[str, float, str]:
        """Analizar dirección de la tendencia con información mejorada del Brain Max"""
        
        direction = 'neutral'
        strength = 50.0
        description_parts = []
        
        # Análisis básico de Moving Averages
        if sma_20 > sma_50 and ema_12 > ema_26:
            direction = 'bullish'
            strength += 20
            description_parts.append("MA alcista")
        elif sma_20 < sma_50 and ema_12 < ema_26:
            direction = 'bearish'
            strength += 20
            description_parts.append("MA bajista")
        
        # Análisis de precio vs MA
        if price > sma_20:
            if direction == 'bullish':
                strength += 10
                description_parts.append("Precio sobre MA20")
        elif price < sma_20:
            if direction == 'bearish':
                strength += 10
                description_parts.append("Precio bajo MA20")
        
        # Análisis ADX
        if adx > 25:
            strength += 15
            description_parts.append("Tendencia fuerte")
        elif adx < 20:
            strength -= 10
            description_parts.append("Tendencia débil")
        
        # INTEGRAR INFORMACIÓN DEL BRAIN MAX
        if brain_max_info:
            # Análisis de RSI del Brain Max
            rsi = brain_max_info.get('rsi', 50)
            if rsi < 30:
                if direction == 'bullish':
                    strength += 15
                    description_parts.append("RSI sobrevendido + MA alcista")
                elif direction == 'neutral':
                    direction = 'bullish'
                    strength += 10
                    description_parts.append("RSI sobrevendido")
            elif rsi > 70:
                if direction == 'bearish':
                    strength += 15
                    description_parts.append("RSI sobrecomprado + MA bajista")
                elif direction == 'neutral':
                    direction = 'bearish'
                    strength += 10
                    description_parts.append("RSI sobrecomprado")
            
            # Análisis de MACD del Brain Max
            macd = brain_max_info.get('macd', 0)
            macd_signal = brain_max_info.get('macd_signal', 0)
            if macd > macd_signal:
                if direction == 'bullish':
                    strength += 10
                    description_parts.append("MACD alcista")
                elif direction == 'neutral':
                    direction = 'bullish'
                    strength += 5
                    description_parts.append("MACD alcista")
            elif macd < macd_signal:
                if direction == 'bearish':
                    strength += 10
                    description_parts.append("MACD bajista")
                elif direction == 'neutral':
                    direction = 'bearish'
                    strength += 5
                    description_parts.append("MACD bajista")
            
            # Análisis de Bollinger Bands del Brain Max
            bb_position = brain_max_info.get('bb_position', 0.5)
            if bb_position < 0.2:  # Precio cerca de la banda inferior
                if direction == 'bullish':
                    strength += 10
                    description_parts.append("Precio bajo BB + MA alcista")
                elif direction == 'neutral':
                    direction = 'bullish'
                    strength += 5
                    description_parts.append("Precio bajo BB")
            elif bb_position > 0.8:  # Precio cerca de la banda superior
                if direction == 'bearish':
                    strength += 10
                    description_parts.append("Precio alto BB + MA bajista")
                elif direction == 'neutral':
                    direction = 'bearish'
                    strength += 5
                    description_parts.append("Precio alto BB")
            
            # Análisis de momentum del Brain Max
            momentum = brain_max_info.get('momentum', 0)
            momentum_acceleration = brain_max_info.get('momentum_acceleration', 0)
            if momentum > 0 and momentum_acceleration > 0:
                if direction == 'bullish':
                    strength += 8
                    description_parts.append("Momentum alcista acelerando")
            elif momentum < 0 and momentum_acceleration < 0:
                if direction == 'bearish':
                    strength += 8
                    description_parts.append("Momentum bajista acelerando")
            
            # Análisis de trend strength del Brain Max
            trend_strength = brain_max_info.get('trend_strength', 0)
            if trend_strength > 2.0:  # Tendencia muy fuerte
                strength += 12
                description_parts.append("Fuerza de tendencia alta")
            elif trend_strength > 1.0:  # Tendencia moderada
                strength += 6
                description_parts.append("Fuerza de tendencia moderada")
            
            # Análisis de trend direction del Brain Max
            trend_direction = brain_max_info.get('trend_direction', 0)
            if trend_direction == 1 and direction == 'bullish':
                strength += 8
                description_parts.append("Dirección de tendencia confirmada")
            elif trend_direction == -1 and direction == 'bearish':
                strength += 8
                description_parts.append("Dirección de tendencia confirmada")
            elif trend_direction != 0 and direction == 'neutral':
                # Usar la dirección del Brain Max si no hay dirección clara
                direction = 'bullish' if trend_direction == 1 else 'bearish'
                strength += 5
                description_parts.append("Dirección basada en Brain Max")
        
        # Limitar fuerza
        strength = max(min(strength, 100.0), 0.0)
        
        description = " + ".join(description_parts) if description_parts else "Tendencia neutra"
        
        return direction, strength, description

    def _calculate_stop_loss(self, signal_type: str, price: float, bb_lower: float, bb_upper: float) -> float:
        """Calcular stop loss"""
        if signal_type == 'buy':
            return bb_lower * 0.995  # 0.5% por debajo del BB inferior
        elif signal_type == 'sell':
            return bb_upper * 1.005  # 0.5% por encima del BB superior
        else:
            return price * 0.99  # 1% por debajo del precio actual

    def _calculate_take_profit(self, signal_type: str, price: float, bb_lower: float, bb_upper: float) -> float:
        """Calcular take profit"""
        if signal_type == 'buy':
            return bb_upper * 1.005  # 0.5% por encima del BB superior
        elif signal_type == 'sell':
            return bb_lower * 0.995  # 0.5% por debajo del BB inferior
        else:
            return price * 1.01  # 1% por encima del precio actual

    def _calculate_support_resistance(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Tuple[float, float]:
        """Calcular niveles de soporte y resistencia"""
        try:
            # Usar Bollinger Bands como aproximación
            bb_lower = indicators['bb_lower'].iloc[-1] if 'bb_lower' in indicators and not indicators['bb_lower'].empty else data['Close'].iloc[-1] * 0.98
            bb_upper = indicators['bb_upper'].iloc[-1] if 'bb_upper' in indicators and not indicators['bb_upper'].empty else data['Close'].iloc[-1] * 1.02
            
            return bb_lower, bb_upper
        except Exception as e:
            logger.error(f"Error calculando soporte/resistencia: {e}")
            current_price = data['Close'].iloc[-1]
            return current_price * 0.98, current_price * 1.02 