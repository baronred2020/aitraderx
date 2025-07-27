import asyncio
import logging
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TechnicalSignal:
    """Estructura para señales técnicas"""
    pair: str
    signal_type: str  # 'buy', 'sell', 'hold'
    strength: str     # 'strong', 'medium', 'weak'
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    reasoning: str
    timestamp: datetime

@dataclass
class TrendAnalysis:
    """Estructura para análisis de tendencias"""
    pair: str
    direction: str    # 'bullish', 'bearish', 'neutral'
    strength: float
    timeframe: str
    support: float
    resistance: float
    description: str
    timestamp: datetime

class TechnicalAnalysisService:
    """
    Servicio de análisis técnico que usa datos reales de Yahoo Finance
    para generar señales y tendencias basadas en indicadores técnicos reales.
    """
    
    def __init__(self):
        self.symbol_mapping = {
            'EURUSD': 'EURUSD=X',
            'GBPUSD': 'GBPUSD=X',
            'USDJPY': 'USDJPY=X',
            'USDCAD': 'USDCAD=X',
            'AUDUSD': 'AUDUSD=X'
        }
    
    async def get_historical_data(self, pair: str, period: str = "30d") -> pd.DataFrame:
        """Obtener datos históricos reales"""
        try:
            symbol = self.symbol_mapping.get(pair, pair)
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                raise ValueError(f"No se pudieron obtener datos para {pair}")
            
            logger.info(f"Obtenidos {len(data)} registros históricos para {pair}")
            return data
            
        except Exception as e:
            logger.error(f"Error obteniendo datos históricos para {pair}: {e}")
            raise
    
    async def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calcular indicadores técnicos usando pandas"""
        try:
            indicators = {}
            
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
            
            # ADX (Average Directional Index)
            indicators['adx'] = self._calculate_adx(data)
            
            logger.info(f"Indicadores técnicos calculados para {len(data)} períodos")
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculando indicadores técnicos: {e}")
            raise
    
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
        """Calcular ADX (simplificado)"""
        # Simplificación del ADX usando la diferencia de precios
        high_low = data['High'] - data['Low']
        high_close = abs(data['High'] - data['Close'].shift())
        low_close = abs(data['Low'] - data['Close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # ADX simplificado basado en la volatilidad
        adx = (atr / data['Close']) * 100
        return adx
    
    async def generate_real_signals(self, pair: str, limit: int = 10) -> List[TechnicalSignal]:
        """Generar señales reales basadas en análisis técnico"""
        try:
            # Obtener datos históricos
            data = await self.get_historical_data(pair, "60d")
            indicators = await self.calculate_technical_indicators(data)
            
            signals = []
            current_price = data['Close'].iloc[-1]
            
            # Analizar los últimos períodos para generar señales
            for i in range(min(limit, len(data) - 50)):  # Necesitamos al menos 50 períodos para indicadores
                idx = -(i + 1)
                
                if idx < 50:  # Evitar índices fuera de rango
                    continue
                
                # Obtener valores actuales de indicadores
                rsi = indicators['rsi'].iloc[idx] if not pd.isna(indicators['rsi'].iloc[idx]) else 50
                macd = indicators['macd'].iloc[idx] if not pd.isna(indicators['macd'].iloc[idx]) else 0
                macd_signal = indicators['macd_signal'].iloc[idx] if not pd.isna(indicators['macd_signal'].iloc[idx]) else 0
                bb_upper = indicators['bb_upper'].iloc[idx] if not pd.isna(indicators['bb_upper'].iloc[idx]) else current_price * 1.02
                bb_lower = indicators['bb_lower'].iloc[idx] if not pd.isna(indicators['bb_lower'].iloc[idx]) else current_price * 0.98
                sma_20 = indicators['sma_20'].iloc[idx] if not pd.isna(indicators['sma_20'].iloc[idx]) else current_price
                sma_50 = indicators['sma_50'].iloc[idx] if not pd.isna(indicators['sma_50'].iloc[idx]) else current_price
                stoch_k = indicators['stoch_k'].iloc[idx] if not pd.isna(indicators['stoch_k'].iloc[idx]) else 50
                adx = indicators['adx'].iloc[idx] if not pd.isna(indicators['adx'].iloc[idx]) else 25
                price = data['Close'].iloc[idx]
                
                # Lógica de señales basada en indicadores reales
                signal_type, strength, confidence, reasoning = self._analyze_technical_signals(
                    price, rsi, macd, macd_signal, bb_upper, bb_lower, 
                    sma_20, sma_50, stoch_k, adx
                )
                
                # Calcular niveles de entrada, stop loss y take profit
                entry_price = price
                stop_loss = self._calculate_stop_loss(signal_type, price, bb_lower, bb_upper)
                take_profit = self._calculate_take_profit(signal_type, price, bb_lower, bb_upper)
                
                signal = TechnicalSignal(
                    pair=pair,
                    signal_type=signal_type,
                    strength=strength,
                    confidence=confidence,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reasoning=reasoning,
                    timestamp=data.index[idx]
                )
                
                signals.append(signal)
                
                if len(signals) >= limit:
                    break
            
            logger.info(f"Generadas {len(signals)} señales reales para {pair}")
            return signals
            
        except Exception as e:
            logger.error(f"Error generando señales reales para {pair}: {e}")
            raise
    
    async def generate_real_trends(self, pair: str, limit: int = 10) -> List[TrendAnalysis]:
        """Generar análisis de tendencias reales"""
        try:
            # Obtener datos históricos
            data = await self.get_historical_data(pair, "60d")
            indicators = await self.calculate_technical_indicators(data)
            
            trends = []
            
            # Analizar tendencias en diferentes timeframes
            timeframes = ['1H', '4H', '1D']
            
            for timeframe in timeframes:
                # Calcular tendencia basada en moving averages y ADX
                sma_20 = indicators['sma_20'].iloc[-1] if not pd.isna(indicators['sma_20'].iloc[-1]) else data['Close'].iloc[-1]
                sma_50 = indicators['sma_50'].iloc[-1] if not pd.isna(indicators['sma_50'].iloc[-1]) else data['Close'].iloc[-1]
                ema_12 = indicators['ema_12'].iloc[-1] if not pd.isna(indicators['ema_12'].iloc[-1]) else data['Close'].iloc[-1]
                ema_26 = indicators['ema_26'].iloc[-1] if not pd.isna(indicators['ema_26'].iloc[-1]) else data['Close'].iloc[-1]
                adx = indicators['adx'].iloc[-1] if not pd.isna(indicators['adx'].iloc[-1]) else 25
                current_price = data['Close'].iloc[-1]
                
                # Determinar dirección de tendencia
                direction, strength, description = self._analyze_trend_direction(
                    current_price, sma_20, sma_50, ema_12, ema_26, adx
                )
                
                # Calcular niveles de soporte y resistencia
                support, resistance = self._calculate_support_resistance(data, indicators)
                
                trend = TrendAnalysis(
                    pair=pair,
                    direction=direction,
                    strength=strength,
                    timeframe=timeframe,
                    support=support,
                    resistance=resistance,
                    description=description,
                    timestamp=datetime.now()
                )
                
                trends.append(trend)
                
                if len(trends) >= limit:
                    break
            
            logger.info(f"Generados {len(trends)} análisis de tendencias reales para {pair}")
            return trends
            
        except Exception as e:
            logger.error(f"Error generando tendencias reales para {pair}: {e}")
            raise
    
    def _analyze_technical_signals(self, price: float, rsi: float, macd: float, 
                                 macd_signal: float, bb_upper: float, bb_lower: float,
                                 sma_20: float, sma_50: float, stoch_k: float, adx: float) -> Tuple[str, str, float, str]:
        """Analizar señales técnicas basadas en indicadores reales"""
        
        # Inicializar contadores
        buy_signals = 0
        sell_signals = 0
        total_signals = 0
        reasoning_parts = []
        
        # RSI Analysis
        if not pd.isna(rsi):
            if rsi < 30:
                buy_signals += 1
                reasoning_parts.append("RSI oversold")
            elif rsi > 70:
                sell_signals += 1
                reasoning_parts.append("RSI overbought")
            total_signals += 1
        
        # MACD Analysis
        if not pd.isna(macd) and not pd.isna(macd_signal):
            if macd > macd_signal:
                buy_signals += 1
                reasoning_parts.append("MACD bullish crossover")
            else:
                sell_signals += 1
                reasoning_parts.append("MACD bearish crossover")
            total_signals += 1
        
        # Bollinger Bands Analysis
        if not pd.isna(bb_upper) and not pd.isna(bb_lower):
            if price <= bb_lower:
                buy_signals += 1
                reasoning_parts.append("Price at BB lower band")
            elif price >= bb_upper:
                sell_signals += 1
                reasoning_parts.append("Price at BB upper band")
            total_signals += 1
        
        # Moving Average Analysis
        if not pd.isna(sma_20) and not pd.isna(sma_50):
            if sma_20 > sma_50:
                buy_signals += 1
                reasoning_parts.append("SMA bullish alignment")
            else:
                sell_signals += 1
                reasoning_parts.append("SMA bearish alignment")
            total_signals += 1
        
        # Stochastic Analysis
        if not pd.isna(stoch_k):
            if stoch_k < 20:
                buy_signals += 1
                reasoning_parts.append("Stochastic oversold")
            elif stoch_k > 80:
                sell_signals += 1
                reasoning_parts.append("Stochastic overbought")
            total_signals += 1
        
        # Determinar señal final
        if total_signals == 0:
            return 'hold', 'weak', 50.0, "Insufficient data"
        
        buy_ratio = buy_signals / total_signals
        sell_ratio = sell_signals / total_signals
        
        if buy_ratio > 0.6:
            signal_type = 'buy'
            confidence = 60 + (buy_ratio * 30)
            strength = 'strong' if buy_ratio > 0.8 else 'medium'
        elif sell_ratio > 0.6:
            signal_type = 'sell'
            confidence = 60 + (sell_ratio * 30)
            strength = 'strong' if sell_ratio > 0.8 else 'medium'
        else:
            signal_type = 'hold'
            confidence = 50.0
            strength = 'weak'
        
        reasoning = " | ".join(reasoning_parts[:3])  # Máximo 3 razones
        
        return signal_type, strength, confidence, reasoning
    
    def _analyze_trend_direction(self, price: float, sma_20: float, sma_50: float, 
                               ema_12: float, ema_26: float, adx: float) -> Tuple[str, float, str]:
        """Analizar dirección de tendencia"""
        
        # Contadores para determinar tendencia
        bullish_signals = 0
        bearish_signals = 0
        total_signals = 0
        reasoning_parts = []
        
        # Moving Average Analysis
        if not pd.isna(sma_20) and not pd.isna(sma_50):
            if sma_20 > sma_50:
                bullish_signals += 1
                reasoning_parts.append("SMA bullish")
            else:
                bearish_signals += 1
                reasoning_parts.append("SMA bearish")
            total_signals += 1
        
        # EMA Analysis
        if not pd.isna(ema_12) and not pd.isna(ema_26):
            if ema_12 > ema_26:
                bullish_signals += 1
                reasoning_parts.append("EMA bullish")
            else:
                bearish_signals += 1
                reasoning_parts.append("EMA bearish")
            total_signals += 1
        
        # Price vs Moving Averages
        if not pd.isna(sma_20):
            if price > sma_20:
                bullish_signals += 1
                reasoning_parts.append("Price above SMA20")
            else:
                bearish_signals += 1
                reasoning_parts.append("Price below SMA20")
            total_signals += 1
        
        # Determinar dirección final
        if total_signals == 0:
            return 'neutral', 50.0, "Insufficient data"
        
        bullish_ratio = bullish_signals / total_signals
        bearish_ratio = bearish_signals / total_signals
        
        if bullish_ratio > 0.6:
            direction = 'bullish'
            strength = 50 + (bullish_ratio * 40)
        elif bearish_ratio > 0.6:
            direction = 'bearish'
            strength = 50 + (bearish_ratio * 40)
        else:
            direction = 'neutral'
            strength = 50.0
        
        description = " | ".join(reasoning_parts[:2])
        
        return direction, strength, description
    
    def _calculate_stop_loss(self, signal_type: str, price: float, bb_lower: float, bb_upper: float) -> float:
        """Calcular stop loss basado en Bollinger Bands"""
        if signal_type == 'buy':
            return bb_lower if not pd.isna(bb_lower) else price * 0.995
        elif signal_type == 'sell':
            return bb_upper if not pd.isna(bb_upper) else price * 1.005
        else:
            return price
    
    def _calculate_take_profit(self, signal_type: str, price: float, bb_lower: float, bb_upper: float) -> float:
        """Calcular take profit basado en Bollinger Bands"""
        if signal_type == 'buy':
            return bb_upper if not pd.isna(bb_upper) else price * 1.015
        elif signal_type == 'sell':
            return bb_lower if not pd.isna(bb_lower) else price * 0.985
        else:
            return price
    
    def _calculate_support_resistance(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Tuple[float, float]:
        """Calcular niveles de soporte y resistencia"""
        current_price = data['Close'].iloc[-1]
        
        # Usar Bollinger Bands como soporte/resistencia
        bb_lower = indicators['bb_lower'].iloc[-1] if not pd.isna(indicators['bb_lower'].iloc[-1]) else current_price * 0.99
        bb_upper = indicators['bb_upper'].iloc[-1] if not pd.isna(indicators['bb_upper'].iloc[-1]) else current_price * 1.01
        
        support = bb_lower
        resistance = bb_upper
        
        return support, resistance 