"""
Brain Trader Service
===================
Servicio para manejar predicciones, señales y tendencias de Brain Trader
"""

import asyncio
import logging
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Importar el servicio de análisis técnico
try:
    from services.technical_analysis_service import TechnicalAnalysisService
    technical_analysis_service = TechnicalAnalysisService()
except ImportError as e:
    logging.error(f"Error importing TechnicalAnalysisService: {e}")
    technical_analysis_service = None

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
            current_price = ticker.info.get('regularMarketPrice')
            
            if current_price:
                return float(current_price)
            else:
                # Fallback a precio base si no se puede obtener
                return self.base_prices.get(pair, 1.0)
                
        except Exception as e:
            logger.error(f"Error obteniendo precio real para {pair}: {e}")
            return self.base_prices.get(pair, 1.0)

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
                                direction, confidence, reasoning = self._analyze_full_technical(indicators, base_price)
                                reasoning = f'{style.replace("_", " ").title()}: {reasoning}'
                            
                            # Calcular precio objetivo usando volatilidad real y dirección
                            volatility = data['Close'].pct_change().std()
                            if direction == 'up':
                                target_price = base_price * (1 + random.uniform(0.001, volatility*2))
                            elif direction == 'down':
                                target_price = base_price * (1 - random.uniform(0.001, volatility*2))
                            else:  # sideways
                                target_price = base_price * (1 + random.uniform(-volatility, volatility))
                            
                        else:
                            # Fallback a generación aleatoria
                            direction = random.choice(['up', 'down', 'sideways'])
                            confidence = random.uniform(70, 95)
                            reasoning = f'{style.replace("_", " ").title()}: Análisis fallback - {direction.upper()}'
                            
                            # Calcular target_price según dirección
                            if direction == 'up':
                                target_price = base_price + random.uniform(0.001, 0.01)
                            elif direction == 'down':
                                target_price = base_price - random.uniform(0.001, 0.01)
                            else:  # sideways
                                target_price = base_price + random.uniform(-0.005, 0.005)
                            
                    except Exception as e:
                        logger.error(f"Error en análisis técnico para predicciones: {e}")
                        # Fallback a generación aleatoria
                        direction = random.choice(['up', 'down', 'sideways'])
                        confidence = random.uniform(70, 95)
                        reasoning = f'{style.replace("_", " ").title()}: Error en análisis - {direction.upper()}'
                        
                        # Calcular target_price según dirección
                        if direction == 'up':
                            target_price = base_price + random.uniform(0.001, 0.01)
                        elif direction == 'down':
                            target_price = base_price - random.uniform(0.001, 0.01)
                        else:  # sideways
                            target_price = base_price + random.uniform(-0.005, 0.005)
                else:
                    # Sin servicio de análisis técnico, usar generación aleatoria
                    direction = random.choice(['up', 'down', 'sideways'])
                    confidence = random.uniform(70, 95)
                    reasoning = f'{style.replace("_", " ").title()}: Sin análisis técnico - {direction.upper()}'
                    
                    # Calcular target_price según dirección
                    if direction == 'up':
                        target_price = base_price + random.uniform(0.001, 0.01)
                    elif direction == 'down':
                        target_price = base_price - random.uniform(0.001, 0.01)
                    else:  # sideways
                        target_price = base_price + random.uniform(-0.005, 0.005)
                
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