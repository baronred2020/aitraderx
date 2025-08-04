"""
Ultra Quality Signals Service
=============================
Servicio para generar señales de trading de ultra calidad integrando
todos los modelos avanzados y herramientas disponibles en la app.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import sys
import os

# Agregar el directorio de modelos al path
models_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
if models_path not in sys.path:
    sys.path.insert(0, models_path)

try:
    from Modelo_Brain_Max import HybridForexAI, create_advanced_technical_indicators
    BRAIN_MAX_AVAILABLE = True
except ImportError:
    BRAIN_MAX_AVAILABLE = False
    logging.warning("Modelo_Brain_Max no disponible")

# Configurar logging
logger = logging.getLogger(__name__)

@dataclass
class UltraQualitySignal:
    """Señal de trading de ultra calidad"""
    id: str
    pair: str
    signal_type: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0-100
    entry_price: float
    stop_loss: float
    take_profit: float
    reasoning: str
    brain_type: str
    timestamp: str
    quality_score: float  # 0-100
    risk_reward_ratio: float
    market_conditions: Dict[str, Any]
    technical_analysis: Dict[str, Any]
    ai_consensus: Dict[str, Any]
    volatility_analysis: Dict[str, Any]
    volume_analysis: Dict[str, Any]
    trend_analysis: Dict[str, Any]
    support_resistance: Dict[str, Any]
    momentum_analysis: Dict[str, Any]
    style: str  # 'scalping', 'day_trading', 'swing_trading', 'position_trading'

class UltraQualitySignalsService:
    """Servicio para generar señales de ultra calidad"""
    
    def __init__(self):
        self.brain_max_ai = None
        self.technical_indicators = {}
        self.signal_history = []
        self.quality_thresholds = {
            'scalping': {
                'min_confidence': 60.0,
                'min_quality_score': 65.0,
                'min_risk_reward': 1.0,
                'max_volatility': 0.05
            },
            'day_trading': {
                'min_confidence': 55.0,
                'min_quality_score': 60.0,
                'min_risk_reward': 0.8,
                'max_volatility': 0.08
            },
            'swing_trading': {
                'min_confidence': 50.0,
                'min_quality_score': 55.0,
                'min_risk_reward': 0.6,
                'max_volatility': 0.10
            },
            'position_trading': {
                'min_confidence': 45.0,
                'min_quality_score': 50.0,
                'min_risk_reward': 0.5,
                'max_volatility': 0.15
            }
        }
        
        # Inicializar Brain Max AI si está disponible
        if BRAIN_MAX_AVAILABLE:
            try:
                self.brain_max_ai = HybridForexAI(symbol='EURUSD', use_lstm=True)
                logger.info("✅ Brain Max AI inicializado para señales ultra calidad")
            except Exception as e:
                logger.error(f"❌ Error inicializando Brain Max AI: {e}")
    
    async def generate_ultra_quality_signal(self, pair: str, style: str = 'day_trading') -> Optional[UltraQualitySignal]:
        """Generar señal de ultra calidad integrando todos los modelos"""
        try:
            logger.info(f"🎯 Generando señal ultra calidad para {pair} - {style}")
            
            # 1. Obtener datos históricos
            historical_data = await self._get_historical_data(pair)
            if historical_data.empty:
                logger.warning(f"⚠️ No hay datos históricos para {pair}")
                return None
            
            # 2. Calcular indicadores técnicos avanzados
            technical_indicators = self._calculate_advanced_indicators(historical_data)
            
            # 3. Análisis de condiciones de mercado
            market_conditions = self._analyze_market_conditions(historical_data, technical_indicators)
            
            # 4. Análisis de volatilidad
            volatility_analysis = self._analyze_volatility(historical_data, technical_indicators)
            
            # 5. Análisis de volumen
            volume_analysis = self._analyze_volume(historical_data, technical_indicators)
            
            # 6. Análisis de tendencia
            trend_analysis = self._analyze_trend(historical_data, technical_indicators)
            
            # 7. Análisis de soporte y resistencia
            support_resistance = self._analyze_support_resistance(historical_data, technical_indicators)
            
            # 8. Análisis de momentum
            momentum_analysis = self._analyze_momentum(historical_data, technical_indicators)
            
            # 9. Consenso de IA (Brain Max + otros modelos)
            ai_consensus = await self._get_ai_consensus(pair, style, historical_data, technical_indicators)
            
            # 10. Generar señal integrada
            signal = self._generate_integrated_signal(
                pair, style, historical_data, technical_indicators,
                market_conditions, volatility_analysis, volume_analysis,
                trend_analysis, support_resistance, momentum_analysis, ai_consensus
            )
            
            # 11. Validar calidad de la señal
            if signal and self._validate_signal_quality(signal, style):
                logger.info(f"✅ Señal ultra calidad generada: {signal.signal_type} - {signal.confidence:.1f}%")
                return signal
            else:
                logger.info(f"⚠️ Señal no cumple criterios de calidad para {style}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error generando señal ultra calidad: {e}")
            return None
    
    async def _get_historical_data(self, pair: str) -> pd.DataFrame:
        """Obtener datos históricos de alta calidad"""
        try:
            # Usar yfinance para datos reales
            import yfinance as yf
            
            # Mapear pares a símbolos de yfinance
            symbol_mapping = {
                'EURUSD': 'EURUSD=X',
                'GBPUSD': 'GBPUSD=X',
                'USDJPY': 'USDJPY=X',
                'AUDUSD': 'AUDUSD=X',
                'USDCAD': 'USDCAD=X'
            }
            
            symbol = symbol_mapping.get(pair, f"{pair}=X")
            ticker = yf.Ticker(symbol)
            
            # Obtener datos de los últimos 60 días con intervalo de 1 hora
            data = ticker.history(period='60d', interval='1h')
            
            if data.empty:
                logger.warning(f"⚠️ No se pudieron obtener datos para {symbol}")
                return pd.DataFrame()
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo datos históricos: {e}")
            return pd.DataFrame()
    
    def _calculate_advanced_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calcular indicadores técnicos avanzados"""
        try:
            indicators = {}
            
            # Usar la función avanzada de Brain Max
            if BRAIN_MAX_AVAILABLE:
                advanced_data = create_advanced_technical_indicators(data.copy())
                
                # Extraer valores actuales
                current = advanced_data.iloc[-1]
                
                indicators = {
                    'rsi': current.get('rsi', 50),
                    'macd': current.get('macd', 0),
                    'macd_signal': current.get('macd_signal', 0),
                    'macd_hist': current.get('macd_hist', 0),
                    'bb_upper': current.get('bb_upper', data['Close'].iloc[-1] * 1.02),
                    'bb_lower': current.get('bb_lower', data['Close'].iloc[-1] * 0.98),
                    'bb_position': current.get('bb_position', 0.5),
                    'sma_5': current.get('sma_5', data['Close'].iloc[-1]),
                    'sma_20': current.get('sma_20', data['Close'].iloc[-1]),
                    'sma_50': current.get('sma_50', data['Close'].iloc[-1]),
                    'ema_12': current.get('ema_12', data['Close'].iloc[-1]),
                    'ema_26': current.get('ema_26', data['Close'].iloc[-1]),
                    'volatility': current.get('volatility', 0),
                    'volatility_ratio': current.get('volatility_ratio', 1),
                    'volume_ratio': current.get('volume_ratio', 1),
                    'volume_trend': current.get('volume_trend', 1),
                    'momentum': current.get('momentum', 0),
                    'momentum_ratio': current.get('momentum_ratio', 0),
                    'trend_strength': current.get('trend_strength', 0),
                    'trend_direction': current.get('trend_direction', 0),
                    'support_level': current.get('support_level', data['Low'].iloc[-1]),
                    'resistance_level': current.get('resistance_level', data['High'].iloc[-1]),
                    'price_position': current.get('price_position', 0.5)
                }
            else:
                # Fallback a indicadores básicos
                indicators = self._calculate_basic_indicators(data)
            
            return indicators
            
        except Exception as e:
            logger.error(f"❌ Error calculando indicadores avanzados: {e}")
            return self._calculate_basic_indicators(data)
    
    def _calculate_basic_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calcular indicadores básicos como fallback"""
        try:
            current_price = data['Close'].iloc[-1]
            
            # RSI básico
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rs = rs.replace([np.inf, -np.inf], 0)
            rsi = 100 - (100 / (1 + rs))
            rsi = rsi.fillna(50)
            
            # MACD básico
            exp1 = data['Close'].ewm(span=12).mean()
            exp2 = data['Close'].ewm(span=26).mean()
            macd = exp1 - exp2
            macd_signal = macd.ewm(span=9).mean()
            
            # Medias móviles
            sma_20 = data['Close'].rolling(window=20).mean()
            sma_50 = data['Close'].rolling(window=50).mean()
            
            return {
                'rsi': float(rsi.iloc[-1]),
                'macd': float(macd.iloc[-1]),
                'macd_signal': float(macd_signal.iloc[-1]),
                'sma_20': float(sma_20.iloc[-1]),
                'sma_50': float(sma_50.iloc[-1]),
                'volatility': float(data['Close'].rolling(window=20).std().iloc[-1]),
                'support_level': float(data['Low'].rolling(window=20).min().iloc[-1]),
                'resistance_level': float(data['High'].rolling(window=20).max().iloc[-1])
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculando indicadores básicos: {e}")
            return {}
    
    def _analyze_market_conditions(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Analizar condiciones generales del mercado"""
        try:
            current_price = data['Close'].iloc[-1]
            rsi = indicators.get('rsi', 50)
            volatility = indicators.get('volatility', 0)
            
            # Determinar condición del mercado
            if rsi < 30:
                market_condition = 'oversold'
            elif rsi > 70:
                market_condition = 'overbought'
            else:
                market_condition = 'neutral'
            
            # Análisis de volatilidad
            avg_volatility = data['Close'].rolling(window=20).std().mean()
            volatility_regime = 'high' if volatility > avg_volatility * 1.5 else 'low' if volatility < avg_volatility * 0.5 else 'normal'
            
            return {
                'condition': market_condition,
                'volatility_regime': volatility_regime,
                'trend_strength': indicators.get('trend_strength', 0),
                'momentum': indicators.get('momentum', 0),
                'market_sentiment': 'bullish' if rsi > 50 else 'bearish' if rsi < 50 else 'neutral'
            }
            
        except Exception as e:
            logger.error(f"❌ Error analizando condiciones de mercado: {e}")
            return {'condition': 'neutral', 'volatility_regime': 'normal'}
    
    def _analyze_volatility(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Análisis detallado de volatilidad"""
        try:
            current_volatility = indicators.get('volatility', 0)
            volatility_ratio = indicators.get('volatility_ratio', 1)
            
            # Calcular volatilidad histórica
            returns = data['Close'].pct_change().dropna()
            historical_volatility = returns.std() * np.sqrt(252)  # Anualizada
            
            # Análisis de volatilidad
            volatility_analysis = {
                'current_volatility': current_volatility,
                'historical_volatility': historical_volatility,
                'volatility_ratio': volatility_ratio,
                'volatility_trend': 'increasing' if volatility_ratio > 1.2 else 'decreasing' if volatility_ratio < 0.8 else 'stable',
                'risk_level': 'high' if current_volatility > historical_volatility * 1.5 else 'low' if current_volatility < historical_volatility * 0.5 else 'medium'
            }
            
            return volatility_analysis
            
        except Exception as e:
            logger.error(f"❌ Error analizando volatilidad: {e}")
            return {'current_volatility': 0, 'risk_level': 'medium'}
    
    def _analyze_volume(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Análisis detallado de volumen"""
        try:
            volume_ratio = indicators.get('volume_ratio', 1)
            volume_trend = indicators.get('volume_trend', 1)
            
            # Análisis de volumen
            avg_volume = data['Volume'].rolling(window=20).mean().iloc[-1]
            current_volume = data['Volume'].iloc[-1]
            
            volume_analysis = {
                'current_volume': current_volume,
                'average_volume': avg_volume,
                'volume_ratio': volume_ratio,
                'volume_trend': volume_trend,
                'volume_signal': 'high' if volume_ratio > 1.5 else 'low' if volume_ratio < 0.5 else 'normal',
                'volume_confirmation': 'strong' if volume_ratio > 2.0 else 'weak' if volume_ratio < 0.3 else 'moderate'
            }
            
            return volume_analysis
            
        except Exception as e:
            logger.error(f"❌ Error analizando volumen: {e}")
            return {'volume_signal': 'normal', 'volume_confirmation': 'moderate'}
    
    def _analyze_trend(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Análisis detallado de tendencia"""
        try:
            sma_20 = indicators.get('sma_20', data['Close'].iloc[-1])
            sma_50 = indicators.get('sma_50', data['Close'].iloc[-1])
            ema_12 = indicators.get('ema_12', data['Close'].iloc[-1])
            ema_26 = indicators.get('ema_26', data['Close'].iloc[-1])
            current_price = data['Close'].iloc[-1]
            trend_direction = indicators.get('trend_direction', 0)
            trend_strength = indicators.get('trend_strength', 0)
            
            # Determinar dirección de tendencia
            if current_price > sma_20 > sma_50:
                trend = 'strong_uptrend'
            elif current_price > sma_20 and sma_20 < sma_50:
                trend = 'weak_uptrend'
            elif current_price < sma_20 < sma_50:
                trend = 'strong_downtrend'
            elif current_price < sma_20 and sma_20 > sma_50:
                trend = 'weak_downtrend'
            else:
                trend = 'sideways'
            
            # Análisis de EMA
            ema_trend = 'bullish' if ema_12 > ema_26 else 'bearish'
            
            trend_analysis = {
                'trend': trend,
                'trend_direction': trend_direction,
                'trend_strength': trend_strength,
                'ema_trend': ema_trend,
                'price_vs_sma20': 'above' if current_price > sma_20 else 'below',
                'price_vs_sma50': 'above' if current_price > sma_50 else 'below',
                'sma_alignment': 'bullish' if sma_20 > sma_50 else 'bearish'
            }
            
            return trend_analysis
            
        except Exception as e:
            logger.error(f"❌ Error analizando tendencia: {e}")
            return {'trend': 'sideways', 'trend_strength': 0}
    
    def _analyze_support_resistance(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Análisis de soporte y resistencia"""
        try:
            current_price = data['Close'].iloc[-1]
            support_level = indicators.get('support_level', data['Low'].rolling(window=20).min().iloc[-1])
            resistance_level = indicators.get('resistance_level', data['High'].rolling(window=20).max().iloc[-1])
            price_position = indicators.get('price_position', 0.5)
            
            # Calcular distancia a soporte y resistencia
            distance_to_support = (current_price - support_level) / current_price
            distance_to_resistance = (resistance_level - current_price) / current_price
            
            # Determinar proximidad
            proximity_to_support = 'near' if distance_to_support < 0.01 else 'far'
            proximity_to_resistance = 'near' if distance_to_resistance < 0.01 else 'far'
            
            support_resistance_analysis = {
                'support_level': support_level,
                'resistance_level': resistance_level,
                'price_position': price_position,
                'distance_to_support': distance_to_support,
                'distance_to_resistance': distance_to_resistance,
                'proximity_to_support': proximity_to_support,
                'proximity_to_resistance': proximity_to_resistance,
                'breakout_potential': 'high' if proximity_to_resistance == 'near' else 'low',
                'bounce_potential': 'high' if proximity_to_support == 'near' else 'low'
            }
            
            return support_resistance_analysis
            
        except Exception as e:
            logger.error(f"❌ Error analizando soporte/resistencia: {e}")
            return {'price_position': 0.5, 'breakout_potential': 'low'}
    
    def _analyze_momentum(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Análisis de momentum"""
        try:
            rsi = indicators.get('rsi', 50)
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            macd_hist = indicators.get('macd_hist', 0)
            momentum = indicators.get('momentum', 0)
            momentum_ratio = indicators.get('momentum_ratio', 0)
            
            # Análisis de RSI
            rsi_signal = 'oversold' if rsi < 30 else 'overbought' if rsi > 70 else 'neutral'
            
            # Análisis de MACD
            macd_signal_direction = 'bullish' if macd > macd_signal else 'bearish'
            macd_strength = 'strong' if abs(macd_hist) > abs(macd) * 0.5 else 'weak'
            
            # Análisis de momentum
            momentum_signal = 'strong_bullish' if momentum > 0 and momentum_ratio > 1.5 else \
                             'strong_bearish' if momentum < 0 and momentum_ratio < -1.5 else \
                             'weak_bullish' if momentum > 0 else 'weak_bearish'
            
            momentum_analysis = {
                'rsi_signal': rsi_signal,
                'rsi_value': rsi,
                'macd_signal': macd_signal_direction,
                'macd_strength': macd_strength,
                'macd_histogram': macd_hist,
                'momentum_signal': momentum_signal,
                'momentum_strength': abs(momentum),
                'momentum_ratio': momentum_ratio,
                'overall_momentum': 'bullish' if rsi > 50 and macd > macd_signal and momentum > 0 else \
                                   'bearish' if rsi < 50 and macd < macd_signal and momentum < 0 else 'neutral'
            }
            
            return momentum_analysis
            
        except Exception as e:
            logger.error(f"❌ Error analizando momentum: {e}")
            return {'overall_momentum': 'neutral'}
    
    async def _get_ai_consensus(self, pair: str, style: str, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Obtener consenso de todos los modelos de IA"""
        try:
            ai_consensus = {
                'brain_max_prediction': None,
                'technical_consensus': None,
                'ensemble_confidence': 0,
                'signal_agreement': 0,
                'models_used': []
            }
            
            # 1. Predicción de Brain Max
            if self.brain_max_ai:
                try:
                    brain_max_result = self.brain_max_ai.predict(style, data)
                    if brain_max_result:
                        ai_consensus['brain_max_prediction'] = {
                            'signal': brain_max_result.get('signal', 'HOLD'),
                            'confidence': brain_max_result.get('confidence', 0),
                            'target_price': brain_max_result.get('target_price', 0),
                            'stop_loss': brain_max_result.get('stop_loss', 0)
                        }
                        ai_consensus['models_used'].append('brain_max')
                except Exception as e:
                    logger.warning(f"⚠️ Error en predicción Brain Max: {e}")
            
            # 2. Consenso técnico
            technical_signals = []
            
            # RSI
            rsi = indicators.get('rsi', 50)
            if rsi < 30:
                technical_signals.append(('buy', 0.8))
            elif rsi > 70:
                technical_signals.append(('sell', 0.8))
            
            # MACD
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            if macd > macd_signal:
                technical_signals.append(('buy', 0.7))
            elif macd < macd_signal:
                technical_signals.append(('sell', 0.7))
            
            # Bollinger Bands
            bb_position = indicators.get('bb_position', 0.5)
            if bb_position < 0.2:
                technical_signals.append(('buy', 0.6))
            elif bb_position > 0.8:
                technical_signals.append(('sell', 0.6))
            
            # Calcular consenso técnico
            if technical_signals:
                buy_signals = [conf for signal, conf in technical_signals if signal == 'buy']
                sell_signals = [conf for signal, conf in technical_signals if signal == 'sell']
                
                if buy_signals and not sell_signals:
                    ai_consensus['technical_consensus'] = ('buy', np.mean(buy_signals))
                elif sell_signals and not buy_signals:
                    ai_consensus['technical_consensus'] = ('sell', np.mean(sell_signals))
                else:
                    ai_consensus['technical_consensus'] = ('hold', 0.5)
            
            # 3. Calcular confianza del ensemble
            confidences = []
            if ai_consensus['brain_max_prediction']:
                confidences.append(ai_consensus['brain_max_prediction']['confidence'])
            if ai_consensus['technical_consensus']:
                confidences.append(ai_consensus['technical_consensus'][1] * 100)
            
            if confidences:
                ai_consensus['ensemble_confidence'] = np.mean(confidences)
            
            # 4. Calcular acuerdo entre señales
            signals = []
            if ai_consensus['brain_max_prediction']:
                signals.append(ai_consensus['brain_max_prediction']['signal'].lower())
            if ai_consensus['technical_consensus']:
                signals.append(ai_consensus['technical_consensus'][0])
            
            if len(set(signals)) == 1:
                ai_consensus['signal_agreement'] = 100
            elif len(signals) > 1:
                ai_consensus['signal_agreement'] = 50
            
            return ai_consensus
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo consenso de IA: {e}")
            return {'ensemble_confidence': 0, 'signal_agreement': 0}
    
    def _generate_integrated_signal(self, pair: str, style: str, data: pd.DataFrame, 
                                  indicators: Dict[str, Any], market_conditions: Dict[str, Any],
                                  volatility_analysis: Dict[str, Any], volume_analysis: Dict[str, Any],
                                  trend_analysis: Dict[str, Any], support_resistance: Dict[str, Any],
                                  momentum_analysis: Dict[str, Any], ai_consensus: Dict[str, Any]) -> Optional[UltraQualitySignal]:
        """Generar señal integrada de ultra calidad"""
        try:
            current_price = data['Close'].iloc[-1]
            
            # 1. Determinar señal base
            signal_type = self._determine_signal_type(
                indicators, market_conditions, trend_analysis, 
                momentum_analysis, ai_consensus
            )
            
            # 2. Calcular confianza integrada
            confidence = self._calculate_integrated_confidence(
                indicators, market_conditions, volatility_analysis,
                volume_analysis, trend_analysis, momentum_analysis, ai_consensus
            )
            
            # 3. Calcular stop loss y take profit
            stop_loss, take_profit = self._calculate_risk_management(
                signal_type, current_price, indicators, support_resistance, style
            )
            
            # 4. Calcular ratio riesgo/beneficio
            risk_reward_ratio = abs(take_profit - current_price) / abs(stop_loss - current_price) if stop_loss != current_price else 0
            
            # 5. Calcular score de calidad
            quality_score = self._calculate_quality_score(
                confidence, risk_reward_ratio, market_conditions,
                volatility_analysis, volume_analysis, ai_consensus
            )
            
            # 6. Generar razonamiento detallado
            reasoning = self._generate_detailed_reasoning(
                signal_type, indicators, market_conditions, trend_analysis,
                momentum_analysis, ai_consensus, quality_score
            )
            
            # 7. Crear señal de ultra calidad
            signal = UltraQualitySignal(
                id=f"ultra_{pair}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                pair=pair,
                signal_type=signal_type,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reasoning=reasoning,
                brain_type='ultra_quality',
                timestamp=datetime.now().isoformat(),
                quality_score=quality_score,
                risk_reward_ratio=risk_reward_ratio,
                market_conditions=market_conditions,
                technical_analysis=indicators,
                ai_consensus=ai_consensus,
                volatility_analysis=volatility_analysis,
                volume_analysis=volume_analysis,
                trend_analysis=trend_analysis,
                support_resistance=support_resistance,
                momentum_analysis=momentum_analysis,
                style=style
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Error generando señal integrada: {e}")
            return None
    
    def _determine_signal_type(self, indicators: Dict[str, Any], market_conditions: Dict[str, Any],
                              trend_analysis: Dict[str, Any], momentum_analysis: Dict[str, Any],
                              ai_consensus: Dict[str, Any]) -> str:
        """Determinar tipo de señal basado en múltiples factores"""
        try:
            # Puntuación para cada tipo de señal
            buy_score = 0
            sell_score = 0
            
            # 1. Análisis técnico básico
            rsi = indicators.get('rsi', 50)
            if rsi < 30:
                buy_score += 2
            elif rsi > 70:
                sell_score += 2
            
            # 2. Análisis de tendencia
            trend = trend_analysis.get('trend', 'sideways')
            if 'uptrend' in trend:
                buy_score += 1
            elif 'downtrend' in trend:
                sell_score += 1
            
            # 3. Análisis de momentum
            momentum = momentum_analysis.get('overall_momentum', 'neutral')
            if momentum == 'bullish':
                buy_score += 1
            elif momentum == 'bearish':
                sell_score += 1
            
            # 4. Consenso de IA
            if ai_consensus.get('brain_max_prediction'):
                brain_signal = ai_consensus['brain_max_prediction']['signal'].lower()
                if brain_signal == 'buy':
                    buy_score += 2
                elif brain_signal == 'sell':
                    sell_score += 2
            
            if ai_consensus.get('technical_consensus'):
                tech_signal = ai_consensus['technical_consensus'][0]
                if tech_signal == 'buy':
                    buy_score += 1
                elif tech_signal == 'sell':
                    sell_score += 1
            
            # 5. Análisis técnico adicional (sin Brain Max)
            rsi = indicators.get('rsi', 50)
            if rsi < 25:
                buy_score += 2
            elif rsi > 75:
                sell_score += 2
            elif rsi < 35:
                buy_score += 1
            elif rsi > 65:
                sell_score += 1
            
            # 6. Condiciones de mercado
            market_condition = market_conditions.get('condition', 'neutral')
            if market_condition == 'oversold':
                buy_score += 1
            elif market_condition == 'overbought':
                sell_score += 1
            
            # Determinar señal final
            if buy_score > sell_score and buy_score >= 2:
                return 'buy'
            elif sell_score > buy_score and sell_score >= 2:
                return 'sell'
            else:
                return 'hold'
                
        except Exception as e:
            logger.error(f"❌ Error determinando tipo de señal: {e}")
            return 'hold'
    
    def _calculate_integrated_confidence(self, indicators: Dict[str, Any], market_conditions: Dict[str, Any],
                                       volatility_analysis: Dict[str, Any], volume_analysis: Dict[str, Any],
                                       trend_analysis: Dict[str, Any], momentum_analysis: Dict[str, Any],
                                       ai_consensus: Dict[str, Any]) -> float:
        """Calcular confianza integrada"""
        try:
            confidences = []
            
            # 1. Confianza del consenso de IA
            if ai_consensus.get('ensemble_confidence', 0) > 0:
                confidences.append(ai_consensus['ensemble_confidence'])
            
            # 2. Confianza basada en RSI
            rsi = indicators.get('rsi', 50)
            if rsi < 20 or rsi > 80:
                confidences.append(85.0)
            elif rsi < 30 or rsi > 70:
                confidences.append(75.0)
            else:
                confidences.append(60.0)
            
            # 3. Confianza basada en tendencia
            trend_strength = trend_analysis.get('trend_strength', 0)
            if trend_strength > 2.0:
                confidences.append(80.0)
            elif trend_strength > 1.0:
                confidences.append(70.0)
            else:
                confidences.append(50.0)
            
            # 4. Confianza basada en volumen
            volume_confirmation = volume_analysis.get('volume_confirmation', 'moderate')
            if volume_confirmation == 'strong':
                confidences.append(85.0)
            elif volume_confirmation == 'moderate':
                confidences.append(70.0)
            else:
                confidences.append(50.0)
            
            # 5. Confianza basada en acuerdo de señales
            signal_agreement = ai_consensus.get('signal_agreement', 0)
            confidences.append(signal_agreement)
            
            # Calcular confianza promedio ponderada
            if confidences:
                # Dar más peso a la confianza del ensemble de IA
                weights = [2.0 if i == 0 and ai_consensus.get('ensemble_confidence', 0) > 0 else 1.0 
                          for i in range(len(confidences))]
                weighted_confidence = np.average(confidences, weights=weights)
                return min(weighted_confidence, 100.0)
            else:
                return 50.0
                
        except Exception as e:
            logger.error(f"❌ Error calculando confianza integrada: {e}")
            return 50.0
    
    def _calculate_risk_management(self, signal_type: str, current_price: float, 
                                 indicators: Dict[str, Any], support_resistance: Dict[str, Any],
                                 style: str) -> Tuple[float, float]:
        """Calcular stop loss y take profit optimizados"""
        try:
            support_level = support_resistance.get('support_level', current_price * 0.98)
            resistance_level = support_resistance.get('resistance_level', current_price * 1.02)
            volatility = indicators.get('volatility', current_price * 0.01)
            
            # Multiplicadores según el estilo
            style_multipliers = {
                'scalping': {'stop': 0.5, 'profit': 1.0},
                'day_trading': {'stop': 1.0, 'profit': 1.5},
                'swing_trading': {'stop': 1.5, 'profit': 2.0},
                'position_trading': {'stop': 2.0, 'profit': 3.0}
            }
            
            multiplier = style_multipliers.get(style, {'stop': 1.0, 'profit': 1.5})
            
            if signal_type == 'buy':
                # Stop loss: precio actual - (volatilidad * multiplicador)
                stop_loss = current_price - (volatility * multiplier['stop'])
                # Take profit: precio actual + (volatilidad * multiplicador * ratio)
                take_profit = current_price + (volatility * multiplier['profit'])
                
                # Ajustar a niveles de soporte/resistencia
                if support_level > stop_loss:
                    stop_loss = support_level * 0.995  # Justo debajo del soporte
                if resistance_level < take_profit:
                    take_profit = resistance_level * 1.005  # Justo arriba de la resistencia
                    
            elif signal_type == 'sell':
                # Stop loss: precio actual + (volatilidad * multiplicador)
                stop_loss = current_price + (volatility * multiplier['stop'])
                # Take profit: precio actual - (volatilidad * multiplicador * ratio)
                take_profit = current_price - (volatility * multiplier['profit'])
                
                # Ajustar a niveles de soporte/resistencia
                if resistance_level < stop_loss:
                    stop_loss = resistance_level * 1.005  # Justo arriba de la resistencia
                if support_level > take_profit:
                    take_profit = support_level * 0.995  # Justo debajo del soporte
                    
            else:  # hold
                stop_loss = current_price
                take_profit = current_price
            
            return stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"❌ Error calculando gestión de riesgo: {e}")
            return current_price * 0.98, current_price * 1.02
    
    def _calculate_quality_score(self, confidence: float, risk_reward_ratio: float,
                               market_conditions: Dict[str, Any], volatility_analysis: Dict[str, Any],
                               volume_analysis: Dict[str, Any], ai_consensus: Dict[str, Any]) -> float:
        """Calcular score de calidad de la señal"""
        try:
            score = 0
            
            # 1. Confianza (40% del score)
            score += confidence * 0.4
            
            # 2. Ratio riesgo/beneficio (25% del score)
            if risk_reward_ratio >= 2.0:
                score += 25
            elif risk_reward_ratio >= 1.5:
                score += 20
            elif risk_reward_ratio >= 1.0:
                score += 15
            else:
                score += 10
            
            # 3. Condiciones de mercado (15% del score)
            market_condition = market_conditions.get('condition', 'neutral')
            if market_condition in ['oversold', 'overbought']:
                score += 15
            elif market_condition == 'neutral':
                score += 10
            else:
                score += 5
            
            # 4. Análisis de volatilidad (10% del score)
            risk_level = volatility_analysis.get('risk_level', 'medium')
            if risk_level == 'medium':
                score += 10
            elif risk_level == 'low':
                score += 8
            else:
                score += 5
            
            # 5. Confirmación de volumen (10% del score)
            volume_confirmation = volume_analysis.get('volume_confirmation', 'moderate')
            if volume_confirmation == 'strong':
                score += 10
            elif volume_confirmation == 'moderate':
                score += 8
            else:
                score += 5
            
            return min(score, 100.0)
            
        except Exception as e:
            logger.error(f"❌ Error calculando score de calidad: {e}")
            return 50.0
    
    def _generate_detailed_reasoning(self, signal_type: str, indicators: Dict[str, Any],
                                   market_conditions: Dict[str, Any], trend_analysis: Dict[str, Any],
                                   momentum_analysis: Dict[str, Any], ai_consensus: Dict[str, Any],
                                   quality_score: float) -> str:
        """Generar razonamiento detallado de la señal"""
        try:
            reasoning_parts = []
            
            # 1. Señal principal
            reasoning_parts.append(f"Señal {signal_type.upper()} generada con {quality_score:.1f}% de calidad")
            
            # 2. Análisis técnico
            rsi = indicators.get('rsi', 50)
            if rsi < 30:
                reasoning_parts.append("RSI en zona de sobreventa")
            elif rsi > 70:
                reasoning_parts.append("RSI en zona de sobrecompra")
            
            # 3. Análisis de tendencia
            trend = trend_analysis.get('trend', 'sideways')
            if 'uptrend' in trend:
                reasoning_parts.append("Tendencia alcista confirmada")
            elif 'downtrend' in trend:
                reasoning_parts.append("Tendencia bajista confirmada")
            
            # 4. Análisis de momentum
            momentum = momentum_analysis.get('overall_momentum', 'neutral')
            if momentum == 'bullish':
                reasoning_parts.append("Momentum alcista")
            elif momentum == 'bearish':
                reasoning_parts.append("Momentum bajista")
            
            # 5. Consenso de IA
            if ai_consensus.get('brain_max_prediction'):
                brain_confidence = ai_consensus['brain_max_prediction']['confidence']
                reasoning_parts.append(f"Brain Max AI confirma con {brain_confidence:.1f}% de confianza")
            
            # 6. Condiciones de mercado
            market_condition = market_conditions.get('condition', 'neutral')
            if market_condition != 'neutral':
                reasoning_parts.append(f"Mercado en condición de {market_condition}")
            
            # 7. Acuerdo de señales
            signal_agreement = ai_consensus.get('signal_agreement', 0)
            if signal_agreement > 80:
                reasoning_parts.append("Alto acuerdo entre modelos de IA")
            elif signal_agreement > 50:
                reasoning_parts.append("Acuerdo moderado entre modelos")
            
            return " | ".join(reasoning_parts)
            
        except Exception as e:
            logger.error(f"❌ Error generando razonamiento: {e}")
            return f"Señal {signal_type.upper()} basada en análisis técnico y AI"
    
    def _validate_signal_quality(self, signal: UltraQualitySignal, style: str) -> bool:
        """Validar que la señal cumple los criterios de calidad"""
        try:
            thresholds = self.quality_thresholds.get(style, self.quality_thresholds['day_trading'])
            
            # Verificar confianza mínima
            if signal.confidence < thresholds['min_confidence']:
                return False
            
            # Verificar score de calidad mínimo
            if signal.quality_score < thresholds['min_quality_score']:
                return False
            
            # Verificar ratio riesgo/beneficio mínimo
            if signal.risk_reward_ratio < thresholds['min_risk_reward']:
                return False
            
            # Verificar volatilidad máxima
            current_volatility = signal.volatility_analysis.get('current_volatility', 0)
            if current_volatility > thresholds['max_volatility']:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error validando calidad de señal: {e}")
            return False

# Instancia global del servicio
ultra_quality_service = UltraQualitySignalsService() 