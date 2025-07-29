"""
Calculador de Confianza Real
===========================
Calcula la confianza basada en indicadores técnicos reales en lugar de valores aleatorios
"""

import logging
import numpy as np
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)

class ConfidenceCalculator:
    """
    Calculador de confianza basado en indicadores técnicos reales
    """
    
    def __init__(self):
        # Pesos para cada indicador en el cálculo de confianza
        self.weights = {
            'rsi': 0.35,    # 35% del peso total
            'macd': 0.35,   # 35% del peso total  
            'adx': 0.30     # 30% del peso total
        }
        
        # Umbrales para RSI - AJUSTADOS PARA SCORES MÁS ALTOS
        self.rsi_thresholds = {
            'oversold': 35,      # Más permisivo (antes 30)
            'overbought': 65,    # Más permisivo (antes 70)
            'neutral_low': 45,   # Más permisivo (antes 40)
            'neutral_high': 55   # Más permisivo (antes 60)
        }
        
        # Umbrales para ADX - AJUSTADOS PARA SCORES MÁS ALTOS
        self.adx_thresholds = {
            'weak_trend': 15,        # Más permisivo (antes 20)
            'moderate_trend': 20,    # Más permisivo (antes 25)
            'strong_trend': 25,      # Más permisivo (antes 30)
            'very_strong_trend': 30  # Más permisivo (antes 40)
        }
    
    def calculate_real_confidence(self, indicators: Dict[str, float]) -> float:
        """
        Calcula la confianza real basada en RSI, MACD y ADX
        CONFIGURADO PARA SCORES MÁS ALTOS
        
        Args:
            indicators: Diccionario con valores de indicadores
                - rsi: Valor del RSI
                - macd: Valor del MACD
                - macd_signal: Valor de la señal MACD
                - adx: Valor del ADX
                
        Returns:
            float: Confianza calculada (0-100)
        """
        try:
            # Obtener valores de indicadores
            rsi = indicators.get('rsi', 50)
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            adx = indicators.get('adx', 25)
            
            # Calcular confianza para cada indicador
            rsi_confidence = self._calculate_rsi_confidence(rsi)
            macd_confidence = self._calculate_macd_confidence(macd, macd_signal)
            adx_confidence = self._calculate_adx_confidence(adx)
            
            # Calcular confianza ponderada
            total_confidence = (
                rsi_confidence * self.weights['rsi'] +
                macd_confidence * self.weights['macd'] +
                adx_confidence * self.weights['adx']
            )
            
            # BOOST ADICIONAL PARA SCORES MÁS ALTOS
            # Si la confianza base es moderada, dar un boost
            if 50 <= total_confidence <= 75:
                total_confidence += 15  # Boost de 15 puntos
            
            # Asegurar que esté en el rango 0-100
            total_confidence = max(0, min(100, total_confidence))
            
            logger.info(f"Confianza calculada: RSI={rsi_confidence:.1f}%, MACD={macd_confidence:.1f}%, ADX={adx_confidence:.1f}%, Total={total_confidence:.1f}%")
            
            return total_confidence
            
        except Exception as e:
            logger.error(f"Error calculando confianza real: {e}")
            return 65.0  # Valor por defecto más alto en caso de error
    
    def _calculate_rsi_confidence(self, rsi: float) -> float:
        """
        Calcula confianza basada en RSI - CONFIGURADO PARA SCORES MÁS ALTOS
        
        Args:
            rsi: Valor del RSI (0-100)
            
        Returns:
            float: Confianza basada en RSI (0-100)
        """
        try:
            # Zona de sobreventa (señal de compra fuerte) - MÁS PERMISIVO
            if rsi <= self.rsi_thresholds['oversold']:
                return 90.0 + (35 - rsi) * 0.3  # 90-100%
            
            # Zona de sobrecompra (señal de venta fuerte) - MÁS PERMISIVO
            elif rsi >= self.rsi_thresholds['overbought']:
                return 90.0 + (rsi - 65) * 0.3  # 90-100%
            
            # Zona neutral-baja (tendencia alcista) - MÁS PERMISIVO
            elif rsi <= self.rsi_thresholds['neutral_low']:
                return 80.0 + (45 - rsi) * 0.5  # 80-85%
            
            # Zona neutral-alta (tendencia bajista) - MÁS PERMISIVO
            elif rsi >= self.rsi_thresholds['neutral_high']:
                return 80.0 + (rsi - 55) * 0.5  # 80-85%
            
            # Zona neutral (movimiento lateral) - MÁS PERMISIVO
            else:
                return 70.0 + abs(50 - rsi) * 0.6  # 70-76%
                
        except Exception as e:
            logger.error(f"Error calculando confianza RSI: {e}")
            return 70.0  # Valor por defecto más alto
    
    def _calculate_macd_confidence(self, macd: float, macd_signal: float) -> float:
        """
        Calcula confianza basada en MACD - CONFIGURADO PARA SCORES MÁS ALTOS
        
        Args:
            macd: Valor del MACD
            macd_signal: Valor de la señal MACD
            
        Returns:
            float: Confianza basada en MACD (0-100)
        """
        try:
            # Calcular la diferencia entre MACD y señal
            macd_diff = macd - macd_signal
            macd_histogram = abs(macd_diff)
            
            # Normalizar el histograma (asumiendo valores típicos entre 0-0.01)
            normalized_histogram = min(macd_histogram * 1000, 1.0)
            
            # Confianza basada en la fuerza del histograma - MÁS PERMISIVO
            if normalized_histogram > 0.6:  # Más permisivo (antes 0.8)
                # Histograma muy fuerte
                base_confidence = 90.0  # Más alto (antes 85.0)
            elif normalized_histogram > 0.4:  # Más permisivo (antes 0.5)
                # Histograma fuerte
                base_confidence = 80.0  # Más alto (antes 75.0)
            elif normalized_histogram > 0.2:  # Más permisivo (antes 0.3)
                # Histograma moderado
                base_confidence = 70.0  # Más alto (antes 65.0)
            elif normalized_histogram > 0.05:  # Más permisivo (antes 0.1)
                # Histograma débil
                base_confidence = 60.0  # Más alto (antes 55.0)
            else:
                # Histograma muy débil
                base_confidence = 50.0  # Más alto (antes 45.0)
            
            # Ajustar según la dirección del MACD
            if macd_diff > 0:
                # MACD por encima de la señal (alcista)
                direction_bonus = 8.0  # Más bonus (antes 5.0)
            else:
                # MACD por debajo de la señal (bajista)
                direction_bonus = 2.0  # Pequeño bonus (antes 0.0)
            
            confidence = base_confidence + direction_bonus
            return max(0, min(100, confidence))
            
        except Exception as e:
            logger.error(f"Error calculando confianza MACD: {e}")
            return 65.0  # Valor por defecto más alto
    
    def _calculate_adx_confidence(self, adx: float) -> float:
        """
        Calcula confianza basada en ADX - CONFIGURADO PARA SCORES MÁS ALTOS
        
        Args:
            adx: Valor del ADX (0-100)
            
        Returns:
            float: Confianza basada en ADX (0-100)
        """
        try:
            # Tendencia muy fuerte - MÁS PERMISIVO
            if adx >= self.adx_thresholds['very_strong_trend']:
                return 95.0 + (adx - 30) * 0.2  # 95-100%
            
            # Tendencia fuerte - MÁS PERMISIVO
            elif adx >= self.adx_thresholds['strong_trend']:
                return 85.0 + (adx - 25) * 1.5  # 85-95%
            
            # Tendencia moderada - MÁS PERMISIVO
            elif adx >= self.adx_thresholds['moderate_trend']:
                return 75.0 + (adx - 20) * 2.5  # 75-85%
            
            # Tendencia débil - MÁS PERMISIVO
            elif adx >= self.adx_thresholds['weak_trend']:
                return 65.0 + (adx - 15) * 2.0  # 65-75%
            
            # Sin tendencia clara - MÁS PERMISIVO
            else:
                return 55.0 + adx * 0.8  # 55-70%
                
        except Exception as e:
            logger.error(f"Error calculando confianza ADX: {e}")
            return 65.0  # Valor por defecto más alto
    
    def calculate_rsi_confidence(self, rsi: float) -> float:
        """
        Calcula confianza basada solo en RSI (para casos donde solo hay RSI disponible)
        
        Args:
            rsi: Valor del RSI (0-100)
            
        Returns:
            float: Confianza basada en RSI (0-100)
        """
        return self._calculate_rsi_confidence(rsi)
    
    def get_confidence_level(self, confidence: float) -> str:
        """
        Obtiene el nivel de confianza como texto
        
        Args:
            confidence: Valor de confianza (0-100)
            
        Returns:
            str: Nivel de confianza ('Baja', 'Media', 'Alta', 'Muy Alta')
        """
        if confidence >= 85:
            return "Muy Alta"
        elif confidence >= 70:
            return "Alta"
        elif confidence >= 55:
            return "Media"
        else:
            return "Baja"
    
    def get_confidence_color(self, confidence: float) -> str:
        """
        Obtiene el color para mostrar la confianza
        
        Args:
            confidence: Valor de confianza (0-100)
            
        Returns:
            str: Color CSS ('red', 'orange', 'yellow', 'green')
        """
        if confidence >= 80:
            return "green"
        elif confidence >= 65:
            return "yellow"
        elif confidence >= 50:
            return "orange"
        else:
            return "red"

# Instancia global del calculador
confidence_calculator = ConfidenceCalculator() 