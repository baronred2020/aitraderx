#!/usr/bin/env python3
"""
Servicio Mega Mind para colaboración de cerebros de IA
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime
from typing import Dict, List, Any, Optional
import asyncio

class MegaMindService:
    """
    Servicio que coordina la colaboración entre múltiples cerebros de IA
    """
    
    def __init__(self):
        self.brain_weights = {
            'brain_max': 0.35,
            'brain_ultra': 0.40,
            'brain_predictor': 0.25
        }
        self.collaboration_threshold = 0.75
        self.consensus_threshold = 0.60
        
    async def generate_collaborative_signals(self, market_data: pd.DataFrame, strategy_style: str) -> List[Dict[str, Any]]:
        """
        Genera señales colaborativas usando múltiples cerebros
        """
        try:
            # Obtener predicciones de cada cerebro
            brain_predictions = await self._get_brain_predictions(market_data, strategy_style)
            
            # Calcular consenso y colaboración
            consensus_score = self._calculate_consensus(brain_predictions)
            collaboration_score = self._calculate_collaboration_score(brain_predictions)
            
            # Generar señal final basada en colaboración
            final_signal = self._generate_final_signal(brain_predictions, consensus_score, collaboration_score)
            
            return [final_signal]
            
        except Exception as e:
            print(f"❌ Error en Mega Mind: {e}")
            # Fallback: señal neutral
            return [{
                'signal': 'hold',
                'confidence': 50.0,
                'reason': f'Error en colaboración: {str(e)}',
                'collaboration_score': 0.0,
                'consensus_score': 0.0,
                'brain_contributions': {}
            }]
    
    async def _get_brain_predictions(self, market_data: pd.DataFrame, strategy_style: str) -> Dict[str, Dict]:
        """
        Obtiene predicciones de cada cerebro individual
        """
        predictions = {}
        
        # Simular predicciones de cada cerebro
        for brain_name in self.brain_weights.keys():
            try:
                # Simular predicción del cerebro
                prediction = self._simulate_brain_prediction(brain_name, market_data, strategy_style)
                predictions[brain_name] = prediction
            except Exception as e:
                print(f"⚠️ Error obteniendo predicción de {brain_name}: {e}")
                # Predicción neutral como fallback
                predictions[brain_name] = {
                    'signal': 'hold',
                    'confidence': 50.0,
                    'direction': 'sideways',
                    'reason': f'Error en {brain_name}'
                }
        
        return predictions
    
    def _simulate_brain_prediction(self, brain_name: str, market_data: pd.DataFrame, strategy_style: str) -> Dict[str, Any]:
        """
        Simula predicción de un cerebro específico
        """
        # Simular diferentes comportamientos por cerebro
        if brain_name == 'brain_max':
            # Brain Max: Análisis técnico completo
            confidence = random.uniform(65, 85)
            if confidence > 75:
                signal = random.choice(['buy', 'sell'])
                direction = 'up' if signal == 'buy' else 'down'
            else:
                signal = 'hold'
                direction = 'sideways'
                
        elif brain_name == 'brain_ultra':
            # Brain Ultra: Especializado en scalping
            confidence = random.uniform(70, 90)
            if strategy_style == 'scalping' and confidence > 80:
                signal = random.choice(['buy', 'sell'])
                direction = 'up' if signal == 'buy' else 'down'
            else:
                signal = 'hold'
                direction = 'sideways'
                
        elif brain_name == 'brain_predictor':
            # Brain Predictor: ML predictivo
            confidence = random.uniform(60, 80)
            if confidence > 70:
                signal = random.choice(['buy', 'sell'])
                direction = 'up' if signal == 'buy' else 'down'
            else:
                signal = 'hold'
                direction = 'sideways'
        
        else:
            # Fallback
            signal = 'hold'
            direction = 'sideways'
            confidence = 50.0
        
        return {
            'signal': signal,
            'confidence': confidence,
            'direction': direction,
            'reason': f'Predicción de {brain_name}'
        }
    
    def _calculate_consensus(self, predictions: Dict[str, Dict]) -> float:
        """
        Calcula el nivel de consenso entre cerebros
        """
        signals = [pred['signal'] for pred in predictions.values()]
        
        # Contar señales
        signal_counts = {}
        for signal in signals:
            signal_counts[signal] = signal_counts.get(signal, 0) + 1
        
        # Calcular consenso
        total_predictions = len(signals)
        max_consensus = max(signal_counts.values()) if signal_counts else 0
        consensus_score = max_consensus / total_predictions if total_predictions > 0 else 0
        
        return consensus_score
    
    def _calculate_collaboration_score(self, predictions: Dict[str, Dict]) -> float:
        """
        Calcula la puntuación de colaboración basada en confianza y consenso
        """
        # Promedio de confianzas
        avg_confidence = np.mean([pred['confidence'] for pred in predictions.values()])
        
        # Consenso
        consensus = self._calculate_consensus(predictions)
        
        # Puntuación de colaboración (promedio ponderado)
        collaboration_score = (avg_confidence * 0.6 + consensus * 0.4) / 100
        
        return collaboration_score
    
    def _generate_final_signal(self, predictions: Dict[str, Dict], consensus_score: float, collaboration_score: float) -> Dict[str, Any]:
        """
        Genera la señal final basada en colaboración de cerebros
        """
        # Si hay alto consenso, usar la señal más común
        if consensus_score >= self.consensus_threshold:
            signals = [pred['signal'] for pred in predictions.values()]
            final_signal = max(set(signals), key=signals.count)
        else:
            # Si no hay consenso, usar promedio ponderado
            final_signal = self._weighted_signal_decision(predictions)
        
        # Calcular confianza final
        final_confidence = self._calculate_final_confidence(predictions, collaboration_score)
        
        # Contribuciones de cada cerebro
        brain_contributions = {}
        for brain_name, pred in predictions.items():
            brain_contributions[brain_name] = {
                'signal': pred['signal'],
                'confidence': pred['confidence'],
                'weight': self.brain_weights.get(brain_name, 0.0)
            }
        
        return {
            'signal': final_signal,
            'confidence': final_confidence,
            'collaboration_score': collaboration_score,
            'consensus_score': consensus_score,
            'brain_contributions': brain_contributions,
            'reason': f'Colaboración de {len(predictions)} cerebros (consenso: {consensus_score:.2%})',
            'timestamp': datetime.now().isoformat()
        }
    
    def _weighted_signal_decision(self, predictions: Dict[str, Dict]) -> str:
        """
        Toma decisión basada en pesos de cerebros
        """
        buy_score = 0.0
        sell_score = 0.0
        
        for brain_name, pred in predictions.items():
            weight = self.brain_weights.get(brain_name, 0.0)
            confidence = pred['confidence'] / 100.0
            
            if pred['signal'] == 'buy':
                buy_score += weight * confidence
            elif pred['signal'] == 'sell':
                sell_score += weight * confidence
        
        # Decisión basada en puntuaciones
        if buy_score > sell_score and buy_score > 0.3:
            return 'buy'
        elif sell_score > buy_score and sell_score > 0.3:
            return 'sell'
        else:
            return 'hold'
    
    def _calculate_final_confidence(self, predictions: Dict[str, Dict], collaboration_score: float) -> float:
        """
        Calcula la confianza final basada en colaboración
        """
        # Promedio de confianzas
        avg_confidence = np.mean([pred['confidence'] for pred in predictions.values()])
        
        # Ajustar por colaboración
        final_confidence = avg_confidence * collaboration_score
        
        # Limitar entre 0 y 100
        return max(0.0, min(100.0, final_confidence))
    
    async def get_collaboration_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado de colaboración de Mega Mind
        """
        return {
            'status': 'active',
            'brain_count': len(self.brain_weights),
            'brain_weights': self.brain_weights,
            'collaboration_threshold': self.collaboration_threshold,
            'consensus_threshold': self.consensus_threshold,
            'last_update': datetime.now().isoformat()
        }
    
    async def get_brain_performance(self) -> Dict[str, Any]:
        """
        Obtiene métricas de rendimiento de cada cerebro
        """
        performance = {}
        
        for brain_name in self.brain_weights.keys():
            # Simular métricas de rendimiento
            performance[brain_name] = {
                'accuracy': random.uniform(0.65, 0.85),
                'win_rate': random.uniform(0.60, 0.80),
                'profit_factor': random.uniform(1.2, 2.5),
                'max_drawdown': random.uniform(0.05, 0.15),
                'total_trades': random.randint(100, 500),
                'last_update': datetime.now().isoformat()
            }
        
        return performance 