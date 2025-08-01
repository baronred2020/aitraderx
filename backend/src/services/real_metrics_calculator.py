#!/usr/bin/env python3
"""
Servicio para calcular métricas reales de Precision y Win Rate
basadas en el historial de predicciones completadas.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import yfinance as yf

from config.database_config import db_config

logger = logging.getLogger(__name__)

class RealMetricsCalculator:
    """Calculador de métricas reales basadas en predicciones completadas"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def calculate_real_metrics_for_user(self, user_id: str, brain_type: str = None, pair: str = None, style: str = None) -> Dict[str, Any]:
        """
        Calcular métricas reales para un usuario basadas en predicciones completadas
        
        Args:
            user_id: ID del usuario
            brain_type: Tipo de brain (opcional, para filtrar)
            pair: Par de divisas (opcional, para filtrar)
            style: Estilo de trading (opcional, para filtrar)
        
        Returns:
            Dict con métricas calculadas
        """
        try:
            # Obtener predicciones completadas
            completed_predictions = await self._get_completed_predictions(user_id, brain_type, pair, style)
            
            if not completed_predictions:
                return {
                    'total_predictions': 0,
                    'successful_predictions': 0,
                    'win_rate': 0.0,
                    'precision': 0.0,
                    'average_confidence': 0.0,
                    'average_success_percentage': 0.0,
                    'best_pair': None,
                    'best_brain_type': None,
                    'best_style': None,
                    'recent_performance': [],
                    'metrics_by_pair': {},
                    'metrics_by_brain': {},
                    'metrics_by_style': {}
                }
            
            # Calcular métricas generales
            total_predictions = len(completed_predictions)
            successful_predictions = len([p for p in completed_predictions if p['prediction_success']])
            win_rate = (successful_predictions / total_predictions) * 100 if total_predictions > 0 else 0
            
            # Calcular precisión basada en el porcentaje de éxito promedio
            success_percentages = [p['success_percentage'] for p in completed_predictions if p['success_percentage'] is not None]
            average_success_percentage = sum(success_percentages) / len(success_percentages) if success_percentages else 0
            precision = average_success_percentage  # La precisión es el porcentaje de éxito promedio
            
            # Calcular confianza promedio
            confidences = [p['confidence'] for p in completed_predictions if p['confidence'] is not None]
            average_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Encontrar mejor par
            pair_performance = {}
            for pred in completed_predictions:
                pair = pred['pair']
                if pair not in pair_performance:
                    pair_performance[pair] = {'total': 0, 'successful': 0}
                pair_performance[pair]['total'] += 1
                if pred['prediction_success']:
                    pair_performance[pair]['successful'] += 1
            
            best_pair = None
            best_pair_rate = 0
            for pair, stats in pair_performance.items():
                rate = (stats['successful'] / stats['total']) * 100
                if rate > best_pair_rate:
                    best_pair_rate = rate
                    best_pair = pair
            
            # Encontrar mejor brain type
            brain_performance = {}
            for pred in completed_predictions:
                brain = pred['brain_type']
                if brain not in brain_performance:
                    brain_performance[brain] = {'total': 0, 'successful': 0}
                brain_performance[brain]['total'] += 1
                if pred['prediction_success']:
                    brain_performance[brain]['successful'] += 1
            
            best_brain_type = None
            best_brain_rate = 0
            for brain, stats in brain_performance.items():
                rate = (stats['successful'] / stats['total']) * 100
                if rate > best_brain_rate:
                    best_brain_rate = rate
                    best_brain_type = brain
            
            # Calcular métricas por par
            metrics_by_pair = {}
            for pair, stats in pair_performance.items():
                pair_success_percentages = [p['success_percentage'] for p in completed_predictions 
                                          if p['pair'] == pair and p['success_percentage'] is not None]
                avg_success = sum(pair_success_percentages) / len(pair_success_percentages) if pair_success_percentages else 0
                
                metrics_by_pair[pair] = {
                    'total_predictions': stats['total'],
                    'successful_predictions': stats['successful'],
                    'win_rate': (stats['successful'] / stats['total']) * 100,
                    'precision': avg_success,
                    'average_confidence': sum([p['confidence'] for p in completed_predictions if p['pair'] == pair]) / stats['total']
                }
            
            # Calcular métricas por brain type
            metrics_by_brain = {}
            for brain, stats in brain_performance.items():
                brain_success_percentages = [p['success_percentage'] for p in completed_predictions 
                                           if p['brain_type'] == brain and p['success_percentage'] is not None]
                avg_success = sum(brain_success_percentages) / len(brain_success_percentages) if brain_success_percentages else 0
                
                metrics_by_brain[brain] = {
                    'total_predictions': stats['total'],
                    'successful_predictions': stats['successful'],
                    'win_rate': (stats['successful'] / stats['total']) * 100,
                    'precision': avg_success,
                    'average_confidence': sum([p['confidence'] for p in completed_predictions if p['brain_type'] == brain]) / stats['total']
                }
            
            # Obtener rendimiento reciente (últimas 10 predicciones)
            recent_predictions = sorted(completed_predictions, key=lambda x: x['created_at'], reverse=True)[:10]
            recent_performance = []
            for pred in recent_predictions:
                recent_performance.append({
                    'id': pred['id'],
                    'pair': pred['pair'],
                    'direction': pred['direction'],
                    'prediction_success': pred['prediction_success'],
                    'success_percentage': pred['success_percentage'],
                    'confidence': pred['confidence'],
                    'brain_type': pred['brain_type'],
                    'created_at': pred['created_at']
                })
            
            return {
                'total_predictions': total_predictions,
                'successful_predictions': successful_predictions,
                'win_rate': win_rate,
                'precision': precision,
                'average_confidence': average_confidence,
                'average_success_percentage': average_success_percentage,
                'best_pair': best_pair,
                'best_brain_type': best_brain_type,
                'recent_performance': recent_performance,
                'metrics_by_pair': metrics_by_pair,
                'metrics_by_brain': metrics_by_brain
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating real metrics for user {user_id}: {e}")
            return {}
    
    async def _get_completed_predictions(self, user_id: str, brain_type: str = None, pair: str = None, style: str = None) -> List[Dict[str, Any]]:
        """
        Obtener predicciones completadas con resultados reales
        """
        try:
            with db_config.get_connection() as connection:
                # Construir query base
                query = """
                    SELECT 
                        id, pair, direction, current_price, confidence, `precision`, win_rate,
                        timeframe, reasoning, brain_type, created_at, expires_at,
                        is_completed, actual_price_at_expiry, prediction_success, success_percentage
                    FROM user_predictions 
                    WHERE user_id = %s AND is_completed = 1 AND prediction_success IS NOT NULL
                """
                params = [user_id]
                
                # Agregar filtros opcionales
                if brain_type:
                    query += " AND brain_type = %s"
                    params.append(brain_type)
                
                if pair:
                    query += " AND pair = %s"
                    params.append(pair)
                
                query += " ORDER BY created_at DESC"
                
                cursor = connection.cursor()
                cursor.execute(query, params)
                results = cursor.fetchall()
                cursor.close()
                
                completed_predictions = []
                for row in results:
                    completed_predictions.append({
                        'id': row[0],
                        'pair': row[1],
                        'direction': row[2],
                        'current_price': float(row[3]) if row[3] else 0.0,
                        'confidence': float(row[4]) if row[4] else 0.0,
                        'precision': float(row[5]) if row[5] else 0.0,
                        'win_rate': float(row[6]) if row[6] else 0.0,
                        'timeframe': row[7],
                        'reasoning': row[8],
                        'brain_type': row[9],
                        'created_at': row[10].isoformat() if row[10] else "",
                        'expires_at': row[11].isoformat() if row[11] else "",
                        'is_completed': bool(row[12]),
                        'actual_price_at_expiry': float(row[13]) if row[13] else None,
                        'prediction_success': bool(row[14]) if row[14] is not None else None,
                        'success_percentage': float(row[15]) if row[15] else None
                    })
                
                return completed_predictions
                
        except Exception as e:
            self.logger.error(f"Error getting completed predictions: {e}")
            return []
    
    async def update_prediction_with_real_result(self, prediction_id: int) -> bool:
        """
        Actualizar una predicción con el resultado real del precio
        """
        try:
            with db_config.get_connection() as connection:
                cursor = connection.cursor()
                
                # Obtener la predicción
                cursor.execute("""
                    SELECT pair, direction, current_price, expires_at, created_at
                    FROM user_predictions 
                    WHERE id = %s AND is_completed = 0
                """, (prediction_id,))
                
                prediction = cursor.fetchone()
                if not prediction:
                    cursor.close()
                    return False
                
                pair, direction, current_price, expires_at, created_at = prediction
                
                # Obtener precio real actual
                actual_price = await self._get_current_price(pair)
                if actual_price is None:
                    cursor.close()
                    return False
                
                # Calcular éxito de la predicción
                prediction_success = self._calculate_prediction_success(direction, current_price, actual_price)
                success_percentage = self._calculate_success_percentage(direction, current_price, actual_price)
                
                # Actualizar la predicción
                cursor.execute("""
                    UPDATE user_predictions 
                    SET is_completed = 1, 
                        actual_price_at_expiry = %s,
                        prediction_success = %s,
                        success_percentage = %s
                    WHERE id = %s
                """, (actual_price, prediction_success, success_percentage, prediction_id))
                
                connection.commit()
                cursor.close()
                
                self.logger.info(f"Predicción {prediction_id} actualizada: éxito={prediction_success}, porcentaje={success_percentage:.2f}%")
                return True
                
        except Exception as e:
            self.logger.error(f"Error updating prediction {prediction_id}: {e}")
            return False
    
    async def _get_current_price(self, pair: str) -> Optional[float]:
        """
        Obtener precio actual real usando yfinance
        """
        try:
            # Mapear pares de forex a símbolos de yfinance
            pair_mapping = {
                'EURUSD': 'EURUSD=X',
                'GBPUSD': 'GBPUSD=X',
                'USDJPY': 'USDJPY=X',
                'AUDUSD': 'AUDUSD=X',
                'USDCAD': 'USDCAD=X'
            }
            
            symbol = pair_mapping.get(pair, 'EURUSD=X')
            ticker = yf.Ticker(symbol)
            current_price = ticker.info.get('regularMarketPrice')
            
            return current_price
            
        except Exception as e:
            self.logger.error(f"Error getting current price for {pair}: {e}")
            return None
    
    def _calculate_prediction_success(self, direction: str, current_price: float, actual_price: float) -> bool:
        """
        Calcular si la predicción fue exitosa
        """
        try:
            if direction == 'up':
                return actual_price > current_price
            elif direction == 'down':
                return actual_price < current_price
            else:  # sideways
                threshold = current_price * 0.001  # 0.1% threshold
                return abs(actual_price - current_price) <= threshold
        except Exception as e:
            self.logger.error(f"Error calculating prediction success: {e}")
            return False
    
    def _calculate_success_percentage(self, direction: str, current_price: float, actual_price: float) -> float:
        """
        Calcular porcentaje de éxito de la predicción
        """
        try:
            # Convertir a float para evitar problemas con decimal.Decimal
            current_price = float(current_price)
            actual_price = float(actual_price)
            
            if direction == 'up':
                if actual_price > current_price:
                    return 100.0
                else:
                    movement = (actual_price - current_price) / current_price
                    return max(0, min(100, (movement + 1) * 100))
            elif direction == 'down':
                if actual_price < current_price:
                    return 100.0
                else:
                    movement = (current_price - actual_price) / current_price
                    return max(0, min(100, (movement + 1) * 100))
            else:  # sideways
                threshold = current_price * 0.001
                deviation = abs(actual_price - current_price)
                if deviation <= threshold:
                    return 100.0
                else:
                    return max(0, 100 - (deviation / threshold) * 100)
        except Exception as e:
            self.logger.error(f"Error calculating success percentage: {e}")
            return 0.0
    
    async def complete_expired_predictions(self, user_id: str = None) -> Dict[str, Any]:
        """
        Completar todas las predicciones expiradas con resultados reales
        """
        try:
            with db_config.get_connection() as connection:
                cursor = connection.cursor()
                
                # Obtener predicciones expiradas no completadas
                query = """
                    SELECT id FROM user_predictions 
                    WHERE expires_at < NOW() AND is_completed = 0
                """
                params = []
                
                if user_id:
                    query += " AND user_id = %s"
                    params.append(user_id)
                
                cursor.execute(query, params)
                expired_predictions = cursor.fetchall()
                cursor.close()
                
                completed_count = 0
                for (prediction_id,) in expired_predictions:
                    if await self.update_prediction_with_real_result(prediction_id):
                        completed_count += 1
                
                return {
                    'total_expired': len(expired_predictions),
                    'completed': completed_count,
                    'failed': len(expired_predictions) - completed_count
                }
                
        except Exception as e:
            self.logger.error(f"Error completing expired predictions: {e}")
            return {'total_expired': 0, 'completed': 0, 'failed': 0} 