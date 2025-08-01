"""
Prediction Service for AI Trading System
"""
import logging
import numpy as np
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import yfinance as yf

# Importar configuración de base de datos
from config.database_config import db_config

logger = logging.getLogger(__name__)

# Importar BrainTraderService para usar modelos entrenados
try:
    from services.brain_trader_service import BrainTraderService
    brain_trader_service = BrainTraderService()
    logger.info("BrainTraderService importado correctamente")
except ImportError as e:
    logging.error(f"Error importing BrainTraderService: {e}")
    brain_trader_service = None

class PredictionService:
    """Servicio para manejar predicciones"""
    
    def __init__(self, db_session=None):
        self.logger = logging.getLogger(__name__)
        self.db_session = db_session
        self.style_timeframes = {
            'scalping': '5M',
            'day_trading': '15M',
            'swing_trading': '1H',
            'position_trading': '1D'
        }
        self.style_durations = {
            'scalping': 5,
            'day_trading': 15,
            'swing_trading': 60,
            'position_trading': 1440
        }
    
    async def can_generate_prediction(self, user_id: str, style: str = "day_trading", plan_type: str = "starter") -> Dict[str, Any]:
        """Verificar si el usuario puede generar una predicción"""
        try:
            # Obtener límites del plan
            plan_limits = self._get_user_plan_limits(user_id, plan_type)
            max_predictions = plan_limits['max_predictions_per_day']
            predictions_used_today = self._get_predictions_used_today(user_id)
            
            # Verificar si tiene predicciones ilimitadas
            has_unlimited = plan_limits['has_unlimited']
            
            if has_unlimited:
                return {
                    "can_generate": True,
                    "remaining_predictions": -1,  # Ilimitado
                    "max_predictions_per_day": -1,
                    "has_active_prediction": False,
                    "active_prediction_expires": None,
                    "plan_type": plan_type,
                    "analysis_type": "technical_analysis",
                    "timeframe": self.style_timeframes.get(style, "15M"),
                    "duration_minutes": self.style_durations.get(style, 15),
                    "has_unlimited": True
                }
            
            # Calcular predicciones restantes
            remaining_predictions = max_predictions - predictions_used_today
            can_generate = remaining_predictions > 0
            
            self.logger.info(f"Usuario {user_id} - Plan: {plan_type}, Usadas: {predictions_used_today}, Máximo: {max_predictions}, Restantes: {remaining_predictions}, Puede generar: {can_generate}")
            
            return {
                "can_generate": can_generate,
                "remaining_predictions": max(0, remaining_predictions),
                "max_predictions_per_day": max_predictions,
                "has_active_prediction": False,
                "active_prediction_expires": None,
                "plan_type": plan_type,
                "analysis_type": "technical_analysis",
                "timeframe": self.style_timeframes.get(style, "15M"),
                "duration_minutes": self.style_durations.get(style, 15),
                "has_unlimited": False
            }
            
        except Exception as e:
            self.logger.error(f"Error checking if can generate prediction: {e}")
            return {
                "can_generate": False,
                "remaining_predictions": 0,
                "max_predictions_per_day": 5,
                "has_active_prediction": False,
                "active_prediction_expires": None,
                "plan_type": plan_type,
                "analysis_type": "technical_analysis",
                "timeframe": "15M",
                "duration_minutes": 15,
                "has_unlimited": False
            }
    
    def increment_prediction_usage(self, user_id: str) -> bool:
        """Incrementar el contador de uso de predicciones del usuario"""
        try:
            # Mock increment
            return True
        except Exception as e:
            self.logger.error(f"Error incrementing prediction usage: {e}")
            return False
    
    def get_prediction_limits(self, user_id: str, style: str = "day_trading", plan_type: str = "starter") -> Dict:
        """Obtener límites de predicciones del usuario desde la base de datos"""
        try:
            # Obtener límites del plan
            plan_limits = self._get_user_plan_limits(user_id, plan_type)
            has_unlimited = plan_limits['has_unlimited']
            max_predictions = plan_limits['max_predictions_per_day']
            
            # Si tiene predicciones ilimitadas
            if has_unlimited:
                return {
                    "can_generate": True,
                    "remaining_predictions": -1,  # Ilimitado
                    "max_predictions_per_day": -1,  # Ilimitado
                    "has_active_prediction": True,
                    "active_prediction_expires": (datetime.now() + timedelta(minutes=15)).isoformat(),
                    "plan_type": plan_type,
                    "analysis_type": "technical_analysis",
                    "timeframe": self.style_timeframes.get(style, "15M"),
                    "duration_minutes": self.style_durations.get(style, 15),
                    "has_unlimited": True
                }
            
            # Obtener predicciones usadas hoy desde la base de datos
            predictions_used_today = self._get_predictions_used_today(user_id)
            remaining_predictions = max(0, max_predictions - predictions_used_today)
            
            return {
                "can_generate": remaining_predictions > 0,
                "remaining_predictions": remaining_predictions,
                "max_predictions_per_day": max_predictions,
                "has_active_prediction": True,
                "active_prediction_expires": (datetime.now() + timedelta(minutes=15)).isoformat(),
                "plan_type": plan_type,
                "analysis_type": "technical_analysis",
                "timeframe": self.style_timeframes.get(style, "15M"),
                "duration_minutes": self.style_durations.get(style, 15),
                "has_unlimited": False
            }
        except Exception as e:
            self.logger.error(f"Error getting prediction limits: {e}")
            return {}
    
    async def generate_prediction(self, user_id: str, pair: str, brain_type: str, style: str) -> Dict:
        """Generar una nueva predicción"""
        try:
            # Obtener precio real usando yfinance
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
                current_price = ticker.info.get('regularMarketPrice', 1.0925)
                
                # Si no se puede obtener el precio real, usar un precio simulado
                if not current_price or current_price <= 0:
                    current_price = 1.0925 + random.uniform(-0.01, 0.01)
                    
            except Exception as e:
                self.logger.warning(f"No se pudo obtener precio real para {pair}: {e}")
                # Fallback a precio simulado
                current_price = 1.0925 + random.uniform(-0.01, 0.01)
            
            # Usar BrainTraderService para brain_max, brain_ultra, brain_predictor
            if brain_trader_service and brain_type in ['brain_max', 'brain_ultra', 'brain_predictor']:
                self.logger.info(f"Usando {brain_type} con modelos entrenados para {pair}/{style}")
                
                if brain_type == 'brain_max':
                    prediction_result = await brain_trader_service._get_brain_max_prediction(pair, style, current_price)
                elif brain_type == 'brain_ultra':
                    prediction_result = await brain_trader_service._get_brain_ultra_prediction(pair, style, current_price)
                elif brain_type == 'brain_predictor':
                    prediction_result = await brain_trader_service._get_brain_predictor_prediction(pair, style, current_price)
                else:
                    prediction_result = None
                
                if prediction_result and prediction_result.get('direction'):
                    # Usar predicción del modelo entrenado
                    direction = prediction_result['direction']
                    confidence = prediction_result.get('confidence', 75.0)
                    precision = prediction_result.get('precision', 0.0)
                    win_rate = prediction_result.get('win_rate', 0.0)
                    reasoning = prediction_result.get('reasoning', f"Predicción de {brain_type} para {pair}")
                    
                    self.logger.info(f"Predicción generada con {brain_type}: {direction} - {confidence:.2f}% - Precision: {precision:.1f}% - Win Rate: {win_rate:.1f}%")
                else:
                    # No usar fallback - solo datos reales
                    self.logger.error(f"No se pudo obtener predicción real de {brain_type}")
                    return {}
            else:
                # No usar fallback - solo datos reales
                self.logger.error(f"Brain type {brain_type} no soportado o no disponible")
                return {}
            
            # Crear predicción con precio actual real
            prediction = {
                "id": None,  # Se asignará después de guardar
                "pair": pair,
                "direction": direction,
                "current_price": current_price,  # Precio real capturado
                "confidence": confidence,
                "precision": precision if 'precision' in locals() else 0.0,
                "win_rate": win_rate if 'win_rate' in locals() else 0.0,
                "timeframe": self.style_timeframes.get(style, "15M"),
                "reasoning": reasoning,
                "brain_type": brain_type,
                "created_at": datetime.now().isoformat(),
                "expires_at": (datetime.now() + timedelta(minutes=self.style_durations.get(style, 15))).isoformat(),
                "time_remaining": self.style_durations.get(style, 15) * 60,
                "is_completed": False,
                "actual_price_at_expiry": None,
                "prediction_success": None,  # ✅ Siempre None al crear
                "success_percentage": None   # ✅ Siempre None al crear
            }
            
            # ✅ Guardar predicción en la base de datos y obtener el ID real
            prediction_id = self._save_prediction_to_db(user_id, prediction)
            if prediction_id > 0:
                prediction["id"] = prediction_id
                self.logger.info(f"Predicción guardada con ID: {prediction_id}")
            else:
                self.logger.error("Error: No se pudo obtener ID de la predicción guardada")
            
            # Actualizar contador de uso
            self._update_prediction_usage(user_id)
            
            return prediction
        except Exception as e:
            self.logger.error(f"Error generating prediction: {e}")
            return {}
    
    def _save_prediction_to_db(self, user_id: str, prediction: Dict) -> int:
        """Guardar predicción en la tabla predictions existente y retornar el ID"""
        try:
            with db_config.get_connection() as connection:
                # ✅ Incluir todos los campos necesarios
                insert_query = """
                    INSERT INTO user_predictions 
                    (user_id, pair, direction, current_price, confidence, 
                     `precision`, win_rate, timeframe, reasoning, brain_type, created_at, expires_at, is_completed,
                     actual_price_at_expiry, prediction_success, success_percentage) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                cursor = connection.cursor()
                cursor.execute(insert_query, (
                    user_id,
                    prediction['pair'],
                    prediction['direction'],
                    prediction['current_price'],
                    prediction['confidence'],
                    prediction['precision'],
                    prediction['win_rate'],
                    prediction['timeframe'],
                    prediction['reasoning'],
                    prediction['brain_type'],
                    prediction['created_at'],
                    prediction['expires_at'],
                    prediction['is_completed'],
                    prediction['actual_price_at_expiry'],
                    prediction['prediction_success'],
                    prediction['success_percentage']
                ))
                
                # ✅ Obtener el ID de la predicción insertada
                prediction_id = cursor.lastrowid
                
                connection.commit()
                cursor.close()
                return prediction_id
                
        except Exception as e:
            self.logger.error(f"Error saving prediction to database: {e}")
            return 0
    
    async def get_active_prediction(self, user_id: str, style: str = "day_trading") -> Optional[Dict]:
        """Obtener predicción activa del usuario"""
        try:
            # Mock active prediction
            return None  # No active prediction for now
        except Exception as e:
            self.logger.error(f"Error getting active prediction: {e}")
            return None
    
    async def get_prediction_history(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Obtener historial de predicciones del usuario"""
        try:
            with db_config.get_connection() as connection:
                query = """
                    SELECT 
                        id,
                        pair,
                        direction,
                        current_price,
                        confidence,
                        `precision`,
                        win_rate,
                        timeframe,
                        reasoning,
                        brain_type,
                        created_at,
                        expires_at,
                        is_completed,
                        actual_price_at_expiry,
                        prediction_success,
                        success_percentage
                    FROM user_predictions 
                    WHERE user_id = %s 
                    ORDER BY created_at DESC 
                    LIMIT %s
                """
                
                cursor = connection.cursor()
                cursor.execute(query, (user_id, limit))
                results = cursor.fetchall()
                cursor.close()
                
                history = []
                for row in results:
                    history.append({
                        "id": row[0],
                        "pair": row[1],
                        "direction": row[2],
                        "current_price": float(row[3]) if row[3] else 0.0,
                        "confidence": float(row[4]) if row[4] else 0.0,
                        "precision": float(row[5]) if row[5] else 0.0,
                        "win_rate": float(row[6]) if row[6] else 0.0,
                        "timeframe": row[7],
                        "reasoning": row[8],
                        "brain_type": row[9],
                        "created_at": row[10].isoformat() if row[10] else "",
                        "expires_at": row[11].isoformat() if row[11] else "",
                        "is_completed": bool(row[12]),
                        "actual_price_at_expiry": float(row[13]) if row[13] else None,
                        "prediction_success": bool(row[14]) if row[14] is not None else None,
                        "success_percentage": float(row[15]) if row[15] else None
                    })
                
                return history
                
        except Exception as e:
            self.logger.error(f"Error getting prediction history: {e}")
            return []
    
    async def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Obtener estadísticas del usuario"""
        try:
            with db_config.get_connection() as connection:
                cursor = connection.cursor()
                
                # Obtener total de predicciones
                cursor.execute("SELECT COUNT(*) FROM user_predictions WHERE user_id = %s", (user_id,))
                total_predictions = cursor.fetchone()[0]
                
                # Obtener predicciones exitosas
                cursor.execute("SELECT COUNT(*) FROM user_predictions WHERE user_id = %s AND prediction_success = 1", (user_id,))
                successful_predictions = cursor.fetchone()[0]
                
                # Obtener predicciones de hoy
                today = datetime.now().date()
                cursor.execute("SELECT COUNT(*) FROM user_predictions WHERE user_id = %s AND DATE(created_at) = %s", (user_id, today))
                total_predictions_today = cursor.fetchone()[0]
                
                # Obtener mejor par
                cursor.execute("""
                    SELECT pair, COUNT(*) as count 
                    FROM user_predictions 
                    WHERE user_id = %s 
                    GROUP BY pair 
                    ORDER BY count DESC 
                    LIMIT 1
                """, (user_id,))
                best_pair_result = cursor.fetchone()
                best_pair = best_pair_result[0] if best_pair_result else None
                
                # Calcular porcentaje de éxito promedio
                cursor.execute("""
                    SELECT AVG(success_percentage) 
                    FROM user_predictions 
                    WHERE user_id = %s AND success_percentage IS NOT NULL
                """, (user_id,))
                avg_success_result = cursor.fetchone()
                average_success_percentage = float(avg_success_result[0]) if avg_success_result and avg_success_result[0] else 0.0
                
                cursor.close()
                
                # Calcular tasa de éxito
                success_rate = (successful_predictions / total_predictions * 100) if total_predictions > 0 else 0.0
                
                return {
                    "total_predictions": total_predictions,
                    "successful_predictions": successful_predictions,
                    "success_rate": success_rate,
                    "average_success_percentage": average_success_percentage,
                    "best_pair": best_pair,
                    "total_predictions_today": total_predictions_today
                }
                
        except Exception as e:
            self.logger.error(f"Error getting user stats: {e}")
            return {
                "total_predictions": 0,
                "successful_predictions": 0,
                "success_rate": 0.0,
                "average_success_percentage": 0.0,
                "best_pair": None,
                "total_predictions_today": 0
            }
    
    async def complete_expired_predictions(self, user_id: str) -> Dict:
        """Completar predicciones expiradas del usuario"""
        try:
            with db_config.get_connection() as connection:
                cursor = connection.cursor()
                
                # ✅ Buscar predicciones expiradas que no han sido completadas
                query = """
                    SELECT id, pair, direction, current_price, target_price, expires_at
                    FROM user_predictions 
                    WHERE user_id = %s 
                    AND is_completed = FALSE 
                    AND expires_at < NOW()
                    AND actual_price_at_expiry IS NULL
                """
                
                cursor.execute(query, (user_id,))
                expired_predictions = cursor.fetchall()
                
                completed_count = 0
                
                for prediction in expired_predictions:
                    pred_id, pair, direction, current_price, target_price, expires_at = prediction
                    
                    try:
                        # ✅ Obtener precio real actual usando yfinance
                        pair_mapping = {
                            'EURUSD': 'EURUSD=X',
                            'GBPUSD': 'GBPUSD=X',
                            'USDJPY': 'USDJPY=X',
                            'AUDUSD': 'AUDUSD=X',
                            'USDCAD': 'USDCAD=X'
                        }
                        
                        symbol = pair_mapping.get(pair, 'EURUSD=X')
                        ticker = yf.Ticker(symbol)
                        actual_price = ticker.info.get('regularMarketPrice', current_price)
                        
                        # ✅ Si no se puede obtener el precio real, usar el precio actual
                        if not actual_price or actual_price <= 0:
                            actual_price = current_price
                        
                        # ✅ Calcular si la predicción fue exitosa
                        prediction_success = self._calculate_prediction_success(
                            direction, current_price, target_price, actual_price
                        )
                        
                        # ✅ Calcular porcentaje de éxito
                        success_percentage = self._calculate_success_percentage(
                            direction, current_price, target_price, actual_price
                        )
                        
                        # ✅ Actualizar la predicción en la base de datos
                        update_query = """
                            UPDATE user_predictions 
                            SET is_completed = TRUE,
                                actual_price_at_expiry = %s,
                                prediction_success = %s,
                                success_percentage = %s
                            WHERE id = %s
                        """
                        
                        cursor.execute(update_query, (
                            actual_price,
                            prediction_success,
                            success_percentage,
                            pred_id
                        ))
                        
                        completed_count += 1
                        
                    except Exception as e:
                        self.logger.error(f"Error completing prediction {pred_id}: {e}")
                        continue
                
                connection.commit()
                cursor.close()
                
                return {
                    "success": True,
                    "completed": completed_count,
                    "total": len(expired_predictions),
                    "message": f"Se completaron {completed_count} predicciones expiradas"
                }
                
        except Exception as e:
            self.logger.error(f"Error completing expired predictions: {e}")
            return {"success": False, "message": "Error completando predicciones", "completed": 0}
    
    def _calculate_prediction_success(self, direction: str, current_price: float, target_price: float, actual_price: float) -> bool:
        """Calcular si una predicción fue exitosa"""
        try:
            if direction == 'up':
                return actual_price >= target_price
            elif direction == 'down':
                return actual_price <= target_price
            else:  # sideways
                threshold = current_price * 0.001  # 0.1% threshold
                return abs(actual_price - current_price) <= threshold
        except Exception as e:
            self.logger.error(f"Error calculating prediction success: {e}")
            return False
    
    def _calculate_success_percentage(self, direction: str, current_price: float, target_price: float, actual_price: float) -> float:
        """Calcular porcentaje de éxito de una predicción"""
        try:
            if direction == 'up':
                if actual_price >= target_price:
                    return 100.0
                else:
                    movement = (actual_price - current_price) / (target_price - current_price)
                    return max(0, min(100, movement * 100))
            elif direction == 'down':
                if actual_price <= target_price:
                    return 100.0
                else:
                    movement = (current_price - actual_price) / (current_price - target_price)
                    return max(0, min(100, movement * 100))
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
    
    def _has_unlimited_predictions(self, user_id: str, plan_type: str = None) -> bool:
        """Verificar si el usuario tiene predicciones ilimitadas"""
        # TODO: Implementar verificación real de rol de usuario
        # Por ahora, verificar por plan_type
        unlimited_plans = ['institutional', 'admin']
        return plan_type in unlimited_plans
    
    def _get_user_plan_limits(self, user_id: str, plan_type: str = "starter") -> Dict:
        """Obtener límites según el plan del usuario"""
        plan_limits = {
            'starter': 5,
            'trader': 20,
            'expert': 50,
            'premium': 100,
            'institutional': -1,  # Sin límite
            'admin': -1  # Sin límite
        }
        
        max_predictions = plan_limits.get(plan_type, 5)
        return {
            'max_predictions_per_day': max_predictions,
            'has_unlimited': max_predictions == -1
        } 
    
    def _get_predictions_used_today(self, user_id: str) -> int:
        """Obtener el número de predicciones usadas hoy desde la base de datos"""
        try:
            with db_config.get_connection() as connection:
                # Para desarrollo, usar un user_id fijo si no se proporciona uno válido
                if user_id is None or user_id == 0:
                    user_id_str = "0bb94f45-4299-4506-b8c4-9d12d438c79c"  # Usuario demo para desarrollo
                else:
                    user_id_str = str(user_id)
                today = datetime.now().date()
                query = """
                    SELECT COUNT(*) as count 
                    FROM user_predictions 
                    WHERE user_id = %s 
                    AND DATE(created_at) = %s
                """
                
                cursor = connection.cursor()
                cursor.execute(query, (user_id_str, today))
                result = cursor.fetchone()
                cursor.close()
                
                count = result[0] if result else 0
                self.logger.info(f"Predicciones usadas hoy para usuario {user_id_str}: {count}")
                return count
                
        except Exception as e:
            self.logger.error(f"Error getting predictions used today: {e}")
            return 0
    
    def _update_prediction_usage(self, user_id: str) -> bool:
        """Actualizar el contador de uso de predicciones en la base de datos"""
        try:
            # Para la tabla predictions existente, no necesitamos una tabla separada de límites
            # El contador se calcula dinámicamente desde la tabla predictions
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating prediction usage: {e}")
            return False 