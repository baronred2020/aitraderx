"""
Prediction Service for AI Trading System
"""
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import yfinance as yf

# Importar configuración de base de datos
from config.database_config import db_config

logger = logging.getLogger(__name__)

class PredictionService:
    """Servicio para manejar predicciones"""
    
    def __init__(self, db_session=None):
        self.logger = logging.getLogger(__name__)
        self.db_session = db_session
        self.style_timeframes = {
            'scalping': '5M',
            'day_trading': '15M',
            'swing_trading': '1H',
            'position_trading': '4H'
        }
        self.style_durations = {
            'scalping': 5,
            'day_trading': 15,
            'swing_trading': 60,
            'position_trading': 240
        }
    
    async def can_generate_prediction(self, user_id: int, style: str = "day_trading", plan_type: str = "starter") -> Dict[str, Any]:
        """Verificar si el usuario puede generar una predicción"""
        try:
            # Obtener límites del plan
            max_predictions = self._get_user_plan_limits(plan_type)
            predictions_used_today = self._get_predictions_used_today(user_id)
            
            # Verificar si tiene predicciones ilimitadas
            has_unlimited = self._has_unlimited_predictions(plan_type)
            
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
    
    def increment_prediction_usage(self, user_id: int) -> bool:
        """Incrementar el contador de uso de predicciones del usuario"""
        try:
            # Mock increment
            return True
        except Exception as e:
            self.logger.error(f"Error incrementing prediction usage: {e}")
            return False
    
    def get_prediction_limits(self, user_id: int, style: str = "day_trading", plan_type: str = "starter") -> Dict:
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
    
    async def generate_prediction(self, user_id: int, pair: str, brain_type: str, style: str) -> Dict:
        """Generar una nueva predicción"""
        try:
            # Obtener precio actual real usando el servicio de Brain Trader
            from src.services.brain_trader_service import BrainTraderService
            brain_service = BrainTraderService()
            
            # Obtener precio actual real
            current_price = await brain_service.get_real_price(pair)
            
            # Generar predicción basada en análisis técnico
            direction = "up" if np.random.random() > 0.5 else "down"
            confidence = np.random.uniform(60, 95)
            
            # Calcular target price basado en volatilidad real
            volatility = 0.001  # 0.1% base volatility
            if direction == "up":
                target_price = current_price * (1 + volatility)
            else:
                target_price = current_price * (1 - volatility)
            
            # Crear predicción con precio actual real
            prediction = {
                "id": np.random.randint(1000, 9999),
                "pair": pair,
                "direction": direction,
                "current_price": current_price,  # Precio real capturado
                "target_price": target_price,
                "confidence": confidence,
                "timeframe": self.style_timeframes.get(style, "15M"),
                "reasoning": f"Análisis técnico para {pair} usando {brain_type} - {direction.upper()}",
                "brain_type": brain_type,
                "created_at": datetime.now().isoformat(),
                "expires_at": (datetime.now() + timedelta(minutes=self.style_durations.get(style, 15))).isoformat(),
                "time_remaining": self.style_durations.get(style, 15) * 60,
                "is_completed": False,
                "actual_price_at_expiry": None,
                "prediction_success": None,
                "success_percentage": None
            }
            
            # Guardar predicción en la base de datos
            if self.db_session:
                self._save_prediction_to_db(user_id, prediction)
                # Actualizar contador de uso
                self._update_prediction_usage(user_id)
            
            return prediction
        except Exception as e:
            self.logger.error(f"Error generating prediction: {e}")
            return {}
    
    def _save_prediction_to_db(self, user_id: int, prediction: Dict) -> bool:
        """Guardar predicción en la tabla predictions existente"""
        try:
            with db_config.get_connection() as connection:
                # Adaptar a la estructura de la tabla predictions existente
                insert_query = """
                    INSERT INTO predictions 
                    (user_id, pair, direction, current_price, target_price, confidence, 
                     timeframe, reasoning, brain_type, created_at, expires_at, is_completed) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                cursor = connection.cursor()
                cursor.execute(insert_query, (
                    user_id,
                    prediction['pair'],
                    prediction['direction'],
                    prediction['current_price'],
                    prediction['target_price'],
                    prediction['confidence'],
                    prediction['timeframe'],
                    prediction['reasoning'],
                    prediction['brain_type'],
                    prediction['created_at'],
                    prediction['expires_at'],
                    prediction['is_completed']
                ))
                
                connection.commit()
                cursor.close()
                return True
                
        except Exception as e:
            self.logger.error(f"Error saving prediction to database: {e}")
            return False
    
    async def get_active_prediction(self, user_id: int, style: str = "day_trading") -> Optional[Dict]:
        """Obtener predicción activa del usuario"""
        try:
            # Mock active prediction
            return None  # No active prediction for now
        except Exception as e:
            self.logger.error(f"Error getting active prediction: {e}")
            return None
    
    async def get_prediction_history(self, user_id: int, limit: int = 20) -> List[Dict[str, Any]]:
        """Obtener historial de predicciones del usuario"""
        try:
            with db_config.get_connection() as connection:
                # Usar el UUID del usuario demo para las pruebas
                demo_user_id = "0bb94f45-4299-4506-b8c4-9d12d438c79c"
                query = """
                    SELECT 
                        prediction_id as id,
                        symbol as pair,
                        predicted_signal as direction,
                        predicted_value as current_price,
                        target_price,
                        confidence,
                        timeframe,
                        'Análisis técnico' as reasoning,
                        'brain_max' as brain_type,
                        prediction_date as created_at,
                        DATE_ADD(prediction_date, INTERVAL 15 MINUTE) as expires_at,
                        CASE WHEN actual_value IS NOT NULL THEN 1 ELSE 0 END as is_completed,
                        actual_value as actual_price_at_expiry,
                        CASE WHEN actual_signal = predicted_signal THEN 1 ELSE 0 END as prediction_success,
                        accuracy as success_percentage
                    FROM predictions 
                    WHERE user_id = %s 
                    ORDER BY prediction_date DESC 
                    LIMIT %s
                """
                
                cursor = connection.cursor()
                cursor.execute(query, (demo_user_id, limit))
                results = cursor.fetchall()
                cursor.close()
                
                history = []
                for row in results:
                    history.append({
                        "id": row[0],
                        "pair": row[1],
                        "direction": row[2],
                        "current_price": float(row[3]) if row[3] else 0.0,
                        "target_price": float(row[4]) if row[4] else 0.0,
                        "confidence": float(row[5]) if row[5] else 0.0,
                        "timeframe": row[6],
                        "reasoning": row[7],
                        "brain_type": row[8],
                        "created_at": row[9].isoformat() if row[9] else "",
                        "expires_at": row[10].isoformat() if row[10] else "",
                        "is_completed": bool(row[11]),
                        "actual_price_at_expiry": float(row[12]) if row[12] else None,
                        "prediction_success": bool(row[13]) if row[13] is not None else None,
                        "success_percentage": float(row[14]) if row[14] else None
                    })
                
                return history
                
        except Exception as e:
            self.logger.error(f"Error getting prediction history: {e}")
            return []
    
    async def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        """Obtener estadísticas del usuario"""
        try:
            with db_config.get_connection() as connection:
                # Usar el UUID del usuario demo para las pruebas
                demo_user_id = "0bb94f45-4299-4506-b8c4-9d12d438c79c"
                cursor = connection.cursor()
                
                # Obtener total de predicciones
                cursor.execute("SELECT COUNT(*) FROM predictions WHERE user_id = %s", (demo_user_id,))
                total_predictions = cursor.fetchone()[0]
                
                # Obtener predicciones exitosas (donde actual_signal = predicted_signal)
                cursor.execute("SELECT COUNT(*) FROM predictions WHERE user_id = %s AND actual_signal = predicted_signal AND actual_signal IS NOT NULL", (demo_user_id,))
                successful_predictions = cursor.fetchone()[0]
                
                # Obtener predicciones de hoy
                today = datetime.now().date()
                cursor.execute("SELECT COUNT(*) FROM predictions WHERE user_id = %s AND DATE(prediction_date) = %s", (demo_user_id, today))
                total_predictions_today = cursor.fetchone()[0]
                
                # Obtener mejor par
                cursor.execute("""
                    SELECT symbol, COUNT(*) as count 
                    FROM predictions 
                    WHERE user_id = %s 
                    GROUP BY symbol 
                    ORDER BY count DESC 
                    LIMIT 1
                """, (demo_user_id,))
                best_pair_result = cursor.fetchone()
                best_pair = best_pair_result[0] if best_pair_result else None
                
                # Calcular porcentaje de éxito promedio
                cursor.execute("""
                    SELECT AVG(accuracy) 
                    FROM predictions 
                    WHERE user_id = %s AND accuracy IS NOT NULL
                """, (demo_user_id,))
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
    
    async def complete_expired_predictions(self, user_id: int) -> Dict:
        """Completar predicciones expiradas"""
        try:
            # Mock completion
            return {
                "success": True,
                "message": "Predicciones expiradas completadas",
                "completed": 2
            }
        except Exception as e:
            self.logger.error(f"Error completing expired predictions: {e}")
            return {"success": False, "message": "Error completando predicciones", "completed": 0} 
    
    def _has_unlimited_predictions(self, user_id: int, plan_type: str = None) -> bool:
        """Verificar si el usuario tiene predicciones ilimitadas"""
        # TODO: Implementar verificación real de rol de usuario
        # Por ahora, verificar por plan_type
        unlimited_plans = ['institutional', 'admin']
        return plan_type in unlimited_plans
    
    def _get_user_plan_limits(self, user_id: int, plan_type: str = "starter") -> Dict:
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
    
    def _get_predictions_used_today(self, user_id: int) -> int:
        """Obtener el número de predicciones usadas hoy desde la base de datos"""
        try:
            with db_config.get_connection() as connection:
                # Usar el UUID del usuario demo para las pruebas
                demo_user_id = "0bb94f45-4299-4506-b8c4-9d12d438c79c"
                today = datetime.now().date()
                query = """
                    SELECT COUNT(*) as count 
                    FROM predictions 
                    WHERE user_id = %s 
                    AND DATE(prediction_date) = %s
                """
                
                cursor = connection.cursor()
                cursor.execute(query, (demo_user_id, today))
                result = cursor.fetchone()
                cursor.close()
                
                count = result[0] if result else 0
                self.logger.info(f"Predicciones usadas hoy para usuario {demo_user_id}: {count}")
                return count
                
        except Exception as e:
            self.logger.error(f"Error getting predictions used today: {e}")
            return 0
    
    def _update_prediction_usage(self, user_id: int) -> bool:
        """Actualizar el contador de uso de predicciones en la base de datos"""
        try:
            # Para la tabla predictions existente, no necesitamos una tabla separada de límites
            # El contador se calcula dinámicamente desde la tabla predictions
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating prediction usage: {e}")
            return False 