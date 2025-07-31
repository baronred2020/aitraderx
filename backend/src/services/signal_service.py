"""
Signal Service
=============
Servicio para manejar señales de trading y sus límites por suscripción
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import os
from models.signal_models import UserSignal, UserSignalLimit, SignalType, SignalStrength

logger = logging.getLogger(__name__)

class SignalService:
    """Servicio para manejar señales de trading"""
    
    def __init__(self, database_url: str = None):
        self.logger = logging.getLogger(__name__)
        
        # Configurar conexión a base de datos
        if database_url:
            self.database_url = database_url
        else:
            # Usar variables de entorno por defecto
            db_host = os.getenv('DB_HOST', 'localhost')
            db_user = os.getenv('DB_USER', 'root')
            db_password = os.getenv('DB_PASSWORD', 'root')
            db_name = os.getenv('DB_NAME', 'trading_db')
            self.database_url = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"
        
        # Crear engine y session factory
        self.engine = create_engine(self.database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
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
        
        # Probar conexión
        self._test_connection()
    
    def _test_connection(self):
        """Verificar conexión a la base de datos"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            self.logger.info("✅ Conexión a MySQL establecida para SignalService")
        except Exception as e:
            self.logger.error(f"❌ Error conectando a MySQL en SignalService: {e}")
            raise
    
    def get_db_session(self) -> Session:
        """Obtener sesión de base de datos"""
        return self.SessionLocal()
    
    async def can_generate_signal(self, user_id: str, style: str = "day_trading", plan_type: str = "starter") -> Dict[str, Any]:
        """Verificar si el usuario puede generar una señal"""
        try:
            # Obtener límites del plan
            plan_limits = self._get_user_plan_limits(user_id, plan_type)
            max_signals = plan_limits['max_signals_per_day']
            signals_used_today = self._get_signals_used_today(user_id)
            
            # Verificar si tiene señales ilimitadas
            has_unlimited = plan_limits['has_unlimited']
            
            if has_unlimited:
                return {
                    "can_generate": True,
                    "remaining_signals": -1,  # Ilimitado
                    "max_signals_per_day": -1,
                    "has_active_signal": False,
                    "active_signal_expires": None,
                    "plan_type": plan_type,
                    "analysis_type": "signal_analysis",
                    "timeframe": self.style_timeframes.get(style, "15M"),
                    "duration_minutes": self.style_durations.get(style, 15),
                    "has_unlimited": True
                }
            
            # Calcular señales restantes
            remaining_signals = max_signals - signals_used_today
            can_generate = remaining_signals > 0
            
            self.logger.info(f"Usuario {user_id} - Plan: {plan_type}, Señales usadas: {signals_used_today}, Máximo: {max_signals}, Restantes: {remaining_signals}, Puede generar: {can_generate}")
            
            return {
                "can_generate": can_generate,
                "remaining_signals": max(0, remaining_signals),
                "max_signals_per_day": max_signals,
                "has_active_signal": False,
                "active_signal_expires": None,
                "plan_type": plan_type,
                "analysis_type": "signal_analysis",
                "timeframe": self.style_timeframes.get(style, "15M"),
                "duration_minutes": self.style_durations.get(style, 15),
                "has_unlimited": False
            }
            
        except Exception as e:
            self.logger.error(f"Error checking if can generate signal: {e}")
            return {
                "can_generate": False,
                "remaining_signals": 0,
                "max_signals_per_day": 5,
                "has_active_signal": False,
                "active_signal_expires": None,
                "plan_type": plan_type,
                "analysis_type": "signal_analysis",
                "timeframe": "15M",
                "duration_minutes": 15,
                "has_unlimited": False
            }
    
    def increment_signal_usage(self, user_id: str) -> bool:
        """Incrementar el contador de uso de señales del usuario"""
        try:
            return self._update_signal_usage(user_id)
        except Exception as e:
            self.logger.error(f"Error incrementing signal usage: {e}")
            return False
    
    def get_signal_limits(self, user_id: str, style: str = "day_trading", plan_type: str = "starter") -> Dict:
        """Obtener límites de señales del usuario desde la base de datos"""
        try:
            # Obtener límites del plan
            plan_limits = self._get_user_plan_limits(user_id, plan_type)
            has_unlimited = plan_limits['has_unlimited']
            max_signals = plan_limits['max_signals_per_day']
            
            # Si tiene señales ilimitadas
            if has_unlimited:
                return {
                    "can_generate": True,
                    "remaining_signals": -1,  # Ilimitado
                    "max_signals_per_day": -1,  # Ilimitado
                    "has_active_signal": True,
                    "active_signal_expires": (datetime.now() + timedelta(minutes=15)).isoformat(),
                    "plan_type": plan_type,
                    "analysis_type": "signal_analysis",
                    "timeframe": self.style_timeframes.get(style, "15M"),
                    "duration_minutes": self.style_durations.get(style, 15),
                    "has_unlimited": True
                }
            
            # Obtener señales usadas hoy desde la base de datos
            signals_used_today = self._get_signals_used_today(user_id)
            remaining_signals = max(0, max_signals - signals_used_today)
            
            return {
                "can_generate": remaining_signals > 0,
                "remaining_signals": remaining_signals,
                "max_signals_per_day": max_signals,
                "has_active_signal": False,
                "active_signal_expires": None,
                "plan_type": plan_type,
                "analysis_type": "signal_analysis",
                "timeframe": self.style_timeframes.get(style, "15M"),
                "duration_minutes": self.style_durations.get(style, 15),
                "has_unlimited": False
            }
            
        except Exception as e:
            self.logger.error(f"Error getting signal limits: {e}")
            return {
                "can_generate": False,
                "remaining_signals": 0,
                "max_signals_per_day": 5,
                "has_active_signal": False,
                "active_signal_expires": None,
                "plan_type": plan_type,
                "analysis_type": "signal_analysis",
                "timeframe": "15M",
                "duration_minutes": 15,
                "has_unlimited": False
            }
    
    async def save_signal_to_db(self, user_id: str, signal_data: Dict) -> int:
        """Guardar señal en la base de datos"""
        try:
            # Mock save for now
            signal_id = 1  # Mock ID
            self.logger.info(f"Señal guardada para usuario {user_id} con ID {signal_id}")
            return signal_id
        except Exception as e:
            self.logger.error(f"Error saving signal to database: {e}")
            return 0
    
    async def get_signal_history(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Obtener historial de señales del usuario"""
        try:
            # Mock data for now
            return []
        except Exception as e:
            self.logger.error(f"Error getting signal history: {e}")
            return []
    
    async def get_user_signal_stats(self, user_id: str) -> Dict[str, Any]:
        """Obtener estadísticas de señales del usuario"""
        try:
            # Mock stats for now
            return {
                "total_signals": 0,
                "successful_signals": 0,
                "success_rate": 0.0,
                "best_pair": "EURUSD",
                "signals_today": 0
            }
        except Exception as e:
            self.logger.error(f"Error getting user signal stats: {e}")
            return {
                "total_signals": 0,
                "successful_signals": 0,
                "success_rate": 0.0,
                "best_pair": "EURUSD",
                "signals_today": 0
            }
    
    def _has_unlimited_signals(self, user_id: str, plan_type: str = None) -> bool:
        """Verificar si el usuario tiene señales ilimitadas"""
        if plan_type in ['institutional', 'admin']:
            return True
        return False
    
    def _get_user_plan_limits(self, user_id: str, plan_type: str = "starter") -> Dict:
        """Obtener límites del plan del usuario"""
        plan_limits = {
            'starter': 5,
            'trader': 20,
            'expert': 50,
            'premium': 100,
            'institutional': -1,  # Sin límite
            'admin': -1  # Sin límite
        }
        
        max_signals = plan_limits.get(plan_type, 5)
        has_unlimited = max_signals == -1
        
        return {
            'max_signals_per_day': max_signals,
            'has_unlimited': has_unlimited
        }
    
    def _get_signals_used_today(self, user_id: str) -> int:
        """Obtener número de señales usadas hoy por el usuario"""
        try:
            with self.get_db_session() as session:
                # Consultar la tabla user_signal_limits
                query = text("""
                    SELECT signals_used_today 
                    FROM user_signal_limits 
                    WHERE user_id = :user_id
                """)
                
                result = session.execute(query, {"user_id": user_id}).fetchone()
                
                if result:
                    signals_used = result[0] or 0
                    self.logger.info(f"Usuario {user_id} ha usado {signals_used} señales hoy")
                    return signals_used
                else:
                    self.logger.warning(f"No se encontraron límites para usuario {user_id}")
                    return 0
                
        except Exception as e:
            self.logger.error(f"Error getting signals used today: {e}")
            return 0
    
    def _update_signal_usage(self, user_id: str) -> bool:
        """Actualizar el contador de uso de señales"""
        try:
            with self.get_db_session() as session:
                # Actualizar el contador de señales usadas hoy
                query = text("""
                    UPDATE user_signal_limits 
                    SET signals_used_today = signals_used_today + 1,
                        last_reset_date = CURRENT_DATE
                    WHERE user_id = :user_id
                """)
                
                result = session.execute(query, {"user_id": user_id})
                session.commit()
                
                if result.rowcount > 0:
                    self.logger.info(f"Contador de señales actualizado para usuario {user_id}")
                    return True
                else:
                    self.logger.warning(f"No se pudo actualizar contador para usuario {user_id}")
                    return False
                
        except Exception as e:
            self.logger.error(f"Error updating signal usage: {e}")
            return False 