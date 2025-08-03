"""
Servicio de Entrenamiento de Reinforcement Learning
==================================================
Servicio para manejar el entrenamiento de RL con límites de producción
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import and_, desc, create_engine
import uuid
import os
import json

from models.database_models import RLTrainingSession, User
from config.database_config import DatabaseConfig

logger = logging.getLogger(__name__)

class RLTrainingService:
    """Servicio para manejar entrenamiento de RL"""
    
    def __init__(self, db_config: DatabaseConfig):
        self.db_config = db_config
        self.active_sessions: Dict[str, Dict] = {}  # session_id -> session_info
        
        # Crear engine y session factory para SQLAlchemy
        connection_string = f"mysql+pymysql://{db_config.user}:{db_config.password}@{db_config.host}:{db_config.port}/{db_config.database}"
        self.engine = create_engine(connection_string)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def get_session(self) -> Session:
        """Obtiene una sesión de SQLAlchemy"""
        return self.SessionLocal()
        
    def can_user_train(self, user_id: str, db: Session) -> Dict[str, any]:
        """
        Verifica si un usuario puede iniciar un entrenamiento
        - Máximo 1 entrenamiento por semana
        - Límites de episodios según el plan
        """
        try:
            # Verificar si hay una sesión activa
            active_session = db.query(RLTrainingSession).filter(
                and_(
                    RLTrainingSession.user_id == user_id,
                    RLTrainingSession.status == "running"
                )
            ).first()
            
            if active_session:
                return {
                    "can_train": False,
                    "reason": "Ya tienes una sesión de entrenamiento activa",
                    "session_id": active_session.session_id
                }
            
            # Verificar entrenamiento de la última semana
            week_ago = datetime.utcnow() - timedelta(days=7)
            recent_session = db.query(RLTrainingSession).filter(
                and_(
                    RLTrainingSession.user_id == user_id,
                    RLTrainingSession.started_at >= week_ago,
                    RLTrainingSession.status.in_(["completed", "failed"])
                )
            ).order_by(desc(RLTrainingSession.started_at)).first()
            
            if recent_session:
                days_until_next = 7 - (datetime.utcnow() - recent_session.started_at).days
                return {
                    "can_train": False,
                    "reason": f"Ya realizaste un entrenamiento esta semana. Puedes entrenar nuevamente en {days_until_next} días",
                    "days_until_next": days_until_next
                }
            
            return {"can_train": True, "reason": "Puedes iniciar entrenamiento"}
            
        except Exception as e:
            logger.error(f"Error verificando capacidad de entrenamiento: {e}")
            return {"can_train": False, "reason": "Error interno del servidor"}
    
    def validate_training_params(self, episodes: int, user_plan: str = "starter") -> Dict[str, any]:
        """
        Valida los parámetros de entrenamiento según el plan del usuario
        """
        # Límites por plan
        limits = {
            "starter": {"min": 100, "max": 1000, "recommended": 500},
            "basic": {"min": 100, "max": 2000, "recommended": 1000},
            "pro": {"min": 100, "max": 5000, "recommended": 2000},
            "elite": {"min": 100, "max": 10000, "recommended": 5000}
        }
        
        plan_limits = limits.get(user_plan, limits["starter"])
        
        if episodes < plan_limits["min"]:
            return {
                "valid": False,
                "reason": f"Mínimo {plan_limits['min']} episodios para tu plan"
            }
        
        if episodes > plan_limits["max"]:
            return {
                "valid": False,
                "reason": f"Máximo {plan_limits['max']} episodios para tu plan"
            }
        
        return {
            "valid": True,
            "limits": plan_limits,
            "estimated_minutes": self._estimate_training_time(episodes)
        }
    
    def _estimate_training_time(self, episodes: int) -> int:
        """Estima el tiempo de entrenamiento en minutos"""
        # Estimación basada en episodios
        base_time_per_episode = 0.1  # minutos por episodio
        return max(5, int(episodes * base_time_per_episode))
    
    async def start_training(
        self, 
        user_id: str, 
        episodes: int, 
        db: Session,
        algorithm: str = "dqn",
        trading_pair: str = "EURUSD",
        timeframe: str = "1h"
    ) -> Dict[str, any]:
        """
        Inicia una sesión de entrenamiento
        """
        try:
            # Crear sesión en base de datos
            session = RLTrainingSession(
                user_id=user_id,
                episodes=episodes,
                algorithm=algorithm,
                trading_pair=trading_pair,
                timeframe=timeframe,
                total_episodes=episodes,
                estimated_duration_minutes=self._estimate_training_time(episodes)
            )
            
            db.add(session)
            db.commit()
            db.refresh(session)
            
            # Iniciar entrenamiento en background
            asyncio.create_task(self._run_training(session.session_id, episodes))
            
            # Guardar en sesiones activas
            self.active_sessions[session.session_id] = {
                "session_id": session.session_id,
                "user_id": user_id,
                "episodes": episodes,
                "current_episode": 0,
                "started_at": datetime.utcnow(),
                "status": "running"
            }
            
            return {
                "success": True,
                "session_id": session.session_id,
                "message": "Entrenamiento iniciado correctamente"
            }
            
        except Exception as e:
            logger.error(f"Error iniciando entrenamiento: {e}")
            db.rollback()
            return {
                "success": False,
                "error": "Error interno del servidor"
            }
    
    async def _run_training(self, session_id: str, total_episodes: int):
        """
        Ejecuta el entrenamiento en background
        """
        try:
            # Simular entrenamiento con actualizaciones de progreso
            for episode in range(1, total_episodes + 1):
                # Simular tiempo de entrenamiento por episodio
                await asyncio.sleep(0.1)  # 100ms por episodio para simulación
                
                # Actualizar progreso
                progress = episode / total_episodes
                self._update_session_progress(session_id, episode, progress)
                
                # Verificar si la sesión fue cancelada
                if session_id not in self.active_sessions:
                    break
            
            # Marcar como completado
            self._complete_session(session_id, success=True)
            
        except Exception as e:
            logger.error(f"Error en entrenamiento {session_id}: {e}")
            self._complete_session(session_id, success=False, error=str(e))
    
    def _update_session_progress(self, session_id: str, current_episode: int, progress: float):
        """Actualiza el progreso de la sesión en la base de datos"""
        try:
            db = self.get_session()
            session = db.query(RLTrainingSession).filter(
                RLTrainingSession.session_id == session_id
            ).first()
            
            if session:
                session.current_episode = current_episode
                session.progress = progress
                session.updated_at = datetime.utcnow()
                db.commit()
                
                # Actualizar sesión activa
                if session_id in self.active_sessions:
                    self.active_sessions[session_id]["current_episode"] = current_episode
                    
        except Exception as e:
            logger.error(f"Error actualizando progreso: {e}")
        finally:
            db.close()
    
    def _complete_session(self, session_id: str, success: bool, error: str = None):
        """Marca la sesión como completada"""
        try:
            db = self.get_session()
            session = db.query(RLTrainingSession).filter(
                RLTrainingSession.session_id == session_id
            ).first()
            
            if session:
                session.status = "completed" if success else "failed"
                session.progress = 1.0 if success else session.progress
                session.completed_at = datetime.utcnow()
                session.error_message = error
                
                # Calcular duración real
                if session.started_at:
                    duration = session.completed_at - session.started_at
                    session.actual_duration_minutes = duration.total_seconds() / 60
                
                # Generar resultados simulados
                if success:
                    session.final_reward = 0.85
                    session.win_rate = 0.72
                    session.sharpe_ratio = 1.2
                    session.max_drawdown = 0.15
                    
                    # Generar rutas de archivos
                    session.model_path = f"models/rl_trained/{session_id}/model.pkl"
                    session.training_log_path = f"logs/rl_training/{session_id}/training.log"
                
                db.commit()
                
        except Exception as e:
            logger.error(f"Error completando sesión: {e}")
        finally:
            db.close()
            
        # Remover de sesiones activas
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
    
    def get_training_progress(self, session_id: str) -> Dict[str, any]:
        """Obtiene el progreso actual del entrenamiento"""
        try:
            # Verificar sesión activa en memoria
            if session_id in self.active_sessions:
                session_info = self.active_sessions[session_id]
                return {
                    "is_training": True,
                    "progress": session_info["current_episode"] / session_info["episodes"],
                    "current_episode": session_info["current_episode"],
                    "total_episodes": session_info["episodes"],
                    "status": "running"
                }
            
            # Verificar en base de datos
            db = self.get_session()
            session = db.query(RLTrainingSession).filter(
                RLTrainingSession.session_id == session_id
            ).first()
            
            if session:
                return {
                    "is_training": session.status == "running",
                    "progress": session.progress,
                    "current_episode": session.current_episode,
                    "total_episodes": session.total_episodes,
                    "status": session.status,
                    "error_message": session.error_message
                }
            
            return {"is_training": False, "error": "Sesión no encontrada"}
            
        except Exception as e:
            logger.error(f"Error obteniendo progreso: {e}")
            return {"is_training": False, "error": "Error interno"}
        finally:
            db.close()
    
    def cancel_training(self, session_id: str, user_id: str) -> Dict[str, any]:
        """Cancela un entrenamiento en curso"""
        try:
            # Verificar que el usuario sea dueño de la sesión
            db = self.get_session()
            session = db.query(RLTrainingSession).filter(
                and_(
                    RLTrainingSession.session_id == session_id,
                    RLTrainingSession.user_id == user_id,
                    RLTrainingSession.status == "running"
                )
            ).first()
            
            if not session:
                return {"success": False, "error": "Sesión no encontrada o no autorizada"}
            
            # Marcar como cancelada
            session.status = "cancelled"
            session.completed_at = datetime.utcnow()
            db.commit()
            
            # Remover de sesiones activas
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            return {"success": True, "message": "Entrenamiento cancelado"}
            
        except Exception as e:
            logger.error(f"Error cancelando entrenamiento: {e}")
            return {"success": False, "error": "Error interno"}
        finally:
            db.close()
    
    def get_user_training_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Obtiene el historial de entrenamientos del usuario"""
        try:
            db = self.get_session()
            sessions = db.query(RLTrainingSession).filter(
                RLTrainingSession.user_id == user_id
            ).order_by(desc(RLTrainingSession.started_at)).limit(limit).all()
            
            return [
                {
                    "session_id": session.session_id,
                    "episodes": session.episodes,
                    "algorithm": session.algorithm,
                    "status": session.status,
                    "progress": session.progress,
                    "started_at": session.started_at.isoformat() if session.started_at else None,
                    "completed_at": session.completed_at.isoformat() if session.completed_at else None,
                    "final_reward": session.final_reward,
                    "win_rate": session.win_rate
                }
                for session in sessions
            ]
            
        except Exception as e:
            logger.error(f"Error obteniendo historial: {e}")
            return []
        finally:
            db.close() 