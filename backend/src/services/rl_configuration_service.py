"""
Servicio de Configuración de Reinforcement Learning
==================================================
Servicio para manejar configuraciones avanzadas de RL por usuario
"""

import logging
from typing import Dict, Optional
from datetime import datetime
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import and_, create_engine

from models.database_models import RLUserConfiguration, User
from config.database_config import DatabaseConfig

logger = logging.getLogger(__name__)

class RLConfigurationService:
    """Servicio para manejar configuraciones de RL"""
    
    def __init__(self, db_config: DatabaseConfig):
        self.db_config = db_config
        
        # Crear engine y session factory para SQLAlchemy
        connection_string = f"mysql+pymysql://{db_config.user}:{db_config.password}@{db_config.host}:{db_config.port}/{db_config.database}"
        self.engine = create_engine(connection_string)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def get_session(self) -> Session:
        """Obtiene una sesión de SQLAlchemy"""
        return self.SessionLocal()
    
    def get_user_configuration(self, user_id: str) -> Dict:
        """Obtiene la configuración actual del usuario"""
        try:
            db = self.get_session()
            config = db.query(RLUserConfiguration).filter(
                RLUserConfiguration.user_id == user_id
            ).first()
            
            if config:
                return {
                    "success": True,
                    "configuration": config.to_dict()
                }
            else:
                # Crear configuración por defecto
                default_config = self._create_default_configuration(user_id, db)
                return {
                    "success": True,
                    "configuration": default_config.to_dict()
                }
                
        except Exception as e:
            logger.error(f"Error obteniendo configuración: {e}")
            return {
                "success": False,
                "error": "Error interno del servidor"
            }
        finally:
            db.close()
    
    def save_user_configuration(
        self,
        user_id: str,
        max_drawdown_percentage: float = 15.0,
        max_position_size_percentage: float = 5.0,
        min_confidence_threshold: float = 70.0,
        retraining_frequency: str = "monthly",
        retraining_enabled: bool = False
    ) -> Dict:
        """Guarda la configuración del usuario"""
        try:
            db = self.get_session()
            
            # Obtener tipo de suscripción del usuario
            user_subscription = self._get_user_subscription_type(user_id, db)
            
            # Validar parámetros según el tipo de suscripción
            validation_result = self._validate_configuration_parameters(
                max_drawdown_percentage,
                max_position_size_percentage,
                min_confidence_threshold,
                retraining_frequency,
                retraining_enabled,
                user_subscription
            )
            
            if not validation_result["valid"]:
                return {
                    "success": False,
                    "error": validation_result["reason"]
                }
            
            # Buscar configuración existente
            config = db.query(RLUserConfiguration).filter(
                RLUserConfiguration.user_id == user_id
            ).first()
            
            if config:
                # Actualizar configuración existente
                config.max_drawdown_percentage = max_drawdown_percentage
                config.max_position_size_percentage = max_position_size_percentage
                config.min_confidence_threshold = min_confidence_threshold
                config.retraining_frequency = retraining_frequency
                config.retraining_enabled = retraining_enabled
                config.updated_at = datetime.utcnow()
            else:
                # Crear nueva configuración
                config = RLUserConfiguration(
                    user_id=user_id,
                    max_drawdown_percentage=max_drawdown_percentage,
                    max_position_size_percentage=max_position_size_percentage,
                    min_confidence_threshold=min_confidence_threshold,
                    retraining_frequency=retraining_frequency,
                    retraining_enabled=retraining_enabled
                )
                db.add(config)
            
            db.commit()
            db.refresh(config)
            
            logger.info(f"Configuración guardada para usuario {user_id}")
            
            return {
                "success": True,
                "configuration": config.to_dict(),
                "message": "Configuración guardada correctamente"
            }
            
        except Exception as e:
            logger.error(f"Error guardando configuración: {e}")
            db.rollback()
            return {
                "success": False,
                "error": "Error interno del servidor"
            }
        finally:
            db.close()
    
    def reset_user_configuration(self, user_id: str) -> Dict:
        """Restaura la configuración del usuario a valores por defecto"""
        try:
            db = self.get_session()
            
            # Buscar configuración existente
            config = db.query(RLUserConfiguration).filter(
                RLUserConfiguration.user_id == user_id
            ).first()
            
            if config:
                # Restaurar valores por defecto
                config.max_drawdown_percentage = 15.0
                config.max_position_size_percentage = 5.0
                config.min_confidence_threshold = 70.0
                config.retraining_frequency = "monthly"
                config.retraining_enabled = False
                config.updated_at = datetime.utcnow()
                
                db.commit()
                db.refresh(config)
                
                logger.info(f"Configuración restaurada para usuario {user_id}")
                
                return {
                    "success": True,
                    "configuration": config.to_dict(),
                    "message": "Configuración restaurada a valores por defecto"
                }
            else:
                return {
                    "success": False,
                    "error": "No se encontró configuración para restaurar"
                }
                
        except Exception as e:
            logger.error(f"Error restaurando configuración: {e}")
            db.rollback()
            return {
                "success": False,
                "error": "Error interno del servidor"
            }
        finally:
            db.close()
    
    def _create_default_configuration(self, user_id: str, db: Session) -> RLUserConfiguration:
        """Crea una configuración por defecto para el usuario"""
        config = RLUserConfiguration(
            user_id=user_id,
            max_drawdown_percentage=15.0,
            max_position_size_percentage=5.0,
            min_confidence_threshold=70.0,
            retraining_frequency="monthly",
            retraining_enabled=False
        )
        
        db.add(config)
        db.commit()
        db.refresh(config)
        
        logger.info(f"Configuración por defecto creada para usuario {user_id}")
        return config
    
    def _get_user_subscription_type(self, user_id: str, db: Session) -> str:
        """Obtiene el tipo de suscripción del usuario"""
        try:
            from models.database_models import UserSubscription
            
            # Buscar suscripción activa del usuario
            subscription = db.query(UserSubscription).filter(
                and_(
                    UserSubscription.user_id == user_id,
                    UserSubscription.status == "active"
                )
            ).first()
            
            if subscription:
                return subscription.plan_type
            else:
                return "freemium"  # Por defecto si no hay suscripción
                
        except Exception as e:
            logger.error(f"Error obteniendo tipo de suscripción: {e}")
            return "freemium"
    
    def _validate_configuration_parameters(
        self,
        max_drawdown_percentage: float,
        max_position_size_percentage: float,
        min_confidence_threshold: float,
        retraining_frequency: str,
        retraining_enabled: bool,
        user_subscription: str
    ) -> Dict:
        """Valida los parámetros de configuración"""
        
        # Validar drawdown máximo
        if not (5.0 <= max_drawdown_percentage <= 25.0):
            return {
                "valid": False,
                "reason": "El drawdown máximo debe estar entre 5% y 25%"
            }
        
        # Validar tamaño de posición
        if not (1.0 <= max_position_size_percentage <= 10.0):
            return {
                "valid": False,
                "reason": "El tamaño máximo de posición debe estar entre 1% y 10%"
            }
        
        # Validar umbral de confianza
        if not (50.0 <= min_confidence_threshold <= 90.0):
            return {
                "valid": False,
                "reason": "El umbral de confianza debe estar entre 50% y 90%"
            }
        
        # Validar frecuencia de reentrenamiento según suscripción
        if user_subscription == "premium":
            # Premium: solo mensual, puede activar/desactivar
            if retraining_frequency != "monthly":
                return {
                    "valid": False,
                    "reason": "Suscripción Premium: solo permite frecuencia mensual"
                }
        elif user_subscription == "institutional":
            # Institutional: semanal o mensual, puede activar/desactivar
            valid_frequencies = ["weekly", "monthly"]
            if retraining_frequency not in valid_frequencies:
                return {
                    "valid": False,
                    "reason": f"Suscripción Institutional: frecuencia debe ser una de: {', '.join(valid_frequencies)}"
                }
        else:
            # Freemium/Basic: no tiene acceso a reentrenamiento
            if retraining_enabled:
                return {
                    "valid": False,
                    "reason": "Tu suscripción no incluye reentrenamiento automático"
                }
        
        return {"valid": True}
    
    def get_configuration_limits(self, user_id: str = None) -> Dict:
        """Obtiene los límites válidos para cada parámetro según la suscripción"""
        try:
            user_subscription = "freemium"  # Por defecto
            
            if user_id:
                db = self.get_session()
                try:
                    user_subscription = self._get_user_subscription_type(user_id, db)
                finally:
                    db.close()
            
            # Configurar opciones de reentrenamiento según suscripción
            if user_subscription == "premium":
                retraining_options = [
                    {"value": "monthly", "label": "Mensual"}
                ]
                retraining_default = "monthly"
            elif user_subscription == "institutional":
                retraining_options = [
                    {"value": "weekly", "label": "Semanal"},
                    {"value": "monthly", "label": "Mensual"}
                ]
                retraining_default = "monthly"
            else:
                retraining_options = []
                retraining_default = None
            
            return {
                "max_drawdown_percentage": {
                    "min": 5.0,
                    "max": 25.0,
                    "default": 15.0,
                    "step": 1.0
                },
                "max_position_size_percentage": {
                    "min": 1.0,
                    "max": 10.0,
                    "default": 5.0,
                    "step": 0.5
                },
                "min_confidence_threshold": {
                    "min": 50.0,
                    "max": 90.0,
                    "default": 70.0,
                    "step": 5.0
                },
                "retraining_frequency": {
                    "options": retraining_options,
                    "default": retraining_default,
                    "available": user_subscription in ["premium", "institutional"]
                },
                "retraining_enabled": {
                    "default": False,
                    "available": user_subscription in ["premium", "institutional"]
                },
                "user_subscription": user_subscription
            }
        except Exception as e:
            logger.error(f"Error getting configuration limits: {e}")
            return {
                "max_drawdown_percentage": {"min": 5.0, "max": 25.0, "default": 15.0, "step": 1.0},
                "max_position_size_percentage": {"min": 1.0, "max": 10.0, "default": 5.0, "step": 0.5},
                "min_confidence_threshold": {"min": 50.0, "max": 90.0, "default": 70.0, "step": 5.0},
                "retraining_frequency": {"options": [], "default": None, "available": False},
                "retraining_enabled": {"default": False, "available": False},
                "user_subscription": "freemium"
            } 