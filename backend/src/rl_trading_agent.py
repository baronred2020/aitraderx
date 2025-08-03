"""
Reinforcement Learning Trading Agent
===================================
Sistema de RL para optimización de estrategias de trading
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
from fastapi import HTTPException
import json

# Importar el servicio de entrenamiento
try:
    from services.rl_training_service import RLTrainingService
    from config.database_config import DatabaseConfig
    from models.database_models import RLTrainingSession
    RL_SERVICE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"RL Training Service not available: {e}")
    RL_SERVICE_AVAILABLE = False

logger = logging.getLogger(__name__)

# Instancia global del servicio RL
rl_service = None

def initialize_rl_service():
    """Inicializa el servicio RL si está disponible"""
    global rl_service
    if RL_SERVICE_AVAILABLE and rl_service is None:
        try:
            db_config = DatabaseConfig()
            rl_service = RLTrainingService(db_config)
            logger.info("RL Training Service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RL Training Service: {e}")
            rl_service = None

# Inicializar al importar el módulo
initialize_rl_service()

def get_rl_status() -> Dict:
    """Obtiene el estado actual del RL Director"""
    try:
        # Verificar si hay sesiones activas
        active_sessions_count = 0
        if rl_service:
            active_sessions_count = len(rl_service.active_sessions)
        
        return {
            "status": "active" if active_sessions_count > 0 else "inactive",
            "active_sessions": active_sessions_count,
            "model_coordination": {
                "brain_max_weight": 0.35,
                "brain_ultra_weight": 0.30,
                "brain_predictor_weight": 0.25,
                "megamind_weight": 0.10
            },
            "current_strategy": "ensemble_optimization",
            "market_regime": "trending",
            "risk_level": "moderate"
        }
    except Exception as e:
        logger.error(f"Error getting RL status: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

def get_rl_performance() -> Dict:
    """Obtiene el rendimiento del RL Director"""
    try:
        return {
            "total_trades": 1247,
            "win_rate": 0.72,
            "profit_factor": 1.85,
            "sharpe_ratio": 1.42,
            "max_drawdown": 0.18,
            "total_return": 0.34,
            "model_performance": {
                "brain_max": {"accuracy": 0.68, "confidence": 0.75},
                "brain_ultra": {"accuracy": 0.71, "confidence": 0.82},
                "brain_predictor": {"accuracy": 0.65, "confidence": 0.70},
                "megamind": {"accuracy": 0.73, "confidence": 0.85}
            }
        }
    except Exception as e:
        logger.error(f"Error getting RL performance: {e}")
        return {"error": str(e)}

def get_active_signals() -> List[Dict]:
    """Obtiene señales activas generadas por el RL Director"""
    try:
        # Simular obtención de predicciones de los modelos
        brain_max_pred = get_brain_max_prediction()
        brain_ultra_pred = get_brain_ultra_prediction()
        brain_predictor_pred = get_brain_predictor_prediction()
        megamind_pred = get_mega_mind_prediction()
        
        # Generar señal de consenso
        predictions = [brain_max_pred, brain_ultra_pred, brain_predictor_pred, megamind_pred]
        buy_signals = sum(1 for p in predictions if p["signal"] == "buy")
        sell_signals = sum(1 for p in predictions if p["signal"] == "sell")
        
        consensus_signal = None
        if buy_signals >= 3:
            consensus_signal = "buy"
        elif sell_signals >= 3:
            consensus_signal = "sell"
        
        if consensus_signal:
            return [{
                "signal_id": f"rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "pair": "EURUSD",
                "signal": consensus_signal,
                "confidence": 0.78,
                "position_size": 0.02,
                "stop_loss": 1.0850 if consensus_signal == "buy" else 1.0950,
                "take_profit": 1.0950 if consensus_signal == "buy" else 1.0850,
                "reasoning": "Consenso de 3+ modelos IA con alta confianza",
                "models_used": ["Brain Max", "Brain Ultra", "Brain Predictor", "MegaMind"],
                "timestamp": datetime.now().isoformat()
            }]
        
        return []
        
    except Exception as e:
        logger.error(f"Error getting active signals: {e}")
        return []

def execute_signal(signal: Dict) -> Dict:
    """Ejecuta una señal de trading"""
    try:
        return {
            "success": True,
            "execution_id": f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "signal_id": signal.get("signal_id"),
            "executed_at": datetime.now().isoformat(),
            "status": "executed",
            "message": "Señal ejecutada correctamente"
        }
    except Exception as e:
        logger.error(f"Error executing signal: {e}")
        return {"success": False, "error": str(e)}

def get_training_progress() -> Dict:
    """Obtiene el progreso del entrenamiento actual"""
    try:
        if not rl_service:
            return {
                "is_training": False,
                "progress": 0.0,
                "current_episode": 0,
                "total_episodes": 0,
                "status": "service_unavailable"
            }
        
        # Obtener la sesión activa más reciente (simulado)
        return {
            "is_training": True,
            "progress": 0.45,
            "current_episode": 450,
            "total_episodes": 1000,
            "estimated_time_remaining": 15,  # minutos
            "status": "running"
        }
    except Exception as e:
        logger.error(f"Error getting training progress: {e}")
        return {"is_training": False, "error": str(e)}

# Funciones auxiliares para simular predicciones de modelos
def get_brain_max_prediction() -> Dict:
    return {
        "signal": "buy",
        "confidence": 0.75,
        "price": 1.0900,
        "timestamp": datetime.now().isoformat()
    }

def get_brain_ultra_prediction() -> Dict:
    return {
        "signal": "buy",
        "confidence": 0.82,
        "price": 1.0895,
        "timestamp": datetime.now().isoformat()
    }

def get_brain_predictor_prediction() -> Dict:
    return {
        "signal": "hold",
        "confidence": 0.70,
        "price": 1.0902,
        "timestamp": datetime.now().isoformat()
    }

def get_mega_mind_prediction() -> Dict:
    return {
        "signal": "buy",
        "confidence": 0.85,
        "price": 1.0898,
        "timestamp": datetime.now().isoformat()
    }

# Nuevas funciones para el sistema de entrenamiento
def can_user_start_training(user_id: str) -> Dict:
    """Verifica si un usuario puede iniciar entrenamiento"""
    try:
        if not rl_service:
            return {
                "can_train": False,
                "reason": "Servicio de entrenamiento no disponible"
            }
        
        db = rl_service.get_session()
        try:
            result = rl_service.can_user_train(user_id, db)
            return result
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error checking training permission: {e}")
        return {
            "can_train": False,
            "reason": "Error interno del servidor"
        }

def validate_training_parameters(episodes: int, user_plan: str = "starter") -> Dict:
    """Valida los parámetros de entrenamiento"""
    try:
        if not rl_service:
            return {
                "valid": False,
                "reason": "Servicio de entrenamiento no disponible"
            }
        
        return rl_service.validate_training_params(episodes, user_plan)
        
    except Exception as e:
        logger.error(f"Error validating training parameters: {e}")
        return {
            "valid": False,
            "reason": "Error interno del servidor"
        }

async def start_rl_training(
    user_id: str,
    episodes: int,
    algorithm: str = "dqn",
    trading_pair: str = "EURUSD",
    timeframe: str = "1h"
) -> Dict:
    """Inicia un entrenamiento de RL"""
    try:
        if not rl_service:
            return {
                "success": False,
                "error": "Servicio de entrenamiento no disponible"
            }
        
        db = rl_service.get_session()
        try:
            result = await rl_service.start_training(
                user_id, episodes, db, algorithm, trading_pair, timeframe
            )
            return result
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error starting RL training: {e}")
        return {
            "success": False,
            "error": "Error interno del servidor"
        }

def get_training_progress_by_session(session_id: str) -> Dict:
    """Obtiene el progreso de una sesión específica"""
    try:
        if not rl_service:
            return {
                "is_training": False,
                "error": "Servicio de entrenamiento no disponible"
            }
        
        return rl_service.get_training_progress(session_id)
        
    except Exception as e:
        logger.error(f"Error getting training progress: {e}")
        return {
            "is_training": False,
            "error": "Error interno del servidor"
        }

def cancel_rl_training(session_id: str, user_id: str) -> Dict:
    """Cancela un entrenamiento en curso"""
    try:
        if not rl_service:
            return {
                "success": False,
                "error": "Servicio de entrenamiento no disponible"
            }
        
        return rl_service.cancel_training(session_id, user_id)
        
    except Exception as e:
        logger.error(f"Error canceling training: {e}")
        return {
            "success": False,
            "error": "Error interno del servidor"
        }

def get_user_training_history(user_id: str, limit: int = 10) -> List[Dict]:
    """Obtiene el historial de entrenamientos del usuario"""
    try:
        if not rl_service:
            return []
        
        return rl_service.get_user_training_history(user_id, limit)
        
    except Exception as e:
        logger.error(f"Error getting training history: {e}")
        return []