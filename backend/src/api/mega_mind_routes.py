"""
MegaMind API Routes
===================
Rutas para el sistema de Cerebros Colaborativos MegaMind
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from datetime import datetime
import asyncio
import logging

# Configurar logging
logger = logging.getLogger(__name__)

# Importar el servicio MegaMind
try:
    from services.mega_mind_service import MegaMindService
    mega_mind_service = MegaMindService()
except ImportError as e:
    logger.error(f"Error importing MegaMindService: {e}")
    # Crear un servicio mock para desarrollo
    class MockMegaMindService:
        def __init__(self):
            self.brain_collaboration = None
            self.brain_fusion = None
            self.brain_arena = None
            self.brain_evolution = None
            self.brain_orchestration = None
            self.brain_gamification = None
            self.brain_personalization = None
            
        async def get_mega_mind_predictions(self, pair: str, style: str, limit: int):
            return []
            
        async def get_brain_collaboration(self, pair: str):
            return {}
            
        async def get_brain_arena_results(self, pair: str):
            return {}
            
        async def get_brain_evolution_status(self):
            return {}
            
        async def get_brain_orchestration_status(self):
            return {}
            
        async def get_mega_mind_performance(self):
            return {}
            
        async def configure_brain(self, brain_type: str, config: dict):
            return {}
            
        async def train_brain(self, brain_type: str, training_data: dict):
            return {}
    
    mega_mind_service = MockMegaMindService()

# Crear router
router = APIRouter(prefix="/api/v1/mega-mind", tags=["MegaMind"])

# Modelos Pydantic para requests/responses
class MegaMindPredictionResponse(BaseModel):
    pair: str
    direction: str
    confidence: float
    target_price: float
    timeframe: str
    reasoning: str
    brain_type: str
    fusion_method: str
    collaboration_score: float
    fusion_details: dict
    timestamp: str

class BrainCollaborationResponse(BaseModel):
    pair: str
    collaboration_score: float
    consensus_level: float
    brain_synergy: dict
    conflict_resolution: str
    performance_metrics: dict
    timestamp: str

class BrainArenaResponse(BaseModel):
    pair: str
    competition_round: int
    arena_results: dict
    champion: str
    overall_performance: dict
    timestamp: str

class BrainEvolutionResponse(BaseModel):
    evolution_phase: str
    generation: int
    improvement_rate: float
    evolution_metrics: dict
    next_evolution_trigger: str
    timestamp: str

class BrainOrchestrationResponse(BaseModel):
    orchestration_mode: str
    coordination_score: float
    orchestration_metrics: dict
    active_strategies: int
    timestamp: str

class MegaMindPerformanceResponse(BaseModel):
    overall_accuracy: float
    prediction_success_rate: float
    risk_adjusted_returns: float
    sharpe_ratio: float
    win_rate: float
    brain_levels: dict
    achievements: list
    timestamp: str

class BrainConfigRequest(BaseModel):
    trading_params: dict
    market_preferences: dict
    specializations: dict
    consensus_weight: float

class BrainTrainingRequest(BaseModel):
    training_data: dict
    training_params: dict

class BrainConfigResponse(BaseModel):
    config_applied: bool
    brain_type: str
    config_version: str
    timestamp: str

class BrainTrainingResponse(BaseModel):
    training_completed: bool
    brain_type: str
    training_metrics: dict
    new_accuracy: float
    timestamp: str

# Rutas principales
@router.get("/predictions")
async def get_mega_mind_predictions(
    pair: str,
    style: str = "day_trading",
    limit: int = 10
) -> List[MegaMindPredictionResponse]:
    """Obtener predicciones del sistema MegaMind"""
    try:
        # Validar parámetros
        if not pair:
            raise HTTPException(status_code=400, detail="Pair parameter is required")
        
        if limit > 50:
            limit = 50  # Limitar a 50 predicciones máximo
        
        # Obtener predicciones del servicio
        predictions = await mega_mind_service.get_mega_mind_predictions(pair, style, limit)
        
        # Convertir a formato de respuesta
        response_predictions = []
        for pred in predictions:
            response_pred = MegaMindPredictionResponse(
                pair=pred['pair'],
                direction=pred['direction'],
                confidence=pred['confidence'],
                target_price=pred['target_price'],
                timeframe=pred['timeframe'],
                reasoning=pred['reasoning'],
                brain_type=pred['brain_type'],
                fusion_method=pred['fusion_method'],
                collaboration_score=pred['collaboration_score'],
                fusion_details=pred.get('fusion_details', {}),
                timestamp=pred['timestamp']
            )
            response_predictions.append(response_pred)
        
        return response_predictions
        
    except Exception as e:
        logger.error(f"Error getting MEGA MIND predictions: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting MEGA MIND predictions: {str(e)}")

@router.get("/collaboration")
async def get_brain_collaboration(pair: str = "EURUSD") -> BrainCollaborationResponse:
    """Obtener análisis de colaboración entre cerebros"""
    try:
        collaboration = await mega_mind_service.get_brain_collaboration(pair)
        
        return BrainCollaborationResponse(
            pair=collaboration['pair'],
            collaboration_score=collaboration['collaboration_score'],
            consensus_level=collaboration['consensus_level'],
            brain_synergy=collaboration['brain_synergy'],
            conflict_resolution=collaboration['conflict_resolution'],
            performance_metrics=collaboration['performance_metrics'],
            timestamp=collaboration['timestamp']
        )
        
    except Exception as e:
        logger.error(f"Error getting brain collaboration: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting brain collaboration: {str(e)}")

@router.get("/arena")
async def get_brain_arena_results(pair: str = "EURUSD") -> BrainArenaResponse:
    """Obtener resultados de la arena de cerebros"""
    try:
        arena_results = await mega_mind_service.get_brain_arena_results(pair)
        
        return BrainArenaResponse(
            pair=arena_results['pair'],
            competition_round=arena_results['competition_round'],
            arena_results=arena_results['arena_results'],
            champion=arena_results['champion'],
            overall_performance=arena_results['overall_performance'],
            timestamp=arena_results['timestamp']
        )
        
    except Exception as e:
        logger.error(f"Error getting brain arena results: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting brain arena results: {str(e)}")

@router.get("/evolution")
async def get_brain_evolution_status() -> BrainEvolutionResponse:
    """Obtener estado de evolución de cerebros"""
    try:
        evolution_status = await mega_mind_service.get_brain_evolution_status()
        
        return BrainEvolutionResponse(
            evolution_phase=evolution_status['evolution_phase'],
            generation=evolution_status['generation'],
            improvement_rate=evolution_status['improvement_rate'],
            evolution_metrics=evolution_status['evolution_metrics'],
            next_evolution_trigger=evolution_status['next_evolution_trigger'],
            timestamp=evolution_status['timestamp']
        )
        
    except Exception as e:
        logger.error(f"Error getting brain evolution status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting brain evolution status: {str(e)}")

@router.get("/orchestration")
async def get_brain_orchestration_status() -> BrainOrchestrationResponse:
    """Obtener estado de orquestación de cerebros"""
    try:
        orchestration_status = await mega_mind_service.get_brain_orchestration_status()
        
        return BrainOrchestrationResponse(
            orchestration_mode=orchestration_status['orchestration_mode'],
            coordination_score=orchestration_status['coordination_score'],
            orchestration_metrics=orchestration_status['orchestration_metrics'],
            active_strategies=orchestration_status['active_strategies'],
            timestamp=orchestration_status['timestamp']
        )
        
    except Exception as e:
        logger.error(f"Error getting brain orchestration status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting brain orchestration status: {str(e)}")

@router.get("/performance")
async def get_mega_mind_performance() -> MegaMindPerformanceResponse:
    """Obtener métricas de rendimiento del sistema MegaMind"""
    try:
        performance = await mega_mind_service.get_mega_mind_performance()
        
        return MegaMindPerformanceResponse(
            overall_accuracy=performance['overall_accuracy'],
            prediction_success_rate=performance['prediction_success_rate'],
            risk_adjusted_returns=performance['risk_adjusted_returns'],
            sharpe_ratio=performance['sharpe_ratio'],
            win_rate=performance['win_rate'],
            brain_levels=performance['brain_levels'],
            achievements=performance['achievements'],
            timestamp=performance['timestamp']
        )
        
    except Exception as e:
        logger.error(f"Error getting MegaMind performance: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting MegaMind performance: {str(e)}")

# Rutas de configuración y entrenamiento
@router.post("/configure-brain")
async def configure_brain(
    brain_type: str,
    config: BrainConfigRequest
) -> BrainConfigResponse:
    """Configurar un cerebro específico"""
    try:
        result = await mega_mind_service.configure_brain(brain_type, config.dict())
        
        return BrainConfigResponse(
            config_applied=result['config_applied'],
            brain_type=result['brain_type'],
            config_version=result['config_version'],
            timestamp=result['timestamp']
        )
        
    except Exception as e:
        logger.error(f"Error configuring brain: {e}")
        raise HTTPException(status_code=500, detail=f"Error configuring brain: {str(e)}")

@router.post("/train-brain")
async def train_brain(
    brain_type: str,
    training_data: BrainTrainingRequest
) -> BrainTrainingResponse:
    """Entrenar un cerebro específico"""
    try:
        result = await mega_mind_service.train_brain(brain_type, training_data.dict())
        
        return BrainTrainingResponse(
            training_completed=result['training_completed'],
            brain_type=result['brain_type'],
            training_metrics=result['training_metrics'],
            new_accuracy=result['new_accuracy'],
            timestamp=result['timestamp']
        )
        
    except Exception as e:
        logger.error(f"Error training brain: {e}")
        raise HTTPException(status_code=500, detail=f"Error training brain: {str(e)}")

# Rutas de información del sistema
@router.get("/brain-states")
async def get_brain_states():
    """Obtener estados de todos los cerebros"""
    try:
        return {
            "brain_states": mega_mind_service.brain_states,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting brain states: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting brain states: {str(e)}")

@router.get("/brain-config")
async def get_brain_config():
    """Obtener configuración actual de cerebros"""
    try:
        return {
            "fusion_weights": getattr(mega_mind_service, 'fusion_weights', {}),
            "collaboration_config": getattr(mega_mind_service, 'collaboration_config', {}),
            "evolution_config": getattr(mega_mind_service, 'evolution_config', {}),
            "gamification_config": getattr(mega_mind_service, 'gamification_config', {}),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting brain config: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting brain config: {str(e)}")

# Rutas de salud y estado
@router.get("/health")
async def health_check():
    """Verificar salud del sistema MegaMind"""
    try:
        return {
            "status": "healthy",
            "service": "MegaMind",
            "components": {
                "brain_collaboration": mega_mind_service.brain_collaboration is not None,
                "brain_fusion": mega_mind_service.brain_fusion is not None,
                "brain_arena": mega_mind_service.brain_arena is not None,
                "brain_evolution": mega_mind_service.brain_evolution is not None,
                "brain_orchestration": mega_mind_service.brain_orchestration is not None,
                "brain_gamification": mega_mind_service.brain_gamification is not None,
                "brain_personalization": mega_mind_service.brain_personalization is not None
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        raise HTTPException(status_code=500, detail=f"Error in health check: {str(e)}")

@router.get("/config")
async def get_config():
    """Obtener configuración del sistema MegaMind"""
    try:
        return {
            "system_name": "MegaMind - Sistema de Cerebros Colaborativos",
            "version": "1.0.0",
            "description": "Sistema avanzado de IA con cerebros colaborativos para trading",
            "features": [
                "Brain Collaboration",
                "Brain Fusion", 
                "Brain Arena",
                "Brain Evolution",
                "Brain Orchestration",
                "Brain Gamification",
                "Brain Personalization"
            ],
            "brains": ["brain_max", "brain_ultra", "brain_predictor"],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting config: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting config: {str(e)}") 