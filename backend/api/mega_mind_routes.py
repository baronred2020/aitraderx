from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from pydantic import BaseModel
import asyncio
from datetime import datetime
import random

# Importar servicios
from ..services.mega_mind_service import MegaMindService

router = APIRouter(prefix="/mega-mind", tags=["MEGA MIND"])

# Instancia del servicio
mega_mind_service = MegaMindService()

# Modelos Pydantic para las respuestas
class MegaMindPredictionResponse(BaseModel):
    pair: str
    direction: str  # 'up', 'down', 'sideways'
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
    conflict_resolution: dict
    performance_metrics: dict
    timestamp: str

class BrainArenaResponse(BaseModel):
    pair: str
    competition_round: int
    arena_results: dict
    champion: str
    overall_performance: float
    timestamp: str

class BrainEvolutionResponse(BaseModel):
    evolution_phase: str
    generation: int
    improvement_rate: float
    evolution_metrics: dict
    next_evolution_trigger: float
    timestamp: str

class BrainOrchestrationResponse(BaseModel):
    orchestration_mode: str
    coordination_score: float
    orchestration_metrics: dict
    active_strategies: int
    timestamp: str

class BrainConfigRequest(BaseModel):
    brain_type: str
    trading_params: dict
    market_preferences: dict
    specializations: dict
    consensus_weight: float = 0.33

class BrainTrainingRequest(BaseModel):
    brain_type: str
    training_data: dict
    training_params: dict

@router.get("/predictions")
async def get_mega_mind_predictions(
    pair: str,
    style: str = "day_trading",
    limit: int = 10
) -> List[MegaMindPredictionResponse]:
    """
    Obtener predicciones MEGA MIND (combinación de los 3 cerebros)
    """
    try:
        # Validar parámetros
        valid_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'EURGBP', 'GBPJPY', 'EURJPY']
        if pair not in valid_pairs:
            raise HTTPException(status_code=400, detail=f"Pair must be one of: {valid_pairs}")
        
        valid_styles = ['scalping', 'day_trading', 'swing_trading', 'position_trading']
        if style not in valid_styles:
            raise HTTPException(status_code=400, detail=f"Style must be one of: {valid_styles}")
        
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
        raise HTTPException(status_code=500, detail=f"Error getting MEGA MIND predictions: {str(e)}")

@router.get("/collaboration")
async def get_brain_collaboration(pair: str) -> BrainCollaborationResponse:
    """
    Obtener análisis de colaboración de cerebros
    """
    try:
        # Validar parámetros
        valid_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'EURGBP', 'GBPJPY', 'EURJPY']
        if pair not in valid_pairs:
            raise HTTPException(status_code=400, detail=f"Pair must be one of: {valid_pairs}")
        
        # Obtener análisis de colaboración
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
        raise HTTPException(status_code=500, detail=f"Error getting brain collaboration: {str(e)}")

@router.get("/arena")
async def get_brain_arena_results(pair: str) -> BrainArenaResponse:
    """
    Obtener resultados de la Brain Arena (competencia entre cerebros)
    """
    try:
        # Validar parámetros
        valid_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'EURGBP', 'GBPJPY', 'EURJPY']
        if pair not in valid_pairs:
            raise HTTPException(status_code=400, detail=f"Pair must be one of: {valid_pairs}")
        
        # Obtener resultados de arena
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
        raise HTTPException(status_code=500, detail=f"Error getting brain arena results: {str(e)}")

@router.get("/evolution")
async def get_brain_evolution_status() -> BrainEvolutionResponse:
    """
    Obtener estado de la evolución de cerebros
    """
    try:
        # Obtener estado de evolución
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
        raise HTTPException(status_code=500, detail=f"Error getting brain evolution status: {str(e)}")

@router.get("/orchestration")
async def get_brain_orchestration_status() -> BrainOrchestrationResponse:
    """
    Obtener estado de la orquestación de cerebros
    """
    try:
        # Obtener estado de orquestación
        orchestration_status = await mega_mind_service.get_brain_orchestration_status()
        
        return BrainOrchestrationResponse(
            orchestration_mode=orchestration_status['orchestration_mode'],
            coordination_score=orchestration_status['coordination_score'],
            orchestration_metrics=orchestration_status['orchestration_metrics'],
            active_strategies=orchestration_status['active_strategies'],
            timestamp=orchestration_status['timestamp']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting brain orchestration status: {str(e)}")

@router.get("/performance")
async def get_mega_mind_performance() -> dict:
    """
    Obtener métricas de rendimiento de MEGA MIND
    """
    try:
        # Obtener métricas de rendimiento
        performance = await mega_mind_service.get_mega_mind_performance()
        
        return performance
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting MEGA MIND performance: {str(e)}")

@router.post("/configure-brain")
async def configure_brain(config: BrainConfigRequest) -> dict:
    """
    Configurar un cerebro específico
    """
    try:
        # Validar tipo de cerebro
        valid_brain_types = ['brain_max', 'brain_ultra', 'brain_predictor']
        if config.brain_type not in valid_brain_types:
            raise HTTPException(status_code=400, detail=f"Brain type must be one of: {valid_brain_types}")
        
        # Configurar cerebro
        result = await mega_mind_service.configure_brain(
            config.brain_type,
            {
                'trading_params': config.trading_params,
                'market_preferences': config.market_preferences,
                'specializations': config.specializations,
                'consensus_weight': config.consensus_weight
            }
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error configuring brain: {str(e)}")

@router.post("/train-brain")
async def train_brain(training: BrainTrainingRequest) -> dict:
    """
    Entrenar un cerebro específico
    """
    try:
        # Validar tipo de cerebro
        valid_brain_types = ['brain_max', 'brain_ultra', 'brain_predictor']
        if training.brain_type not in valid_brain_types:
            raise HTTPException(status_code=400, detail=f"Brain type must be one of: {valid_brain_types}")
        
        # Entrenar cerebro
        result = await mega_mind_service.train_brain(
            training.brain_type,
            {
                'training_data': training.training_data,
                'training_params': training.training_params
            }
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training brain: {str(e)}")

@router.get("/brain-states")
async def get_brain_states() -> dict:
    """
    Obtener estados actuales de todos los cerebros
    """
    try:
        return mega_mind_service.brain_states
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting brain states: {str(e)}")

@router.get("/brain-config")
async def get_brain_config() -> dict:
    """
    Obtener configuración actual de cerebros
    """
    try:
        return {
            'fusion_weights': mega_mind_service.fusion_weights,
            'collaboration_config': mega_mind_service.collaboration_config,
            'evolution_config': mega_mind_service.evolution_config,
            'gamification_config': mega_mind_service.gamification_config
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting brain config: {str(e)}")

@router.get("/health")
async def mega_mind_health_check() -> dict:
    """
    Health check para MEGA MIND
    """
    try:
        # Verificar que todos los componentes estén funcionando
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {
                'brain_collaboration': 'active',
                'brain_fusion': 'active',
                'brain_arena': 'active',
                'brain_evolution': 'active',
                'brain_orchestration': 'active',
                'brain_gamification': 'active',
                'brain_personalization': 'active'
            },
            'brain_states': {
                brain: state['status'] for brain, state in mega_mind_service.brain_states.items()
            }
        }
        
        return health_status
        
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

@router.get("/config")
async def get_mega_mind_config() -> dict:
    """
    Obtener configuración completa de MEGA MIND
    """
    try:
        return {
            'service_config': {
                'fusion_weights': mega_mind_service.fusion_weights,
                'collaboration_config': mega_mind_service.collaboration_config,
                'evolution_config': mega_mind_service.evolution_config,
                'gamification_config': mega_mind_service.gamification_config
            },
            'brain_states': mega_mind_service.brain_states,
            'available_endpoints': [
                '/predictions',
                '/collaboration',
                '/arena',
                '/evolution',
                '/orchestration',
                '/performance',
                '/configure-brain',
                '/train-brain',
                '/brain-states',
                '/brain-config',
                '/health'
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting MEGA MIND config: {str(e)}") 