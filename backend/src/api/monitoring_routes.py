from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import uuid
import random

router = APIRouter(prefix="/brain-trader/monitoring", tags=["Monitoring Agents"])

# Modelos Pydantic para las respuestas
class MonitoringAlertResponse(BaseModel):
    id: str
    agent_type: str
    severity: str
    category: str
    title: str
    description: str
    pair: Optional[str] = None
    brain_type: Optional[str] = None
    timestamp: datetime
    is_read: bool
    action_required: bool
    metadata: Optional[dict] = None

class MonitoringAgentStatusResponse(BaseModel):
    agent_type: str
    status: str
    last_check: datetime
    alerts_count: int
    is_active: bool
    performance_metrics: Optional[dict] = None

class MonitoringSystemStatusResponse(BaseModel):
    overall_status: str
    active_agents: int
    total_alerts: int
    critical_alerts: int
    last_updated: datetime
    agents_status: List[MonitoringAgentStatusResponse]

class MonitoringConfigResponse(BaseModel):
    enabled: bool
    auto_refresh: bool
    refresh_interval: int
    alert_thresholds: dict
    agent_settings: dict
    notification_settings: dict

# Datos simulados para demostración
mock_alerts = [
    {
        "id": str(uuid.uuid4()),
        "agent_type": "technical",
        "severity": "high",
        "category": "pattern_detection",
        "title": "Patrón de reversión detectado",
        "description": "Se detectó un patrón de doble techo en EURUSD que sugiere una posible reversión",
        "pair": "EURUSD",
        "brain_type": "brain_max",
        "timestamp": datetime.now(),
        "is_read": False,
        "action_required": True,
        "metadata": {"pattern_type": "double_top", "confidence": 0.85}
    },
    {
        "id": str(uuid.uuid4()),
        "agent_type": "ai",
        "severity": "medium",
        "category": "model_performance",
        "title": "Degradación de precisión del modelo",
        "description": "El modelo Brain Ultra muestra una precisión del 72%, por debajo del umbral del 75%",
        "pair": "EURUSD",
        "brain_type": "brain_ultra",
        "timestamp": datetime.now(),
        "is_read": False,
        "action_required": False,
        "metadata": {"current_accuracy": 0.72, "threshold": 0.75}
    },
    {
        "id": str(uuid.uuid4()),
        "agent_type": "risk",
        "severity": "critical",
        "category": "drawdown_alert",
        "title": "Drawdown crítico detectado",
        "description": "El drawdown actual del 8.5% excede el límite del 5% establecido",
        "pair": "EURUSD",
        "brain_type": "brain_predictor",
        "timestamp": datetime.now(),
        "is_read": False,
        "action_required": True,
        "metadata": {"current_drawdown": 0.085, "limit": 0.05}
    },
    {
        "id": str(uuid.uuid4()),
        "agent_type": "temporal",
        "severity": "low",
        "category": "timeframe_analysis",
        "title": "Análisis de múltiples timeframes",
        "description": "Divergencia detectada entre timeframes de 1H y 4H para EURUSD",
        "pair": "EURUSD",
        "brain_type": "brain_max",
        "timestamp": datetime.now(),
        "is_read": True,
        "action_required": False,
        "metadata": {"timeframes": ["1H", "4H"], "divergence_type": "bearish"}
    },
    {
        "id": str(uuid.uuid4()),
        "agent_type": "fundamental",
        "severity": "medium",
        "category": "news_impact",
        "title": "Noticia económica importante",
        "description": "Anuncio del BCE sobre tasas de interés puede impactar EURUSD",
        "pair": "EURUSD",
        "brain_type": "brain_ultra",
        "timestamp": datetime.now(),
        "is_read": False,
        "action_required": False,
        "metadata": {"news_source": "Reuters", "impact_score": 0.7}
    }
]

mock_agents_status = [
    {
        "agent_type": "technical",
        "status": "active",
        "last_check": datetime.now(),
        "alerts_count": 12,
        "is_active": True,
        "performance_metrics": {"accuracy": 0.88, "response_time": 0.15}
    },
    {
        "agent_type": "ai",
        "status": "active",
        "last_check": datetime.now(),
        "alerts_count": 8,
        "is_active": True,
        "performance_metrics": {"model_accuracy": 0.72, "training_status": "stable"}
    },
    {
        "agent_type": "risk",
        "status": "warning",
        "last_check": datetime.now(),
        "alerts_count": 3,
        "is_active": True,
        "performance_metrics": {"risk_score": 0.65, "drawdown": 0.085}
    },
    {
        "agent_type": "temporal",
        "status": "active",
        "last_check": datetime.now(),
        "alerts_count": 5,
        "is_active": True,
        "performance_metrics": {"timeframe_consistency": 0.82}
    },
    {
        "agent_type": "fundamental",
        "status": "active",
        "last_check": datetime.now(),
        "alerts_count": 7,
        "is_active": True,
        "performance_metrics": {"news_impact_accuracy": 0.75}
    }
]

mock_config = {
    "enabled": True,
    "auto_refresh": True,
    "refresh_interval": 30,
    "alert_thresholds": {
        "technical": {"min_confidence": 0.75, "max_alerts_per_hour": 10},
        "ai": {"min_accuracy": 0.70, "max_model_errors": 5},
        "risk": {"max_drawdown": 0.05, "max_risk_score": 0.8},
        "temporal": {"min_timeframe_consistency": 0.80},
        "fundamental": {"min_news_impact": 0.6}
    },
    "agent_settings": {
        "technical": {"enabled": True, "check_interval": 60},
        "ai": {"enabled": True, "check_interval": 120},
        "risk": {"enabled": True, "check_interval": 30},
        "temporal": {"enabled": True, "check_interval": 300},
        "fundamental": {"enabled": True, "check_interval": 600}
    },
    "notification_settings": {
        "email_enabled": True,
        "push_enabled": True,
        "critical_alerts_only": False
    }
}

@router.get("/alerts")
async def get_monitoring_alerts(
    agent_type: Optional[str] = Query(None, description="Filtrar por tipo de agente"),
    severity: Optional[str] = Query(None, description="Filtrar por severidad"),
    limit: int = Query(50, description="Número máximo de alertas a retornar")
) -> List[MonitoringAlertResponse]:
    """
    Obtener alertas de monitoreo con filtros opcionales
    """
    filtered_alerts = mock_alerts.copy()
    
    if agent_type:
        filtered_alerts = [alert for alert in filtered_alerts if alert["agent_type"] == agent_type]
    
    if severity:
        filtered_alerts = [alert for alert in filtered_alerts if alert["severity"] == severity]
    
    # Limitar resultados
    filtered_alerts = filtered_alerts[:limit]
    
    return [MonitoringAlertResponse(**alert) for alert in filtered_alerts]

@router.put("/alerts/{alert_id}/read")
async def mark_alert_as_read(alert_id: str) -> dict:
    """
    Marcar una alerta como leída
    """
    for alert in mock_alerts:
        if alert["id"] == alert_id:
            alert["is_read"] = True
            return {"success": True, "message": "Alerta marcada como leída"}
    
    raise HTTPException(status_code=404, detail="Alerta no encontrada")

@router.get("/status")
async def get_monitoring_system_status() -> MonitoringSystemStatusResponse:
    """
    Obtener el estado general del sistema de monitoreo
    """
    critical_alerts = len([alert for alert in mock_alerts if alert["severity"] == "critical"])
    active_agents = len([agent for agent in mock_agents_status if agent["is_active"]])
    
    overall_status = "warning" if critical_alerts > 0 else "healthy"
    
    return MonitoringSystemStatusResponse(
        overall_status=overall_status,
        active_agents=active_agents,
        total_alerts=len(mock_alerts),
        critical_alerts=critical_alerts,
        last_updated=datetime.now(),
        agents_status=[MonitoringAgentStatusResponse(**agent) for agent in mock_agents_status]
    )

@router.get("/config")
async def get_monitoring_config() -> MonitoringConfigResponse:
    """
    Obtener la configuración actual del sistema de monitoreo
    """
    return MonitoringConfigResponse(**mock_config)

@router.put("/config")
async def update_monitoring_config(config: dict) -> MonitoringConfigResponse:
    """
    Actualizar la configuración del sistema de monitoreo
    """
    mock_config.update(config)
    return MonitoringConfigResponse(**mock_config)

@router.post("/start")
async def start_monitoring(pair: str, brain_type: Optional[str] = None) -> dict:
    """
    Iniciar monitoreo para un par específico
    """
    return {
        "success": True,
        "message": f"Monitoreo iniciado para {pair}" + (f" con cerebro {brain_type}" if brain_type else "")
    }

@router.post("/stop")
async def stop_monitoring(pair: str) -> dict:
    """
    Detener monitoreo para un par específico
    """
    return {
        "success": True,
        "message": f"Monitoreo detenido para {pair}"
    }

@router.get("/health")
async def get_monitoring_health() -> dict:
    """
    Verificar la salud del sistema de monitoreo
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_agents": len([agent for agent in mock_agents_status if agent["is_active"]]),
        "total_alerts": len(mock_alerts),
        "unread_alerts": len([alert for alert in mock_alerts if not alert["is_read"]])
    } 