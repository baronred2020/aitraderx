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
    agent_type: str  # 'technical', 'ai', 'risk', 'temporal', 'fundamental'
    severity: str  # 'low', 'medium', 'high', 'critical'
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
    is_active: bool
    last_check: datetime
    alerts_count: int
    performance_score: float
    status: str  # 'monitoring', 'idle', 'error', 'maintenance'

class MonitoringSystemStatusResponse(BaseModel):
    overall_status: str  # 'healthy', 'warning', 'critical'
    active_agents: int
    total_alerts: int
    unread_alerts: int
    critical_alerts: int
    last_update: datetime
    agents_status: List[MonitoringAgentStatusResponse]

class MonitoringConfigResponse(BaseModel):
    enabled: bool
    check_interval: int  # segundos
    alert_retention_days: int
    max_alerts_per_agent: int
    subscription_limits: dict

# Datos simulados para demostración
mock_alerts = [
    {
        "id": str(uuid.uuid4()),
        "agent_type": "technical",
        "severity": "high",
        "category": "pattern_detection",
        "title": "Patrón de reversión detectado en EURUSD",
        "description": "Se ha detectado un patrón de doble techo en el timeframe de 4H que sugiere una posible reversión bajista.",
        "pair": "EURUSD",
        "brain_type": "brain_max",
        "timestamp": datetime.now(),
        "is_read": False,
        "action_required": True,
        "metadata": {"pattern_type": "double_top", "timeframe": "4H", "confidence": 0.85}
    },
    {
        "id": str(uuid.uuid4()),
        "agent_type": "ai",
        "severity": "medium",
        "category": "model_performance",
        "title": "Disminución en la precisión del modelo Brain Ultra",
        "description": "La precisión del modelo Brain Ultra ha disminuido un 3% en las últimas 24 horas.",
        "pair": "EURUSD",
        "brain_type": "brain_ultra",
        "timestamp": datetime.now(),
        "is_read": False,
        "action_required": False,
        "metadata": {"accuracy_drop": 0.03, "time_period": "24h"}
    },
    {
        "id": str(uuid.uuid4()),
        "agent_type": "risk",
        "severity": "critical",
        "category": "drawdown_alert",
        "title": "Drawdown crítico detectado en GBPUSD",
        "description": "El drawdown actual ha alcanzado el 8%, superando el límite establecido del 5%.",
        "pair": "GBPUSD",
        "brain_type": "brain_predictor",
        "timestamp": datetime.now(),
        "is_read": False,
        "action_required": True,
        "metadata": {"current_drawdown": 0.08, "limit": 0.05}
    },
    {
        "id": str(uuid.uuid4()),
        "agent_type": "temporal",
        "severity": "low",
        "category": "time_analysis",
        "title": "Análisis temporal favorable para USDJPY",
        "description": "El análisis temporal indica condiciones favorables para operaciones largas en USDJPY.",
        "pair": "USDJPY",
        "brain_type": "brain_max",
        "timestamp": datetime.now(),
        "is_read": True,
        "action_required": False,
        "metadata": {"time_condition": "favorable", "recommendation": "long"}
    },
    {
        "id": str(uuid.uuid4()),
        "agent_type": "fundamental",
        "severity": "high",
        "category": "news_impact",
        "title": "Noticia económica de alto impacto detectada",
        "description": "Se ha detectado una noticia sobre tasas de interés que podría afectar significativamente el EURUSD.",
        "pair": "EURUSD",
        "brain_type": None,
        "timestamp": datetime.now(),
        "is_read": False,
        "action_required": True,
        "metadata": {"news_type": "interest_rate", "impact": "high"}
    }
]

mock_agents_status = [
    {
        "agent_type": "technical",
        "is_active": True,
        "last_check": datetime.now(),
        "alerts_count": 2,
        "performance_score": 95.5,
        "status": "monitoring"
    },
    {
        "agent_type": "ai",
        "is_active": True,
        "last_check": datetime.now(),
        "alerts_count": 1,
        "performance_score": 92.3,
        "status": "monitoring"
    },
    {
        "agent_type": "risk",
        "is_active": True,
        "last_check": datetime.now(),
        "alerts_count": 1,
        "performance_score": 98.7,
        "status": "monitoring"
    },
    {
        "agent_type": "temporal",
        "is_active": True,
        "last_check": datetime.now(),
        "alerts_count": 1,
        "performance_score": 89.2,
        "status": "monitoring"
    },
    {
        "agent_type": "fundamental",
        "is_active": True,
        "last_check": datetime.now(),
        "alerts_count": 1,
        "performance_score": 94.1,
        "status": "monitoring"
    }
]

mock_config = {
    "enabled": True,
    "check_interval": 30,
    "alert_retention_days": 30,
    "max_alerts_per_agent": 100,
    "subscription_limits": {
        "starter": {"max_alerts": 50, "max_agents": 2},
        "trader": {"max_alerts": 100, "max_agents": 3},
        "expert": {"max_alerts": 200, "max_agents": 4},
        "premium": {"max_alerts": 500, "max_agents": 5},
        "institutional": {"max_alerts": 1000, "max_agents": 5}
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
    try:
        # Filtrar alertas según los parámetros
        filtered_alerts = mock_alerts.copy()
        
        if agent_type:
            filtered_alerts = [alert for alert in filtered_alerts if alert["agent_type"] == agent_type]
        
        if severity:
            filtered_alerts = [alert for alert in filtered_alerts if alert["severity"] == severity]
        
        # Limitar el número de resultados
        filtered_alerts = filtered_alerts[:limit]
        
        return [MonitoringAlertResponse(**alert) for alert in filtered_alerts]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo alertas: {str(e)}")

@router.put("/alerts/{alert_id}/read")
async def mark_alert_as_read(alert_id: str) -> dict:
    """
    Marcar una alerta como leída
    """
    try:
        # Simular actualización de la alerta
        for alert in mock_alerts:
            if alert["id"] == alert_id:
                alert["is_read"] = True
                break
        
        return {"success": True, "message": "Alerta marcada como leída"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error marcando alerta como leída: {str(e)}")

@router.get("/status")
async def get_monitoring_system_status() -> MonitoringSystemStatusResponse:
    """
    Obtener el estado general del sistema de monitoreo
    """
    try:
        # Calcular estadísticas
        total_alerts = len(mock_alerts)
        unread_alerts = len([alert for alert in mock_alerts if not alert["is_read"]])
        critical_alerts = len([alert for alert in mock_alerts if alert["severity"] == "critical"])
        active_agents = len([agent for agent in mock_agents_status if agent["is_active"]])
        
        # Determinar estado general
        if critical_alerts > 0:
            overall_status = "critical"
        elif unread_alerts > 5:
            overall_status = "warning"
        else:
            overall_status = "healthy"
        
        return MonitoringSystemStatusResponse(
            overall_status=overall_status,
            active_agents=active_agents,
            total_alerts=total_alerts,
            unread_alerts=unread_alerts,
            critical_alerts=critical_alerts,
            last_update=datetime.now(),
            agents_status=[MonitoringAgentStatusResponse(**agent) for agent in mock_agents_status]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo estado del sistema: {str(e)}")

@router.get("/config")
async def get_monitoring_config() -> MonitoringConfigResponse:
    """
    Obtener la configuración del sistema de monitoreo
    """
    try:
        return MonitoringConfigResponse(**mock_config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo configuración: {str(e)}")

@router.put("/config")
async def update_monitoring_config(config_update: dict) -> MonitoringConfigResponse:
    """
    Actualizar la configuración del sistema de monitoreo
    """
    try:
        # Simular actualización de configuración
        mock_config.update(config_update)
        return MonitoringConfigResponse(**mock_config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error actualizando configuración: {str(e)}")

@router.post("/start")
async def start_monitoring(
    pair: str = Query(..., description="Par de divisas a monitorear"),
    brain_type: Optional[str] = Query(None, description="Tipo de cerebro específico")
) -> dict:
    """
    Iniciar monitoreo para un par específico
    """
    try:
        # Simular inicio de monitoreo
        # En una implementación real, aquí se iniciarían los agentes de monitoreo
        
        return {
            "success": True,
            "message": f"Monitoreo iniciado para {pair}" + (f" con cerebro {brain_type}" if brain_type else "")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error iniciando monitoreo: {str(e)}")

@router.post("/stop")
async def stop_monitoring(
    pair: str = Query(..., description="Par de divisas para detener monitoreo")
) -> dict:
    """
    Detener monitoreo para un par específico
    """
    try:
        # Simular detención de monitoreo
        # En una implementación real, aquí se detendrían los agentes de monitoreo
        
        return {
            "success": True,
            "message": f"Monitoreo detenido para {pair}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deteniendo monitoreo: {str(e)}")

@router.get("/health")
async def monitoring_health_check() -> dict:
    """
    Verificar la salud del sistema de monitoreo
    """
    try:
        return {
            "status": "healthy",
            "service": "monitoring_agents",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "active_agents": len([agent for agent in mock_agents_status if agent["is_active"]]),
            "total_alerts": len(mock_alerts)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en health check: {str(e)}") 