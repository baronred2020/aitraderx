from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import uuid
import random
try:
    from services.mcp_monitoring_service import mcp_monitoring_system, AgentType, Severity, MonitoringAlert, AgentStatus
except ImportError:
    from src.services.mcp_monitoring_service import mcp_monitoring_system, AgentType, Severity, MonitoringAlert, AgentStatus

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
    performance_score: float
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
    subscription_limits: dict

# Configuración por defecto OPTIMIZADA
default_config = {
    "enabled": True,
    "auto_refresh": True,
    "refresh_interval": 300,  # 5 minutos en lugar de 30 segundos
    "alert_thresholds": {
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "drawdown_warning": 0.03,
        "drawdown_critical": 0.05,
        "model_confidence": 0.75,
        "model_accuracy": 0.70
    },
    "agent_settings": {
        "technical": {"enabled": True, "interval": 300},    # 5 min en lugar de 30s
        "ai": {"enabled": True, "interval": 600},           # 10 min en lugar de 60s
        "risk": {"enabled": True, "interval": 300},         # 5 min en lugar de 45s
        "temporal": {"enabled": True, "interval": 1800},    # 30 min en lugar de 5 min
        "fundamental": {"enabled": True, "interval": 3600}  # 1 hora en lugar de 10 min
    },
    "cache_settings": {
        "enabled": True,
        "default_ttl": 300,  # 5 minutos de caché por defecto
        "max_cache_size": 1000
    },
    "notification_settings": {
        "email_alerts": False,
        "push_notifications": True,
        "sound_alerts": True
    },
    "subscription_limits": {
        "basic": {"max_alerts": 50, "max_agents": 2},
        "premium": {"max_alerts": 200, "max_agents": 4},
        "expert": {"max_alerts": 500, "max_agents": 5},
        "enterprise": {"max_alerts": 1000, "max_agents": 5}
    }
}

# Almacenamiento temporal de alertas (en producción usar base de datos)
alert_storage = []

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
        # Obtener alertas del sistema MCP
        all_alerts = mcp_monitoring_system.get_all_alerts()
        
        # Filtrar alertas según parámetros
        filtered_alerts = []
        
        for alert in all_alerts:
            if agent_type and alert.agent_type.value != agent_type:
                continue
            if severity and alert.severity.value != severity:
                continue
            filtered_alerts.append(alert)
        
        # Limitar número de resultados
        filtered_alerts = filtered_alerts[:limit]
        
        # Convertir a formato de respuesta
        response_alerts = []
        for alert in filtered_alerts:
            response_alerts.append(MonitoringAlertResponse(
                id=alert.id,
                agent_type=alert.agent_type.value,
                severity=alert.severity.value,
                category=alert.category,
                title=alert.title,
                description=alert.description,
                pair=alert.pair,
                brain_type=alert.brain_type,
                timestamp=alert.timestamp,
                is_read=alert.is_read,
                action_required=alert.action_required,
                metadata=alert.metadata
            ))
        
        return response_alerts
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo alertas: {str(e)}")

@router.put("/alerts/{alert_id}/read")
async def mark_alert_as_read(alert_id: str) -> dict:
    """
    Marcar una alerta como leída
    """
    try:
        # Buscar la alerta en todos los agentes
        for agent in mcp_monitoring_system.agents.values():
            for alert in agent.alerts:
                if alert.id == alert_id:
                    alert.is_read = True
                    return {"success": True, "message": "Alerta marcada como leída"}
        
        raise HTTPException(status_code=404, detail="Alerta no encontrada")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error marcando alerta: {str(e)}")

@router.get("/status")
async def get_monitoring_system_status() -> MonitoringSystemStatusResponse:
    """
    Obtener el estado general del sistema de monitoreo
    """
    try:
        # Obtener estado del sistema MCP
        system_status = mcp_monitoring_system.get_system_status()
        
        # Convertir agentes a formato de respuesta
        agents_status = []
        for agent in mcp_monitoring_system.agents.values():
            status = agent.get_status()
            agents_status.append(MonitoringAgentStatusResponse(
                agent_type=status.agent_type.value,
                status=status.status,
                last_check=status.last_check,
                alerts_count=status.alerts_count,
                is_active=status.is_active,
                performance_score=status.performance_score,
                performance_metrics=status.performance_metrics
            ))
        
        return MonitoringSystemStatusResponse(
            overall_status=system_status["overall_status"],
            active_agents=system_status["active_agents"],
            total_alerts=system_status["total_alerts"],
            critical_alerts=system_status["critical_alerts"],
            last_updated=system_status["last_updated"],
            agents_status=agents_status
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo estado: {str(e)}")

@router.get("/config")
async def get_monitoring_config() -> MonitoringConfigResponse:
    """
    Obtener la configuración actual del sistema de monitoreo
    """
    try:
        return MonitoringConfigResponse(**default_config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo configuración: {str(e)}")

@router.put("/config")
async def update_monitoring_config(config: dict) -> MonitoringConfigResponse:
    """
    Actualizar la configuración del sistema de monitoreo
    """
    try:
        # Actualizar configuración por defecto
        default_config.update(config)
        
        # Aplicar configuración a los agentes MCP
        for agent_type_str, settings in config.get("agent_settings", {}).items():
            if agent_type_str in ["technical", "ai", "risk", "temporal", "fundamental"]:
                agent_type = AgentType(agent_type_str)
                agent = mcp_monitoring_system.agents.get(agent_type)
                if agent and "interval" in settings:
                    agent.monitoring_interval = settings["interval"]
        
        return MonitoringConfigResponse(**default_config)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error actualizando configuración: {str(e)}")

@router.post("/start")
async def start_monitoring(pair: str, brain_type: Optional[str] = None) -> dict:
    """
    Iniciar monitoreo para un par específico
    """
    try:
        await mcp_monitoring_system.start_monitoring(pair, brain_type)
        return {
            "success": True,
            "message": f"Monitoreo iniciado para {pair}" + (f" con cerebro {brain_type}" if brain_type else "")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error iniciando monitoreo: {str(e)}")

@router.post("/stop")
async def stop_monitoring(pair: str) -> dict:
    """
    Detener monitoreo para un par específico
    """
    try:
        await mcp_monitoring_system.stop_monitoring()
        return {
            "success": True,
            "message": f"Monitoreo detenido para {pair}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deteniendo monitoreo: {str(e)}")

@router.get("/health")
async def get_monitoring_health() -> dict:
    """
    Verificar la salud del sistema de monitoreo
    """
    try:
        system_status = mcp_monitoring_system.get_system_status()
        all_alerts = mcp_monitoring_system.get_all_alerts()
        unread_alerts = len([a for a in all_alerts if not a.is_read])
        
        return {
            "status": "healthy",
            "service": "mcp_monitoring_system",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "active_agents": system_status["active_agents"],
            "total_alerts": system_status["total_alerts"],
            "unread_alerts": unread_alerts,
            "monitoring_active": mcp_monitoring_system.monitoring_active
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en health check: {str(e)}")

@router.get("/cache/stats")
async def get_cache_stats() -> dict:
    """
    Obtener estadísticas del sistema de caché
    """
    try:
        from ..services.cache_service import cache_service
        
        stats = cache_service.get_stats()
        info = cache_service.get_info()
        
        return {
            "cache_stats": stats,
            "cache_info": info,
            "optimization_impact": {
                "estimated_api_calls_saved": stats.get("hits", 0),
                "hit_rate_percentage": stats.get("hit_rate", 0),
                "memory_usage_mb": len(cache_service.cache) * 0.001,  # Estimación aproximada
                "cache_efficiency": "high" if stats.get("hit_rate", 0) > 70 else "medium" if stats.get("hit_rate", 0) > 40 else "low"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo estadísticas de caché: {str(e)}")

@router.post("/cache/clear")
async def clear_cache() -> dict:
    """
    Limpiar todo el caché
    """
    try:
        from ..services.cache_service import cache_service
        
        cache_service.clear()
        
        return {
            "success": True,
            "message": "Caché limpiado exitosamente",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error limpiando caché: {str(e)}") 