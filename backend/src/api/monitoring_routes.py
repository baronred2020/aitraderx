from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime, timedelta
import uuid
import random

router = APIRouter(prefix="/api/v1/brain-trader/monitoring", tags=["Monitoring Agents"])

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

# Configuración por defecto
default_config = {
    "enabled": True,
    "auto_refresh": True,
    "refresh_interval": 300,  # 5 minutos
    "alert_thresholds": {
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "drawdown_warning": 0.03,
        "drawdown_critical": 0.05,
        "model_confidence": 0.75,
        "model_accuracy": 0.70
    },
    "agent_settings": {
        "technical": {"enabled": True, "interval": 300},
        "ai": {"enabled": True, "interval": 600},
        "risk": {"enabled": True, "interval": 300},
        "temporal": {"enabled": True, "interval": 1800},
        "fundamental": {"enabled": True, "interval": 3600}
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

# Almacenamiento temporal de alertas
alert_storage = []

# Estado del sistema de monitoreo
monitoring_state = {
    "enabled": True,
    "agents": {
        "technical": {"active": True, "last_check": datetime.now(), "alerts": 0, "performance": 85.0},
        "ai": {"active": True, "last_check": datetime.now(), "alerts": 0, "performance": 92.0},
        "risk": {"active": True, "last_check": datetime.now(), "alerts": 0, "performance": 88.0},
        "temporal": {"active": True, "last_check": datetime.now(), "alerts": 0, "performance": 90.0},
        "fundamental": {"active": True, "last_check": datetime.now(), "alerts": 0, "performance": 87.0}
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
        # Filtrar alertas según parámetros
        filtered_alerts = alert_storage.copy()
        
        if agent_type:
            filtered_alerts = [alert for alert in filtered_alerts if alert["agent_type"] == agent_type]
        
        if severity:
            filtered_alerts = [alert for alert in filtered_alerts if alert["severity"] == severity]
        
        # Limitar resultados
        filtered_alerts = filtered_alerts[:limit]
        
        # Convertir a respuesta
        return [
            MonitoringAlertResponse(
                id=alert["id"],
                agent_type=alert["agent_type"],
                severity=alert["severity"],
                category=alert["category"],
                title=alert["title"],
                description=alert["description"],
                pair=alert.get("pair"),
                brain_type=alert.get("brain_type"),
                timestamp=alert["timestamp"],
                is_read=alert["is_read"],
                action_required=alert["action_required"],
                metadata=alert.get("metadata")
            )
            for alert in filtered_alerts
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo alertas: {str(e)}")

@router.put("/alerts/{alert_id}/read")
async def mark_alert_as_read(alert_id: str) -> dict:
    """
    Marcar una alerta como leída
    """
    try:
        for alert in alert_storage:
            if alert["id"] == alert_id:
                alert["is_read"] = True
                return {"success": True, "message": "Alerta marcada como leída"}
        
        raise HTTPException(status_code=404, detail="Alerta no encontrada")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error marcando alerta: {str(e)}")

@router.get("/status")
async def get_monitoring_system_status() -> MonitoringSystemStatusResponse:
    """
    Obtener estado general del sistema de monitoreo
    """
    try:
        # Calcular métricas
        active_agents = sum(1 for agent in monitoring_state["agents"].values() if agent["active"])
        total_alerts = len(alert_storage)
        critical_alerts = len([alert for alert in alert_storage if alert["severity"] == "critical"])
        
        # Determinar estado general
        if critical_alerts > 0:
            overall_status = "critical"
        elif total_alerts > 10:
            overall_status = "warning"
        else:
            overall_status = "healthy"
        
        # Crear estado de agentes
        agents_status = []
        for agent_type, agent_data in monitoring_state["agents"].items():
            agents_status.append(
                MonitoringAgentStatusResponse(
                    agent_type=agent_type,
                    status="active" if agent_data["active"] else "inactive",
                    last_check=agent_data["last_check"],
                    alerts_count=agent_data["alerts"],
                    is_active=agent_data["active"],
                    performance_score=agent_data["performance"],
                    performance_metrics={
                        "accuracy": agent_data["performance"],
                        "reliability": agent_data["performance"] * 0.95
                    }
                )
            )
        
        return MonitoringSystemStatusResponse(
            overall_status=overall_status,
            active_agents=active_agents,
            total_alerts=total_alerts,
            critical_alerts=critical_alerts,
            last_updated=datetime.now(),
            agents_status=agents_status
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo estado: {str(e)}")

@router.get("/config")
async def get_monitoring_config() -> MonitoringConfigResponse:
    """
    Obtener configuración actual del monitoreo
    """
    try:
        return MonitoringConfigResponse(**default_config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo configuración: {str(e)}")

@router.put("/config")
async def update_monitoring_config(config: dict) -> MonitoringConfigResponse:
    """
    Actualizar configuración del monitoreo
    """
    try:
        # Actualizar configuración
        default_config.update(config)
        monitoring_state["enabled"] = config.get("enabled", True)
        
        return MonitoringConfigResponse(**default_config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error actualizando configuración: {str(e)}")

@router.post("/start")
async def start_monitoring(pair: str, brain_type: Optional[str] = None) -> dict:
    """
    Iniciar monitoreo para un par específico
    """
    try:
        # Simular inicio de monitoreo
        alert_id = str(uuid.uuid4())
        new_alert = {
            "id": alert_id,
            "agent_type": "technical",
            "severity": "info",
            "category": "monitoring_started",
            "title": f"Monitoreo iniciado para {pair}",
            "description": f"Se ha iniciado el monitoreo para el par {pair}",
            "pair": pair,
            "brain_type": brain_type,
            "timestamp": datetime.now(),
            "is_read": False,
            "action_required": False,
            "metadata": {"pair": pair, "brain_type": brain_type}
        }
        
        alert_storage.append(new_alert)
        
        return {
            "success": True,
            "message": f"Monitoreo iniciado para {pair}",
            "alert_id": alert_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error iniciando monitoreo: {str(e)}")

@router.post("/stop")
async def stop_monitoring(pair: str) -> dict:
    """
    Detener monitoreo para un par específico
    """
    try:
        # Simular detención de monitoreo
        alert_id = str(uuid.uuid4())
        new_alert = {
            "id": alert_id,
            "agent_type": "technical",
            "severity": "info",
            "category": "monitoring_stopped",
            "title": f"Monitoreo detenido para {pair}",
            "description": f"Se ha detenido el monitoreo para el par {pair}",
            "pair": pair,
            "timestamp": datetime.now(),
            "is_read": False,
            "action_required": False,
            "metadata": {"pair": pair}
        }
        
        alert_storage.append(new_alert)
        
        return {
            "success": True,
            "message": f"Monitoreo detenido para {pair}",
            "alert_id": alert_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deteniendo monitoreo: {str(e)}")

@router.get("/health")
async def get_monitoring_health() -> dict:
    """
    Verificar salud del sistema de monitoreo
    """
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "uptime": "24h",
            "active_agents": len([a for a in monitoring_state["agents"].values() if a["active"]]),
            "total_alerts": len(alert_storage),
            "system_load": "low"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error verificando salud: {str(e)}")

# Función para generar alertas de ejemplo (para testing)
def generate_sample_alerts():
    """Generar algunas alertas de ejemplo para testing"""
    sample_alerts = [
        {
            "id": str(uuid.uuid4()),
            "agent_type": "technical",
            "severity": "warning",
            "category": "rsi_alert",
            "title": "RSI en zona de sobrecompra",
            "description": "El RSI de EURUSD está en 75, indicando posible sobrecompra",
            "pair": "EURUSD",
            "timestamp": datetime.now() - timedelta(minutes=5),
            "is_read": False,
            "action_required": True,
            "metadata": {"rsi_value": 75, "threshold": 70}
        },
        {
            "id": str(uuid.uuid4()),
            "agent_type": "ai",
            "severity": "info",
            "category": "model_update",
            "title": "Modelo Brain Max actualizado",
            "description": "El modelo Brain Max ha sido actualizado con nuevos datos",
            "brain_type": "brain_max",
            "timestamp": datetime.now() - timedelta(minutes=15),
            "is_read": True,
            "action_required": False,
            "metadata": {"model_version": "2.1.0", "accuracy_improvement": 0.05}
        },
        {
            "id": str(uuid.uuid4()),
            "agent_type": "risk",
            "severity": "critical",
            "category": "drawdown_alert",
            "title": "Drawdown crítico detectado",
            "description": "Se ha detectado un drawdown del 5% en el portafolio",
            "timestamp": datetime.now() - timedelta(minutes=2),
            "is_read": False,
            "action_required": True,
            "metadata": {"drawdown_percentage": 5.2, "threshold": 5.0}
        }
    ]
    
    alert_storage.extend(sample_alerts)

# Generar alertas de ejemplo al importar el módulo
generate_sample_alerts() 