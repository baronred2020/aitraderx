import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import numpy as np
from dataclasses import dataclass
from enum import Enum
from .cache_service import cache_service, cached

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentType(Enum):
    TECHNICAL = "technical"
    AI = "ai"
    RISK = "risk"
    TEMPORAL = "temporal"
    FUNDAMENTAL = "fundamental"

class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class MonitoringAlert:
    id: str
    agent_type: AgentType
    severity: Severity
    category: str
    title: str
    description: str
    timestamp: datetime
    pair: Optional[str] = None
    brain_type: Optional[str] = None
    is_read: bool = False
    action_required: bool = False
    metadata: Optional[Dict] = None

@dataclass
class AgentStatus:
    agent_type: AgentType
    status: str
    last_check: datetime
    alerts_count: int
    is_active: bool
    performance_score: float
    performance_metrics: Optional[Dict] = None

class BaseMCPAgent:
    """Agente MCP base con funcionalidades comunes"""
    
    def __init__(self, agent_type: AgentType, name: str):
        self.agent_type = agent_type
        self.name = name
        self.is_active = False
        self.alerts: List[MonitoringAlert] = []
        self.monitoring_interval = 30  # segundos
        self.last_check = datetime.now()
        self.performance_score = 100.0
        self.monitoring_task: Optional[asyncio.Task] = None
        
    async def start_monitoring(self, symbol: str, brain_type: Optional[str] = None):
        """Inicia el monitoreo en segundo plano"""
        if self.is_active:
            logger.warning(f"Agente {self.name} ya está monitoreando")
            return
            
        self.is_active = True
        self.monitoring_task = asyncio.create_task(
            self._monitoring_loop(symbol, brain_type)
        )
        logger.info(f"Agente {self.name} inició monitoreo para {symbol}")
        
    async def stop_monitoring(self):
        """Detiene el monitoreo"""
        self.is_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info(f"Agente {self.name} detuvo monitoreo")
        
    async def _monitoring_loop(self, symbol: str, brain_type: Optional[str] = None):
        """Loop principal de monitoreo"""
        while self.is_active:
            try:
                await self._check_conditions(symbol, brain_type)
                self.last_check = datetime.now()
                await asyncio.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Error en agente {self.name}: {e}")
                await asyncio.sleep(5)  # Espera corta antes de reintentar
                
    async def _check_conditions(self, symbol: str, brain_type: Optional[str] = None):
        """Método abstracto para verificar condiciones específicas"""
        raise NotImplementedError
        
    def add_alert(self, alert: MonitoringAlert):
        """Añade una nueva alerta"""
        self.alerts.append(alert)
        logger.info(f"Nueva alerta {alert.severity.value} en {self.name}: {alert.title}")
        
    def get_alerts(self) -> List[MonitoringAlert]:
        """Retorna todas las alertas del agente"""
        return self.alerts.copy()
        
    def get_status(self) -> AgentStatus:
        """Retorna el estado actual del agente"""
        return AgentStatus(
            agent_type=self.agent_type,
            status="monitoring" if self.is_active else "idle",
            last_check=self.last_check,
            alerts_count=len(self.alerts),
            is_active=self.is_active,
            performance_score=self.performance_score,
            performance_metrics=self._get_performance_metrics()
        )
        
    def _get_performance_metrics(self) -> Dict:
        """Métricas de rendimiento específicas del agente"""
        return {
            "alerts_generated": len(self.alerts),
            "critical_alerts": len([a for a in self.alerts if a.severity == Severity.CRITICAL]),
            "uptime_percentage": self.performance_score
        }

class TechnicalMCPAgent(BaseMCPAgent):
    """Agente MCP para monitoreo técnico"""
    
    def __init__(self):
        super().__init__(AgentType.TECHNICAL, "Technical Monitoring Agent")
        self.monitoring_interval = 300  # 5 minutos en lugar de 30 segundos
        self.rsi_thresholds = {"overbought": 70, "oversold": 30}
        self.macd_thresholds = {"signal_strength": 0.5}
        
    async def _check_conditions(self, symbol: str, brain_type: Optional[str] = None):
        """Verifica condiciones técnicas"""
        try:
            # Simular obtención de datos técnicos
            rsi_value = await self._get_rsi_value(symbol)
            macd_data = await self._get_macd_data(symbol)
            
            # Verificar RSI
            if rsi_value > self.rsi_thresholds["overbought"]:
                self.add_alert(MonitoringAlert(
                    id=f"tech_rsi_{datetime.now().timestamp()}",
                    agent_type=self.agent_type,
                    severity=Severity.HIGH,
                    category="overbought_condition",
                    title=f"RSI en sobrecompra para {symbol}",
                    description=f"RSI en {rsi_value:.2f}, por encima del umbral de {self.rsi_thresholds['overbought']}",
                    pair=symbol,
                    brain_type=brain_type,
                    timestamp=datetime.now(),
                    action_required=True,
                    metadata={"rsi_value": rsi_value, "threshold": self.rsi_thresholds["overbought"]}
                ))
                
            elif rsi_value < self.rsi_thresholds["oversold"]:
                self.add_alert(MonitoringAlert(
                    id=f"tech_rsi_{datetime.now().timestamp()}",
                    agent_type=self.agent_type,
                    severity=Severity.MEDIUM,
                    category="oversold_condition",
                    title=f"RSI en sobreventa para {symbol}",
                    description=f"RSI en {rsi_value:.2f}, por debajo del umbral de {self.rsi_thresholds['oversold']}",
                    pair=symbol,
                    brain_type=brain_type,
                    timestamp=datetime.now(),
                    action_required=False,
                    metadata={"rsi_value": rsi_value, "threshold": self.rsi_thresholds["oversold"]}
                ))
                
        except Exception as e:
            logger.error(f"Error en verificación técnica: {e}")
            
    @cached(ttl=300, key_prefix="rsi")  # Cache por 5 minutos
    async def _get_rsi_value(self, symbol: str) -> float:
        """Obtiene el valor RSI actual (simulado) con caché"""
        # TODO: Integrar con API real de datos de mercado
        logger.debug(f"Fetching RSI for {symbol} from API")
        return np.random.uniform(20, 80)
        
    @cached(ttl=300, key_prefix="macd")  # Cache por 5 minutos
    async def _get_macd_data(self, symbol: str) -> Dict:
        """Obtiene datos MACD (simulado) con caché"""
        # TODO: Integrar con API real de datos de mercado
        logger.debug(f"Fetching MACD for {symbol} from API")
        return {
            "macd": np.random.uniform(-0.5, 0.5),
            "signal": np.random.uniform(-0.5, 0.5),
            "histogram": np.random.uniform(-0.3, 0.3)
        }

class AIMCPAgent(BaseMCPAgent):
    """Agente MCP para monitoreo de IA"""
    
    def __init__(self):
        super().__init__(AgentType.AI, "AI Monitoring Agent")
        self.monitoring_interval = 600  # 10 minutos en lugar de 60 segundos
        self.confidence_threshold = 0.75
        self.accuracy_threshold = 0.70
        
    async def _check_conditions(self, symbol: str, brain_type: Optional[str] = None):
        """Verifica condiciones de IA"""
        try:
            # Simular obtención de métricas de IA
            model_confidence = await self._get_model_confidence(brain_type)
            model_accuracy = await self._get_model_accuracy(brain_type)
            
            # Verificar confianza del modelo
            if model_confidence < self.confidence_threshold:
                self.add_alert(MonitoringAlert(
                    id=f"ai_confidence_{datetime.now().timestamp()}",
                    agent_type=self.agent_type,
                    severity=Severity.HIGH,
                    category="low_confidence",
                    title=f"Confianza baja en modelo {brain_type}",
                    description=f"Confianza del modelo: {model_confidence:.2%}, por debajo del umbral de {self.confidence_threshold:.2%}",
                    pair=symbol,
                    brain_type=brain_type,
                    timestamp=datetime.now(),
                    action_required=True,
                    metadata={"confidence": model_confidence, "threshold": self.confidence_threshold}
                ))
                
            # Verificar precisión del modelo
            if model_accuracy < self.accuracy_threshold:
                self.add_alert(MonitoringAlert(
                    id=f"ai_accuracy_{datetime.now().timestamp()}",
                    agent_type=self.agent_type,
                    severity=Severity.MEDIUM,
                    category="low_accuracy",
                    title=f"Precisión degradada en modelo {brain_type}",
                    description=f"Precisión del modelo: {model_accuracy:.2%}, por debajo del umbral de {self.accuracy_threshold:.2%}",
                    pair=symbol,
                    brain_type=brain_type,
                    timestamp=datetime.now(),
                    action_required=False,
                    metadata={"accuracy": model_accuracy, "threshold": self.accuracy_threshold}
                ))
                
        except Exception as e:
            logger.error(f"Error en verificación de IA: {e}")
            
    async def _get_model_confidence(self, brain_type: Optional[str]) -> float:
        """Obtiene la confianza del modelo (simulado)"""
        # TODO: Integrar con modelos reales de Brain Trader
        return np.random.uniform(0.6, 0.95)
        
    async def _get_model_accuracy(self, brain_type: Optional[str]) -> float:
        """Obtiene la precisión del modelo (simulado)"""
        # TODO: Integrar con modelos reales de Brain Trader
        return np.random.uniform(0.65, 0.90)

class RiskMCPAgent(BaseMCPAgent):
    """Agente MCP para monitoreo de riesgo"""
    
    def __init__(self):
        super().__init__(AgentType.RISK, "Risk Monitoring Agent")
        self.monitoring_interval = 300  # 5 minutos en lugar de 45 segundos
        self.drawdown_warning_threshold = 0.03  # 3%
        self.drawdown_critical_threshold = 0.05  # 5%
        
    async def _check_conditions(self, symbol: str, brain_type: Optional[str] = None):
        """Verifica condiciones de riesgo"""
        try:
            # Simular obtención de métricas de riesgo
            current_drawdown = await self._get_current_drawdown(symbol)
            portfolio_value = await self._get_portfolio_value()
            
            # Verificar drawdown crítico
            if current_drawdown > self.drawdown_critical_threshold:
                self.add_alert(MonitoringAlert(
                    id=f"risk_drawdown_{datetime.now().timestamp()}",
                    agent_type=self.agent_type,
                    severity=Severity.CRITICAL,
                    category="critical_drawdown",
                    title=f"Drawdown crítico detectado para {symbol}",
                    description=f"Drawdown actual: {current_drawdown:.2%}, excede el límite crítico de {self.drawdown_critical_threshold:.2%}",
                    pair=symbol,
                    brain_type=brain_type,
                    timestamp=datetime.now(),
                    action_required=True,
                    metadata={"drawdown": current_drawdown, "threshold": self.drawdown_critical_threshold}
                ))
                
            # Verificar drawdown de advertencia
            elif current_drawdown > self.drawdown_warning_threshold:
                self.add_alert(MonitoringAlert(
                    id=f"risk_drawdown_{datetime.now().timestamp()}",
                    agent_type=self.agent_type,
                    severity=Severity.HIGH,
                    category="warning_drawdown",
                    title=f"Drawdown de advertencia para {symbol}",
                    description=f"Drawdown actual: {current_drawdown:.2%}, excede el límite de advertencia de {self.drawdown_warning_threshold:.2%}",
                    pair=symbol,
                    brain_type=brain_type,
                    timestamp=datetime.now(),
                    action_required=True,
                    metadata={"drawdown": current_drawdown, "threshold": self.drawdown_warning_threshold}
                ))
                
        except Exception as e:
            logger.error(f"Error en verificación de riesgo: {e}")
            
    async def _get_current_drawdown(self, symbol: str) -> float:
        """Obtiene el drawdown actual (simulado)"""
        # TODO: Integrar con sistema real de trading
        return np.random.uniform(0.01, 0.08)
        
    async def _get_portfolio_value(self) -> float:
        """Obtiene el valor del portafolio (simulado)"""
        # TODO: Integrar con sistema real de trading
        return np.random.uniform(10000, 50000)

class TemporalMCPAgent(BaseMCPAgent):
    """Agente MCP para monitoreo temporal"""
    
    def __init__(self):
        super().__init__(AgentType.TEMPORAL, "Temporal Monitoring Agent")
        self.monitoring_interval = 1800  # 30 minutos en lugar de 5 minutos
        
    async def _check_conditions(self, symbol: str, brain_type: Optional[str] = None):
        """Verifica condiciones temporales"""
        try:
            current_time = datetime.now()
            session_info = await self._get_market_session_info()
            
            # Verificar cambio de sesión
            if session_info["session_ending"]:
                self.add_alert(MonitoringAlert(
                    id=f"temporal_session_{datetime.now().timestamp()}",
                    agent_type=self.agent_type,
                    severity=Severity.MEDIUM,
                    category="session_change",
                    title=f"Cambio de sesión próximo para {symbol}",
                    description=f"La sesión {session_info['current_session']} termina en {session_info['time_remaining']} minutos",
                    pair=symbol,
                    brain_type=brain_type,
                    timestamp=datetime.now(),
                    action_required=False,
                    metadata={"current_session": session_info["current_session"], "time_remaining": session_info["time_remaining"]}
                ))
                
            # Verificar eventos económicos próximos
            upcoming_events = await self._get_upcoming_events()
            for event in upcoming_events:
                if event["time_until"] <= 60:  # 1 hora
                    self.add_alert(MonitoringAlert(
                        id=f"temporal_event_{datetime.now().timestamp()}",
                        agent_type=self.agent_type,
                        severity=Severity.HIGH,
                        category="economic_event",
                        title=f"Evento económico próximo: {event['name']}",
                        description=f"Evento en {event['time_until']} minutos: {event['description']}",
                        pair=symbol,
                        brain_type=brain_type,
                        timestamp=datetime.now(),
                        action_required=True,
                        metadata={"event": event}
                    ))
                    
        except Exception as e:
            logger.error(f"Error en verificación temporal: {e}")
            
    async def _get_market_session_info(self) -> Dict:
        """Obtiene información de sesiones de mercado (simulado)"""
        # TODO: Integrar con API de horarios de mercado
        sessions = ["asian", "european", "american"]
        current_session = np.random.choice(sessions)
        return {
            "current_session": current_session,
            "session_ending": np.random.choice([True, False]),
            "time_remaining": np.random.randint(0, 120)
        }
        
    async def _get_upcoming_events(self) -> List[Dict]:
        """Obtiene eventos económicos próximos (simulado)"""
        # TODO: Integrar con API de calendario económico
        events = [
            {"name": "NFP", "description": "Non-Farm Payrolls", "time_until": np.random.randint(30, 180)},
            {"name": "CPI", "description": "Consumer Price Index", "time_until": np.random.randint(60, 240)},
            {"name": "FOMC", "description": "Federal Reserve Meeting", "time_until": np.random.randint(120, 360)}
        ]
        return [event for event in events if event["time_until"] <= 180]

class FundamentalMCPAgent(BaseMCPAgent):
    """Agente MCP para monitoreo fundamental"""
    
    def __init__(self):
        super().__init__(AgentType.FUNDAMENTAL, "Fundamental Monitoring Agent")
        self.monitoring_interval = 3600  # 1 hora en lugar de 10 minutos
        
    async def _check_conditions(self, symbol: str, brain_type: Optional[str] = None):
        """Verifica condiciones fundamentales"""
        try:
            # Simular análisis de sentimiento de noticias
            news_sentiment = await self._get_news_sentiment(symbol)
            economic_indicators = await self._get_economic_indicators()
            
            # Verificar sentimiento negativo extremo
            if news_sentiment < -0.7:
                self.add_alert(MonitoringAlert(
                    id=f"fundamental_sentiment_{datetime.now().timestamp()}",
                    agent_type=self.agent_type,
                    severity=Severity.HIGH,
                    category="negative_sentiment",
                    title=f"Sentimiento negativo extremo para {symbol}",
                    description=f"Sentimiento de noticias: {news_sentiment:.2f}, indica posible impacto negativo",
                    pair=symbol,
                    brain_type=brain_type,
                    timestamp=datetime.now(),
                    action_required=True,
                    metadata={"sentiment": news_sentiment, "threshold": -0.7}
                ))
                
            # Verificar indicadores económicos críticos
            for indicator in economic_indicators:
                if indicator["status"] == "critical":
                    self.add_alert(MonitoringAlert(
                        id=f"fundamental_indicator_{datetime.now().timestamp()}",
                        agent_type=self.agent_type,
                        severity=Severity.CRITICAL,
                        category="economic_indicator",
                        title=f"Indicador económico crítico: {indicator['name']}",
                        description=f"Indicador {indicator['name']}: {indicator['value']} - {indicator['description']}",
                        pair=symbol,
                        brain_type=brain_type,
                        timestamp=datetime.now(),
                        action_required=True,
                        metadata={"indicator": indicator}
                    ))
                    
        except Exception as e:
            logger.error(f"Error en verificación fundamental: {e}")
            
    async def _get_news_sentiment(self, symbol: str) -> float:
        """Obtiene el sentimiento de noticias (simulado)"""
        # TODO: Integrar con API de análisis de sentimiento
        return np.random.uniform(-1.0, 1.0)
        
    async def _get_economic_indicators(self) -> List[Dict]:
        """Obtiene indicadores económicos (simulado)"""
        # TODO: Integrar con API de indicadores económicos
        indicators = [
            {"name": "GDP", "value": "2.1%", "status": "normal", "description": "Crecimiento estable"},
            {"name": "Inflation", "value": "3.2%", "status": "warning", "description": "Inflación elevada"},
            {"name": "Unemployment", "value": "4.5%", "status": "critical", "description": "Desempleo en aumento"}
        ]
        return indicators

class MCPMonitoringSystem:
    """Sistema principal de monitoreo MCP"""
    
    def __init__(self):
        self.agents = {
            AgentType.TECHNICAL: TechnicalMCPAgent(),
            AgentType.AI: AIMCPAgent(),
            AgentType.RISK: RiskMCPAgent(),
            AgentType.TEMPORAL: TemporalMCPAgent(),
            AgentType.FUNDAMENTAL: FundamentalMCPAgent()
        }
        self.monitoring_active = False
        
    async def start_monitoring(self, symbol: str, brain_type: Optional[str] = None):
        """Inicia monitoreo para todos los agentes"""
        if self.monitoring_active:
            logger.warning("Sistema de monitoreo ya está activo")
            return
            
        self.monitoring_active = True
        tasks = []
        
        for agent in self.agents.values():
            task = asyncio.create_task(agent.start_monitoring(symbol, brain_type))
            tasks.append(task)
            
        logger.info(f"Monitoreo MCP iniciado para {symbol} con {len(self.agents)} agentes")
        
    async def stop_monitoring(self):
        """Detiene monitoreo para todos los agentes"""
        self.monitoring_active = False
        
        for agent in self.agents.values():
            await agent.stop_monitoring()
            
        logger.info("Monitoreo MCP detenido")
        
    def get_all_alerts(self) -> List[MonitoringAlert]:
        """Obtiene alertas de todos los agentes"""
        all_alerts = []
        for agent in self.agents.values():
            all_alerts.extend(agent.get_alerts())
        return all_alerts
        
    def get_system_status(self) -> Dict:
        """Obtiene el estado del sistema completo"""
        agents_status = [agent.get_status() for agent in self.agents.values()]
        total_alerts = len(self.get_all_alerts())
        critical_alerts = len([a for a in self.get_all_alerts() if a.severity == Severity.CRITICAL])
        
        return {
            "overall_status": "healthy" if critical_alerts == 0 else "warning" if critical_alerts < 3 else "critical",
            "active_agents": len([a for a in agents_status if a.is_active]),
            "total_alerts": total_alerts,
            "critical_alerts": critical_alerts,
            "last_updated": datetime.now(),
            "agents_status": agents_status
        }
        
    def get_agent_status(self, agent_type: AgentType) -> Optional[AgentStatus]:
        """Obtiene el estado de un agente específico"""
        agent = self.agents.get(agent_type)
        return agent.get_status() if agent else None

# Instancia global del sistema de monitoreo
mcp_monitoring_system = MCPMonitoringSystem() 