"""
Sistema de Suscripciones - Modelos de Datos
===========================================
Modelos para gestionar planes de suscripción y usuarios
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field
import uuid

class PlanType(str, Enum):
    """Tipos de planes de suscripción"""
    FREEMIUM = "freemium"
    BASIC = "basic"
    PRO = "pro"
    ELITE = "elite"

class SubscriptionStatus(str, Enum):
    """Estados de suscripción"""
    ACTIVE = "active"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    PENDING = "pending"
    TRIAL = "trial"

class AICapabilities(BaseModel):
    """Capacidades de IA por plan"""
    traditional_ai: bool = False
    reinforcement_learning: bool = False
    ensemble_ai: bool = False
    lstm_predictions: bool = False
    custom_models: bool = False
    auto_training: bool = False

class APILimits(BaseModel):
    """Límites de API por plan"""
    daily_requests: int = 100
    prediction_days: int = 3
    backtest_days: int = 30
    trading_pairs: int = 1
    alerts_limit: int = 3
    portfolio_size: int = 1

class UIFeatures(BaseModel):
    """Características de UI por plan"""
    advanced_charts: bool = False
    multiple_timeframes: bool = False
    rl_dashboard: bool = False
    ai_monitor: bool = False
    mt4_integration: bool = False
    api_access: bool = False
    custom_reports: bool = False
    priority_support: bool = False

class SubscriptionPlan(BaseModel):
    """Modelo para planes de suscripción"""
    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    plan_type: PlanType
    price: float
    currency: str = "USD"
    billing_cycle: str = "monthly"
    
    # Capacidades de IA
    ai_capabilities: AICapabilities
    
    # Límites de API
    api_limits: APILimits
    
    # Características de UI
    ui_features: UIFeatures
    
    # Descripción y beneficios
    description: str
    benefits: List[str]
    
    # Configuración de límites
    max_indicators: int = 1
    max_predictions_per_day: int = 10
    max_backtests_per_month: int = 5
    max_portfolios: int = 1
    
    # Configuración de soporte
    support_level: str = "email"
    response_time_hours: int = 48
    
    class Config:
        use_enum_values = True

class UserSubscription(BaseModel):
    """Modelo para suscripciones de usuarios"""
    subscription_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    plan_id: str
    plan_type: PlanType
    
    # Fechas
    start_date: datetime
    end_date: datetime
    trial_end_date: Optional[datetime] = None
    
    # Estado
    status: SubscriptionStatus
    is_trial: bool = False
    
    # Métricas de uso
    usage_metrics: Dict[str, Any] = Field(default_factory=dict)
    
    # Configuración de pagos
    payment_method: Optional[str] = None
    auto_renew: bool = True
    
    # Historial
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        use_enum_values = True

class UsageMetrics(BaseModel):
    """Métricas de uso del usuario"""
    user_id: str
    date: datetime = Field(default_factory=datetime.now)
    
    # Métricas de API
    api_requests_today: int = 0
    predictions_made_today: int = 0
    backtests_run_today: int = 0
    alerts_created: int = 0
    
    # Métricas de IA
    ai_models_used: List[str] = Field(default_factory=list)
    rl_episodes_trained: int = 0
    custom_models_created: int = 0
    
    # Métricas de trading
    trades_executed: int = 0
    portfolio_value: float = 0.0
    profit_loss: float = 0.0

class SubscriptionUpgrade(BaseModel):
    """Modelo para solicitudes de upgrade"""
    user_id: str
    current_plan: PlanType
    target_plan: PlanType
    reason: Optional[str] = None
    requested_at: datetime = Field(default_factory=datetime.now)

# Configuraciones predefinidas de planes
DEFAULT_PLANS = {
    PlanType.FREEMIUM: SubscriptionPlan(
        name="Freemium",
        plan_type=PlanType.FREEMIUM,
        price=0.0,
        description="Plan gratuito para empezar con trading básico",
        benefits=[
            "Señales básicas de trading",
            "1 indicador técnico (RSI)",
            "Predicciones limitadas (3 días)",
            "Backtesting básico (30 días)",
            "1 par de trading (EUR/USD)",
            "3 alertas básicas"
        ],
        ai_capabilities=AICapabilities(
            traditional_ai=True,
            reinforcement_learning=False,
            ensemble_ai=False,
            lstm_predictions=False,
            custom_models=False,
            auto_training=False
        ),
        api_limits=APILimits(
            daily_requests=100,
            prediction_days=3,
            backtest_days=30,
            trading_pairs=1,
            alerts_limit=3,
            portfolio_size=1
        ),
        ui_features=UIFeatures(
            advanced_charts=False,
            multiple_timeframes=False,
            rl_dashboard=False,
            ai_monitor=False,
            mt4_integration=False,
            api_access=False,
            custom_reports=False,
            priority_support=False
        ),
        max_indicators=1,
        max_predictions_per_day=10,
        max_backtests_per_month=5,
        max_portfolios=1,
        support_level="community",
        response_time_hours=72
    ),
    
    PlanType.BASIC: SubscriptionPlan(
        name="Básico",
        plan_type=PlanType.BASIC,
        price=29.0,
        description="Plan para traders serios que quieren más herramientas",
        benefits=[
            "AI Tradicional completa",
            "3 indicadores técnicos (RSI, MACD, Bollinger)",
            "Predicciones mejoradas (7 días)",
            "Backtesting avanzado (90 días)",
            "5 pares de trading",
            "10 alertas avanzadas",
            "Análisis básico de tendencias",
            "Reportes mensuales"
        ],
        ai_capabilities=AICapabilities(
            traditional_ai=True,
            reinforcement_learning=False,
            ensemble_ai=False,
            lstm_predictions=False,
            custom_models=False,
            auto_training=False
        ),
        api_limits=APILimits(
            daily_requests=500,
            prediction_days=7,
            backtest_days=90,
            trading_pairs=5,
            alerts_limit=10,
            portfolio_size=3
        ),
        ui_features=UIFeatures(
            advanced_charts=True,
            multiple_timeframes=False,
            rl_dashboard=False,
            ai_monitor=False,
            mt4_integration=False,
            api_access=False,
            custom_reports=False,
            priority_support=False
        ),
        max_indicators=3,
        max_predictions_per_day=50,
        max_backtests_per_month=20,
        max_portfolios=3,
        support_level="email",
        response_time_hours=48
    ),
    
    PlanType.PRO: SubscriptionPlan(
        name="Pro",
        plan_type=PlanType.PRO,
        price=99.0,
        description="Plan para traders profesionales y semi-profesionales",
        benefits=[
            "AI Tradicional Premium + LSTM",
            "Reinforcement Learning (DQN)",
            "Todos los indicadores técnicos",
            "Predicciones avanzadas (14 días)",
            "Backtesting profesional",
            "Todos los pares de trading",
            "Ensemble AI (Tradicional + RL)",
            "Risk Management básico",
            "Portfolio Optimization básico",
            "Integración MT4 básica",
            "Reportes semanales"
        ],
        ai_capabilities=AICapabilities(
            traditional_ai=True,
            reinforcement_learning=True,
            ensemble_ai=True,
            lstm_predictions=True,
            custom_models=False,
            auto_training=True
        ),
        api_limits=APILimits(
            daily_requests=2000,
            prediction_days=14,
            backtest_days=365,
            trading_pairs=50,
            alerts_limit=50,
            portfolio_size=10
        ),
        ui_features=UIFeatures(
            advanced_charts=True,
            multiple_timeframes=True,
            rl_dashboard=True,
            ai_monitor=True,
            mt4_integration=True,
            api_access=False,
            custom_reports=True,
            priority_support=False
        ),
        max_indicators=10,
        max_predictions_per_day=200,
        max_backtests_per_month=100,
        max_portfolios=10,
        support_level="email",
        response_time_hours=24
    ),
    
    PlanType.ELITE: SubscriptionPlan(
        name="Elite",
        plan_type=PlanType.ELITE,
        price=299.0,
        description="Plan para traders institucionales y fondos",
        benefits=[
            "AI Tradicional Elite + máxima precisión",
            "Reinforcement Learning completo (DQN + PPO)",
            "Ensemble AI avanzado optimizado",
            "Predicciones elite (30 días)",
            "Backtesting institucional",
            "Todos los instrumentos (Forex, Stocks, Crypto)",
            "Risk Management avanzado",
            "Portfolio Optimization avanzado",
            "Auto-Trading con AI",
            "Custom Models personalizados",
            "Integración MT4 completa",
            "API personalizada",
            "Soporte prioritario 24/7"
        ],
        ai_capabilities=AICapabilities(
            traditional_ai=True,
            reinforcement_learning=True,
            ensemble_ai=True,
            lstm_predictions=True,
            custom_models=True,
            auto_training=True
        ),
        api_limits=APILimits(
            daily_requests=10000,
            prediction_days=30,
            backtest_days=1825,  # 5 años
            trading_pairs=1000,
            alerts_limit=200,
            portfolio_size=100
        ),
        ui_features=UIFeatures(
            advanced_charts=True,
            multiple_timeframes=True,
            rl_dashboard=True,
            ai_monitor=True,
            mt4_integration=True,
            api_access=True,
            custom_reports=True,
            priority_support=True
        ),
        max_indicators=50,
        max_predictions_per_day=1000,
        max_backtests_per_month=500,
        max_portfolios=100,
        support_level="phone",
        response_time_hours=4
    )
} 