"""
Configuración del Sistema de Suscripciones
==========================================
Configuraciones y constantes para el sistema de suscripciones
"""

from typing import Dict, List, Any
from models.subscription import PlanType

# Configuración de directorios
SUBSCRIPTION_DATA_DIR = "data/subscriptions"
SUBSCRIPTION_LOGS_DIR = "logs/subscriptions"

# Configuración de archivos
SUBSCRIPTION_FILES = {
    "plans": "plans.json",
    "subscriptions": "subscriptions.json",
    "usage": "usage.json",
    "logs": "subscription_activity.log"
}

# Configuración de límites por defecto
DEFAULT_LIMITS = {
    PlanType.FREEMIUM: {
        "daily_requests": 100,
        "predictions_per_day": 10,
        "backtests_per_month": 5,
        "alerts_limit": 3,
        "trading_pairs": 1,
        "portfolios": 1,
        "indicators": 1
    },
    PlanType.BASIC: {
        "daily_requests": 500,
        "predictions_per_day": 50,
        "backtests_per_month": 20,
        "alerts_limit": 10,
        "trading_pairs": 5,
        "portfolios": 3,
        "indicators": 3
    },
    PlanType.PRO: {
        "daily_requests": 2000,
        "predictions_per_day": 200,
        "backtests_per_month": 100,
        "alerts_limit": 50,
        "trading_pairs": 50,
        "portfolios": 10,
        "indicators": 10
    },
    PlanType.ELITE: {
        "daily_requests": 10000,
        "predictions_per_day": 1000,
        "backtests_per_month": 500,
        "alerts_limit": 200,
        "trading_pairs": 1000,
        "portfolios": 100,
        "indicators": 50
    }
}

# Configuración de características por plan
PLAN_FEATURES = {
    PlanType.FREEMIUM: {
        "ai_capabilities": {
            "traditional_ai": True,
            "reinforcement_learning": False,
            "ensemble_ai": False,
            "lstm_predictions": False,
            "custom_models": False,
            "auto_training": False
        },
        "ui_features": {
            "advanced_charts": False,
            "multiple_timeframes": False,
            "rl_dashboard": False,
            "ai_monitor": False,
            "mt4_integration": False,
            "api_access": False,
            "custom_reports": False,
            "priority_support": False
        },
        "api_limits": {
            "prediction_days": 3,
            "backtest_days": 30,
            "portfolio_size": 1
        }
    },
    PlanType.BASIC: {
        "ai_capabilities": {
            "traditional_ai": True,
            "reinforcement_learning": False,
            "ensemble_ai": False,
            "lstm_predictions": False,
            "custom_models": False,
            "auto_training": False
        },
        "ui_features": {
            "advanced_charts": True,
            "multiple_timeframes": False,
            "rl_dashboard": False,
            "ai_monitor": False,
            "mt4_integration": False,
            "api_access": False,
            "custom_reports": False,
            "priority_support": False
        },
        "api_limits": {
            "prediction_days": 7,
            "backtest_days": 90,
            "portfolio_size": 3
        }
    },
    PlanType.PRO: {
        "ai_capabilities": {
            "traditional_ai": True,
            "reinforcement_learning": True,
            "ensemble_ai": True,
            "lstm_predictions": True,
            "custom_models": False,
            "auto_training": True
        },
        "ui_features": {
            "advanced_charts": True,
            "multiple_timeframes": True,
            "rl_dashboard": True,
            "ai_monitor": True,
            "mt4_integration": True,
            "api_access": False,
            "custom_reports": True,
            "priority_support": False
        },
        "api_limits": {
            "prediction_days": 14,
            "backtest_days": 365,
            "portfolio_size": 10
        }
    },
    PlanType.ELITE: {
        "ai_capabilities": {
            "traditional_ai": True,
            "reinforcement_learning": True,
            "ensemble_ai": True,
            "lstm_predictions": True,
            "custom_models": True,
            "auto_training": True
        },
        "ui_features": {
            "advanced_charts": True,
            "multiple_timeframes": True,
            "rl_dashboard": True,
            "ai_monitor": True,
            "mt4_integration": True,
            "api_access": True,
            "custom_reports": True,
            "priority_support": True
        },
        "api_limits": {
            "prediction_days": 30,
            "backtest_days": 1825,  # 5 años
            "portfolio_size": 100
        }
    }
}

# Configuración de soporte por plan
SUPPORT_CONFIG = {
    PlanType.FREEMIUM: {
        "level": "community",
        "response_time_hours": 72,
        "channels": ["community_forum"],
        "priority": "low"
    },
    PlanType.BASIC: {
        "level": "email",
        "response_time_hours": 48,
        "channels": ["email", "community_forum"],
        "priority": "normal"
    },
    PlanType.PRO: {
        "level": "email",
        "response_time_hours": 24,
        "channels": ["email", "chat", "community_forum"],
        "priority": "high"
    },
    PlanType.ELITE: {
        "level": "phone",
        "response_time_hours": 4,
        "channels": ["phone", "email", "chat", "dedicated_manager"],
        "priority": "urgent"
    }
}

# Configuración de precios
PRICING_CONFIG = {
    PlanType.FREEMIUM: {
        "price": 0.0,
        "currency": "USD",
        "billing_cycle": "monthly",
        "trial_days": 0,
        "discounts": {}
    },
    PlanType.BASIC: {
        "price": 29.0,
        "currency": "USD",
        "billing_cycle": "monthly",
        "trial_days": 7,
        "discounts": {
            "annual": 0.20,  # 20% descuento anual
            "quarterly": 0.10  # 10% descuento trimestral
        }
    },
    PlanType.PRO: {
        "price": 99.0,
        "currency": "USD",
        "billing_cycle": "monthly",
        "trial_days": 14,
        "discounts": {
            "annual": 0.25,  # 25% descuento anual
            "quarterly": 0.15  # 15% descuento trimestral
        }
    },
    PlanType.ELITE: {
        "price": 299.0,
        "currency": "USD",
        "billing_cycle": "monthly",
        "trial_days": 30,
        "discounts": {
            "annual": 0.30,  # 30% descuento anual
            "quarterly": 0.20  # 20% descuento trimestral
        }
    }
}

# Configuración de beneficios por plan
BENEFITS_CONFIG = {
    PlanType.FREEMIUM: [
        "Señales básicas de trading",
        "1 indicador técnico (RSI)",
        "Predicciones limitadas (3 días)",
        "Backtesting básico (30 días)",
        "1 par de trading (EUR/USD)",
        "3 alertas básicas"
    ],
    PlanType.BASIC: [
        "AI Tradicional completa",
        "3 indicadores técnicos (RSI, MACD, Bollinger)",
        "Predicciones mejoradas (7 días)",
        "Backtesting avanzado (90 días)",
        "5 pares de trading",
        "10 alertas avanzadas",
        "Análisis básico de tendencias",
        "Reportes mensuales"
    ],
    PlanType.PRO: [
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
    PlanType.ELITE: [
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
    ]
}

# Configuración de mensajes de error
ERROR_MESSAGES = {
    "no_subscription": "Usuario sin suscripción activa",
    "plan_not_found": "Plan no encontrado",
    "feature_not_available": "Característica no disponible en este plan",
    "limit_exceeded": "Límite alcanzado para esta característica",
    "upgrade_required": "Se requiere upgrade para acceder a esta característica",
    "trial_expired": "Período de prueba expirado",
    "subscription_expired": "Suscripción expirada",
    "payment_required": "Pago requerido para continuar"
}

# Configuración de mensajes de upgrade
UPGRADE_MESSAGES = {
    PlanType.FREEMIUM: {
        "title": "Upgrade a Básico",
        "message": "Obtén acceso a más herramientas de trading",
        "cta": "Upgrade por $29/mes"
    },
    PlanType.BASIC: {
        "title": "Upgrade a Pro",
        "message": "Accede a AI avanzada y Reinforcement Learning",
        "cta": "Upgrade por $99/mes"
    },
    PlanType.PRO: {
        "title": "Upgrade a Elite",
        "message": "Acceso completo a todas las características",
        "cta": "Upgrade por $299/mes"
    }
}

# Configuración de métricas de uso
USAGE_METRICS_CONFIG = {
    "reset_schedule": "daily",  # daily, weekly, monthly
    "tracking_enabled": True,
    "metrics_to_track": [
        "api_requests",
        "predictions",
        "backtests",
        "alerts",
        "rl_episodes",
        "custom_models",
        "trades"
    ]
}

# Configuración de notificaciones
NOTIFICATION_CONFIG = {
    "usage_warnings": {
        "enabled": True,
        "thresholds": {
            "api_requests": 0.8,  # 80% del límite
            "predictions": 0.8,
            "backtests": 0.8,
            "alerts": 0.8
        }
    },
    "expiry_notifications": {
        "enabled": True,
        "days_before": [7, 3, 1]  # Notificar 7, 3 y 1 día antes
    }
} 