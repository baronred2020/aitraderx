"""
Migración: Insertar planes por defecto
======================================
Insertar los 4 planes de suscripción por defecto
"""

from alembic import op
import sqlalchemy as sa
from datetime import datetime
import uuid

# revision identifiers, used by Alembic.
revision = '002_insert_default_plans'
down_revision = '001_create_subscription_tables'
branch_labels = None
depends_on = None

def upgrade():
    """Insertar planes por defecto"""
    
    # Obtener conexión
    connection = op.get_bind()
    
    # Plan FREEMIUM
    freemium_plan = {
        'plan_id': str(uuid.uuid4()),
        'name': 'Freemium',
        'plan_type': 'freemium',
        'price': 0.0,
        'currency': 'USD',
        'billing_cycle': 'monthly',
        
        # Capacidades de IA
        'traditional_ai': True,
        'reinforcement_learning': False,
        'ensemble_ai': False,
        'lstm_predictions': False,
        'custom_models': False,
        'auto_training': False,
        
        # Límites de API
        'daily_requests': 100,
        'prediction_days': 3,
        'backtest_days': 30,
        'trading_pairs': 1,
        'alerts_limit': 3,
        'portfolio_size': 1,
        
        # Características de UI
        'advanced_charts': False,
        'multiple_timeframes': False,
        'rl_dashboard': False,
        'ai_monitor': False,
        'mt4_integration': False,
        'api_access': False,
        'custom_reports': False,
        'priority_support': False,
        
        # Configuración de límites
        'max_indicators': 1,
        'max_predictions_per_day': 10,
        'max_backtests_per_month': 5,
        'max_portfolios': 1,
        
        # Configuración de soporte
        'support_level': 'community',
        'response_time_hours': 72,
        
        # Descripción y beneficios
        'description': 'Plan gratuito para empezar con trading básico',
        'benefits': [
            'Señales básicas de trading',
            '1 indicador técnico (RSI)',
            'Predicciones limitadas (3 días)',
            'Backtesting básico (30 días)',
            '1 par de trading (EUR/USD)',
            '3 alertas básicas'
        ],
        
        # Timestamps
        'created_at': datetime.utcnow(),
        'updated_at': datetime.utcnow()
    }
    
    # Plan BÁSICO
    basic_plan = {
        'plan_id': str(uuid.uuid4()),
        'name': 'Básico',
        'plan_type': 'basic',
        'price': 29.0,
        'currency': 'USD',
        'billing_cycle': 'monthly',
        
        # Capacidades de IA
        'traditional_ai': True,
        'reinforcement_learning': False,
        'ensemble_ai': False,
        'lstm_predictions': False,
        'custom_models': False,
        'auto_training': False,
        
        # Límites de API
        'daily_requests': 500,
        'prediction_days': 7,
        'backtest_days': 90,
        'trading_pairs': 5,
        'alerts_limit': 10,
        'portfolio_size': 3,
        
        # Características de UI
        'advanced_charts': True,
        'multiple_timeframes': False,
        'rl_dashboard': False,
        'ai_monitor': False,
        'mt4_integration': False,
        'api_access': False,
        'custom_reports': False,
        'priority_support': False,
        
        # Configuración de límites
        'max_indicators': 3,
        'max_predictions_per_day': 50,
        'max_backtests_per_month': 20,
        'max_portfolios': 3,
        
        # Configuración de soporte
        'support_level': 'email',
        'response_time_hours': 48,
        
        # Descripción y beneficios
        'description': 'Plan para traders serios que quieren más herramientas',
        'benefits': [
            'AI Tradicional completa',
            '3 indicadores técnicos (RSI, MACD, Bollinger)',
            'Predicciones mejoradas (7 días)',
            'Backtesting avanzado (90 días)',
            '5 pares de trading',
            '10 alertas avanzadas',
            'Análisis básico de tendencias',
            'Reportes mensuales'
        ],
        
        # Timestamps
        'created_at': datetime.utcnow(),
        'updated_at': datetime.utcnow()
    }
    
    # Plan PRO
    pro_plan = {
        'plan_id': str(uuid.uuid4()),
        'name': 'Pro',
        'plan_type': 'pro',
        'price': 99.0,
        'currency': 'USD',
        'billing_cycle': 'monthly',
        
        # Capacidades de IA
        'traditional_ai': True,
        'reinforcement_learning': True,
        'ensemble_ai': True,
        'lstm_predictions': True,
        'custom_models': False,
        'auto_training': True,
        
        # Límites de API
        'daily_requests': 2000,
        'prediction_days': 14,
        'backtest_days': 365,
        'trading_pairs': 50,
        'alerts_limit': 50,
        'portfolio_size': 10,
        
        # Características de UI
        'advanced_charts': True,
        'multiple_timeframes': True,
        'rl_dashboard': True,
        'ai_monitor': True,
        'mt4_integration': True,
        'api_access': False,
        'custom_reports': True,
        'priority_support': False,
        
        # Configuración de límites
        'max_indicators': 10,
        'max_predictions_per_day': 200,
        'max_backtests_per_month': 100,
        'max_portfolios': 10,
        
        # Configuración de soporte
        'support_level': 'email',
        'response_time_hours': 24,
        
        # Descripción y beneficios
        'description': 'Plan para traders profesionales y semi-profesionales',
        'benefits': [
            'AI Tradicional Premium + LSTM',
            'Reinforcement Learning (DQN)',
            'Todos los indicadores técnicos',
            'Predicciones avanzadas (14 días)',
            'Backtesting profesional',
            'Todos los pares de trading',
            'Ensemble AI (Tradicional + RL)',
            'Risk Management básico',
            'Portfolio Optimization básico',
            'Integración MT4 básica',
            'Reportes semanales'
        ],
        
        # Timestamps
        'created_at': datetime.utcnow(),
        'updated_at': datetime.utcnow()
    }
    
    # Plan ELITE
    elite_plan = {
        'plan_id': str(uuid.uuid4()),
        'name': 'Elite',
        'plan_type': 'elite',
        'price': 299.0,
        'currency': 'USD',
        'billing_cycle': 'monthly',
        
        # Capacidades de IA
        'traditional_ai': True,
        'reinforcement_learning': True,
        'ensemble_ai': True,
        'lstm_predictions': True,
        'custom_models': True,
        'auto_training': True,
        
        # Límites de API
        'daily_requests': 10000,
        'prediction_days': 30,
        'backtest_days': 1825,  # 5 años
        'trading_pairs': 1000,
        'alerts_limit': 200,
        'portfolio_size': 100,
        
        # Características de UI
        'advanced_charts': True,
        'multiple_timeframes': True,
        'rl_dashboard': True,
        'ai_monitor': True,
        'mt4_integration': True,
        'api_access': True,
        'custom_reports': True,
        'priority_support': True,
        
        # Configuración de límites
        'max_indicators': 50,
        'max_predictions_per_day': 1000,
        'max_backtests_per_month': 500,
        'max_portfolios': 100,
        
        # Configuración de soporte
        'support_level': 'phone',
        'response_time_hours': 4,
        
        # Descripción y beneficios
        'description': 'Plan para traders institucionales y fondos',
        'benefits': [
            'AI Tradicional Elite + máxima precisión',
            'Reinforcement Learning completo (DQN + PPO)',
            'Ensemble AI avanzado optimizado',
            'Predicciones elite (30 días)',
            'Backtesting institucional',
            'Todos los instrumentos (Forex, Stocks, Crypto)',
            'Risk Management avanzado',
            'Portfolio Optimization avanzado',
            'Auto-Trading con AI',
            'Custom Models personalizados',
            'Integración MT4 completa',
            'API personalizada',
            'Soporte prioritario 24/7'
        ],
        
        # Timestamps
        'created_at': datetime.utcnow(),
        'updated_at': datetime.utcnow()
    }
    
    # Insertar planes
    plans = [freemium_plan, basic_plan, pro_plan, elite_plan]
    
    for plan in plans:
        connection.execute(
            sa.text("""
                INSERT INTO subscription_plans (
                    plan_id, name, plan_type, price, currency, billing_cycle,
                    traditional_ai, reinforcement_learning, ensemble_ai, lstm_predictions,
                    custom_models, auto_training, daily_requests, prediction_days,
                    backtest_days, trading_pairs, alerts_limit, portfolio_size,
                    advanced_charts, multiple_timeframes, rl_dashboard, ai_monitor,
                    mt4_integration, api_access, custom_reports, priority_support,
                    max_indicators, max_predictions_per_day, max_backtests_per_month,
                    max_portfolios, support_level, response_time_hours, description,
                    benefits, created_at, updated_at
                ) VALUES (
                    :plan_id, :name, :plan_type, :price, :currency, :billing_cycle,
                    :traditional_ai, :reinforcement_learning, :ensemble_ai, :lstm_predictions,
                    :custom_models, :auto_training, :daily_requests, :prediction_days,
                    :backtest_days, :trading_pairs, :alerts_limit, :portfolio_size,
                    :advanced_charts, :multiple_timeframes, :rl_dashboard, :ai_monitor,
                    :mt4_integration, :api_access, :custom_reports, :priority_support,
                    :max_indicators, :max_predictions_per_day, :max_backtests_per_month,
                    :max_portfolios, :support_level, :response_time_hours, :description,
                    :benefits, :created_at, :updated_at
                )
            """),
            plan
        )

def downgrade():
    """Eliminar planes por defecto"""
    connection = op.get_bind()
    connection.execute(
        sa.text("DELETE FROM subscription_plans WHERE plan_type IN ('freemium', 'basic', 'pro', 'elite')")
    ) 