"""
Migración: Agregar plan Institutional
====================================
Agregar el plan Institutional con precio de 1199.99 USD
"""

from alembic import op
import sqlalchemy as sa
from datetime import datetime
import uuid

# revision identifiers, used by Alembic.
revision = '004_add_institutional_plan'
down_revision = '003_update_plan_names'
branch_labels = None
depends_on = None

def upgrade():
    """Agregar plan Institutional"""
    
    # Obtener conexión
    connection = op.get_bind()
    
    # Verificar si el plan Institutional ya existe
    result = connection.execute(
        sa.text("""
            SELECT COUNT(*) as count 
            FROM subscription_plans 
            WHERE plan_type = 'institutional'
        """)
    ).fetchone()
    
    if result.count == 0:
        # Insertar plan Institutional
        connection.execute(
            sa.text("""
                INSERT INTO subscription_plans (
                    plan_id, name, plan_type, price, description,
                    traditional_ai, reinforcement_learning, ensemble_ai, lstm_predictions, custom_models, auto_training,
                    daily_requests, prediction_days, backtest_days, trading_pairs, alerts_limit, portfolio_size,
                    advanced_charts, multiple_timeframes, rl_dashboard, ai_monitor, mt4_integration, api_access, custom_reports, priority_support,
                    max_indicators, max_predictions_per_day, max_backtests_per_month, max_portfolios,
                    support_level, response_time_hours,
                    created_at, updated_at
                ) VALUES (
                    :plan_id, :name, :plan_type, :price, :description,
                    :traditional_ai, :reinforcement_learning, :ensemble_ai, :lstm_predictions, :custom_models, :auto_training,
                    :daily_requests, :prediction_days, :backtest_days, :trading_pairs, :alerts_limit, :portfolio_size,
                    :advanced_charts, :multiple_timeframes, :rl_dashboard, :ai_monitor, :mt4_integration, :api_access, :custom_reports, :priority_support,
                    :max_indicators, :max_predictions_per_day, :max_backtests_per_month, :max_portfolios,
                    :support_level, :response_time_hours,
                    NOW(), NOW()
                )
            """),
            {
                'plan_id': str(uuid.uuid4()),
                'name': 'Institutional',
                'plan_type': 'institutional',
                'price': 1199.99,
                'description': 'Plan para empresas de trading y fondos institucionales',
                'traditional_ai': True,
                'reinforcement_learning': True,
                'ensemble_ai': True,
                'lstm_predictions': True,
                'custom_models': True,
                'auto_training': True,
                'daily_requests': 50000,
                'prediction_days': 60,
                'backtest_days': 365,
                'trading_pairs': 50,
                'alerts_limit': 1000,
                'portfolio_size': 100,
                'advanced_charts': True,
                'multiple_timeframes': True,
                'rl_dashboard': True,
                'ai_monitor': True,
                'mt4_integration': True,
                'api_access': True,
                'custom_reports': True,
                'priority_support': True,
                'max_indicators': 50,
                'max_predictions_per_day': 5000,
                'max_backtests_per_month': 2000,
                'max_portfolios': 100,
                'support_level': 'dedicated',
                'response_time_hours': 1
            }
        )
        print("✅ Plan Institutional agregado exitosamente")
    else:
        print("ℹ️ Plan Institutional ya existe, actualizando precio...")
        # Actualizar precio del plan Institutional
        connection.execute(
            sa.text("""
                UPDATE subscription_plans 
                SET price = 1199.99, updated_at = NOW()
                WHERE plan_type = 'institutional'
            """)
        )

def downgrade():
    """Eliminar plan Institutional"""
    
    # Obtener conexión
    connection = op.get_bind()
    
    # Eliminar plan Institutional
    connection.execute(
        sa.text("""
            DELETE FROM subscription_plans 
            WHERE plan_type = 'institutional'
        """)
    ) 