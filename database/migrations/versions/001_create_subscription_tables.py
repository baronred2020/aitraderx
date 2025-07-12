"""
Migración: Crear tablas de suscripciones
========================================
Migración inicial para el sistema de suscripciones
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = '001_create_subscription_tables'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    """Crear tablas de suscripciones"""
    
    # Tabla: subscription_plans
    op.create_table('subscription_plans',
        sa.Column('plan_id', sa.String(36), primary_key=True),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('plan_type', sa.String(20), nullable=False, unique=True),
        sa.Column('price', sa.Float, nullable=False),
        sa.Column('currency', sa.String(3), default='USD'),
        sa.Column('billing_cycle', sa.String(20), default='monthly'),
        
        # Capacidades de IA
        sa.Column('traditional_ai', sa.Boolean, default=False),
        sa.Column('reinforcement_learning', sa.Boolean, default=False),
        sa.Column('ensemble_ai', sa.Boolean, default=False),
        sa.Column('lstm_predictions', sa.Boolean, default=False),
        sa.Column('custom_models', sa.Boolean, default=False),
        sa.Column('auto_training', sa.Boolean, default=False),
        
        # Límites de API
        sa.Column('daily_requests', sa.Integer, default=100),
        sa.Column('prediction_days', sa.Integer, default=3),
        sa.Column('backtest_days', sa.Integer, default=30),
        sa.Column('trading_pairs', sa.Integer, default=1),
        sa.Column('alerts_limit', sa.Integer, default=3),
        sa.Column('portfolio_size', sa.Integer, default=1),
        
        # Características de UI
        sa.Column('advanced_charts', sa.Boolean, default=False),
        sa.Column('multiple_timeframes', sa.Boolean, default=False),
        sa.Column('rl_dashboard', sa.Boolean, default=False),
        sa.Column('ai_monitor', sa.Boolean, default=False),
        sa.Column('mt4_integration', sa.Boolean, default=False),
        sa.Column('api_access', sa.Boolean, default=False),
        sa.Column('custom_reports', sa.Boolean, default=False),
        sa.Column('priority_support', sa.Boolean, default=False),
        
        # Configuración de límites
        sa.Column('max_indicators', sa.Integer, default=1),
        sa.Column('max_predictions_per_day', sa.Integer, default=10),
        sa.Column('max_backtests_per_month', sa.Integer, default=5),
        sa.Column('max_portfolios', sa.Integer, default=1),
        
        # Configuración de soporte
        sa.Column('support_level', sa.String(20), default='email'),
        sa.Column('response_time_hours', sa.Integer, default=48),
        
        # Descripción y beneficios
        sa.Column('description', sa.Text),
        sa.Column('benefits', sa.JSON),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, default=sa.func.now(), onupdate=sa.func.now())
    )
    
    # Tabla: user_subscriptions
    op.create_table('user_subscriptions',
        sa.Column('subscription_id', sa.String(36), primary_key=True),
        sa.Column('user_id', sa.String(100), nullable=False, index=True),
        sa.Column('plan_id', sa.String(36), nullable=False, index=True),
        sa.Column('plan_type', sa.String(20), nullable=False),
        
        # Fechas
        sa.Column('start_date', sa.DateTime, nullable=False),
        sa.Column('end_date', sa.DateTime, nullable=False),
        sa.Column('trial_end_date', sa.DateTime, nullable=True),
        
        # Estado
        sa.Column('status', sa.String(20), nullable=False, default='active'),
        sa.Column('is_trial', sa.Boolean, default=False),
        
        # Métricas de uso
        sa.Column('usage_metrics', sa.JSON, default=dict),
        
        # Configuración de pagos
        sa.Column('payment_method', sa.String(50), nullable=True),
        sa.Column('auto_renew', sa.Boolean, default=True),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, default=sa.func.now(), onupdate=sa.func.now()),
        
        # Foreign Keys
        sa.ForeignKeyConstraint(['plan_id'], ['subscription_plans.plan_id'])
    )
    
    # Tabla: usage_metrics
    op.create_table('usage_metrics',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('user_id', sa.String(100), nullable=False, index=True),
        sa.Column('subscription_id', sa.String(36), nullable=False, index=True),
        sa.Column('date', sa.DateTime, nullable=False, default=sa.func.now()),
        
        # Métricas de API
        sa.Column('api_requests_today', sa.Integer, default=0),
        sa.Column('predictions_made_today', sa.Integer, default=0),
        sa.Column('backtests_run_today', sa.Integer, default=0),
        sa.Column('alerts_created', sa.Integer, default=0),
        
        # Métricas de IA
        sa.Column('ai_models_used', sa.JSON, default=list),
        sa.Column('rl_episodes_trained', sa.Integer, default=0),
        sa.Column('custom_models_created', sa.Integer, default=0),
        
        # Métricas de trading
        sa.Column('trades_executed', sa.Integer, default=0),
        sa.Column('portfolio_value', sa.Float, default=0.0),
        sa.Column('profit_loss', sa.Float, default=0.0),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, default=sa.func.now(), onupdate=sa.func.now()),
        
        # Foreign Keys
        sa.ForeignKeyConstraint(['subscription_id'], ['user_subscriptions.subscription_id'])
    )
    
    # Tabla: subscription_upgrades
    op.create_table('subscription_upgrades',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('user_id', sa.String(100), nullable=False, index=True),
        sa.Column('current_plan', sa.String(20), nullable=False),
        sa.Column('target_plan', sa.String(20), nullable=False),
        sa.Column('reason', sa.Text, nullable=True),
        sa.Column('status', sa.String(20), default='pending'),
        
        # Timestamps
        sa.Column('requested_at', sa.DateTime, default=sa.func.now()),
        sa.Column('processed_at', sa.DateTime, nullable=True),
        sa.Column('created_at', sa.DateTime, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, default=sa.func.now(), onupdate=sa.func.now())
    )
    
    # Tabla: subscription_payments
    op.create_table('subscription_payments',
        sa.Column('payment_id', sa.String(36), primary_key=True),
        sa.Column('subscription_id', sa.String(36), nullable=False, index=True),
        sa.Column('user_id', sa.String(100), nullable=False, index=True),
        
        # Información del pago
        sa.Column('amount', sa.Float, nullable=False),
        sa.Column('currency', sa.String(3), default='USD'),
        sa.Column('payment_method', sa.String(50), nullable=False),
        sa.Column('payment_provider', sa.String(50), nullable=False),
        sa.Column('provider_payment_id', sa.String(100), nullable=True),
        
        # Estado del pago
        sa.Column('status', sa.String(20), default='pending'),
        sa.Column('billing_cycle', sa.String(20), default='monthly'),
        
        # Timestamps
        sa.Column('payment_date', sa.DateTime, nullable=False),
        sa.Column('created_at', sa.DateTime, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, default=sa.func.now(), onupdate=sa.func.now()),
        
        # Foreign Keys
        sa.ForeignKeyConstraint(['subscription_id'], ['user_subscriptions.subscription_id'])
    )
    
    # Tabla: subscription_audit
    op.create_table('subscription_audit',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('user_id', sa.String(100), nullable=False, index=True),
        sa.Column('subscription_id', sa.String(36), nullable=False, index=True),
        
        # Información de la acción
        sa.Column('action', sa.String(50), nullable=False),
        sa.Column('old_plan', sa.String(20), nullable=True),
        sa.Column('new_plan', sa.String(20), nullable=True),
        sa.Column('reason', sa.Text, nullable=True),
        
        # Metadatos
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('user_agent', sa.Text, nullable=True),
        
        # Timestamps
        sa.Column('action_date', sa.DateTime, default=sa.func.now()),
        sa.Column('created_at', sa.DateTime, default=sa.func.now()),
        
        # Foreign Keys
        sa.ForeignKeyConstraint(['subscription_id'], ['user_subscriptions.subscription_id'])
    )

def downgrade():
    """Eliminar tablas de suscripciones"""
    op.drop_table('subscription_audit')
    op.drop_table('subscription_payments')
    op.drop_table('subscription_upgrades')
    op.drop_table('usage_metrics')
    op.drop_table('user_subscriptions')
    op.drop_table('subscription_plans') 