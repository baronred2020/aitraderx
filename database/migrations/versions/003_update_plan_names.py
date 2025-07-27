"""
Migración: Actualizar nombres de planes de suscripción
====================================================
Actualizar los nombres de los planes para hacerlos más atractivos
"""

from alembic import op
import sqlalchemy as sa
from datetime import datetime
import uuid

# revision identifiers, used by Alembic.
revision = '003_update_plan_names'
down_revision = '002_insert_default_plans'
branch_labels = None
depends_on = None

def upgrade():
    """Actualizar nombres de planes"""
    
    # Obtener conexión
    connection = op.get_bind()
    
    # Actualizar plan FREEMIUM a STARTER
    connection.execute(
        sa.text("""
            UPDATE subscription_plans 
            SET name = 'Starter', plan_type = 'starter', updated_at = NOW()
            WHERE plan_type = 'freemium'
        """)
    )
    
    # Actualizar plan BASIC a TRADER
    connection.execute(
        sa.text("""
            UPDATE subscription_plans 
            SET name = 'Trader', plan_type = 'trader', updated_at = NOW()
            WHERE plan_type = 'basic'
        """)
    )
    
    # Actualizar plan PRO a EXPERT
    connection.execute(
        sa.text("""
            UPDATE subscription_plans 
            SET name = 'Expert', plan_type = 'expert', updated_at = NOW()
            WHERE plan_type = 'pro'
        """)
    )
    
    # Actualizar plan ELITE a PREMIUM
    connection.execute(
        sa.text("""
            UPDATE subscription_plans 
            SET name = 'Premium', plan_type = 'premium', updated_at = NOW()
            WHERE plan_type = 'elite'
        """)
    )
    
    # Actualizar precio del plan INSTITUTIONAL
    connection.execute(
        sa.text("""
            UPDATE subscription_plans 
            SET price = 1199.99, updated_at = NOW()
            WHERE plan_type = 'institutional'
        """)
    )
    
    # Actualizar suscripciones de usuarios para usar los nuevos nombres
    connection.execute(
        sa.text("""
            UPDATE user_subscriptions 
            SET plan_type = 'starter', updated_at = NOW()
            WHERE plan_type = 'freemium'
        """)
    )
    
    connection.execute(
        sa.text("""
            UPDATE user_subscriptions 
            SET plan_type = 'trader', updated_at = NOW()
            WHERE plan_type = 'basic'
        """)
    )
    
    connection.execute(
        sa.text("""
            UPDATE user_subscriptions 
            SET plan_type = 'expert', updated_at = NOW()
            WHERE plan_type = 'pro'
        """)
    )
    
    connection.execute(
        sa.text("""
            UPDATE user_subscriptions 
            SET plan_type = 'premium', updated_at = NOW()
            WHERE plan_type = 'elite'
        """)
    )

def downgrade():
    """Revertir cambios"""
    
    # Obtener conexión
    connection = op.get_bind()
    
    # Revertir STARTER a FREEMIUM
    connection.execute(
        sa.text("""
            UPDATE subscription_plans 
            SET name = 'Freemium', plan_type = 'freemium', updated_at = NOW()
            WHERE plan_type = 'starter'
        """)
    )
    
    # Revertir TRADER a BASIC
    connection.execute(
        sa.text("""
            UPDATE subscription_plans 
            SET name = 'Básico', plan_type = 'basic', updated_at = NOW()
            WHERE plan_type = 'trader'
        """)
    )
    
    # Revertir EXPERT a PRO
    connection.execute(
        sa.text("""
            UPDATE subscription_plans 
            SET name = 'Pro', plan_type = 'pro', updated_at = NOW()
            WHERE plan_type = 'expert'
        """)
    )
    
    # Revertir PREMIUM a ELITE
    connection.execute(
        sa.text("""
            UPDATE subscription_plans 
            SET name = 'Elite', plan_type = 'elite', updated_at = NOW()
            WHERE plan_type = 'premium'
        """)
    )
    
    # Revertir precio del plan INSTITUTIONAL
    connection.execute(
        sa.text("""
            UPDATE subscription_plans 
            SET price = 999.0, updated_at = NOW()
            WHERE plan_type = 'institutional'
        """)
    )
    
    # Revertir suscripciones de usuarios
    connection.execute(
        sa.text("""
            UPDATE user_subscriptions 
            SET plan_type = 'freemium', updated_at = NOW()
            WHERE plan_type = 'starter'
        """)
    )
    
    connection.execute(
        sa.text("""
            UPDATE user_subscriptions 
            SET plan_type = 'basic', updated_at = NOW()
            WHERE plan_type = 'trader'
        """)
    )
    
    connection.execute(
        sa.text("""
            UPDATE user_subscriptions 
            SET plan_type = 'pro', updated_at = NOW()
            WHERE plan_type = 'expert'
        """)
    )
    
    connection.execute(
        sa.text("""
            UPDATE user_subscriptions 
            SET plan_type = 'elite', updated_at = NOW()
            WHERE plan_type = 'premium'
        """)
    ) 