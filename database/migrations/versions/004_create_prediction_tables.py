"""
Create prediction tables for Day Trading system
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.types import DECIMAL
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = '004_create_prediction_tables'
down_revision = '003_update_plan_names'
branch_labels = None
depends_on = None

def upgrade():
    # Create user_predictions table
    op.create_table('user_predictions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('pair', sa.String(length=10), nullable=False),
        sa.Column('direction', sa.Enum('up', 'down', 'sideways', name='prediction_direction'), nullable=False),
        sa.Column('current_price', DECIMAL(precision=10, scale=5), nullable=False),
        sa.Column('target_price', DECIMAL(precision=10, scale=5), nullable=False),
        sa.Column('confidence', DECIMAL(precision=5, scale=2), nullable=False),
        sa.Column('timeframe', sa.String(length=10), nullable=True, server_default='15M'),
        sa.Column('reasoning', sa.Text(), nullable=True),
        sa.Column('brain_type', sa.String(length=20), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(), nullable=True, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('expires_at', sa.TIMESTAMP(), nullable=True),
        sa.Column('is_completed', sa.Boolean(), nullable=True, server_default='0'),
        sa.Column('actual_price_at_expiry', DECIMAL(precision=10, scale=5), nullable=True),
        sa.Column('prediction_success', sa.Boolean(), nullable=True),
        sa.Column('success_percentage', DECIMAL(precision=5, scale=2), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE')
    )

    # Create user_prediction_limits table
    op.create_table('user_prediction_limits',
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('plan_type', sa.String(length=20), nullable=False),
        sa.Column('max_predictions_per_day', sa.Integer(), nullable=False),
        sa.Column('predictions_used_today', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('last_reset_date', sa.Date(), nullable=True, server_default=sa.text('CURRENT_DATE')),
        sa.PrimaryKeyConstraint('user_id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE')
    )

    # Create indexes for better performance
    op.create_index('idx_user_predictions_user_id', 'user_predictions', ['user_id'])
    op.create_index('idx_user_predictions_created_at', 'user_predictions', ['created_at'])
    op.create_index('idx_user_predictions_is_completed', 'user_predictions', ['is_completed'])

def downgrade():
    # Drop indexes
    op.drop_index('idx_user_predictions_is_completed', table_name='user_predictions')
    op.drop_index('idx_user_predictions_created_at', table_name='user_predictions')
    op.drop_index('idx_user_predictions_user_id', table_name='user_predictions')
    
    # Drop tables
    op.drop_table('user_prediction_limits')
    op.drop_table('user_predictions')
    
    # Drop enum
    op.execute("DROP TYPE prediction_direction") 