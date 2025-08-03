"""Create RL Training Sessions table

Revision ID: 015
Revises: 014
Create Date: 2025-01-27 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = '015'
down_revision = '014'
branch_labels = None
depends_on = None


def upgrade():
    # Create rl_training_sessions table
    op.create_table('rl_training_sessions',
        sa.Column('session_id', sa.String(36), nullable=False),
        sa.Column('user_id', sa.String(36), nullable=False),
        sa.Column('episodes', sa.Integer(), nullable=False),
        sa.Column('algorithm', sa.String(20), nullable=True),
        sa.Column('trading_pair', sa.String(10), nullable=True),
        sa.Column('timeframe', sa.String(10), nullable=True),
        sa.Column('status', sa.String(20), nullable=True),
        sa.Column('progress', sa.Float(), nullable=True),
        sa.Column('current_episode', sa.Integer(), nullable=True),
        sa.Column('total_episodes', sa.Integer(), nullable=False),
        sa.Column('final_reward', sa.Float(), nullable=True),
        sa.Column('win_rate', sa.Float(), nullable=True),
        sa.Column('sharpe_ratio', sa.Float(), nullable=True),
        sa.Column('max_drawdown', sa.Float(), nullable=True),
        sa.Column('model_path', sa.String(500), nullable=True),
        sa.Column('training_log_path', sa.String(500), nullable=True),
        sa.Column('estimated_duration_minutes', sa.Integer(), nullable=True),
        sa.Column('actual_duration_minutes', sa.Float(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.user_id'], ),
        sa.PrimaryKeyConstraint('session_id')
    )
    
    # Create indexes
    op.create_index('ix_rl_training_sessions_user_id', 'rl_training_sessions', ['user_id'])
    op.create_index('ix_rl_training_sessions_status', 'rl_training_sessions', ['status'])
    op.create_index('ix_rl_training_sessions_started_at', 'rl_training_sessions', ['started_at'])


def downgrade():
    # Drop indexes
    op.drop_index('ix_rl_training_sessions_started_at', table_name='rl_training_sessions')
    op.drop_index('ix_rl_training_sessions_status', table_name='rl_training_sessions')
    op.drop_index('ix_rl_training_sessions_user_id', table_name='rl_training_sessions')
    
    # Drop table
    op.drop_table('rl_training_sessions') 