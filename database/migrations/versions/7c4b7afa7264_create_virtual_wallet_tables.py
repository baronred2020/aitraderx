"""
Revision ID: 7c4b7afa7264
Revises: 002_insert_default_plans
Create Date: 2025-07-11 20:27:40.707233

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '7c4b7afa7264'
down_revision = '002_insert_default_plans'
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        'virtual_wallets',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.user_id'), nullable=False, index=True, unique=True),
        sa.Column('balance', sa.Numeric(12, 2), nullable=False, default=10000.00),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, nullable=False, server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_table(
        'wallet_transactions',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('wallet_id', sa.Integer, sa.ForeignKey('virtual_wallets.id'), nullable=False, index=True),
        sa.Column('type', sa.String(20), nullable=False),
        sa.Column('amount', sa.Numeric(12, 2), nullable=False),
        sa.Column('description', sa.String(255)),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
    )

def downgrade():
    op.drop_table('wallet_transactions')
    op.drop_table('virtual_wallets') 