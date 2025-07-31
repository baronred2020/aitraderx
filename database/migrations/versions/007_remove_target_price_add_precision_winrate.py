"""Remove target_price and add precision, win_rate

Revision ID: 007
Revises: a6c2344d1a3a
Create Date: 2025-01-27 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = '007'
down_revision = 'a6c2344d1a3a'
branch_labels = None
depends_on = None


def upgrade():
    # Agregar nuevas columnas
    op.add_column('user_predictions', sa.Column('precision', sa.DECIMAL(5, 2), nullable=False, server_default='0.00'))
    op.add_column('user_predictions', sa.Column('win_rate', sa.DECIMAL(5, 2), nullable=False, server_default='0.00'))
    
    # Eliminar columna target_price
    op.drop_column('user_predictions', 'target_price')


def downgrade():
    # Agregar columna target_price de vuelta
    op.add_column('user_predictions', sa.Column('target_price', sa.DECIMAL(10, 5), nullable=True))
    
    # Eliminar nuevas columnas
    op.drop_column('user_predictions', 'win_rate')
    op.drop_column('user_predictions', 'precision') 