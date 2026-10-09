"""add membase structurer fields: structure_status, structure_token,
structure_started_at, structured_at, related_unit_ids

Revision ID: 8f1a3c6e2b4d
Revises: 3192dd58cb3b
Create Date: 2026-10-09 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = '8f1a3c6e2b4d'
down_revision: Union[str, None] = '3192dd58cb3b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Second, independent background pass (see docs/DECISIONS.md "The
    # 'membase structurer'") -- entirely separate from `status`, which
    # remains the only searchability gate. A unit at
    # structure_status="unstructured" is exactly as searchable as one
    # at "structured".
    op.add_column(
        'memory_units',
        sa.Column('structure_status', sa.Text(), nullable=False, server_default='unstructured'),
    )
    op.add_column('memory_units', sa.Column('structure_token', sa.Text(), nullable=True))
    op.add_column('memory_units', sa.Column('structure_started_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('memory_units', sa.Column('structured_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column(
        'memory_units',
        sa.Column('related_unit_ids', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='[]'),
    )


def downgrade() -> None:
    op.drop_column('memory_units', 'related_unit_ids')
    op.drop_column('memory_units', 'structured_at')
    op.drop_column('memory_units', 'structure_started_at')
    op.drop_column('memory_units', 'structure_token')
    op.drop_column('memory_units', 'structure_status')
