"""streamline memory_unit fields: drop clarification_queue, add
processing lease, explicit metadata, conflict markers

Revision ID: 3192dd58cb3b
Revises: cb394374d39a
Create Date: 2026-10-08 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = '3192dd58cb3b'
down_revision: Union[str, None] = 'cb394374d39a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Clarification is now strictly sequential (one question, re-evaluate
    # after every answer) -- the pre-generated queue is dead weight.
    op.drop_column('memory_units', 'clarification_queue')

    # Short-lived processing lease (see docs/DECISIONS.md "processing
    # lease") -- replaces holding a database row lock for an entire LLM
    # call's duration.
    op.add_column('memory_units', sa.Column('processing_token', sa.Text(), nullable=True))
    op.add_column('memory_units', sa.Column('processing_started_at', sa.DateTime(timezone=True), nullable=True))

    # User-provided structured fields, authoritative over LLM inference
    # for the same field (see docs/DECISIONS.md "Explicit metadata authority").
    op.add_column(
        'memory_units',
        sa.Column('explicit_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
    )

    # Lightweight, advisory conflict markers (see docs/DECISIONS.md
    # "Contradiction detection") -- never auto-deletes/supersedes anything.
    op.add_column(
        'memory_units',
        sa.Column('potential_conflicts', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='[]'),
    )


def downgrade() -> None:
    op.drop_column('memory_units', 'potential_conflicts')
    op.drop_column('memory_units', 'explicit_metadata')
    op.drop_column('memory_units', 'processing_started_at')
    op.drop_column('memory_units', 'processing_token')
    op.add_column(
        'memory_units',
        sa.Column('clarification_queue', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='[]'),
    )
