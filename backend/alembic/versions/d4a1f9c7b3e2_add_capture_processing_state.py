"""add capture processing state: processing_status, processing_token,
processing_started_at, allow_clarification, explicit_metadata

Revision ID: d4a1f9c7b3e2
Revises: 8f1a3c6e2b4d
Create Date: 2026-10-16 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = 'd4a1f9c7b3e2'
down_revision: Union[str, None] = '8f1a3c6e2b4d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Capture-level processing lease (see app/ingestion/pipeline.py
    # "process_capture_fully") -- mirrors memory_units' own
    # processing_token/processing_started_at lease, one level up.
    # server_default='done' is deliberate: every PRE-EXISTING capture
    # already has fully-formed units from the old synchronous pipeline,
    # so it must backfill to "done", not "pending" (which would make
    # POST /api/captures/{id}/retry think it needs re-segmenting). New
    # captures get "pending" from the ORM-level default in app/models.py
    # instead, not from this server_default.
    op.add_column(
        'captures',
        sa.Column('processing_status', sa.Text(), nullable=False, server_default='done'),
    )
    op.add_column('captures', sa.Column('processing_token', sa.Text(), nullable=True))
    op.add_column('captures', sa.Column('processing_started_at', sa.DateTime(timezone=True), nullable=True))
    # Staged at Phase 0 (create_capture()) for process_capture_fully()
    # to read once it runs -- backfilling to true/{} for existing,
    # already-fully-processed captures is harmless (never read again).
    op.add_column(
        'captures',
        sa.Column('allow_clarification', sa.Boolean(), nullable=False, server_default='true'),
    )
    op.add_column(
        'captures',
        sa.Column('explicit_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
    )


def downgrade() -> None:
    op.drop_column('captures', 'explicit_metadata')
    op.drop_column('captures', 'allow_clarification')
    op.drop_column('captures', 'processing_started_at')
    op.drop_column('captures', 'processing_token')
    op.drop_column('captures', 'processing_status')
