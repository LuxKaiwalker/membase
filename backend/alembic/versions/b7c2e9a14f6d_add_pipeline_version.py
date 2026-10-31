"""add pipeline_version to memory_units

Revision ID: b7c2e9a14f6d
Revises: d4a1f9c7b3e2
Create Date: 2026-10-31 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'b7c2e9a14f6d'
down_revision: Union[str, None] = 'd4a1f9c7b3e2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Marks which pipeline code version last derived a unit's
    # attachment-text-dependent state (clarification question,
    # `semantic`, retrieval/searchable text) -- see
    # app/ingestion/pipeline.py's CURRENT_PIPELINE_VERSION and
    # scripts/reprocess_memories.py. server_default='0' so every
    # pre-existing row is treated as "pre-versioning / possibly stale"
    # and becomes eligible for reprocessing the first time
    # CURRENT_PIPELINE_VERSION is bumped above 0 (see docs/DECISIONS.md
    # "Reprocessing pipeline").
    op.add_column(
        'memory_units',
        sa.Column('pipeline_version', sa.Integer(), nullable=False, server_default='0'),
    )


def downgrade() -> None:
    op.drop_column('memory_units', 'pipeline_version')
