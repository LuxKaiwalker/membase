"""add recall session evidence trail

Revision ID: 0c4003935d6f
Revises: 134238d01cd1
Create Date: 2026-11-07 18:36:12.118430

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = '0c4003935d6f'
down_revision: Union[str, None] = '134238d01cd1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # NOTE: autogenerate additionally proposed dropping
    # ix_attachment_chunks_embedding/fts, ix_memory_units_embedding/fts,
    # ix_memory_entities_kind_normalized and
    # ix_unit_associations_source_weight/target_weight. That is the recurring
    # false positive documented in 1653419fcdd7 and abdf3833faaa, now also
    # covering the three indexes added by 134238d01cd1: all of them are
    # hand-added via op.create_index rather than declared as SQLAlchemy Index
    # objects on the models, so every autogenerate run reports them as
    # extraneous. Deliberately NOT included -- dropping them would remove the
    # HNSW/GIN indexes behind every vector and full-text search, plus the
    # composite indexes the graph's traversal and entity-resolution paths use.

    # All four columns are NOT NULL with a server_default so existing rows get
    # a valid empty value without a backfill pass. The defaults are kept on
    # the columns (not dropped after the fact) because every INSERT goes
    # through SQLAlchemy, which supplies them anyway -- the server default
    # exists purely for rows that predate this migration.
    op.add_column(
        'recall_sessions',
        sa.Column(
            'strategy_trail',
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )
    op.add_column(
        'recall_sessions',
        sa.Column(
            'final_evidence',
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )
    # Nullable rather than defaulted: a session that has not produced an answer
    # yet has no confidence category, and "" would be indistinguishable from a
    # real-but-empty value. Same reasoning for abstention_reason.
    op.add_column('recall_sessions', sa.Column('final_confidence', sa.Text(), nullable=True))
    op.add_column('recall_sessions', sa.Column('abstention_reason', sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column('recall_sessions', 'abstention_reason')
    op.drop_column('recall_sessions', 'final_confidence')
    op.drop_column('recall_sessions', 'final_evidence')
    op.drop_column('recall_sessions', 'strategy_trail')
