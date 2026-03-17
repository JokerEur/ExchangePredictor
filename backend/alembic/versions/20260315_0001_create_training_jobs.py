"""create training jobs table

Revision ID: 20260315_0001
Revises:
Create Date: 2026-03-15 00:00:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "20260315_0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "training_jobs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("job_id", sa.String(length=255), nullable=False),
        sa.Column("model_id", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_training_jobs_job_id", "training_jobs", ["job_id"], unique=True)
    op.create_index("ix_training_jobs_model_id", "training_jobs", ["model_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_training_jobs_model_id", table_name="training_jobs")
    op.drop_index("ix_training_jobs_job_id", table_name="training_jobs")
    op.drop_table("training_jobs")
