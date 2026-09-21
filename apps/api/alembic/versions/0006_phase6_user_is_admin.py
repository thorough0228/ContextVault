"""phase6 user.is_admin

Revision ID: 0006_phase6_user_is_admin
Revises: 0005_phase5_conversations
Create Date: 2026-09-15

Adds a ``is_admin`` boolean column to the ``users`` table. Default
``False``. Gates the new ``/api/v1/metrics`` route.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0006_phase6_user_is_admin"
down_revision: Union[str, None] = "0005_phase5_conversations"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column(
            "is_admin",
            sa.Boolean(),
            nullable=False,
            # sa.text("0") renders `DEFAULT 0` — valid on SQLite, rejected
            # by Postgres for a boolean column.
            server_default=sa.false(),
        ),
    )


def downgrade() -> None:
    op.drop_column("users", "is_admin")
