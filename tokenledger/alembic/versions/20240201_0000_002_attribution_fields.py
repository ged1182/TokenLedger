"""Add attribution fields for cost tracking

Adds first-class columns for cost attribution: feature, page, component,
team, project, cost_center, and metadata_extra.

Revision ID: 002
Revises: 001
Create Date: 2024-02-01 00:00:00.000000
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add attribution columns and indexes."""
    # Add attribution columns
    op.add_column(
        "token_ledger_events",
        sa.Column("feature", sa.String(100), nullable=True),
    )
    op.add_column(
        "token_ledger_events",
        sa.Column("page", sa.String(255), nullable=True),
    )
    op.add_column(
        "token_ledger_events",
        sa.Column("component", sa.String(100), nullable=True),
    )
    op.add_column(
        "token_ledger_events",
        sa.Column("team", sa.String(100), nullable=True),
    )
    op.add_column(
        "token_ledger_events",
        sa.Column("project", sa.String(100), nullable=True),
    )
    op.add_column(
        "token_ledger_events",
        sa.Column("cost_center", sa.String(100), nullable=True),
    )
    op.add_column(
        "token_ledger_events",
        sa.Column("metadata_extra", postgresql.JSONB(), nullable=True),
    )

    # Create indexes for attribution queries
    op.create_index(
        "idx_token_ledger_feature",
        "token_ledger_events",
        ["feature", sa.text("timestamp DESC")],
    )

    op.create_index(
        "idx_token_ledger_team",
        "token_ledger_events",
        ["team", sa.text("timestamp DESC")],
    )

    op.create_index(
        "idx_token_ledger_project",
        "token_ledger_events",
        ["project", sa.text("timestamp DESC")],
    )

    op.create_index(
        "idx_token_ledger_cost_center",
        "token_ledger_events",
        ["cost_center", sa.text("timestamp DESC")],
    )

    # Create helper views for attribution analytics
    op.execute("""
        CREATE OR REPLACE VIEW token_ledger_team_costs AS
        SELECT
            COALESCE(team, 'unassigned') as team,
            DATE(timestamp) as date,
            COUNT(*) as request_count,
            SUM(total_tokens) as total_tokens,
            SUM(cost_usd) as total_cost,
            AVG(duration_ms) as avg_latency_ms
        FROM token_ledger_events
        GROUP BY team, DATE(timestamp)
    """)

    op.execute("""
        CREATE OR REPLACE VIEW token_ledger_feature_costs AS
        SELECT
            COALESCE(feature, 'unassigned') as feature,
            DATE(timestamp) as date,
            COUNT(*) as request_count,
            SUM(total_tokens) as total_tokens,
            SUM(cost_usd) as total_cost,
            AVG(duration_ms) as avg_latency_ms
        FROM token_ledger_events
        GROUP BY feature, DATE(timestamp)
    """)

    op.execute("""
        CREATE OR REPLACE VIEW token_ledger_cost_center_costs AS
        SELECT
            COALESCE(cost_center, 'unassigned') as cost_center,
            DATE(timestamp) as date,
            COUNT(*) as request_count,
            SUM(total_tokens) as total_tokens,
            SUM(cost_usd) as total_cost
        FROM token_ledger_events
        GROUP BY cost_center, DATE(timestamp)
    """)


def downgrade() -> None:
    """Remove attribution columns and indexes."""
    # Drop views first
    op.execute("DROP VIEW IF EXISTS token_ledger_cost_center_costs")
    op.execute("DROP VIEW IF EXISTS token_ledger_feature_costs")
    op.execute("DROP VIEW IF EXISTS token_ledger_team_costs")

    # Drop indexes
    op.drop_index("idx_token_ledger_cost_center", table_name="token_ledger_events")
    op.drop_index("idx_token_ledger_project", table_name="token_ledger_events")
    op.drop_index("idx_token_ledger_team", table_name="token_ledger_events")
    op.drop_index("idx_token_ledger_feature", table_name="token_ledger_events")

    # Drop columns
    op.drop_column("token_ledger_events", "metadata_extra")
    op.drop_column("token_ledger_events", "cost_center")
    op.drop_column("token_ledger_events", "project")
    op.drop_column("token_ledger_events", "team")
    op.drop_column("token_ledger_events", "component")
    op.drop_column("token_ledger_events", "page")
    op.drop_column("token_ledger_events", "feature")
