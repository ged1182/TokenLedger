"""Initial TokenLedger schema

Creates the token_ledger_events table with all indexes and views.

Revision ID: 001
Revises: None
Create Date: 2024-01-01 00:00:00.000000
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create initial TokenLedger schema."""
    # Create the events table
    op.create_table(
        "token_ledger_events",
        # Identifiers
        sa.Column("event_id", postgresql.UUID(), primary_key=True),
        sa.Column("trace_id", postgresql.UUID(), nullable=True),
        sa.Column("span_id", postgresql.UUID(), nullable=True),
        sa.Column("parent_span_id", postgresql.UUID(), nullable=True),
        # Timing
        sa.Column(
            "timestamp",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("duration_ms", sa.Float(), nullable=True),
        # Provider & Model
        sa.Column("provider", sa.String(50), nullable=False),
        sa.Column("model", sa.String(100), nullable=False),
        # Token counts
        sa.Column("input_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("output_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("cached_tokens", sa.Integer(), nullable=True, server_default="0"),
        # Cost
        sa.Column("cost_usd", sa.Numeric(12, 8), nullable=True),
        # Request details
        sa.Column("endpoint", sa.String(255), nullable=True),
        sa.Column("request_type", sa.String(50), nullable=True, server_default="'chat'"),
        # User & context
        sa.Column("user_id", sa.String(255), nullable=True),
        sa.Column("session_id", sa.String(255), nullable=True),
        sa.Column("organization_id", sa.String(255), nullable=True),
        # Application context
        sa.Column("app_name", sa.String(100), nullable=True),
        sa.Column("environment", sa.String(50), nullable=True),
        # Status
        sa.Column("status", sa.String(20), nullable=True, server_default="'success'"),
        sa.Column("error_type", sa.String(100), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        # Custom metadata (JSONB for flexible querying)
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        # Request/Response previews (for debugging)
        sa.Column("request_preview", sa.Text(), nullable=True),
        sa.Column("response_preview", sa.Text(), nullable=True),
    )

    # Performance indexes
    op.create_index(
        "idx_token_ledger_timestamp",
        "token_ledger_events",
        [sa.text("timestamp DESC")],
    )

    op.create_index(
        "idx_token_ledger_user",
        "token_ledger_events",
        ["user_id", sa.text("timestamp DESC")],
    )

    op.create_index(
        "idx_token_ledger_model",
        "token_ledger_events",
        ["model", sa.text("timestamp DESC")],
    )

    op.create_index(
        "idx_token_ledger_app",
        "token_ledger_events",
        ["app_name", "environment", sa.text("timestamp DESC")],
    )

    op.create_index(
        "idx_token_ledger_trace",
        "token_ledger_events",
        ["trace_id"],
    )

    op.create_index(
        "idx_token_ledger_status",
        "token_ledger_events",
        ["status", sa.text("timestamp DESC")],
    )

    # Composite index for common dashboard queries
    op.create_index(
        "idx_token_ledger_dashboard",
        "token_ledger_events",
        [sa.text("timestamp DESC"), "model", "user_id"],
        postgresql_include=["cost_usd", "total_tokens"],
    )

    # GIN index for metadata queries
    op.create_index(
        "idx_token_ledger_metadata",
        "token_ledger_events",
        ["metadata"],
        postgresql_using="gin",
    )

    # Create helper views
    op.execute("""
        CREATE OR REPLACE VIEW token_ledger_daily_costs AS
        SELECT
            DATE(timestamp) as date,
            provider,
            model,
            COUNT(*) as request_count,
            SUM(input_tokens) as total_input_tokens,
            SUM(output_tokens) as total_output_tokens,
            SUM(total_tokens) as total_tokens,
            SUM(cost_usd) as total_cost,
            AVG(duration_ms) as avg_latency_ms
        FROM token_ledger_events
        GROUP BY DATE(timestamp), provider, model
    """)

    op.execute("""
        CREATE OR REPLACE VIEW token_ledger_user_costs AS
        SELECT
            COALESCE(user_id, 'anonymous') as user_id,
            COUNT(*) as request_count,
            SUM(total_tokens) as total_tokens,
            SUM(cost_usd) as total_cost,
            MIN(timestamp) as first_request,
            MAX(timestamp) as last_request
        FROM token_ledger_events
        GROUP BY user_id
    """)


def downgrade() -> None:
    """Remove TokenLedger schema."""
    # Drop views first
    op.execute("DROP VIEW IF EXISTS token_ledger_user_costs")
    op.execute("DROP VIEW IF EXISTS token_ledger_daily_costs")

    # Drop indexes (they'll be dropped with the table, but explicit is clearer)
    op.drop_index("idx_token_ledger_metadata", table_name="token_ledger_events")
    op.drop_index("idx_token_ledger_dashboard", table_name="token_ledger_events")
    op.drop_index("idx_token_ledger_status", table_name="token_ledger_events")
    op.drop_index("idx_token_ledger_trace", table_name="token_ledger_events")
    op.drop_index("idx_token_ledger_app", table_name="token_ledger_events")
    op.drop_index("idx_token_ledger_model", table_name="token_ledger_events")
    op.drop_index("idx_token_ledger_user", table_name="token_ledger_events")
    op.drop_index("idx_token_ledger_timestamp", table_name="token_ledger_events")

    # Drop the table
    op.drop_table("token_ledger_events")
