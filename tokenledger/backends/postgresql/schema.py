"""
PostgreSQL Schema Definitions

DDL statements for creating TokenLedger tables and indexes in PostgreSQL.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tokenledger.config import TokenLedgerConfig


def get_create_table_sql(config: TokenLedgerConfig) -> str:
    """
    Generate CREATE TABLE statement for token_ledger_events.

    Args:
        config: TokenLedger configuration with table/schema names

    Returns:
        SQL CREATE TABLE statement
    """
    return f"""
    CREATE TABLE IF NOT EXISTS {config.full_table_name} (
        event_id UUID PRIMARY KEY,
        trace_id UUID,
        span_id UUID,
        parent_span_id UUID,

        timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        duration_ms DOUBLE PRECISION,

        provider VARCHAR(50) NOT NULL,
        model VARCHAR(100) NOT NULL,

        input_tokens INTEGER NOT NULL DEFAULT 0,
        output_tokens INTEGER NOT NULL DEFAULT 0,
        total_tokens INTEGER NOT NULL DEFAULT 0,
        cached_tokens INTEGER DEFAULT 0,

        cost_usd DECIMAL(12, 8),

        endpoint VARCHAR(255),
        request_type VARCHAR(50) DEFAULT 'chat',

        user_id VARCHAR(255),
        session_id VARCHAR(255),
        organization_id VARCHAR(255),

        app_name VARCHAR(100),
        environment VARCHAR(50),

        status VARCHAR(20) DEFAULT 'success',
        error_type VARCHAR(100),
        error_message TEXT,

        metadata JSONB,

        request_preview TEXT,
        response_preview TEXT
    )
    """


def get_create_indexes_sql(config: TokenLedgerConfig) -> str:
    """
    Generate CREATE INDEX statements for common query patterns.

    Args:
        config: TokenLedger configuration with table/schema names

    Returns:
        SQL CREATE INDEX statements
    """
    return f"""
    -- Indexes for common queries
    CREATE INDEX IF NOT EXISTS idx_{config.table_name}_timestamp
        ON {config.full_table_name} (timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_{config.table_name}_user
        ON {config.full_table_name} (user_id, timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_{config.table_name}_model
        ON {config.full_table_name} (model, timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_{config.table_name}_app
        ON {config.full_table_name} (app_name, environment, timestamp DESC);
    """


def get_full_schema_sql(config: TokenLedgerConfig) -> str:
    """
    Generate complete schema creation SQL (table + indexes).

    Args:
        config: TokenLedger configuration with table/schema names

    Returns:
        Complete SQL for schema creation
    """
    return get_create_table_sql(config) + "\n" + get_create_indexes_sql(config)


def get_insert_sql_psycopg2(config: TokenLedgerConfig, columns: tuple[str, ...]) -> str:
    """
    Generate INSERT statement for psycopg2 (uses execute_values with %s).

    Args:
        config: TokenLedger configuration
        columns: Tuple of column names

    Returns:
        SQL INSERT statement for use with execute_values
    """
    return f"""
    INSERT INTO {config.full_table_name}
    ({", ".join(columns)})
    VALUES %s
    ON CONFLICT (event_id) DO NOTHING
    """


def get_insert_sql_psycopg3(config: TokenLedgerConfig, columns: tuple[str, ...]) -> str:
    """
    Generate INSERT statement for psycopg3 (uses %s placeholders).

    Args:
        config: TokenLedger configuration
        columns: Tuple of column names

    Returns:
        SQL INSERT statement for use with executemany
    """
    placeholders = ", ".join(["%s"] * len(columns))
    return f"""
    INSERT INTO {config.full_table_name}
    ({", ".join(columns)})
    VALUES ({placeholders})
    ON CONFLICT (event_id) DO NOTHING
    """


def get_insert_sql_asyncpg(config: TokenLedgerConfig, columns: tuple[str, ...]) -> str:
    """
    Generate INSERT statement for asyncpg (uses $1, $2, ... placeholders).

    Args:
        config: TokenLedger configuration
        columns: Tuple of column names

    Returns:
        SQL INSERT statement for use with executemany
    """
    placeholders = ", ".join(f"${i + 1}" for i in range(len(columns)))
    return f"""
    INSERT INTO {config.full_table_name}
    ({", ".join(columns)})
    VALUES ({placeholders})
    ON CONFLICT (event_id) DO NOTHING
    """


def get_health_check_sql() -> str:
    """
    Get SQL for health check query.

    Returns:
        Simple SELECT 1 query
    """
    return "SELECT 1"
