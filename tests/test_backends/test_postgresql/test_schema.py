"""Tests for PostgreSQL schema module."""

from __future__ import annotations

import pytest

from tokenledger.backends.postgresql.schema import (
    get_create_indexes_sql,
    get_create_table_sql,
    get_full_schema_sql,
    get_health_check_sql,
    get_insert_sql_asyncpg,
    get_insert_sql_psycopg2,
    get_insert_sql_psycopg3,
)
from tokenledger.config import TokenLedgerConfig


@pytest.fixture
def config() -> TokenLedgerConfig:
    """Create a test configuration."""
    return TokenLedgerConfig(
        database_url="postgresql://test:test@localhost/test",
        table_name="test_events",
        schema_name="public",
    )


@pytest.fixture
def custom_schema_config() -> TokenLedgerConfig:
    """Create a test configuration with custom schema."""
    return TokenLedgerConfig(
        database_url="postgresql://test:test@localhost/test",
        table_name="events",
        schema_name="analytics",
    )


class TestGetCreateTableSql:
    """Tests for get_create_table_sql function."""

    def test_contains_create_table(self, config: TokenLedgerConfig) -> None:
        """Test that SQL contains CREATE TABLE."""
        sql = get_create_table_sql(config)
        assert "CREATE TABLE IF NOT EXISTS" in sql

    def test_uses_full_table_name(self, config: TokenLedgerConfig) -> None:
        """Test that SQL uses the full table name from config."""
        sql = get_create_table_sql(config)
        assert config.full_table_name in sql

    def test_contains_required_columns(self, config: TokenLedgerConfig) -> None:
        """Test that SQL contains all required columns."""
        sql = get_create_table_sql(config)

        required_columns = [
            "event_id UUID PRIMARY KEY",
            "trace_id UUID",
            "timestamp TIMESTAMPTZ",
            "provider VARCHAR",
            "model VARCHAR",
            "input_tokens INTEGER",
            "output_tokens INTEGER",
            "total_tokens INTEGER",
            "cached_tokens INTEGER",
            "cost_usd DECIMAL",
            "user_id VARCHAR",
            "app_name VARCHAR",
            "environment VARCHAR",
            "status VARCHAR",
            "metadata JSONB",
        ]

        for col in required_columns:
            assert col in sql, f"Missing column: {col}"

    def test_custom_schema(self, custom_schema_config: TokenLedgerConfig) -> None:
        """Test that custom schema is used."""
        sql = get_create_table_sql(custom_schema_config)
        assert "analytics.events" in sql


class TestGetCreateIndexesSql:
    """Tests for get_create_indexes_sql function."""

    def test_contains_create_index(self, config: TokenLedgerConfig) -> None:
        """Test that SQL contains CREATE INDEX."""
        sql = get_create_indexes_sql(config)
        assert "CREATE INDEX IF NOT EXISTS" in sql

    def test_contains_timestamp_index(self, config: TokenLedgerConfig) -> None:
        """Test that timestamp index is created."""
        sql = get_create_indexes_sql(config)
        assert f"idx_{config.table_name}_timestamp" in sql
        assert "timestamp DESC" in sql

    def test_contains_user_index(self, config: TokenLedgerConfig) -> None:
        """Test that user index is created."""
        sql = get_create_indexes_sql(config)
        assert f"idx_{config.table_name}_user" in sql
        assert "user_id" in sql

    def test_contains_model_index(self, config: TokenLedgerConfig) -> None:
        """Test that model index is created."""
        sql = get_create_indexes_sql(config)
        assert f"idx_{config.table_name}_model" in sql

    def test_contains_app_index(self, config: TokenLedgerConfig) -> None:
        """Test that app index is created."""
        sql = get_create_indexes_sql(config)
        assert f"idx_{config.table_name}_app" in sql
        assert "app_name" in sql
        assert "environment" in sql


class TestGetFullSchemaSql:
    """Tests for get_full_schema_sql function."""

    def test_contains_table_and_indexes(self, config: TokenLedgerConfig) -> None:
        """Test that full schema contains both table and indexes."""
        sql = get_full_schema_sql(config)
        assert "CREATE TABLE IF NOT EXISTS" in sql
        assert "CREATE INDEX IF NOT EXISTS" in sql


class TestGetInsertSqlPsycopg2:
    """Tests for get_insert_sql_psycopg2 function."""

    def test_uses_percent_s_placeholder(self, config: TokenLedgerConfig) -> None:
        """Test that psycopg2 style uses %s placeholder."""
        columns = ("event_id", "provider", "model")
        sql = get_insert_sql_psycopg2(config, columns)
        assert "VALUES %s" in sql

    def test_contains_on_conflict(self, config: TokenLedgerConfig) -> None:
        """Test that upsert syntax is included."""
        columns = ("event_id", "provider")
        sql = get_insert_sql_psycopg2(config, columns)
        assert "ON CONFLICT (event_id) DO NOTHING" in sql

    def test_includes_columns(self, config: TokenLedgerConfig) -> None:
        """Test that columns are included in SQL."""
        columns = ("event_id", "provider", "model")
        sql = get_insert_sql_psycopg2(config, columns)
        for col in columns:
            assert col in sql


class TestGetInsertSqlPsycopg3:
    """Tests for get_insert_sql_psycopg3 function."""

    def test_uses_percent_s_placeholders(self, config: TokenLedgerConfig) -> None:
        """Test that psycopg3 style uses multiple %s placeholders."""
        columns = ("event_id", "provider", "model")
        sql = get_insert_sql_psycopg3(config, columns)
        # Should have 3 %s placeholders for 3 columns
        assert sql.count("%s") == 3

    def test_contains_on_conflict(self, config: TokenLedgerConfig) -> None:
        """Test that upsert syntax is included."""
        columns = ("event_id", "provider")
        sql = get_insert_sql_psycopg3(config, columns)
        assert "ON CONFLICT (event_id) DO NOTHING" in sql


class TestGetInsertSqlAsyncpg:
    """Tests for get_insert_sql_asyncpg function."""

    def test_uses_dollar_placeholders(self, config: TokenLedgerConfig) -> None:
        """Test that asyncpg style uses $1, $2, ... placeholders."""
        columns = ("event_id", "provider", "model")
        sql = get_insert_sql_asyncpg(config, columns)
        assert "$1" in sql
        assert "$2" in sql
        assert "$3" in sql

    def test_placeholder_count_matches_columns(self, config: TokenLedgerConfig) -> None:
        """Test that placeholder count matches column count."""
        columns = ("a", "b", "c", "d", "e")
        sql = get_insert_sql_asyncpg(config, columns)
        # Should have $1 through $5
        for i in range(1, 6):
            assert f"${i}" in sql

    def test_contains_on_conflict(self, config: TokenLedgerConfig) -> None:
        """Test that upsert syntax is included."""
        columns = ("event_id", "provider")
        sql = get_insert_sql_asyncpg(config, columns)
        assert "ON CONFLICT (event_id) DO NOTHING" in sql


class TestGetHealthCheckSql:
    """Tests for get_health_check_sql function."""

    def test_returns_select_1(self) -> None:
        """Test that health check returns SELECT 1."""
        sql = get_health_check_sql()
        assert sql == "SELECT 1"
