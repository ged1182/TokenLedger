"""Pytest configuration and fixtures for TokenLedger tests."""

from __future__ import annotations

import os
import uuid
from collections.abc import Generator
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from tokenledger.config import TokenLedgerConfig

# Integration test database URL (uses port 5433 from docker-compose.test.yml)
INTEGRATION_DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://tokenledger:tokenledger@localhost:5433/tokenledger_test",
)


def _is_postgres_available() -> bool:
    """Check if PostgreSQL is available for integration tests."""
    try:
        import psycopg2

        conn = psycopg2.connect(INTEGRATION_DATABASE_URL, connect_timeout=2)
        conn.close()
        return True
    except Exception:
        return False


# Skip integration tests if database is not available
requires_postgres = pytest.mark.skipif(
    not _is_postgres_available(),
    reason="PostgreSQL not available (run: docker compose -f docker-compose.test.yml up -d)",
)


@pytest.fixture(scope="session")
def integration_database_url() -> str:
    """Get the integration test database URL."""
    return INTEGRATION_DATABASE_URL


@pytest.fixture(scope="session")
def integration_config() -> TokenLedgerConfig:
    """Create configuration for integration tests."""
    return TokenLedgerConfig(
        database_url=INTEGRATION_DATABASE_URL,
        app_name="integration-test",
        environment="test",
        async_mode=False,
        debug=True,
        batch_size=10,
        flush_interval_seconds=0.1,
        sample_rate=1.0,
    )


@pytest.fixture(scope="session")
def async_integration_config() -> TokenLedgerConfig:
    """Create async configuration for integration tests."""
    return TokenLedgerConfig(
        database_url=INTEGRATION_DATABASE_URL,
        app_name="integration-test-async",
        environment="test",
        async_mode=True,
        debug=True,
        batch_size=10,
        flush_interval_seconds=0.1,
        sample_rate=1.0,
    )


@pytest.fixture
def clean_events_table(integration_database_url: str) -> Generator[None, None, None]:
    """Clean the events table before and after each test.

    Drops and recreates the table to ensure schema is up-to-date.
    """
    import psycopg2

    def recreate_table() -> None:
        try:
            conn = psycopg2.connect(integration_database_url)
            with conn.cursor() as cur:
                # Drop table to ensure schema is fresh
                cur.execute("DROP TABLE IF EXISTS token_ledger_events CASCADE")
            conn.commit()
            conn.close()
        except Exception:
            pass

    recreate_table()
    yield
    recreate_table()


@pytest.fixture
def mock_config() -> TokenLedgerConfig:
    """Create a mock configuration for testing."""
    return TokenLedgerConfig(
        database_url="postgresql://test:test@localhost:5432/test",
        app_name="test-app",
        environment="test",
        async_mode=False,
        debug=True,
        batch_size=10,
        flush_interval_seconds=1.0,
        sample_rate=1.0,
    )


@pytest.fixture
def async_mock_config() -> TokenLedgerConfig:
    """Create a mock configuration for async testing."""
    return TokenLedgerConfig(
        database_url="postgresql://test:test@localhost:5432/test",
        app_name="test-app",
        environment="test",
        async_mode=True,
        debug=True,
        batch_size=10,
        flush_interval_seconds=1.0,
        sample_rate=1.0,
    )


@pytest.fixture
def mock_db_connection():
    """Create a mock database connection."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return mock_conn


@pytest.fixture
def mock_async_db():
    """Create a mock async database instance."""
    mock_db = AsyncMock()
    mock_db.initialize = AsyncMock()
    mock_db.insert_events = AsyncMock(return_value=1)
    mock_db.close = AsyncMock()
    mock_db.fetchrow = AsyncMock(return_value=[0, 0, 0, 0, 0])
    mock_db.fetch = AsyncMock(return_value=[])
    return mock_db


@pytest.fixture
def sample_event_data() -> dict[str, Any]:
    """Create sample event data for testing."""
    return {
        "event_id": str(uuid.uuid4()),
        "provider": "openai",
        "model": "gpt-4o",
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
        "cached_tokens": 0,
        "cost_usd": 0.00075,
        "status": "success",
        "user_id": "test-user",
        "session_id": "test-session",
        "app_name": "test-app",
        "environment": "test",
        "timestamp": datetime.now(UTC),
        "duration_ms": 500.0,
    }


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI response object."""
    mock_response = MagicMock()
    mock_response.model = "gpt-4o"
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 50
    return mock_response


@pytest.fixture
def mock_anthropic_response():
    """Create a mock Anthropic response object."""
    mock_response = MagicMock()
    mock_response.model = "claude-3-5-sonnet-20241022"
    # Create usage mock without prompt_tokens (OpenAI style) so it doesn't auto-create
    mock_usage = MagicMock(spec=["input_tokens", "output_tokens", "cache_read_input_tokens"])
    mock_usage.input_tokens = 100
    mock_usage.output_tokens = 50
    mock_usage.cache_read_input_tokens = 20
    mock_response.usage = mock_usage
    return mock_response


@pytest.fixture(autouse=True)
def reset_global_config():
    """Reset global configuration before each test."""
    import tokenledger.config as config_module

    config_module._config = None
    yield
    config_module._config = None


@pytest.fixture(autouse=True)
def reset_global_trackers():
    """Reset global tracker instances before each test."""
    import tokenledger.tracker as tracker_module

    tracker_module._tracker = None
    tracker_module._async_tracker = None
    yield
    tracker_module._tracker = None
    tracker_module._async_tracker = None
