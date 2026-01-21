"""Integration tests for PostgreSQL storage backends.

These tests verify the backend implementations work correctly
with a real PostgreSQL database.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from tests.conftest import requires_postgres
from tokenledger.backends.postgresql import PostgreSQLBackend
from tokenledger.config import TokenLedgerConfig


@pytest.mark.integration
@requires_postgres
class TestPostgreSQLBackendIntegration:
    """Integration tests for PostgreSQLBackend."""

    def test_backend_initialization(
        self,
        integration_config: TokenLedgerConfig,
    ) -> None:
        """Test backend initializes and creates schema."""
        backend = PostgreSQLBackend()
        backend.initialize(integration_config, create_schema=True)

        assert backend.is_initialized is True
        assert backend.name == "PostgreSQL"

        # Verify we can query the database
        assert backend.health_check() is True

        backend.close()
        assert backend.is_initialized is False

    def test_backend_write_single_event(
        self,
        integration_config: TokenLedgerConfig,
        clean_events_table: None,
    ) -> None:
        """Test writing a single event through the backend."""
        backend = PostgreSQLBackend()
        backend.initialize(integration_config)

        event_id = str(uuid.uuid4())
        event = {
            "event_id": event_id,
            "provider": "openai",
            "model": "gpt-4o",
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "cached_tokens": 0,
            "cost_usd": 0.00075,
            "status": "success",
            "timestamp": datetime.now(UTC),
            "app_name": "backend-test",
            "environment": "test",
        }

        count = backend.write_events([event])

        assert count == 1

        # Verify event was written
        import psycopg2

        conn = psycopg2.connect(integration_config.database_url)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT provider, model FROM token_ledger_events WHERE event_id = %s",
                (event_id,),
            )
            row = cur.fetchone()

        conn.close()
        backend.close()

        assert row is not None
        assert row[0] == "openai"
        assert row[1] == "gpt-4o"

    def test_backend_write_batch(
        self,
        integration_config: TokenLedgerConfig,
        clean_events_table: None,
    ) -> None:
        """Test writing a batch of events through the backend."""
        backend = PostgreSQLBackend()
        backend.initialize(integration_config)

        events = []
        for i in range(100):
            events.append(
                {
                    "event_id": str(uuid.uuid4()),
                    "provider": "anthropic",
                    "model": "claude-3-5-sonnet",
                    "input_tokens": 50 + i,
                    "output_tokens": 25 + i,
                    "total_tokens": 75 + (i * 2),
                    "cached_tokens": 0,
                    "cost_usd": 0.001 * (i + 1),
                    "status": "success",
                    "timestamp": datetime.now(UTC),
                    "user_id": f"batch-user-{i}",
                }
            )

        count = backend.write_events(events)

        assert count == 100

        # Verify all events were written
        import psycopg2

        conn = psycopg2.connect(integration_config.database_url)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM token_ledger_events WHERE model = %s",
                ("claude-3-5-sonnet",),
            )
            db_count = cur.fetchone()[0]

        conn.close()
        backend.close()

        assert db_count == 100

    def test_backend_write_empty_list(
        self,
        integration_config: TokenLedgerConfig,
    ) -> None:
        """Test writing an empty list returns 0."""
        backend = PostgreSQLBackend()
        backend.initialize(integration_config)

        count = backend.write_events([])

        assert count == 0

        backend.close()

    def test_backend_health_check(
        self,
        integration_config: TokenLedgerConfig,
    ) -> None:
        """Test health check functionality."""
        backend = PostgreSQLBackend()

        # Health check before initialization
        assert backend.health_check() is False

        backend.initialize(integration_config)

        # Health check after initialization
        assert backend.health_check() is True

        backend.close()

        # Health check after close
        assert backend.health_check() is False

    def test_backend_get_info(
        self,
        integration_config: TokenLedgerConfig,
    ) -> None:
        """Test backend info retrieval."""
        backend = PostgreSQLBackend()
        backend.initialize(integration_config)

        info = backend.get_info()

        assert info.name == "PostgreSQL"
        assert info.supports_async is False
        assert info.capabilities.supports_jsonb is True
        assert info.capabilities.supports_upsert is True
        assert info.capabilities.supports_batch_insert is True

        backend.close()

    def test_backend_idempotent_initialization(
        self,
        integration_config: TokenLedgerConfig,
    ) -> None:
        """Test that multiple initialize calls are idempotent."""
        backend = PostgreSQLBackend()

        backend.initialize(integration_config)
        backend.initialize(integration_config)  # Should not raise
        backend.initialize(integration_config)  # Should not raise

        assert backend.is_initialized is True

        backend.close()

    def test_backend_duplicate_event_handling(
        self,
        integration_config: TokenLedgerConfig,
        clean_events_table: None,
    ) -> None:
        """Test that duplicate events are handled via ON CONFLICT DO NOTHING."""
        backend = PostgreSQLBackend()
        backend.initialize(integration_config)

        event_id = str(uuid.uuid4())
        event = {
            "event_id": event_id,
            "provider": "openai",
            "model": "gpt-4o",
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "cached_tokens": 0,
            "status": "success",
            "timestamp": datetime.now(UTC),
        }

        # Write the same event twice
        count1 = backend.write_events([event])
        count2 = backend.write_events([event])

        # Both should report success (one inserted, one ignored)
        assert count1 == 1
        assert count2 == 1

        # But only one should exist in database
        import psycopg2

        conn = psycopg2.connect(integration_config.database_url)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM token_ledger_events WHERE event_id = %s",
                (event_id,),
            )
            db_count = cur.fetchone()[0]

        conn.close()
        backend.close()

        assert db_count == 1


@pytest.mark.integration
@requires_postgres
class TestAsyncPostgreSQLBackendIntegration:
    """Integration tests for AsyncPostgreSQLBackend."""

    @pytest.mark.asyncio
    async def test_async_backend_initialization(
        self,
        async_integration_config: TokenLedgerConfig,
    ) -> None:
        """Test async backend initializes correctly."""
        from tokenledger.backends.postgresql import AsyncPostgreSQLBackend

        backend = AsyncPostgreSQLBackend()
        await backend.initialize(async_integration_config, create_schema=True)

        assert backend.is_initialized is True
        assert backend.name == "PostgreSQL (Async)"

        # Verify health check
        assert await backend.health_check() is True

        await backend.close()
        assert backend.is_initialized is False

    @pytest.mark.asyncio
    async def test_async_backend_write_events(
        self,
        async_integration_config: TokenLedgerConfig,
        clean_events_table: None,
    ) -> None:
        """Test writing events through async backend."""
        from tokenledger.backends.postgresql import AsyncPostgreSQLBackend

        backend = AsyncPostgreSQLBackend()
        await backend.initialize(async_integration_config)

        events = []
        for i in range(50):
            events.append(
                {
                    "event_id": str(uuid.uuid4()),
                    "provider": "google",
                    "model": "gemini-2.0-flash",
                    "input_tokens": 100 + i,
                    "output_tokens": 50 + i,
                    "total_tokens": 150 + (i * 2),
                    "cached_tokens": 0,
                    "status": "success",
                    "timestamp": datetime.now(UTC),
                }
            )

        count = await backend.write_events(events)

        assert count == 50

        # Verify events were written using async fetch
        result = await backend.fetchval(
            "SELECT COUNT(*) FROM token_ledger_events WHERE model = $1",
            "gemini-2.0-flash",
        )

        assert result == 50

        await backend.close()

    @pytest.mark.asyncio
    async def test_async_backend_pool_operations(
        self,
        async_integration_config: TokenLedgerConfig,
    ) -> None:
        """Test that pool operations work correctly."""
        from tokenledger.backends.postgresql import AsyncPostgreSQLBackend

        backend = AsyncPostgreSQLBackend()
        await backend.initialize(
            async_integration_config,
            min_pool_size=2,
            max_pool_size=5,
        )

        # Verify pool is accessible
        assert backend.pool is not None

        # Execute a simple query
        result = await backend.fetchval("SELECT 1")
        assert result == 1

        # Execute multiple concurrent queries
        import asyncio

        async def query() -> int:
            return await backend.fetchval("SELECT 1")

        results = await asyncio.gather(*[query() for _ in range(10)])
        assert all(r == 1 for r in results)

        await backend.close()
