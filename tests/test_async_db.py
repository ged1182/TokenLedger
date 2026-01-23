"""Tests for the async database module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestAsyncDatabase:
    """Tests for AsyncDatabase class."""

    def test_init_default_config(self) -> None:
        """Test AsyncDatabase initializes with default config."""
        from tokenledger.async_db import AsyncDatabase

        with patch("tokenledger.async_db.get_config") as mock_get_config:
            mock_config = MagicMock()
            mock_get_config.return_value = mock_config
            db = AsyncDatabase()

            assert db.config is mock_config
            assert db._pool is None
            assert db._initialized is False

    def test_init_custom_config(self) -> None:
        """Test AsyncDatabase initializes with custom config."""
        from tokenledger.async_db import AsyncDatabase

        custom_config = MagicMock()
        db = AsyncDatabase(config=custom_config)

        assert db.config is custom_config
        assert db._pool is None
        assert db._initialized is False

    def test_is_initialized_property(self) -> None:
        """Test is_initialized property."""
        from tokenledger.async_db import AsyncDatabase

        db = AsyncDatabase(config=MagicMock())
        assert db.is_initialized is False

        db._initialized = True
        assert db.is_initialized is True

    def test_pool_property(self) -> None:
        """Test pool property."""
        from tokenledger.async_db import AsyncDatabase

        db = AsyncDatabase(config=MagicMock())
        assert db.pool is None

        mock_pool = MagicMock()
        db._pool = mock_pool
        assert db.pool is mock_pool

    @pytest.mark.asyncio
    async def test_initialize_imports_asyncpg(self) -> None:
        """Test initialize raises ImportError if asyncpg not available."""
        from tokenledger.async_db import AsyncDatabase

        db = AsyncDatabase(config=MagicMock(database_url="postgresql://test"))

        with (
            patch.dict("sys.modules", {"asyncpg": None}),
            pytest.raises(ImportError, match="asyncpg is required"),
        ):
            await db.initialize()

    @pytest.mark.asyncio
    async def test_initialize_creates_pool(self) -> None:
        """Test initialize creates connection pool."""
        from tokenledger.async_db import AsyncDatabase

        mock_pool = AsyncMock()
        mock_config = MagicMock(database_url="postgresql://test", debug=False)

        db = AsyncDatabase(config=mock_config)

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create_pool:
            mock_create_pool.return_value = mock_pool
            # Skip table creation
            db._create_tables = AsyncMock()

            await db.initialize(min_size=1, max_size=5, create_tables=False)

            mock_create_pool.assert_called_once_with(
                "postgresql://test",
                min_size=1,
                max_size=5,
            )
            assert db._initialized is True
            assert db._pool is mock_pool

    @pytest.mark.asyncio
    async def test_initialize_skips_if_already_initialized(self) -> None:
        """Test initialize does nothing if already initialized."""
        from tokenledger.async_db import AsyncDatabase

        db = AsyncDatabase(config=MagicMock())
        db._initialized = True

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create_pool:
            await db.initialize()
            mock_create_pool.assert_not_called()

    @pytest.mark.asyncio
    async def test_initialize_creates_tables_by_default(self) -> None:
        """Test initialize creates tables by default."""
        from tokenledger.async_db import AsyncDatabase

        mock_pool = AsyncMock()
        mock_config = MagicMock(database_url="postgresql://test", debug=False)

        db = AsyncDatabase(config=mock_config)
        db._create_tables = AsyncMock()

        with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create_pool:
            mock_create_pool.return_value = mock_pool

            await db.initialize()

            db._create_tables.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_logs_in_debug_mode(self) -> None:
        """Test initialize logs when debug is enabled."""
        from tokenledger.async_db import AsyncDatabase

        mock_pool = AsyncMock()
        mock_config = MagicMock(database_url="postgresql://test", debug=True)

        db = AsyncDatabase(config=mock_config)

        with (
            patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create_pool,
            patch("tokenledger.async_db.logger") as mock_logger,
        ):
            mock_create_pool.return_value = mock_pool
            db._create_tables = AsyncMock()

            await db.initialize(min_size=2, max_size=10)

            mock_logger.info.assert_called()

    @pytest.mark.asyncio
    async def test_acquire_raises_if_not_initialized(self) -> None:
        """Test acquire raises RuntimeError if not initialized."""
        from tokenledger.async_db import AsyncDatabase

        db = AsyncDatabase(config=MagicMock())

        with pytest.raises(RuntimeError, match="Database not initialized"):
            async with db.acquire():
                pass

    @pytest.mark.asyncio
    async def test_acquire_returns_connection(self) -> None:
        """Test acquire returns connection from pool."""
        from tokenledger.async_db import AsyncDatabase

        mock_conn = AsyncMock()
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        db = AsyncDatabase(config=MagicMock())
        db._initialized = True
        db._pool = mock_pool

        async with db.acquire() as conn:
            assert conn is mock_conn

    @pytest.mark.asyncio
    async def test_execute_runs_query(self) -> None:
        """Test execute runs a query."""
        from tokenledger.async_db import AsyncDatabase

        mock_conn = AsyncMock()
        mock_conn.execute.return_value = "OK"

        db = AsyncDatabase(config=MagicMock())
        db._initialized = True

        # Mock acquire context manager
        db.acquire = MagicMock()
        db.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        db.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await db.execute("SELECT 1", "arg1")
        assert result == "OK"
        mock_conn.execute.assert_called_once_with("SELECT 1", "arg1")

    @pytest.mark.asyncio
    async def test_fetch_returns_rows(self) -> None:
        """Test fetch returns all rows."""
        from tokenledger.async_db import AsyncDatabase

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [{"id": 1}, {"id": 2}]

        db = AsyncDatabase(config=MagicMock())
        db._initialized = True

        db.acquire = MagicMock()
        db.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        db.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await db.fetch("SELECT * FROM table")
        assert result == [{"id": 1}, {"id": 2}]

    @pytest.mark.asyncio
    async def test_fetchrow_returns_first_row(self) -> None:
        """Test fetchrow returns first row."""
        from tokenledger.async_db import AsyncDatabase

        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"id": 1}

        db = AsyncDatabase(config=MagicMock())
        db._initialized = True

        db.acquire = MagicMock()
        db.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        db.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await db.fetchrow("SELECT * FROM table LIMIT 1")
        assert result == {"id": 1}

    @pytest.mark.asyncio
    async def test_fetchval_returns_first_value(self) -> None:
        """Test fetchval returns first value."""
        from tokenledger.async_db import AsyncDatabase

        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = 42

        db = AsyncDatabase(config=MagicMock())
        db._initialized = True

        db.acquire = MagicMock()
        db.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        db.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await db.fetchval("SELECT COUNT(*) FROM table")
        assert result == 42

    @pytest.mark.asyncio
    async def test_executemany_runs_multiple(self) -> None:
        """Test executemany runs query with multiple args."""
        from tokenledger.async_db import AsyncDatabase

        mock_conn = AsyncMock()

        db = AsyncDatabase(config=MagicMock())
        db._initialized = True

        db.acquire = MagicMock()
        db.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        db.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        args = [(1,), (2,), (3,)]
        await db.executemany("INSERT INTO table VALUES ($1)", args)
        mock_conn.executemany.assert_called_once_with("INSERT INTO table VALUES ($1)", args)

    @pytest.mark.asyncio
    async def test_insert_events_empty_list(self) -> None:
        """Test insert_events returns 0 for empty list."""
        from tokenledger.async_db import AsyncDatabase

        db = AsyncDatabase(config=MagicMock())
        db._initialized = True

        result = await db.insert_events([])
        assert result == 0

    @pytest.mark.asyncio
    async def test_insert_events_serializes_metadata(self) -> None:
        """Test insert_events serializes metadata to JSON."""
        from tokenledger.async_db import AsyncDatabase

        mock_conn = AsyncMock()
        mock_config = MagicMock(
            full_table_name="public.token_ledger_events",
            debug=False,
        )

        db = AsyncDatabase(config=mock_config)
        db._initialized = True

        db.acquire = MagicMock()
        db.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        db.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        events = [
            {
                "event_id": "123",
                "provider": "openai",
                "model": "gpt-4",
                "metadata": {"key": "value"},
            }
        ]

        result = await db.insert_events(events)
        assert result == 1
        mock_conn.executemany.assert_called_once()

        # Check that metadata was serialized
        call_args = mock_conn.executemany.call_args
        values = call_args[0][1][0]  # First event tuple
        # metadata is at index 23 in the columns list
        assert '{"key": "value"}' in str(values)

    @pytest.mark.asyncio
    async def test_insert_events_handles_string_metadata(self) -> None:
        """Test insert_events handles already-serialized metadata."""
        from tokenledger.async_db import AsyncDatabase

        mock_conn = AsyncMock()
        mock_config = MagicMock(
            full_table_name="public.token_ledger_events",
            debug=False,
        )

        db = AsyncDatabase(config=mock_config)
        db._initialized = True

        db.acquire = MagicMock()
        db.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        db.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        events = [
            {
                "event_id": "123",
                "provider": "openai",
                "model": "gpt-4",
                "metadata": '{"already": "serialized"}',
            }
        ]

        result = await db.insert_events(events)
        assert result == 1

    @pytest.mark.asyncio
    async def test_insert_events_handles_error(self) -> None:
        """Test insert_events handles database errors."""
        from tokenledger.async_db import AsyncDatabase

        mock_conn = AsyncMock()
        mock_conn.executemany.side_effect = Exception("DB error")
        mock_config = MagicMock(
            full_table_name="public.token_ledger_events",
            debug=False,
        )

        db = AsyncDatabase(config=mock_config)
        db._initialized = True

        db.acquire = MagicMock()
        db.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        db.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        events = [{"event_id": "123", "provider": "openai", "model": "gpt-4"}]

        with patch("tokenledger.async_db.logger") as mock_logger:
            result = await db.insert_events(events)
            assert result == 0
            mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_insert_events_logs_in_debug_mode(self) -> None:
        """Test insert_events logs when debug is enabled."""
        from tokenledger.async_db import AsyncDatabase

        mock_conn = AsyncMock()
        mock_config = MagicMock(
            full_table_name="public.token_ledger_events",
            debug=True,
        )

        db = AsyncDatabase(config=mock_config)
        db._initialized = True

        db.acquire = MagicMock()
        db.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        db.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        events = [{"event_id": "123", "provider": "openai", "model": "gpt-4"}]

        with patch("tokenledger.async_db.logger") as mock_logger:
            await db.insert_events(events)
            mock_logger.info.assert_called()

    @pytest.mark.asyncio
    async def test_close_closes_pool(self) -> None:
        """Test close closes the connection pool."""
        from tokenledger.async_db import AsyncDatabase

        mock_pool = AsyncMock()
        mock_config = MagicMock(debug=False)

        db = AsyncDatabase(config=mock_config)
        db._pool = mock_pool
        db._initialized = True

        await db.close()

        mock_pool.close.assert_called_once()
        assert db._pool is None
        assert db._initialized is False

    @pytest.mark.asyncio
    async def test_close_does_nothing_if_no_pool(self) -> None:
        """Test close does nothing if pool is None."""
        from tokenledger.async_db import AsyncDatabase

        db = AsyncDatabase(config=MagicMock(debug=False))
        db._pool = None

        # Should not raise
        await db.close()

    @pytest.mark.asyncio
    async def test_close_logs_in_debug_mode(self) -> None:
        """Test close logs when debug is enabled."""
        from tokenledger.async_db import AsyncDatabase

        mock_pool = AsyncMock()

        db = AsyncDatabase(config=MagicMock(debug=True))
        db._pool = mock_pool
        db._initialized = True

        with patch("tokenledger.async_db.logger") as mock_logger:
            await db.close()
            mock_logger.info.assert_called()

    @pytest.mark.asyncio
    async def test_create_tables_executes_sql(self) -> None:
        """Test _create_tables executes table creation SQL."""
        from tokenledger.async_db import AsyncDatabase

        mock_conn = AsyncMock()
        mock_config = MagicMock(
            full_table_name="public.token_ledger_events",
            table_name="token_ledger_events",
            debug=False,
        )

        db = AsyncDatabase(config=mock_config)
        db._initialized = True

        db.acquire = MagicMock()
        db.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        db.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        await db._create_tables()

        mock_conn.execute.assert_called_once()
        call_sql = mock_conn.execute.call_args[0][0]
        assert "CREATE TABLE IF NOT EXISTS" in call_sql


class TestGlobalAsyncDbFunctions:
    """Tests for global async database functions."""

    @pytest.mark.asyncio
    async def test_get_async_db_creates_instance(self) -> None:
        """Test get_async_db creates a new instance."""
        import tokenledger.async_db as async_db_module

        # Reset global
        async_db_module._async_db = None

        with patch.object(async_db_module, "AsyncDatabase") as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            result = await async_db_module.get_async_db()

            mock_class.assert_called_once()
            assert result is mock_instance

    @pytest.mark.asyncio
    async def test_get_async_db_returns_existing_instance(self) -> None:
        """Test get_async_db returns existing instance."""
        import tokenledger.async_db as async_db_module

        mock_existing = MagicMock()
        async_db_module._async_db = mock_existing

        result = await async_db_module.get_async_db()

        assert result is mock_existing

        # Cleanup
        async_db_module._async_db = None

    @pytest.mark.asyncio
    async def test_init_async_db_initializes(self) -> None:
        """Test init_async_db initializes the database."""
        import tokenledger.async_db as async_db_module

        mock_db = AsyncMock()
        async_db_module._async_db = None

        with patch.object(async_db_module, "get_async_db", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_db

            result = await async_db_module.init_async_db(min_size=3, max_size=15)

            mock_get.assert_called_once()
            mock_db.initialize.assert_called_once_with(min_size=3, max_size=15, create_tables=True)
            assert result is mock_db

        # Cleanup
        async_db_module._async_db = None

    @pytest.mark.asyncio
    async def test_close_async_db_closes_and_clears(self) -> None:
        """Test close_async_db closes and clears the global instance."""
        import tokenledger.async_db as async_db_module

        mock_db = AsyncMock()
        async_db_module._async_db = mock_db

        await async_db_module.close_async_db()

        mock_db.close.assert_called_once()
        assert async_db_module._async_db is None

    @pytest.mark.asyncio
    async def test_close_async_db_does_nothing_if_none(self) -> None:
        """Test close_async_db does nothing if no instance."""
        import tokenledger.async_db as async_db_module

        async_db_module._async_db = None

        # Should not raise
        await async_db_module.close_async_db()

        assert async_db_module._async_db is None
