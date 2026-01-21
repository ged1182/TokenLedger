"""Tests for async PostgreSQL storage backend."""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tokenledger.backends.exceptions import (
    BackendNotInitializedError,
    DriverNotFoundError,
    WriteError,
)
from tokenledger.backends.postgresql.async_storage import AsyncPostgreSQLBackend
from tokenledger.backends.protocol import AsyncStorageBackend, BackendInfo
from tokenledger.config import TokenLedgerConfig


@pytest.fixture
def config() -> TokenLedgerConfig:
    """Create a test configuration."""
    return TokenLedgerConfig(
        database_url="postgresql://test:test@localhost/test",
        app_name="test-app",
        debug=True,
    )


@pytest.fixture
def mock_asyncpg_module() -> MagicMock:
    """Create a complete mock asyncpg module with properly configured pool."""
    # Create mock connection that will be returned from acquire()
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value="INSERT 1")
    mock_conn.executemany = AsyncMock()
    mock_conn.fetch = AsyncMock(return_value=[{"id": 1}, {"id": 2}])
    mock_conn.fetchrow = AsyncMock(return_value={"id": 1, "name": "test"})
    mock_conn.fetchval = AsyncMock(return_value=1)

    # Create mock pool
    mock_pool = MagicMock()
    mock_pool.close = AsyncMock()

    # Create proper async context manager for acquire()
    @asynccontextmanager
    async def mock_acquire():
        yield mock_conn

    mock_pool.acquire = mock_acquire

    # Create the asyncpg module mock
    mock_asyncpg = MagicMock()
    mock_asyncpg.create_pool = AsyncMock(return_value=mock_pool)
    mock_asyncpg.__version__ = "0.29.0"

    # Store references for test assertions
    mock_asyncpg._mock_pool = mock_pool
    mock_asyncpg._mock_conn = mock_conn

    return mock_asyncpg


class TestAsyncPostgreSQLBackendProtocol:
    """Tests for AsyncPostgreSQLBackend protocol compliance."""

    def test_implements_async_storage_backend_protocol(self) -> None:
        """Test that AsyncPostgreSQLBackend implements AsyncStorageBackend protocol."""
        backend = AsyncPostgreSQLBackend()
        assert isinstance(backend, AsyncStorageBackend)


class TestAsyncPostgreSQLBackendProperties:
    """Tests for AsyncPostgreSQLBackend properties."""

    def test_name(self) -> None:
        """Test backend name."""
        backend = AsyncPostgreSQLBackend()
        assert backend.name == "PostgreSQL (Async)"

    def test_is_initialized_default(self) -> None:
        """Test is_initialized is False by default."""
        backend = AsyncPostgreSQLBackend()
        assert backend.is_initialized is False

    def test_capabilities(self) -> None:
        """Test backend capabilities."""
        backend = AsyncPostgreSQLBackend()
        caps = backend.capabilities

        assert caps.supports_jsonb is True
        assert caps.supports_uuid is True
        assert caps.supports_upsert is True
        assert caps.supports_batch_insert is True
        assert caps.max_batch_size == 10000


class TestAsyncPostgreSQLBackendInitialization:
    """Tests for AsyncPostgreSQLBackend initialization."""

    @pytest.mark.asyncio
    async def test_initialize_success(
        self, config: TokenLedgerConfig, mock_asyncpg_module: MagicMock
    ) -> None:
        """Test successful initialization."""
        with patch.dict(sys.modules, {"asyncpg": mock_asyncpg_module}):
            backend = AsyncPostgreSQLBackend()
            await backend.initialize(config)

            assert backend.is_initialized is True
            assert backend._driver_version == "0.29.0"

    @pytest.mark.asyncio
    async def test_initialize_with_custom_pool_size(
        self, config: TokenLedgerConfig, mock_asyncpg_module: MagicMock
    ) -> None:
        """Test initialization with custom pool sizes."""
        with patch.dict(sys.modules, {"asyncpg": mock_asyncpg_module}):
            backend = AsyncPostgreSQLBackend()
            await backend.initialize(config, min_pool_size=5, max_pool_size=20, create_schema=False)

            assert backend._min_pool_size == 5
            assert backend._max_pool_size == 20
            mock_asyncpg_module.create_pool.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_no_driver_raises(self, config: TokenLedgerConfig) -> None:
        """Test initialization raises when asyncpg not available."""
        # Create a fresh sys.modules without asyncpg
        saved_modules = sys.modules.copy()

        # Remove any asyncpg references
        modules_to_remove = [k for k in sys.modules if k == "asyncpg" or k.startswith("asyncpg.")]
        for mod in modules_to_remove:
            sys.modules.pop(mod, None)

        # Mock the import to raise ImportError
        original_import = (
            __builtins__["__import__"]
            if isinstance(__builtins__, dict)
            else __builtins__.__import__
        )

        def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "asyncpg":
                raise ImportError("No module named 'asyncpg'")
            return original_import(name, *args, **kwargs)

        try:
            with patch("builtins.__import__", side_effect=mock_import):
                backend = AsyncPostgreSQLBackend()
                with pytest.raises(DriverNotFoundError) as exc_info:
                    await backend.initialize(config)

                assert "asyncpg" in str(exc_info.value)
        finally:
            # Restore original modules
            sys.modules.update(saved_modules)

    @pytest.mark.asyncio
    async def test_initialize_idempotent(
        self, config: TokenLedgerConfig, mock_asyncpg_module: MagicMock
    ) -> None:
        """Test that initialize is idempotent."""
        with patch.dict(sys.modules, {"asyncpg": mock_asyncpg_module}):
            backend = AsyncPostgreSQLBackend()
            await backend.initialize(config)
            await backend.initialize(config)  # Should not raise

            # create_pool should only be called once
            assert mock_asyncpg_module.create_pool.call_count == 1


class TestAsyncPostgreSQLBackendWriteEvents:
    """Tests for AsyncPostgreSQLBackend write_events."""

    @pytest.mark.asyncio
    async def test_write_events_not_initialized(self) -> None:
        """Test write_events raises when not initialized."""
        backend = AsyncPostgreSQLBackend()
        events = [{"event_id": "test-123"}]

        with pytest.raises(BackendNotInitializedError):
            await backend.write_events(events)

    @pytest.mark.asyncio
    async def test_write_events_empty_list(
        self, config: TokenLedgerConfig, mock_asyncpg_module: MagicMock
    ) -> None:
        """Test write_events with empty list returns 0."""
        with patch.dict(sys.modules, {"asyncpg": mock_asyncpg_module}):
            backend = AsyncPostgreSQLBackend()
            await backend.initialize(config)

            count = await backend.write_events([])

            assert count == 0

    @pytest.mark.asyncio
    async def test_write_events_success(
        self, config: TokenLedgerConfig, mock_asyncpg_module: MagicMock
    ) -> None:
        """Test successful write."""
        with patch.dict(sys.modules, {"asyncpg": mock_asyncpg_module}):
            backend = AsyncPostgreSQLBackend()
            await backend.initialize(config)

            events = [
                {"event_id": "1", "provider": "openai", "model": "gpt-4"},
                {"event_id": "2", "provider": "anthropic", "model": "claude-3"},
            ]

            count = await backend.write_events(events)

            assert count == 2

    @pytest.mark.asyncio
    async def test_write_events_failure(
        self, config: TokenLedgerConfig, mock_asyncpg_module: MagicMock
    ) -> None:
        """Test write failure raises WriteError."""
        # Make executemany raise an error
        mock_asyncpg_module._mock_conn.executemany = AsyncMock(side_effect=Exception("DB Error"))

        with patch.dict(sys.modules, {"asyncpg": mock_asyncpg_module}):
            backend = AsyncPostgreSQLBackend()
            await backend.initialize(config)

            events = [{"event_id": "1"}]

            with pytest.raises(WriteError):
                await backend.write_events(events)


class TestAsyncPostgreSQLBackendHealthCheck:
    """Tests for AsyncPostgreSQLBackend health_check."""

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self) -> None:
        """Test health_check returns False when not initialized."""
        backend = AsyncPostgreSQLBackend()
        result = await backend.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_success(
        self, config: TokenLedgerConfig, mock_asyncpg_module: MagicMock
    ) -> None:
        """Test successful health check."""
        with patch.dict(sys.modules, {"asyncpg": mock_asyncpg_module}):
            backend = AsyncPostgreSQLBackend()
            await backend.initialize(config)

            result = await backend.health_check()

            assert result is True


class TestAsyncPostgreSQLBackendClose:
    """Tests for AsyncPostgreSQLBackend close."""

    @pytest.mark.asyncio
    async def test_close_without_initialize(self) -> None:
        """Test close without initialization doesn't raise."""
        backend = AsyncPostgreSQLBackend()
        await backend.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_close_clears_state(
        self, config: TokenLedgerConfig, mock_asyncpg_module: MagicMock
    ) -> None:
        """Test close clears backend state."""
        with patch.dict(sys.modules, {"asyncpg": mock_asyncpg_module}):
            backend = AsyncPostgreSQLBackend()
            await backend.initialize(config)
            assert backend.is_initialized is True

            await backend.close()

            assert backend.is_initialized is False
            mock_asyncpg_module._mock_pool.close.assert_called_once()


class TestAsyncPostgreSQLBackendGetInfo:
    """Tests for AsyncPostgreSQLBackend get_info."""

    def test_get_info_returns_backend_info(self) -> None:
        """Test get_info returns BackendInfo."""
        backend = AsyncPostgreSQLBackend()
        info = backend.get_info()

        assert isinstance(info, BackendInfo)
        assert info.name == "PostgreSQL (Async)"
        assert info.supports_async is True
        assert info.capabilities.supports_jsonb is True


class TestAsyncPostgreSQLBackendQueryMethods:
    """Tests for AsyncPostgreSQLBackend query methods."""

    @pytest.mark.asyncio
    async def test_execute_method(
        self, config: TokenLedgerConfig, mock_asyncpg_module: MagicMock
    ) -> None:
        """Test execute method."""
        with patch.dict(sys.modules, {"asyncpg": mock_asyncpg_module}):
            backend = AsyncPostgreSQLBackend()
            await backend.initialize(config)

            result = await backend.execute("SELECT 1")

            assert result == "INSERT 1"
            mock_asyncpg_module._mock_conn.execute.assert_called_with("SELECT 1")

    @pytest.mark.asyncio
    async def test_fetch_method(
        self, config: TokenLedgerConfig, mock_asyncpg_module: MagicMock
    ) -> None:
        """Test fetch method."""
        with patch.dict(sys.modules, {"asyncpg": mock_asyncpg_module}):
            backend = AsyncPostgreSQLBackend()
            await backend.initialize(config)

            result = await backend.fetch("SELECT * FROM table")

            assert len(result) == 2
            mock_asyncpg_module._mock_conn.fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetchrow_method(
        self, config: TokenLedgerConfig, mock_asyncpg_module: MagicMock
    ) -> None:
        """Test fetchrow method."""
        with patch.dict(sys.modules, {"asyncpg": mock_asyncpg_module}):
            backend = AsyncPostgreSQLBackend()
            await backend.initialize(config)

            result = await backend.fetchrow("SELECT * FROM table WHERE id = $1", 1)

            assert result["id"] == 1

    @pytest.mark.asyncio
    async def test_fetchval_method(
        self, config: TokenLedgerConfig, mock_asyncpg_module: MagicMock
    ) -> None:
        """Test fetchval method."""
        mock_asyncpg_module._mock_conn.fetchval = AsyncMock(return_value=42)

        with patch.dict(sys.modules, {"asyncpg": mock_asyncpg_module}):
            backend = AsyncPostgreSQLBackend()
            await backend.initialize(config)

            result = await backend.fetchval("SELECT COUNT(*) FROM table")

            assert result == 42

    @pytest.mark.asyncio
    async def test_pool_property(
        self, config: TokenLedgerConfig, mock_asyncpg_module: MagicMock
    ) -> None:
        """Test pool property exposes the underlying pool."""
        with patch.dict(sys.modules, {"asyncpg": mock_asyncpg_module}):
            backend = AsyncPostgreSQLBackend()
            await backend.initialize(config)

            assert backend.pool is mock_asyncpg_module._mock_pool
