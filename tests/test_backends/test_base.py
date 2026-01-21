"""Tests for backend base classes."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from tokenledger.backends.base import BaseAsyncStorageBackend, BaseStorageBackend
from tokenledger.backends.exceptions import (
    BackendNotInitializedError,
    SchemaError,
    WriteError,
)
from tokenledger.backends.protocol import BackendCapabilities, BackendInfo
from tokenledger.config import TokenLedgerConfig


class ConcreteStorageBackend(BaseStorageBackend):
    """Concrete implementation for testing BaseStorageBackend."""

    def __init__(self) -> None:
        super().__init__()
        self._connect_called = False
        self._create_schema_called = False
        self._write_batch_called = False
        self._health_check_result = True
        self._close_called = False
        self._should_fail_connect = False
        self._should_fail_schema = False
        self._should_fail_write = False

    @property
    def name(self) -> str:
        return "TestBackend"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_jsonb=True,
            supports_upsert=True,
            max_batch_size=1000,
        )

    def _connect(self) -> None:
        if self._should_fail_connect:
            raise RuntimeError("Connection failed")
        self._connect_called = True
        self._connection = MagicMock()

    def _create_schema(self) -> None:
        if self._should_fail_schema:
            raise RuntimeError("Schema creation failed")
        self._create_schema_called = True

    def _write_batch(self, events: list[dict[str, Any]]) -> int:
        if self._should_fail_write:
            raise RuntimeError("Write failed")
        self._write_batch_called = True
        return len(events)

    def _health_check(self) -> bool:
        return self._health_check_result

    def _close(self) -> None:
        self._close_called = True

    def _get_version(self) -> str:
        return "1.0.0"

    def _get_driver_name(self) -> str:
        return "test-driver"


class TestBaseStorageBackend:
    """Tests for BaseStorageBackend."""

    @pytest.fixture
    def config(self) -> TokenLedgerConfig:
        """Create a test configuration."""
        return TokenLedgerConfig(
            database_url="postgresql://test:test@localhost/test",
            app_name="test-app",
            debug=True,
        )

    @pytest.fixture
    def backend(self) -> ConcreteStorageBackend:
        """Create a test backend instance."""
        return ConcreteStorageBackend()

    def test_initial_state(self, backend: ConcreteStorageBackend) -> None:
        """Test initial state of backend."""
        assert backend.name == "TestBackend"
        assert backend.is_initialized is False
        assert backend._config is None
        assert backend._connection is None

    def test_initialize_success(
        self, backend: ConcreteStorageBackend, config: TokenLedgerConfig
    ) -> None:
        """Test successful initialization."""
        backend.initialize(config, create_schema=True)

        assert backend.is_initialized is True
        assert backend._config == config
        assert backend._connect_called is True
        assert backend._create_schema_called is True

    def test_initialize_without_schema(
        self, backend: ConcreteStorageBackend, config: TokenLedgerConfig
    ) -> None:
        """Test initialization without schema creation."""
        backend.initialize(config, create_schema=False)

        assert backend.is_initialized is True
        assert backend._connect_called is True
        assert backend._create_schema_called is False

    def test_initialize_idempotent(
        self, backend: ConcreteStorageBackend, config: TokenLedgerConfig
    ) -> None:
        """Test that initialize is idempotent."""
        backend.initialize(config)
        backend.initialize(config)  # Should not raise

        assert backend._connect_called is True

    def test_initialize_schema_error(
        self, backend: ConcreteStorageBackend, config: TokenLedgerConfig
    ) -> None:
        """Test initialization with schema creation failure."""
        backend._should_fail_schema = True

        with pytest.raises(SchemaError) as exc_info:
            backend.initialize(config, create_schema=True)

        assert "TestBackend" in str(exc_info.value)
        assert backend.is_initialized is False
        assert backend._connection is None

    def test_write_events_success(
        self, backend: ConcreteStorageBackend, config: TokenLedgerConfig
    ) -> None:
        """Test successful write_events."""
        backend.initialize(config)

        events = [
            {"event_id": "1", "provider": "openai", "model": "gpt-4"},
            {"event_id": "2", "provider": "openai", "model": "gpt-4"},
        ]

        count = backend.write_events(events)

        assert count == 2
        assert backend._write_batch_called is True

    def test_write_events_empty(
        self, backend: ConcreteStorageBackend, config: TokenLedgerConfig
    ) -> None:
        """Test write_events with empty list."""
        backend.initialize(config)

        count = backend.write_events([])

        assert count == 0
        assert backend._write_batch_called is False

    def test_write_events_not_initialized(self, backend: ConcreteStorageBackend) -> None:
        """Test write_events when not initialized."""
        events = [{"event_id": "1"}]

        with pytest.raises(BackendNotInitializedError) as exc_info:
            backend.write_events(events)

        assert "TestBackend" in str(exc_info.value)

    def test_write_events_failure(
        self, backend: ConcreteStorageBackend, config: TokenLedgerConfig
    ) -> None:
        """Test write_events failure."""
        backend.initialize(config)
        backend._should_fail_write = True

        events = [{"event_id": "1"}]

        with pytest.raises(WriteError) as exc_info:
            backend.write_events(events)

        assert exc_info.value.events_count == 1
        assert exc_info.value.events_written == 0

    def test_health_check_success(
        self, backend: ConcreteStorageBackend, config: TokenLedgerConfig
    ) -> None:
        """Test successful health check."""
        backend.initialize(config)

        result = backend.health_check()

        assert result is True

    def test_health_check_failure(
        self, backend: ConcreteStorageBackend, config: TokenLedgerConfig
    ) -> None:
        """Test health check failure."""
        backend.initialize(config)
        backend._health_check_result = False

        result = backend.health_check()

        assert result is False

    def test_health_check_not_initialized(self, backend: ConcreteStorageBackend) -> None:
        """Test health check when not initialized."""
        result = backend.health_check()

        assert result is False

    def test_close(self, backend: ConcreteStorageBackend, config: TokenLedgerConfig) -> None:
        """Test close method."""
        backend.initialize(config)
        assert backend.is_initialized is True

        backend.close()

        assert backend.is_initialized is False
        assert backend._connection is None
        assert backend._close_called is True

    def test_close_idempotent(self, backend: ConcreteStorageBackend) -> None:
        """Test that close is idempotent."""
        backend.close()  # Should not raise
        backend.close()  # Should not raise

    def test_get_info(self, backend: ConcreteStorageBackend, config: TokenLedgerConfig) -> None:
        """Test get_info method."""
        backend.initialize(config)

        info = backend.get_info()

        assert isinstance(info, BackendInfo)
        assert info.name == "TestBackend"
        assert info.version == "1.0.0"
        assert info.driver == "test-driver"
        assert info.supports_async is False
        assert info.capabilities.supports_jsonb is True

    def test_prepare_event_values(self, backend: ConcreteStorageBackend) -> None:
        """Test _prepare_event_values method."""
        event = {
            "event_id": "test-123",
            "provider": "openai",
            "model": "gpt-4",
            "input_tokens": 100,
            "metadata": {"key": "value"},
        }

        values = backend._prepare_event_values(event)

        assert isinstance(values, tuple)
        assert len(values) == len(backend.COLUMNS)
        assert values[0] == "test-123"  # event_id
        # metadata should be JSON serialized
        assert '"key"' in str(values)

    def test_prepare_event_values_metadata_already_string(
        self, backend: ConcreteStorageBackend
    ) -> None:
        """Test _prepare_event_values with metadata already as string."""
        event = {
            "event_id": "test-123",
            "metadata": '{"key": "value"}',
        }

        values = backend._prepare_event_values(event)

        # Should not double-serialize
        assert values[backend.COLUMNS.index("metadata")] == '{"key": "value"}'


class ConcreteAsyncStorageBackend(BaseAsyncStorageBackend):
    """Concrete implementation for testing BaseAsyncStorageBackend."""

    def __init__(self) -> None:
        super().__init__()
        self._connect_called = False
        self._create_schema_called = False
        self._write_batch_called = False
        self._health_check_result = True
        self._close_called = False
        self._should_fail_schema = False
        self._should_fail_write = False

    @property
    def name(self) -> str:
        return "TestAsyncBackend"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_jsonb=True,
            max_batch_size=5000,
        )

    async def _connect(self) -> None:
        self._connect_called = True
        self._pool = MagicMock()

    async def _create_schema(self) -> None:
        if self._should_fail_schema:
            raise RuntimeError("Schema creation failed")
        self._create_schema_called = True

    async def _write_batch(self, events: list[dict[str, Any]]) -> int:
        if self._should_fail_write:
            raise RuntimeError("Write failed")
        self._write_batch_called = True
        return len(events)

    async def _health_check(self) -> bool:
        return self._health_check_result

    async def _close(self) -> None:
        self._close_called = True

    def _get_version(self) -> str:
        return "2.0.0"

    def _get_driver_name(self) -> str:
        return "async-test-driver"


class TestBaseAsyncStorageBackend:
    """Tests for BaseAsyncStorageBackend."""

    @pytest.fixture
    def config(self) -> TokenLedgerConfig:
        """Create a test configuration."""
        return TokenLedgerConfig(
            database_url="postgresql://test:test@localhost/test",
            app_name="test-app",
            debug=True,
        )

    @pytest.fixture
    def backend(self) -> ConcreteAsyncStorageBackend:
        """Create a test async backend instance."""
        return ConcreteAsyncStorageBackend()

    def test_initial_state(self, backend: ConcreteAsyncStorageBackend) -> None:
        """Test initial state of async backend."""
        assert backend.name == "TestAsyncBackend"
        assert backend.is_initialized is False
        assert backend._config is None
        assert backend._pool is None

    @pytest.mark.asyncio
    async def test_initialize_success(
        self, backend: ConcreteAsyncStorageBackend, config: TokenLedgerConfig
    ) -> None:
        """Test successful async initialization."""
        await backend.initialize(config, create_schema=True)

        assert backend.is_initialized is True
        assert backend._config == config
        assert backend._connect_called is True
        assert backend._create_schema_called is True

    @pytest.mark.asyncio
    async def test_initialize_without_schema(
        self, backend: ConcreteAsyncStorageBackend, config: TokenLedgerConfig
    ) -> None:
        """Test async initialization without schema creation."""
        await backend.initialize(config, create_schema=False)

        assert backend.is_initialized is True
        assert backend._create_schema_called is False

    @pytest.mark.asyncio
    async def test_initialize_schema_error(
        self, backend: ConcreteAsyncStorageBackend, config: TokenLedgerConfig
    ) -> None:
        """Test async initialization with schema failure."""
        backend._should_fail_schema = True

        with pytest.raises(SchemaError):
            await backend.initialize(config, create_schema=True)

        assert backend.is_initialized is False

    @pytest.mark.asyncio
    async def test_write_events_success(
        self, backend: ConcreteAsyncStorageBackend, config: TokenLedgerConfig
    ) -> None:
        """Test successful async write_events."""
        await backend.initialize(config)

        events = [
            {"event_id": "1", "provider": "openai"},
            {"event_id": "2", "provider": "anthropic"},
        ]

        count = await backend.write_events(events)

        assert count == 2
        assert backend._write_batch_called is True

    @pytest.mark.asyncio
    async def test_write_events_not_initialized(self, backend: ConcreteAsyncStorageBackend) -> None:
        """Test async write_events when not initialized."""
        with pytest.raises(BackendNotInitializedError):
            await backend.write_events([{"event_id": "1"}])

    @pytest.mark.asyncio
    async def test_write_events_failure(
        self, backend: ConcreteAsyncStorageBackend, config: TokenLedgerConfig
    ) -> None:
        """Test async write_events failure."""
        await backend.initialize(config)
        backend._should_fail_write = True

        with pytest.raises(WriteError):
            await backend.write_events([{"event_id": "1"}])

    @pytest.mark.asyncio
    async def test_health_check_success(
        self, backend: ConcreteAsyncStorageBackend, config: TokenLedgerConfig
    ) -> None:
        """Test successful async health check."""
        await backend.initialize(config)

        result = await backend.health_check()

        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self, backend: ConcreteAsyncStorageBackend) -> None:
        """Test async health check when not initialized."""
        result = await backend.health_check()

        assert result is False

    @pytest.mark.asyncio
    async def test_close(
        self, backend: ConcreteAsyncStorageBackend, config: TokenLedgerConfig
    ) -> None:
        """Test async close method."""
        await backend.initialize(config)

        await backend.close()

        assert backend.is_initialized is False
        assert backend._pool is None
        assert backend._close_called is True

    def test_get_info(self, backend: ConcreteAsyncStorageBackend) -> None:
        """Test get_info method for async backend."""
        info = backend.get_info()

        assert isinstance(info, BackendInfo)
        assert info.name == "TestAsyncBackend"
        assert info.version == "2.0.0"
        assert info.driver == "async-test-driver"
        assert info.supports_async is True

    def test_columns_shared(self) -> None:
        """Test that COLUMNS is shared between sync and async backends."""
        assert BaseAsyncStorageBackend.COLUMNS == BaseStorageBackend.COLUMNS
