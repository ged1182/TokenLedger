"""Tests for PostgreSQL storage backend."""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tokenledger.backends.exceptions import (
    BackendNotInitializedError,
    DriverNotFoundError,
    WriteError,
)
from tokenledger.backends.postgresql.storage import PostgreSQLBackend
from tokenledger.backends.protocol import BackendInfo, StorageBackend
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
def mock_psycopg2_connection() -> MagicMock:
    """Create a mock psycopg2 connection."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_cursor.fetchone.return_value = (1,)
    return mock_conn


class TestPostgreSQLBackendProtocol:
    """Tests for PostgreSQLBackend protocol compliance."""

    def test_implements_storage_backend_protocol(self) -> None:
        """Test that PostgreSQLBackend implements StorageBackend protocol."""
        backend = PostgreSQLBackend()
        assert isinstance(backend, StorageBackend)


class TestPostgreSQLBackendProperties:
    """Tests for PostgreSQLBackend properties."""

    def test_name(self) -> None:
        """Test backend name."""
        backend = PostgreSQLBackend()
        assert backend.name == "PostgreSQL"

    def test_is_initialized_default(self) -> None:
        """Test is_initialized is False by default."""
        backend = PostgreSQLBackend()
        assert backend.is_initialized is False

    def test_capabilities(self) -> None:
        """Test backend capabilities."""
        backend = PostgreSQLBackend()
        caps = backend.capabilities

        assert caps.supports_jsonb is True
        assert caps.supports_uuid is True
        assert caps.supports_upsert is True
        assert caps.supports_returning is True
        assert caps.supports_batch_insert is True
        assert caps.supports_window_functions is True
        assert caps.max_batch_size == 10000
        assert caps.recommended_batch_size == 1000


class TestPostgreSQLBackendInitialization:
    """Tests for PostgreSQLBackend initialization."""

    def test_initialize_with_psycopg2(
        self, config: TokenLedgerConfig, mock_psycopg2_connection: MagicMock
    ) -> None:
        """Test initialization with psycopg2 driver."""
        mock_execute_values = MagicMock()

        with (
            patch.dict("sys.modules", {"psycopg2": MagicMock(), "psycopg2.extras": MagicMock()}),
            patch("psycopg2.connect", return_value=mock_psycopg2_connection),
            patch("psycopg2.extras.execute_values", mock_execute_values),
        ):
            backend = PostgreSQLBackend()
            backend.initialize(config)

            assert backend.is_initialized is True
            assert backend._use_psycopg2 is True
            assert backend._driver_name == "psycopg2"

    def test_initialize_with_psycopg3_fallback(self) -> None:
        """Test initialization falls back to psycopg3 when psycopg2 unavailable."""
        # This test is covered by TestPostgreSQLBackendPsycopg3 which has a
        # complete psycopg3 fallback test. This is kept for documentation.
        pass

    def test_initialize_no_driver_raises(self, config: TokenLedgerConfig) -> None:
        """Test initialization raises when no driver is available."""
        # Create a fresh sys.modules without psycopg2/psycopg
        saved_modules = sys.modules.copy()

        # Remove any psycopg references
        modules_to_remove = [
            k
            for k in sys.modules
            if k in ("psycopg2", "psycopg") or k.startswith(("psycopg2.", "psycopg."))
        ]
        for mod in modules_to_remove:
            sys.modules.pop(mod, None)

        # Mock the import to raise ImportError
        original_import = (
            __builtins__["__import__"]
            if isinstance(__builtins__, dict)
            else __builtins__.__import__
        )

        def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name in ("psycopg2", "psycopg"):
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        try:
            with patch("builtins.__import__", side_effect=mock_import):
                backend = PostgreSQLBackend()
                with pytest.raises(DriverNotFoundError) as exc_info:
                    backend.initialize(config)

                assert "psycopg2 or psycopg" in str(exc_info.value)
        finally:
            # Restore original modules
            sys.modules.update(saved_modules)

    def test_initialize_idempotent(
        self, config: TokenLedgerConfig, mock_psycopg2_connection: MagicMock
    ) -> None:
        """Test that initialize is idempotent."""
        mock_execute_values = MagicMock()

        with (
            patch.dict("sys.modules", {"psycopg2": MagicMock(), "psycopg2.extras": MagicMock()}),
            patch("psycopg2.connect", return_value=mock_psycopg2_connection) as mock_connect,
            patch("psycopg2.extras.execute_values", mock_execute_values),
        ):
            backend = PostgreSQLBackend()
            backend.initialize(config)
            backend.initialize(config)  # Should not raise

            # connect should only be called once
            assert mock_connect.call_count == 1


class TestPostgreSQLBackendWriteEvents:
    """Tests for PostgreSQLBackend write_events."""

    def test_write_events_not_initialized(self) -> None:
        """Test write_events raises when not initialized."""
        backend = PostgreSQLBackend()
        events = [{"event_id": "test-123"}]

        with pytest.raises(BackendNotInitializedError):
            backend.write_events(events)

    def test_write_events_empty_list(
        self, config: TokenLedgerConfig, mock_psycopg2_connection: MagicMock
    ) -> None:
        """Test write_events with empty list returns 0."""
        with (
            patch.dict("sys.modules", {"psycopg2": MagicMock(), "psycopg2.extras": MagicMock()}),
            patch("psycopg2.connect", return_value=mock_psycopg2_connection),
            patch("psycopg2.extras.execute_values", MagicMock()),
        ):
            backend = PostgreSQLBackend()
            backend.initialize(config)

            count = backend.write_events([])

            assert count == 0

    def test_write_events_success_psycopg2(
        self, config: TokenLedgerConfig, mock_psycopg2_connection: MagicMock
    ) -> None:
        """Test successful write with psycopg2."""
        mock_execute_values = MagicMock()

        with (
            patch.dict("sys.modules", {"psycopg2": MagicMock(), "psycopg2.extras": MagicMock()}),
            patch("psycopg2.connect", return_value=mock_psycopg2_connection),
            patch("psycopg2.extras.execute_values", mock_execute_values),
        ):
            backend = PostgreSQLBackend()
            backend.initialize(config)

            events = [
                {"event_id": "1", "provider": "openai", "model": "gpt-4"},
                {"event_id": "2", "provider": "anthropic", "model": "claude-3"},
            ]

            count = backend.write_events(events)

            assert count == 2
            mock_execute_values.assert_called_once()

    def test_write_events_failure_rolls_back(
        self, config: TokenLedgerConfig, mock_psycopg2_connection: MagicMock
    ) -> None:
        """Test that write failure rolls back transaction."""
        mock_execute_values = MagicMock(side_effect=Exception("DB Error"))

        with (
            patch.dict("sys.modules", {"psycopg2": MagicMock(), "psycopg2.extras": MagicMock()}),
            patch("psycopg2.connect", return_value=mock_psycopg2_connection),
            patch("psycopg2.extras.execute_values", mock_execute_values),
        ):
            backend = PostgreSQLBackend()
            backend.initialize(config)

            events = [{"event_id": "1"}]

            with pytest.raises(WriteError):
                backend.write_events(events)

            mock_psycopg2_connection.rollback.assert_called_once()


class TestPostgreSQLBackendHealthCheck:
    """Tests for PostgreSQLBackend health_check."""

    def test_health_check_not_initialized(self) -> None:
        """Test health_check returns False when not initialized."""
        backend = PostgreSQLBackend()
        assert backend.health_check() is False

    def test_health_check_success(
        self, config: TokenLedgerConfig, mock_psycopg2_connection: MagicMock
    ) -> None:
        """Test successful health check."""
        with (
            patch.dict("sys.modules", {"psycopg2": MagicMock(), "psycopg2.extras": MagicMock()}),
            patch("psycopg2.connect", return_value=mock_psycopg2_connection),
            patch("psycopg2.extras.execute_values", MagicMock()),
        ):
            backend = PostgreSQLBackend()
            backend.initialize(config)

            result = backend.health_check()

            assert result is True


class TestPostgreSQLBackendClose:
    """Tests for PostgreSQLBackend close."""

    def test_close_without_initialize(self) -> None:
        """Test close without initialization doesn't raise."""
        backend = PostgreSQLBackend()
        backend.close()  # Should not raise

    def test_close_clears_state(
        self, config: TokenLedgerConfig, mock_psycopg2_connection: MagicMock
    ) -> None:
        """Test close clears backend state."""
        with (
            patch.dict("sys.modules", {"psycopg2": MagicMock(), "psycopg2.extras": MagicMock()}),
            patch("psycopg2.connect", return_value=mock_psycopg2_connection),
            patch("psycopg2.extras.execute_values", MagicMock()),
        ):
            backend = PostgreSQLBackend()
            backend.initialize(config)
            assert backend.is_initialized is True

            backend.close()

            assert backend.is_initialized is False
            mock_psycopg2_connection.close.assert_called_once()


class TestPostgreSQLBackendGetInfo:
    """Tests for PostgreSQLBackend get_info."""

    def test_get_info_returns_backend_info(self) -> None:
        """Test get_info returns BackendInfo."""
        backend = PostgreSQLBackend()
        info = backend.get_info()

        assert isinstance(info, BackendInfo)
        assert info.name == "PostgreSQL"
        assert info.supports_async is False
        assert info.capabilities.supports_jsonb is True


class TestPostgreSQLBackendPsycopg3:
    """Tests for PostgreSQLBackend with psycopg3."""

    @pytest.fixture
    def mock_psycopg3_connection(self) -> MagicMock:
        """Create a mock psycopg3 connection."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchone.return_value = (1,)
        return mock_conn

    def test_initialize_with_psycopg3_when_psycopg2_unavailable(
        self, config: TokenLedgerConfig, mock_psycopg3_connection: MagicMock
    ) -> None:
        """Test initialization with psycopg3 when psycopg2 is not available."""
        # Create mock psycopg module
        mock_psycopg = MagicMock()
        mock_psycopg.connect = MagicMock(return_value=mock_psycopg3_connection)
        mock_psycopg.__version__ = "3.1.0"

        # Save original import
        original_import = (
            __builtins__["__import__"]
            if isinstance(__builtins__, dict)
            else __builtins__.__import__
        )

        def custom_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "psycopg2":
                raise ImportError("No module named 'psycopg2'")
            if name == "psycopg":
                return mock_psycopg
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=custom_import):
            backend = PostgreSQLBackend()
            backend.initialize(config)

            assert backend.is_initialized is True
            assert backend._use_psycopg2 is False
            assert backend._driver_name == "psycopg"
            assert backend._driver_version == "3.1.0"

    def test_write_events_with_psycopg3(
        self, config: TokenLedgerConfig, mock_psycopg3_connection: MagicMock
    ) -> None:
        """Test write_events uses executemany with psycopg3."""
        mock_psycopg = MagicMock()
        mock_psycopg.connect = MagicMock(return_value=mock_psycopg3_connection)
        mock_psycopg.__version__ = "3.1.0"

        mock_cursor = mock_psycopg3_connection.cursor.return_value.__enter__.return_value

        original_import = (
            __builtins__["__import__"]
            if isinstance(__builtins__, dict)
            else __builtins__.__import__
        )

        def custom_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "psycopg2":
                raise ImportError("No module named 'psycopg2'")
            if name == "psycopg":
                return mock_psycopg
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=custom_import):
            backend = PostgreSQLBackend()
            backend.initialize(config)

            events = [
                {"event_id": "1", "provider": "openai", "model": "gpt-4"},
            ]
            count = backend.write_events(events)

            assert count == 1
            mock_cursor.executemany.assert_called_once()


class TestPostgreSQLBackendReconnect:
    """Tests for PostgreSQLBackend reconnect."""

    def test_reconnect(
        self, config: TokenLedgerConfig, mock_psycopg2_connection: MagicMock
    ) -> None:
        """Test reconnect closes and re-establishes connection."""
        mock_execute_values = MagicMock()
        new_connection = MagicMock()
        new_cursor = MagicMock()
        new_connection.cursor.return_value.__enter__ = MagicMock(return_value=new_cursor)
        new_connection.cursor.return_value.__exit__ = MagicMock(return_value=False)

        connections = [mock_psycopg2_connection, new_connection]
        connect_call_count = [0]

        def mock_connect(*_args: Any, **_kwargs: Any) -> MagicMock:
            conn = connections[connect_call_count[0]]
            connect_call_count[0] = min(connect_call_count[0] + 1, 1)
            return conn

        with (
            patch.dict("sys.modules", {"psycopg2": MagicMock(), "psycopg2.extras": MagicMock()}),
            patch("psycopg2.connect", side_effect=mock_connect),
            patch("psycopg2.extras.execute_values", mock_execute_values),
        ):
            backend = PostgreSQLBackend()
            backend.initialize(config)

            assert backend._connection is mock_psycopg2_connection

            backend.reconnect()

            mock_psycopg2_connection.close.assert_called_once()
            assert backend._connection is new_connection

    def test_reconnect_handles_close_error(
        self, config: TokenLedgerConfig, mock_psycopg2_connection: MagicMock
    ) -> None:
        """Test reconnect handles error when closing connection."""
        mock_execute_values = MagicMock()
        mock_psycopg2_connection.close.side_effect = Exception("Close error")
        new_connection = MagicMock()

        connections = [mock_psycopg2_connection, new_connection]
        connect_call_count = [0]

        def mock_connect(*_args: Any, **_kwargs: Any) -> MagicMock:
            conn = connections[connect_call_count[0]]
            connect_call_count[0] = min(connect_call_count[0] + 1, 1)
            return conn

        with (
            patch.dict("sys.modules", {"psycopg2": MagicMock(), "psycopg2.extras": MagicMock()}),
            patch("psycopg2.connect", side_effect=mock_connect),
            patch("psycopg2.extras.execute_values", mock_execute_values),
        ):
            backend = PostgreSQLBackend()
            backend.initialize(config)

            # Should not raise even if close fails
            backend.reconnect()

            assert backend._connection is new_connection


class TestPostgreSQLBackendGetVersion:
    """Tests for PostgreSQLBackend version methods."""

    def test_get_driver_name(
        self, config: TokenLedgerConfig, mock_psycopg2_connection: MagicMock
    ) -> None:
        """Test _get_driver_name returns correct driver info."""
        mock_execute_values = MagicMock()
        mock_psycopg2 = MagicMock()
        mock_psycopg2.__version__ = "2.9.9"

        with (
            patch.dict("sys.modules", {"psycopg2": mock_psycopg2, "psycopg2.extras": MagicMock()}),
            patch("psycopg2.connect", return_value=mock_psycopg2_connection),
            patch("psycopg2.extras.execute_values", mock_execute_values),
        ):
            backend = PostgreSQLBackend()
            backend.initialize(config)

            driver_name = backend._get_driver_name()

            assert "psycopg2" in driver_name

    def test_get_version_returns_server_version(
        self, config: TokenLedgerConfig, mock_psycopg2_connection: MagicMock
    ) -> None:
        """Test _get_version returns PostgreSQL server version."""
        mock_cursor = mock_psycopg2_connection.cursor.return_value.__enter__.return_value
        mock_cursor.fetchone.return_value = ("PostgreSQL 15.4 on x86_64",)

        with (
            patch.dict("sys.modules", {"psycopg2": MagicMock(), "psycopg2.extras": MagicMock()}),
            patch("psycopg2.connect", return_value=mock_psycopg2_connection),
            patch("psycopg2.extras.execute_values", MagicMock()),
        ):
            backend = PostgreSQLBackend()
            backend.initialize(config)

            version = backend._get_version()

            assert version == "15.4"

    def test_get_version_returns_unknown_when_not_connected(self) -> None:
        """Test _get_version returns unknown when not connected."""
        backend = PostgreSQLBackend()
        assert backend._get_version() == "unknown"

    def test_get_version_returns_unknown_on_error(
        self, config: TokenLedgerConfig, mock_psycopg2_connection: MagicMock
    ) -> None:
        """Test _get_version returns unknown on query error."""
        mock_cursor = mock_psycopg2_connection.cursor.return_value.__enter__.return_value
        mock_cursor.execute.side_effect = Exception("Query error")

        with (
            patch.dict("sys.modules", {"psycopg2": MagicMock(), "psycopg2.extras": MagicMock()}),
            patch("psycopg2.connect", return_value=mock_psycopg2_connection),
            patch("psycopg2.extras.execute_values", MagicMock()),
        ):
            backend = PostgreSQLBackend()
            backend.initialize(config, create_schema=False)

            version = backend._get_version()

            assert version == "unknown"


class TestPostgreSQLBackendHealthCheckEdgeCases:
    """Additional health check tests."""

    def test_health_check_returns_false_on_error(
        self, config: TokenLedgerConfig, mock_psycopg2_connection: MagicMock
    ) -> None:
        """Test health_check returns False when query fails."""
        mock_cursor = mock_psycopg2_connection.cursor.return_value.__enter__.return_value
        mock_cursor.execute.side_effect = Exception("Connection lost")

        with (
            patch.dict("sys.modules", {"psycopg2": MagicMock(), "psycopg2.extras": MagicMock()}),
            patch("psycopg2.connect", return_value=mock_psycopg2_connection),
            patch("psycopg2.extras.execute_values", MagicMock()),
        ):
            backend = PostgreSQLBackend()
            backend.initialize(config, create_schema=False)

            result = backend.health_check()

            assert result is False

    def test_health_check_returns_false_on_wrong_result(
        self, config: TokenLedgerConfig, mock_psycopg2_connection: MagicMock
    ) -> None:
        """Test health_check returns False when result is not 1."""
        mock_cursor = mock_psycopg2_connection.cursor.return_value.__enter__.return_value
        mock_cursor.fetchone.return_value = (0,)  # Not 1

        with (
            patch.dict("sys.modules", {"psycopg2": MagicMock(), "psycopg2.extras": MagicMock()}),
            patch("psycopg2.connect", return_value=mock_psycopg2_connection),
            patch("psycopg2.extras.execute_values", MagicMock()),
        ):
            backend = PostgreSQLBackend()
            backend.initialize(config, create_schema=False)

            result = backend.health_check()

            assert result is False

    def test_health_check_returns_false_on_none_result(
        self, config: TokenLedgerConfig, mock_psycopg2_connection: MagicMock
    ) -> None:
        """Test health_check returns False when result is None."""
        mock_cursor = mock_psycopg2_connection.cursor.return_value.__enter__.return_value
        mock_cursor.fetchone.return_value = None

        with (
            patch.dict("sys.modules", {"psycopg2": MagicMock(), "psycopg2.extras": MagicMock()}),
            patch("psycopg2.connect", return_value=mock_psycopg2_connection),
            patch("psycopg2.extras.execute_values", MagicMock()),
        ):
            backend = PostgreSQLBackend()
            backend.initialize(config, create_schema=False)

            result = backend.health_check()

            assert result is False
