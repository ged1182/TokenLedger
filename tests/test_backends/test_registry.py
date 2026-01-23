"""Tests for backend registry."""

from __future__ import annotations

from typing import Any

import pytest

from tokenledger.backends import registry
from tokenledger.backends.exceptions import BackendNotFoundError
from tokenledger.backends.protocol import BackendInfo, StorageBackend
from tokenledger.config import TokenLedgerConfig


class MockStorageBackend:
    """Mock storage backend for testing registry."""

    def __init__(self) -> None:
        self._initialized = False

    @property
    def name(self) -> str:
        return "MockBackend"

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def initialize(self, config: TokenLedgerConfig, create_schema: bool = True) -> None:
        self._initialized = True

    def write_events(self, events: list[dict[str, Any]]) -> int:
        return len(events)

    def health_check(self) -> bool:
        return True

    def close(self) -> None:
        self._initialized = False

    def get_info(self) -> BackendInfo:
        return BackendInfo(name="MockBackend")


class TestRegistryFunctions:
    """Tests for registry functions."""

    @pytest.fixture(autouse=True)
    def cleanup_registry(self) -> None:
        """Clean up registered backends before and after each test."""
        registry.clear_registered_backends()
        yield
        registry.clear_registered_backends()

    def test_register_storage_backend(self) -> None:
        """Test registering a storage backend."""
        registry.register_storage_backend(
            "mock", "tests.test_backends.test_registry", "MockStorageBackend"
        )

        available = registry.get_available_storage_backends()
        assert "mock" in available
        assert "MockStorageBackend" in available["mock"]

    def test_register_query_backend(self) -> None:
        """Test registering a query backend."""
        registry.register_query_backend(
            "mock", "tests.test_backends.test_registry", "MockStorageBackend"
        )

        available = registry.get_available_query_backends()
        assert "mock" in available

    def test_unregister_storage_backend(self) -> None:
        """Test unregistering a storage backend."""
        registry.register_storage_backend(
            "mock", "tests.test_backends.test_registry", "MockStorageBackend"
        )

        result = registry.unregister_storage_backend("mock")

        assert result is True
        available = registry.get_available_storage_backends()
        assert "mock" not in available

    def test_unregister_nonexistent_backend(self) -> None:
        """Test unregistering a backend that doesn't exist."""
        result = registry.unregister_storage_backend("nonexistent")
        assert result is False

    def test_unregister_query_backend(self) -> None:
        """Test unregistering a query backend."""
        registry.register_query_backend(
            "mock", "tests.test_backends.test_registry", "MockStorageBackend"
        )

        result = registry.unregister_query_backend("mock")

        assert result is True
        available = registry.get_available_query_backends()
        assert "mock" not in available


class TestGetAvailableBackends:
    """Tests for get_available_*_backends functions."""

    @pytest.fixture(autouse=True)
    def cleanup_registry(self) -> None:
        """Clean up registered backends."""
        registry.clear_registered_backends()
        yield
        registry.clear_registered_backends()

    def test_get_available_storage_backends_includes_builtin(self) -> None:
        """Test that built-in backends are always available."""
        available = registry.get_available_storage_backends()

        assert "postgresql" in available
        assert "postgres" in available
        assert "asyncpg" in available

    def test_get_available_query_backends_includes_builtin(self) -> None:
        """Test that built-in query backends are always available."""
        available = registry.get_available_query_backends()

        assert "postgresql" in available
        assert "postgres" in available

    def test_get_available_storage_backends_includes_registered(self) -> None:
        """Test that registered backends are included."""
        registry.register_storage_backend("custom", "some.module", "CustomBackend")

        available = registry.get_available_storage_backends()

        assert "custom" in available
        assert "postgresql" in available  # Built-in still present


class TestLoadBackend:
    """Tests for load_*_backend functions."""

    @pytest.fixture(autouse=True)
    def cleanup_registry(self) -> None:
        """Clean up registered backends."""
        registry.clear_registered_backends()
        yield
        registry.clear_registered_backends()

    def test_load_storage_backend_not_found(self) -> None:
        """Test loading a non-existent backend."""
        with pytest.raises(BackendNotFoundError) as exc_info:
            registry.load_storage_backend("nonexistent_db")

        assert "nonexistent_db" in str(exc_info.value)
        assert "postgresql" in str(exc_info.value)  # Should list available

    def test_load_query_backend_not_found(self) -> None:
        """Test loading a non-existent query backend."""
        with pytest.raises(BackendNotFoundError) as exc_info:
            registry.load_query_backend("nonexistent_db")

        assert "nonexistent_db" in str(exc_info.value)

    def test_load_registered_backend(self) -> None:
        """Test loading a registered backend."""
        registry.register_storage_backend(
            "mock", "tests.test_backends.test_registry", "MockStorageBackend"
        )

        backend_class = registry.load_storage_backend("mock")

        assert backend_class == MockStorageBackend
        # Verify it implements the protocol
        backend = backend_class()
        assert isinstance(backend, StorageBackend)


class TestParseDatabaseUrl:
    """Tests for parse_database_url function."""

    def test_postgresql_url(self) -> None:
        """Test parsing a standard PostgreSQL URL."""
        backend, url = registry.parse_database_url("postgresql://user:pass@localhost:5432/mydb")

        assert backend == "postgresql"
        assert url == "postgresql://user:pass@localhost:5432/mydb"

    def test_postgres_url(self) -> None:
        """Test parsing a postgres:// URL."""
        backend, url = registry.parse_database_url("postgres://user:pass@localhost/mydb")

        assert backend == "postgresql"
        assert url == "postgres://user:pass@localhost/mydb"

    def test_tokenledger_prefixed_url(self) -> None:
        """Test parsing a tokenledger+ prefixed URL."""
        backend, url = registry.parse_database_url(
            "tokenledger+clickhouse://localhost:9000/default"
        )

        assert backend == "clickhouse"
        assert url == "clickhouse://localhost:9000/default"

    def test_tokenledger_postgresql_url(self) -> None:
        """Test parsing a tokenledger+postgresql URL."""
        backend, url = registry.parse_database_url("tokenledger+postgresql://localhost/mydb")

        assert backend == "postgresql"
        assert url == "postgresql://localhost/mydb"

    def test_bigquery_url(self) -> None:
        """Test parsing a BigQuery URL."""
        backend, url = registry.parse_database_url("bigquery://project/dataset")

        assert backend == "bigquery"
        assert url == "bigquery://project/dataset"

    def test_snowflake_url(self) -> None:
        """Test parsing a Snowflake URL."""
        backend, url = registry.parse_database_url("snowflake://account/database")

        assert backend == "snowflake"
        assert url == "snowflake://account/database"

    def test_unknown_scheme(self) -> None:
        """Test parsing an unknown URL scheme."""
        backend, url = registry.parse_database_url("unknown://localhost/db")

        assert backend == "unknown"
        assert url == "unknown://localhost/db"


class TestClearRegisteredBackends:
    """Tests for clear_registered_backends function."""

    def test_clear_registered_backends(self) -> None:
        """Test clearing all registered backends."""
        registry.register_storage_backend("test1", "module", "Class1")
        registry.register_query_backend("test2", "module", "Class2")

        registry.clear_registered_backends()

        # Registered backends should be gone, but built-in should remain
        storage = registry.get_available_storage_backends()
        query = registry.get_available_query_backends()

        assert "test1" not in storage
        assert "test2" not in query
        assert "postgresql" in storage  # Built-in still present
        assert "postgresql" in query


class TestCreateStorageBackend:
    """Tests for create_storage_backend function."""

    @pytest.fixture(autouse=True)
    def cleanup_registry(self) -> None:
        """Clean up registered backends."""
        registry.clear_registered_backends()
        yield
        registry.clear_registered_backends()

    @pytest.fixture
    def config(self) -> TokenLedgerConfig:
        """Create a test configuration."""
        return TokenLedgerConfig(
            database_url="postgresql://test:test@localhost/test",
            app_name="test-app",
        )

    def test_create_storage_backend_registered(self, config: TokenLedgerConfig) -> None:
        """Test creating a registered backend."""
        registry.register_storage_backend(
            "mock", "tests.test_backends.test_registry", "MockStorageBackend"
        )

        backend = registry.create_storage_backend("mock", config)

        assert isinstance(backend, StorageBackend)
        assert backend.is_initialized is True
        assert backend.name == "MockBackend"

    def test_create_storage_backend_not_found(self, config: TokenLedgerConfig) -> None:
        """Test creating a non-existent backend."""
        with pytest.raises(BackendNotFoundError):
            registry.create_storage_backend("nonexistent", config)


class TestDriverNotFoundError:
    """Tests for DriverNotFoundError handling."""

    @pytest.fixture(autouse=True)
    def cleanup_registry(self):
        """Clean up registered backends."""
        registry.clear_registered_backends()
        yield
        registry.clear_registered_backends()

    def test_load_storage_backend_driver_not_found(self) -> None:
        """Test that loading a backend with missing driver raises DriverNotFoundError."""
        from tokenledger.backends.exceptions import DriverNotFoundError

        # Register a backend pointing to a non-existent module
        registry.register_storage_backend(
            "fake_backend", "nonexistent.module.that.does.not.exist", "FakeBackend"
        )

        with pytest.raises(DriverNotFoundError) as exc_info:
            registry.load_storage_backend("fake_backend")

        # Should provide install hint
        assert exc_info.value.install_hint is not None
        assert "fake_backend" in exc_info.value.backend_name

    def test_load_query_backend_driver_not_found(self) -> None:
        """Test that loading a query backend with missing driver raises DriverNotFoundError."""
        from tokenledger.backends.exceptions import DriverNotFoundError

        # Register a backend pointing to a non-existent module
        registry.register_query_backend(
            "fake_query_backend", "nonexistent.module.path", "FakeQueryBackend"
        )

        with pytest.raises(DriverNotFoundError) as exc_info:
            registry.load_query_backend("fake_query_backend")

        assert exc_info.value.backend_name == "fake_query_backend"


class TestEntryPointDiscovery:
    """Tests for entry point discovery error handling."""

    def test_get_available_backends_handles_entry_point_errors(self) -> None:
        """Test that entry point discovery errors are handled gracefully."""
        from unittest.mock import patch

        # Mock entry_points to raise an exception
        with patch.object(registry, "entry_points", side_effect=Exception("Test error")):
            # Should not raise, just log a warning
            storage_backends = registry.get_available_storage_backends()
            query_backends = registry.get_available_query_backends()

            # Built-in backends should still be available
            assert "postgresql" in storage_backends
            assert "postgresql" in query_backends
