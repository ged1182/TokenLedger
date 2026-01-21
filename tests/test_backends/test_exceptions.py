"""Tests for backend exceptions."""

from __future__ import annotations

from tokenledger.backends.exceptions import (
    BackendError,
    BackendNotFoundError,
    BackendNotInitializedError,
    ConnectionError,
    DriverNotFoundError,
    InitializationError,
    ReadError,
    SchemaError,
    WriteError,
)


class TestBackendError:
    """Tests for BackendError base exception."""

    def test_basic_error(self) -> None:
        """Test basic error without backend name."""
        error = BackendError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.backend_name is None

    def test_error_with_backend_name(self) -> None:
        """Test error with backend name prefixed."""
        error = BackendError("Connection failed", backend_name="PostgreSQL")
        assert str(error) == "[PostgreSQL] Connection failed"
        assert error.backend_name == "PostgreSQL"

    def test_error_inheritance(self) -> None:
        """Test that BackendError inherits from Exception."""
        error = BackendError("test")
        assert isinstance(error, Exception)


class TestConnectionError:
    """Tests for ConnectionError."""

    def test_connection_error(self) -> None:
        """Test ConnectionError creation."""
        error = ConnectionError("Cannot connect to database", backend_name="PostgreSQL")
        assert "Cannot connect to database" in str(error)
        assert error.backend_name == "PostgreSQL"

    def test_connection_error_inheritance(self) -> None:
        """Test that ConnectionError inherits from BackendError."""
        error = ConnectionError("test")
        assert isinstance(error, BackendError)


class TestInitializationError:
    """Tests for InitializationError."""

    def test_initialization_error(self) -> None:
        """Test InitializationError creation."""
        error = InitializationError("Failed to initialize pool", backend_name="asyncpg")
        assert "Failed to initialize pool" in str(error)
        assert error.backend_name == "asyncpg"


class TestSchemaError:
    """Tests for SchemaError."""

    def test_schema_error(self) -> None:
        """Test SchemaError creation."""
        error = SchemaError("Table already exists", backend_name="PostgreSQL")
        assert "Table already exists" in str(error)


class TestWriteError:
    """Tests for WriteError."""

    def test_write_error_basic(self) -> None:
        """Test WriteError with basic parameters."""
        error = WriteError("Insert failed", backend_name="PostgreSQL")
        assert "Insert failed" in str(error)
        assert error.events_count == 0
        assert error.events_written == 0

    def test_write_error_with_counts(self) -> None:
        """Test WriteError with event counts."""
        error = WriteError(
            "Partial write failure",
            backend_name="ClickHouse",
            events_count=100,
            events_written=50,
        )
        assert error.events_count == 100
        assert error.events_written == 50


class TestReadError:
    """Tests for ReadError."""

    def test_read_error(self) -> None:
        """Test ReadError creation."""
        error = ReadError("Query timed out", backend_name="BigQuery")
        assert "Query timed out" in str(error)


class TestBackendNotFoundError:
    """Tests for BackendNotFoundError."""

    def test_not_found_error(self) -> None:
        """Test BackendNotFoundError with available backends."""
        error = BackendNotFoundError(
            "unknown_db",
            available_backends=["postgresql", "clickhouse", "bigquery"],
        )
        assert "unknown_db" in str(error)
        assert "bigquery" in str(error)
        assert "clickhouse" in str(error)
        assert "postgresql" in str(error)
        assert error.available_backends == ["postgresql", "clickhouse", "bigquery"]

    def test_not_found_error_no_available(self) -> None:
        """Test BackendNotFoundError with no available backends."""
        error = BackendNotFoundError("mydb")
        assert "mydb" in str(error)
        assert "none" in str(error)
        assert error.available_backends == []


class TestBackendNotInitializedError:
    """Tests for BackendNotInitializedError."""

    def test_not_initialized_error(self) -> None:
        """Test BackendNotInitializedError creation."""
        error = BackendNotInitializedError(backend_name="PostgreSQL")
        assert "not initialized" in str(error).lower()
        assert "initialize()" in str(error)

    def test_not_initialized_error_no_name(self) -> None:
        """Test BackendNotInitializedError without backend name."""
        error = BackendNotInitializedError()
        assert "not initialized" in str(error).lower()


class TestDriverNotFoundError:
    """Tests for DriverNotFoundError."""

    def test_driver_not_found_basic(self) -> None:
        """Test DriverNotFoundError without install hint."""
        error = DriverNotFoundError("psycopg2", backend_name="PostgreSQL")
        assert "psycopg2" in str(error)
        assert error.driver_name == "psycopg2"
        assert error.install_hint is None

    def test_driver_not_found_with_hint(self) -> None:
        """Test DriverNotFoundError with install hint."""
        error = DriverNotFoundError(
            "asyncpg",
            install_hint="pip install tokenledger[asyncpg]",
            backend_name="PostgreSQL",
        )
        assert "asyncpg" in str(error)
        assert "pip install" in str(error)
        assert error.install_hint == "pip install tokenledger[asyncpg]"
