"""Tests for backend protocols."""

from __future__ import annotations

from typing import Any

import pytest

from tokenledger.backends.protocol import (
    AsyncQueryBackend,
    AsyncStorageBackend,
    BackendCapabilities,
    BackendInfo,
    QueryBackend,
    StorageBackend,
)


class TestBackendCapabilities:
    """Tests for BackendCapabilities dataclass."""

    def test_default_values(self) -> None:
        """Test that default values are set correctly."""
        caps = BackendCapabilities()

        assert caps.supports_jsonb is False
        assert caps.supports_uuid is True
        assert caps.supports_decimal is True
        assert caps.supports_upsert is True
        assert caps.supports_returning is False
        assert caps.supports_batch_insert is True
        assert caps.supports_window_functions is True
        assert caps.supports_cte is True
        assert caps.supports_percentile is True
        assert caps.max_batch_size == 10000
        assert caps.recommended_batch_size == 1000
        assert caps.supports_streaming is False
        assert caps.supports_compression is False
        assert caps.supports_partitioning is False

    def test_custom_values(self) -> None:
        """Test that custom values override defaults."""
        caps = BackendCapabilities(
            supports_jsonb=True,
            supports_upsert=False,
            max_batch_size=50000,
            recommended_batch_size=5000,
            supports_compression=True,
        )

        assert caps.supports_jsonb is True
        assert caps.supports_upsert is False
        assert caps.max_batch_size == 50000
        assert caps.recommended_batch_size == 5000
        assert caps.supports_compression is True

    def test_immutable(self) -> None:
        """Test that BackendCapabilities is immutable (frozen)."""
        caps = BackendCapabilities()
        with pytest.raises(AttributeError):
            caps.supports_jsonb = True  # type: ignore[misc]


class TestBackendInfo:
    """Tests for BackendInfo dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        info = BackendInfo(name="Test")

        assert info.name == "Test"
        assert info.version == "unknown"
        assert info.driver == "unknown"
        assert info.supports_async is False
        assert isinstance(info.capabilities, BackendCapabilities)

    def test_custom_values(self) -> None:
        """Test custom values."""
        caps = BackendCapabilities(supports_jsonb=True)
        info = BackendInfo(
            name="PostgreSQL",
            version="15.4",
            driver="psycopg2",
            supports_async=False,
            capabilities=caps,
        )

        assert info.name == "PostgreSQL"
        assert info.version == "15.4"
        assert info.driver == "psycopg2"
        assert info.supports_async is False
        assert info.capabilities.supports_jsonb is True

    def test_to_dict(self) -> None:
        """Test to_dict conversion."""
        caps = BackendCapabilities(supports_jsonb=True, max_batch_size=5000)
        info = BackendInfo(
            name="PostgreSQL",
            version="15.4",
            driver="asyncpg",
            supports_async=True,
            capabilities=caps,
        )

        d = info.to_dict()

        assert d["name"] == "PostgreSQL"
        assert d["version"] == "15.4"
        assert d["driver"] == "asyncpg"
        assert d["supports_async"] is True
        assert d["capabilities"]["supports_jsonb"] is True
        assert d["capabilities"]["max_batch_size"] == 5000

    def test_immutable(self) -> None:
        """Test that BackendInfo is immutable (frozen)."""
        info = BackendInfo(name="Test")
        with pytest.raises(AttributeError):
            info.name = "Changed"  # type: ignore[misc]


class TestStorageBackendProtocol:
    """Tests for StorageBackend protocol."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """Test that StorageBackend is runtime checkable."""

        # Create a class that implements the protocol
        class MockBackend:
            @property
            def name(self) -> str:
                return "Mock"

            @property
            def is_initialized(self) -> bool:
                return True

            def initialize(self, config: Any, create_schema: bool = True) -> None:
                pass

            def write_events(self, events: list[dict[str, Any]]) -> int:
                return len(events)

            def health_check(self) -> bool:
                return True

            def close(self) -> None:
                pass

            def get_info(self) -> BackendInfo:
                return BackendInfo(name="Mock")

        backend = MockBackend()
        assert isinstance(backend, StorageBackend)

    def test_incomplete_implementation_not_protocol(self) -> None:
        """Test that incomplete implementation doesn't satisfy protocol."""

        class IncompleteBackend:
            @property
            def name(self) -> str:
                return "Incomplete"

            # Missing other required methods

        backend = IncompleteBackend()
        assert not isinstance(backend, StorageBackend)


class TestAsyncStorageBackendProtocol:
    """Tests for AsyncStorageBackend protocol."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """Test that AsyncStorageBackend is runtime checkable."""

        class MockAsyncBackend:
            @property
            def name(self) -> str:
                return "MockAsync"

            @property
            def is_initialized(self) -> bool:
                return True

            async def initialize(self, config: Any, create_schema: bool = True) -> None:
                pass

            async def write_events(self, events: list[dict[str, Any]]) -> int:
                return len(events)

            async def health_check(self) -> bool:
                return True

            async def close(self) -> None:
                pass

            def get_info(self) -> BackendInfo:
                return BackendInfo(name="MockAsync", supports_async=True)

        backend = MockAsyncBackend()
        assert isinstance(backend, AsyncStorageBackend)


class TestQueryBackendProtocol:
    """Tests for QueryBackend protocol."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """Test that QueryBackend is runtime checkable."""

        class MockQueryBackend:
            @property
            def name(self) -> str:
                return "MockQuery"

            @property
            def is_initialized(self) -> bool:
                return True

            def initialize(self, config: Any) -> None:
                pass

            def get_cost_summary(
                self,
                days: int = 30,
                user_id: str | None = None,
                model: str | None = None,
                app_name: str | None = None,
            ) -> dict[str, Any]:
                return {}

            def get_costs_by_model(
                self,
                days: int = 30,
                limit: int = 10,
            ) -> list[dict[str, Any]]:
                return []

            def get_costs_by_user(
                self,
                days: int = 30,
                limit: int = 20,
            ) -> list[dict[str, Any]]:
                return []

            def get_daily_costs(
                self,
                days: int = 30,
                user_id: str | None = None,
            ) -> list[dict[str, Any]]:
                return []

            def close(self) -> None:
                pass

        backend = MockQueryBackend()
        assert isinstance(backend, QueryBackend)


class TestAsyncQueryBackendProtocol:
    """Tests for AsyncQueryBackend protocol."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """Test that AsyncQueryBackend is runtime checkable."""

        class MockAsyncQueryBackend:
            @property
            def name(self) -> str:
                return "MockAsyncQuery"

            @property
            def is_initialized(self) -> bool:
                return True

            async def initialize(self, config: Any) -> None:
                pass

            async def get_cost_summary(
                self,
                days: int = 30,
                user_id: str | None = None,
                model: str | None = None,
                app_name: str | None = None,
            ) -> dict[str, Any]:
                return {}

            async def get_costs_by_model(
                self,
                days: int = 30,
                limit: int = 10,
            ) -> list[dict[str, Any]]:
                return []

            async def get_costs_by_user(
                self,
                days: int = 30,
                limit: int = 20,
            ) -> list[dict[str, Any]]:
                return []

            async def get_daily_costs(
                self,
                days: int = 30,
                user_id: str | None = None,
            ) -> list[dict[str, Any]]:
                return []

            async def close(self) -> None:
                pass

        backend = MockAsyncQueryBackend()
        assert isinstance(backend, AsyncQueryBackend)
