"""
TokenLedger Backend Protocols

Defines the contracts for pluggable storage and query backends using Python Protocols (PEP 544).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from tokenledger.config import TokenLedgerConfig


@dataclass(frozen=True)
class BackendCapabilities:
    """
    Describes the capabilities of a database backend.

    Used for feature detection and query optimization.
    """

    # Data type support
    supports_jsonb: bool = False
    supports_uuid: bool = True
    supports_decimal: bool = True

    # Write capabilities
    supports_upsert: bool = True
    supports_returning: bool = False
    supports_batch_insert: bool = True

    # Query capabilities
    supports_window_functions: bool = True
    supports_cte: bool = True
    supports_percentile: bool = True

    # Performance characteristics
    max_batch_size: int = 10000
    recommended_batch_size: int = 1000

    # Advanced features
    supports_streaming: bool = False
    supports_compression: bool = False
    supports_partitioning: bool = False


@dataclass(frozen=True)
class BackendInfo:
    """Information about a backend for introspection and debugging."""

    name: str
    version: str = "unknown"
    driver: str = "unknown"
    supports_async: bool = False
    capabilities: BackendCapabilities = field(default_factory=BackendCapabilities)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "driver": self.driver,
            "supports_async": self.supports_async,
            "capabilities": {
                "supports_jsonb": self.capabilities.supports_jsonb,
                "supports_upsert": self.capabilities.supports_upsert,
                "supports_batch_insert": self.capabilities.supports_batch_insert,
                "max_batch_size": self.capabilities.max_batch_size,
                "recommended_batch_size": self.capabilities.recommended_batch_size,
            },
        }


@runtime_checkable
class StorageBackend(Protocol):
    """
    Protocol defining the interface for TokenLedger storage backends.

    All storage backends must implement these methods. The Protocol approach allows
    third-party implementations without requiring inheritance from a base class.

    Example:
        >>> class MyCustomBackend:
        ...     @property
        ...     def name(self) -> str:
        ...         return "MyBackend"
        ...     @property
        ...     def is_initialized(self) -> bool:
        ...         return self._initialized
        ...     def initialize(self, config, create_schema=True) -> None: ...
        ...     def write_events(self, events) -> int: ...
        ...     def health_check(self) -> bool: ...
        ...     def close(self) -> None: ...
        ...     def get_info(self) -> BackendInfo: ...
        >>>
        >>> assert isinstance(MyCustomBackend(), StorageBackend)  # True via structural subtyping
    """

    @property
    def name(self) -> str:
        """Human-readable name of the backend (e.g., 'PostgreSQL', 'ClickHouse')."""
        ...

    @property
    def is_initialized(self) -> bool:
        """Whether the backend has been initialized."""
        ...

    def initialize(
        self,
        config: TokenLedgerConfig,
        create_schema: bool = True,
    ) -> None:
        """
        Initialize the database backend.

        Args:
            config: TokenLedger configuration with connection details
            create_schema: Whether to create tables/schema if they don't exist

        Raises:
            ConnectionError: If unable to connect to the database
            DriverNotFoundError: If required driver package is not installed
            SchemaError: If schema creation fails
        """
        ...

    def write_events(self, events: list[dict[str, Any]]) -> int:
        """
        Write a batch of events to the storage backend.

        Args:
            events: List of event dictionaries matching the LLMEvent schema

        Returns:
            Number of events successfully written

        Raises:
            BackendNotInitializedError: If the backend is not initialized
            WriteError: If the write operation fails
        """
        ...

    def health_check(self) -> bool:
        """
        Check if the database connection is healthy.

        Returns:
            True if connection is healthy, False otherwise
        """
        ...

    def close(self) -> None:
        """
        Close the database connection and release resources.

        Should be idempotent (safe to call multiple times).
        """
        ...

    def get_info(self) -> BackendInfo:
        """
        Get information about the backend.

        Returns:
            BackendInfo with name, version, driver, and capabilities
        """
        ...


@runtime_checkable
class AsyncStorageBackend(Protocol):
    """
    Protocol for async-capable storage backends.

    Provides async versions of all storage methods.
    """

    @property
    def name(self) -> str:
        """Human-readable name of the backend."""
        ...

    @property
    def is_initialized(self) -> bool:
        """Whether the backend has been initialized."""
        ...

    async def initialize(
        self,
        config: TokenLedgerConfig,
        create_schema: bool = True,
    ) -> None:
        """
        Initialize the async database backend.

        Args:
            config: TokenLedger configuration with connection details
            create_schema: Whether to create tables/schema if they don't exist

        Raises:
            ConnectionError: If unable to connect to the database
            DriverNotFoundError: If required driver package is not installed
            SchemaError: If schema creation fails
        """
        ...

    async def write_events(self, events: list[dict[str, Any]]) -> int:
        """
        Write a batch of events asynchronously.

        Args:
            events: List of event dictionaries matching the LLMEvent schema

        Returns:
            Number of events successfully written

        Raises:
            BackendNotInitializedError: If the backend is not initialized
            WriteError: If the write operation fails
        """
        ...

    async def health_check(self) -> bool:
        """
        Check if the database connection is healthy.

        Returns:
            True if connection is healthy, False otherwise
        """
        ...

    async def close(self) -> None:
        """Close all connections in the pool."""
        ...

    def get_info(self) -> BackendInfo:
        """
        Get information about the backend.

        Returns:
            BackendInfo with name, version, driver, and capabilities
        """
        ...


@runtime_checkable
class QueryBackend(Protocol):
    """
    Protocol for query backends.

    Separated from StorageBackend because:
    1. Write and read paths may use different optimizations
    2. Some backends (Kafka) are write-only
    3. Query syntax varies significantly between databases
    """

    @property
    def name(self) -> str:
        """Human-readable name of the backend."""
        ...

    @property
    def is_initialized(self) -> bool:
        """Whether the backend has been initialized."""
        ...

    def initialize(self, config: TokenLedgerConfig) -> None:
        """Initialize the query backend."""
        ...

    def get_cost_summary(
        self,
        days: int = 30,
        user_id: str | None = None,
        model: str | None = None,
        app_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Get cost summary for a time period.

        Returns dict with:
            - total_cost: float
            - total_tokens: int
            - total_input_tokens: int
            - total_output_tokens: int
            - total_requests: int
            - avg_cost_per_request: float
            - avg_tokens_per_request: float
        """
        ...

    def get_costs_by_model(
        self,
        days: int = 30,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Get cost breakdown by model.

        Returns list of dicts with:
            - model: str
            - provider: str
            - total_cost: float
            - total_requests: int
            - total_tokens: int
        """
        ...

    def get_costs_by_user(
        self,
        days: int = 30,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Get cost breakdown by user.

        Returns list of dicts with:
            - user_id: str
            - total_cost: float
            - total_requests: int
            - total_tokens: int
        """
        ...

    def get_daily_costs(
        self,
        days: int = 30,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get daily cost trends.

        Returns list of dicts with:
            - date: str (ISO format)
            - total_cost: float
            - total_requests: int
            - total_tokens: int
        """
        ...

    def close(self) -> None:
        """Close the query backend."""
        ...


@runtime_checkable
class AsyncQueryBackend(Protocol):
    """Protocol for async query backends."""

    @property
    def name(self) -> str:
        """Human-readable name of the backend."""
        ...

    @property
    def is_initialized(self) -> bool:
        """Whether the backend has been initialized."""
        ...

    async def initialize(self, config: TokenLedgerConfig) -> None:
        """Initialize the query backend."""
        ...

    async def get_cost_summary(
        self,
        days: int = 30,
        user_id: str | None = None,
        model: str | None = None,
        app_name: str | None = None,
    ) -> dict[str, Any]:
        """Get cost summary for a time period."""
        ...

    async def get_costs_by_model(
        self,
        days: int = 30,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get cost breakdown by model."""
        ...

    async def get_costs_by_user(
        self,
        days: int = 30,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Get cost breakdown by user."""
        ...

    async def get_daily_costs(
        self,
        days: int = 30,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get daily cost trends."""
        ...

    async def close(self) -> None:
        """Close the query backend."""
        ...
