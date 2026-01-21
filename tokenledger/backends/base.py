"""
TokenLedger Backend Abstract Base Classes

Provides shared implementation logic for database backends.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from .exceptions import BackendNotInitializedError, SchemaError, WriteError
from .protocol import BackendCapabilities, BackendInfo

if TYPE_CHECKING:
    from tokenledger.config import TokenLedgerConfig

logger = logging.getLogger("tokenledger.backends")


class BaseStorageBackend(ABC):
    """
    Abstract base class for TokenLedger storage backends.

    Provides shared implementation logic while requiring subclasses
    to implement database-specific methods.

    Subclasses must implement:
        - _connect(): Establish database connection
        - _create_schema(): Create database-specific tables
        - _write_batch(): Database-specific batch insert logic
        - _health_check(): Connection health verification
        - _close(): Close connections

    Properties to override:
        - name: Human-readable backend name
        - capabilities: BackendCapabilities instance
    """

    # Column definitions shared across all backends
    COLUMNS = (
        "event_id",
        "trace_id",
        "span_id",
        "parent_span_id",
        "timestamp",
        "duration_ms",
        "provider",
        "model",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_tokens",
        "cost_usd",
        "endpoint",
        "request_type",
        "user_id",
        "session_id",
        "organization_id",
        "app_name",
        "environment",
        "status",
        "error_type",
        "error_message",
        "metadata",
        "request_preview",
        "response_preview",
    )

    def __init__(self) -> None:
        self._config: TokenLedgerConfig | None = None
        self._initialized: bool = False
        self._connection: Any = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the backend."""
        ...

    @property
    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """Return the capabilities of this backend."""
        ...

    @property
    def is_initialized(self) -> bool:
        """Whether the backend has been initialized."""
        return self._initialized

    def initialize(
        self,
        config: TokenLedgerConfig,
        create_schema: bool = True,
    ) -> None:
        """
        Initialize the backend with configuration.

        Args:
            config: TokenLedger configuration
            create_schema: Whether to create tables if they don't exist
        """
        if self._initialized:
            logger.debug(f"{self.name} backend already initialized")
            return

        self._config = config
        self._connect()

        if create_schema:
            try:
                self._create_schema()
            except Exception as e:
                self._close()
                self._connection = None
                raise SchemaError(str(e), backend_name=self.name) from e

        self._initialized = True
        logger.info(f"{self.name} backend initialized")

    @abstractmethod
    def _connect(self) -> None:
        """
        Establish database connection.

        Must be implemented by subclasses.

        Raises:
            ConnectionError: If unable to connect
            DriverNotFoundError: If driver is not installed
        """
        ...

    @abstractmethod
    def _create_schema(self) -> None:
        """
        Create database-specific tables and indexes.

        Must be implemented by subclasses. Should be idempotent.
        """
        ...

    def write_events(self, events: list[dict[str, Any]]) -> int:
        """
        Write events with error handling and logging.

        Args:
            events: List of event dictionaries

        Returns:
            Number of events successfully written

        Raises:
            BackendNotInitializedError: If backend is not initialized
            WriteError: If write operation fails
        """
        if not events:
            return 0

        if not self._initialized:
            raise BackendNotInitializedError(backend_name=self.name)

        try:
            count = self._write_batch(events)
            if self._config and self._config.debug:
                logger.debug(f"{self.name}: Wrote {count} events")
            return count
        except Exception as e:
            logger.error(f"{self.name}: Error writing events: {e}")
            raise WriteError(
                str(e),
                backend_name=self.name,
                events_count=len(events),
                events_written=0,
            ) from e

    @abstractmethod
    def _write_batch(self, events: list[dict[str, Any]]) -> int:
        """
        Database-specific batch write.

        Must be implemented by subclasses.

        Args:
            events: List of event dictionaries

        Returns:
            Number of events written
        """
        ...

    def health_check(self) -> bool:
        """
        Check connection health with error handling.

        Returns:
            True if healthy, False otherwise
        """
        if not self._initialized:
            return False

        try:
            return self._health_check()
        except Exception as e:
            logger.warning(f"{self.name}: Health check failed: {e}")
            return False

    @abstractmethod
    def _health_check(self) -> bool:
        """
        Database-specific health check.

        Must be implemented by subclasses.

        Returns:
            True if healthy, False otherwise
        """
        ...

    def close(self) -> None:
        """Close connection with cleanup."""
        if self._connection:
            try:
                self._close()
            except Exception as e:
                logger.warning(f"{self.name}: Error during close: {e}")
            finally:
                self._connection = None
                self._initialized = False
                logger.info(f"{self.name} backend closed")

    @abstractmethod
    def _close(self) -> None:
        """
        Database-specific close logic.

        Must be implemented by subclasses.
        """
        ...

    def get_info(self) -> BackendInfo:
        """Return information about the backend."""
        return BackendInfo(
            name=self.name,
            version=self._get_version(),
            driver=self._get_driver_name(),
            supports_async=False,
            capabilities=self.capabilities,
        )

    def _get_version(self) -> str:
        """Get backend version. Override in subclasses if available."""
        return "unknown"

    def _get_driver_name(self) -> str:
        """Get driver name. Override in subclasses."""
        return "unknown"

    def _prepare_event_values(self, event: dict[str, Any]) -> tuple[Any, ...]:
        """
        Convert event dict to tuple of values in column order.

        Args:
            event: Event dictionary

        Returns:
            Tuple of values in COLUMNS order
        """
        values: list[Any] = []
        for col in self.COLUMNS:
            val = event.get(col)
            # Serialize metadata to JSON string if needed
            if col == "metadata" and val is not None and not isinstance(val, str):
                val = json.dumps(val)
            values.append(val)
        return tuple(values)


class BaseAsyncStorageBackend(ABC):
    """
    Abstract base class for async TokenLedger storage backends.

    Provides shared implementation logic for async backends.

    Subclasses must implement:
        - _connect(): Establish async database connection/pool
        - _create_schema(): Create database-specific tables
        - _write_batch(): Database-specific async batch insert
        - _health_check(): Connection health verification
        - _close(): Close connections

    Properties to override:
        - name: Human-readable backend name
        - capabilities: BackendCapabilities instance
    """

    COLUMNS = BaseStorageBackend.COLUMNS

    def __init__(self) -> None:
        self._config: TokenLedgerConfig | None = None
        self._initialized: bool = False
        self._pool: Any = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the backend."""
        ...

    @property
    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """Return the capabilities of this backend."""
        ...

    @property
    def is_initialized(self) -> bool:
        """Whether the backend has been initialized."""
        return self._initialized

    async def initialize(
        self,
        config: TokenLedgerConfig,
        create_schema: bool = True,
    ) -> None:
        """
        Initialize the async backend with configuration.

        Args:
            config: TokenLedger configuration
            create_schema: Whether to create tables if they don't exist
        """
        if self._initialized:
            logger.debug(f"{self.name} backend already initialized")
            return

        self._config = config
        await self._connect()

        if create_schema:
            try:
                await self._create_schema()
            except Exception as e:
                await self._close()
                self._pool = None
                raise SchemaError(str(e), backend_name=self.name) from e

        self._initialized = True
        logger.info(f"{self.name} async backend initialized")

    @abstractmethod
    async def _connect(self) -> None:
        """
        Establish async database connection/pool.

        Must be implemented by subclasses.
        """
        ...

    @abstractmethod
    async def _create_schema(self) -> None:
        """
        Create database-specific tables and indexes asynchronously.

        Must be implemented by subclasses.
        """
        ...

    async def write_events(self, events: list[dict[str, Any]]) -> int:
        """
        Write events asynchronously with error handling.

        Args:
            events: List of event dictionaries

        Returns:
            Number of events successfully written
        """
        if not events:
            return 0

        if not self._initialized:
            raise BackendNotInitializedError(backend_name=self.name)

        try:
            count = await self._write_batch(events)
            if self._config and self._config.debug:
                logger.debug(f"{self.name}: Wrote {count} events")
            return count
        except Exception as e:
            logger.error(f"{self.name}: Error writing events: {e}")
            raise WriteError(
                str(e),
                backend_name=self.name,
                events_count=len(events),
                events_written=0,
            ) from e

    @abstractmethod
    async def _write_batch(self, events: list[dict[str, Any]]) -> int:
        """
        Database-specific async batch write.

        Must be implemented by subclasses.
        """
        ...

    async def health_check(self) -> bool:
        """Check connection health with error handling."""
        if not self._initialized:
            return False

        try:
            return await self._health_check()
        except Exception as e:
            logger.warning(f"{self.name}: Health check failed: {e}")
            return False

    @abstractmethod
    async def _health_check(self) -> bool:
        """Database-specific async health check."""
        ...

    async def close(self) -> None:
        """Close connection with cleanup."""
        if self._pool:
            try:
                await self._close()
            except Exception as e:
                logger.warning(f"{self.name}: Error during close: {e}")
            finally:
                self._pool = None
                self._initialized = False
                logger.info(f"{self.name} async backend closed")

    @abstractmethod
    async def _close(self) -> None:
        """Database-specific async close logic."""
        ...

    def get_info(self) -> BackendInfo:
        """Return information about the backend."""
        return BackendInfo(
            name=self.name,
            version=self._get_version(),
            driver=self._get_driver_name(),
            supports_async=True,
            capabilities=self.capabilities,
        )

    def _get_version(self) -> str:
        """Get backend version. Override in subclasses if available."""
        return "unknown"

    def _get_driver_name(self) -> str:
        """Get driver name. Override in subclasses."""
        return "unknown"

    def _prepare_event_values(self, event: dict[str, Any]) -> tuple[Any, ...]:
        """Convert event dict to tuple of values in column order."""
        values: list[Any] = []
        for col in self.COLUMNS:
            val = event.get(col)
            if col == "metadata" and val is not None and not isinstance(val, str):
                val = json.dumps(val)
            values.append(val)
        return tuple(values)
