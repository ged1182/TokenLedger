"""
Async PostgreSQL Storage Backend

Asynchronous storage backend using asyncpg with connection pooling.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from tokenledger.backends.base import BaseAsyncStorageBackend
from tokenledger.backends.exceptions import ConnectionError, DriverNotFoundError
from tokenledger.backends.protocol import BackendCapabilities

from .schema import get_health_check_sql, get_insert_sql_asyncpg, get_schema_statements

if TYPE_CHECKING:
    from tokenledger.config import TokenLedgerConfig

logger = logging.getLogger("tokenledger.backends.postgresql")


class AsyncPostgreSQLBackend(BaseAsyncStorageBackend):
    """
    Async PostgreSQL storage backend using asyncpg with connection pooling.

    Provides high-performance async database operations with automatic
    connection pooling.

    Example:
        >>> from tokenledger.backends.postgresql import AsyncPostgreSQLBackend
        >>> from tokenledger.config import TokenLedgerConfig
        >>>
        >>> config = TokenLedgerConfig(database_url="postgresql://localhost/mydb")
        >>> backend = AsyncPostgreSQLBackend()
        >>> await backend.initialize(config)
        >>>
        >>> events = [{"event_id": "...", "provider": "openai", ...}]
        >>> await backend.write_events(events)
        >>> await backend.close()
    """

    def __init__(self) -> None:
        super().__init__()
        self._driver_version: str = "unknown"
        self._min_pool_size: int = 2
        self._max_pool_size: int = 10

    @property
    def name(self) -> str:
        return "PostgreSQL (Async)"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_jsonb=True,
            supports_uuid=True,
            supports_decimal=True,
            supports_upsert=True,
            supports_returning=True,
            supports_batch_insert=True,
            supports_window_functions=True,
            supports_cte=True,
            supports_percentile=True,
            max_batch_size=10000,
            recommended_batch_size=1000,
        )

    async def initialize(
        self,
        config: TokenLedgerConfig,
        create_schema: bool = True,
        min_pool_size: int = 2,
        max_pool_size: int = 10,
    ) -> None:
        """
        Initialize the async backend with connection pooling.

        Args:
            config: TokenLedger configuration
            create_schema: Whether to create tables if they don't exist
            min_pool_size: Minimum number of connections in the pool
            max_pool_size: Maximum number of connections in the pool
        """
        self._min_pool_size = min_pool_size
        self._max_pool_size = max_pool_size
        await super().initialize(config, create_schema)

    async def _connect(self) -> None:
        """Establish async database connection pool."""
        if not self._config:
            raise ConnectionError("No configuration provided", backend_name=self.name)

        try:
            import asyncpg
        except ImportError as e:
            raise DriverNotFoundError(
                driver_name="asyncpg",
                install_hint="pip install tokenledger[asyncpg]",
                backend_name=self.name,
            ) from e

        try:
            self._driver_version = asyncpg.__version__
        except AttributeError:
            self._driver_version = "unknown"

        self._pool = await asyncpg.create_pool(
            self._config.database_url,
            min_size=self._min_pool_size,
            max_size=self._max_pool_size,
        )

        logger.debug(
            f"Created asyncpg pool with {self._min_pool_size}-{self._max_pool_size} connections"
        )

    async def _create_schema(self) -> None:
        """Create the events table and indexes."""
        if not self._config or not self._pool:
            return

        statements = get_schema_statements(self._config)

        async with self._pool.acquire() as conn:
            for sql in statements:
                await conn.execute(sql)

        if self._config.debug:
            logger.info(f"Created schema for table {self._config.full_table_name}")

    async def _write_batch(self, events: list[dict[str, Any]]) -> int:
        """Write a batch of events using asyncpg."""
        if not self._config or not self._pool:
            return 0

        values = [self._prepare_event_values(event) for event in events]
        sql = get_insert_sql_asyncpg(self._config, self.COLUMNS)

        try:
            async with self._pool.acquire() as conn:
                await conn.executemany(sql, values)
            return len(events)
        except Exception as e:
            logger.error(f"Error writing batch: {e}")
            raise

    async def _health_check(self) -> bool:
        """Check if the database connection pool is healthy."""
        if not self._pool:
            return False

        try:
            async with self._pool.acquire() as conn:
                result = await conn.fetchval(get_health_check_sql())
                return result == 1
        except Exception:
            return False

    async def _close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()

    def _get_version(self) -> str:
        """Get driver version (server version requires async query)."""
        return self._driver_version

    def _get_driver_name(self) -> str:
        """Get the driver name."""
        return f"asyncpg {self._driver_version}"

    @property
    def pool(self) -> Any:
        """Get the underlying asyncpg pool for advanced usage."""
        return self._pool

    async def execute(self, query: str, *args: Any) -> str:
        """
        Execute a query and return the status.

        Args:
            query: SQL query to execute
            *args: Query arguments

        Returns:
            Status string from asyncpg
        """
        if not self._pool:
            raise ConnectionError("Not connected", backend_name=self.name)

        async with self._pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        """
        Execute a query and return all rows.

        Args:
            query: SQL query to execute
            *args: Query arguments

        Returns:
            List of record objects
        """
        if not self._pool:
            raise ConnectionError("Not connected", backend_name=self.name)

        async with self._pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args: Any) -> Any:
        """
        Execute a query and return the first row.

        Args:
            query: SQL query to execute
            *args: Query arguments

        Returns:
            Single record object or None
        """
        if not self._pool:
            raise ConnectionError("Not connected", backend_name=self.name)

        async with self._pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args: Any) -> Any:
        """
        Execute a query and return the first value.

        Args:
            query: SQL query to execute
            *args: Query arguments

        Returns:
            Single value or None
        """
        if not self._pool:
            raise ConnectionError("Not connected", backend_name=self.name)

        async with self._pool.acquire() as conn:
            return await conn.fetchval(query, *args)
