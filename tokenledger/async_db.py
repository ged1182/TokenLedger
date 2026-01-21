"""
TokenLedger Async Database Module
Provides async database operations using asyncpg with connection pooling.
"""

import json
import logging
from contextlib import asynccontextmanager
from typing import Any

from .config import TokenLedgerConfig, get_config

logger = logging.getLogger("tokenledger")


class AsyncDatabase:
    """
    Async database connection pool manager using asyncpg.

    Example:
        >>> from tokenledger.async_db import AsyncDatabase
        >>>
        >>> db = AsyncDatabase()
        >>> await db.initialize()
        >>> async with db.acquire() as conn:
        ...     await conn.fetch("SELECT * FROM token_ledger_events LIMIT 10")
        >>> await db.close()
    """

    def __init__(self, config: TokenLedgerConfig | None = None):
        self.config = config or get_config()
        self._pool = None
        self._initialized = False

    async def initialize(
        self,
        min_size: int = 2,
        max_size: int = 10,
        create_tables: bool = True,
    ) -> None:
        """
        Initialize the async database connection pool.

        Args:
            min_size: Minimum number of connections in the pool
            max_size: Maximum number of connections in the pool
            create_tables: Whether to create the events table if it doesn't exist
        """
        if self._initialized:
            return

        try:
            import asyncpg
        except ImportError as err:
            raise ImportError(
                "asyncpg is required for async database operations. "
                "Install it with: pip install tokenledger[asyncpg]"
            ) from err

        self._pool = await asyncpg.create_pool(
            self.config.database_url,
            min_size=min_size,
            max_size=max_size,
        )

        if create_tables:
            await self._create_tables()

        self._initialized = True

        if self.config.debug:
            logger.info(f"Initialized asyncpg pool with {min_size}-{max_size} connections")

    async def _create_tables(self) -> None:
        """Create the events table if it doesn't exist"""
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.config.full_table_name} (
            event_id UUID PRIMARY KEY,
            trace_id UUID,
            span_id UUID,
            parent_span_id UUID,

            timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            duration_ms DOUBLE PRECISION,

            provider VARCHAR(50) NOT NULL,
            model VARCHAR(100) NOT NULL,

            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            total_tokens INTEGER NOT NULL DEFAULT 0,
            cached_tokens INTEGER DEFAULT 0,

            cost_usd DECIMAL(12, 8),

            endpoint VARCHAR(255),
            request_type VARCHAR(50) DEFAULT 'chat',

            user_id VARCHAR(255),
            session_id VARCHAR(255),
            organization_id VARCHAR(255),

            app_name VARCHAR(100),
            environment VARCHAR(50),

            status VARCHAR(20) DEFAULT 'success',
            error_type VARCHAR(100),
            error_message TEXT,

            metadata JSONB,

            request_preview TEXT,
            response_preview TEXT
        );

        -- Indexes for common queries
        CREATE INDEX IF NOT EXISTS idx_{self.config.table_name}_timestamp
            ON {self.config.full_table_name} (timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_{self.config.table_name}_user
            ON {self.config.full_table_name} (user_id, timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_{self.config.table_name}_model
            ON {self.config.full_table_name} (model, timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_{self.config.table_name}_app
            ON {self.config.full_table_name} (app_name, environment, timestamp DESC);
        """

        async with self.acquire() as conn:
            await conn.execute(create_sql)

        if self.config.debug:
            logger.info(f"Created table {self.config.full_table_name}")

    @asynccontextmanager
    async def acquire(self):
        """
        Acquire a connection from the pool.

        Usage:
            async with db.acquire() as conn:
                await conn.fetch("SELECT * FROM table")
        """
        if not self._initialized or self._pool is None:
            raise RuntimeError("Database not initialized. Call await db.initialize() first.")

        async with self._pool.acquire() as conn:
            yield conn

    async def execute(self, query: str, *args) -> str:
        """Execute a query and return the status."""
        async with self.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args) -> list[Any]:
        """Execute a query and return all rows."""
        async with self.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args) -> Any:
        """Execute a query and return the first row."""
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args) -> Any:
        """Execute a query and return the first value."""
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args)

    async def executemany(self, query: str, args: list) -> None:
        """Execute a query with multiple sets of arguments."""
        async with self.acquire() as conn:
            await conn.executemany(query, args)

    async def insert_events(self, events: list[dict[str, Any]]) -> int:
        """
        Insert multiple events into the database efficiently.

        Args:
            events: List of event dictionaries

        Returns:
            Number of events inserted
        """
        if not events:
            return 0

        columns = [
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
        ]

        # Build values list
        values: list[tuple[Any, ...]] = []
        for event in events:
            row: list[Any] = []
            for col in columns:
                val = event.get(col)
                # Handle metadata JSON serialization
                if col == "metadata" and val is not None:
                    if isinstance(val, str):
                        row.append(val)
                    else:
                        row.append(json.dumps(val))
                else:
                    row.append(val)
            values.append(tuple(row))

        # Build parameterized query with $1, $2, etc.
        placeholders = ", ".join(f"${i + 1}" for i in range(len(columns)))
        insert_sql = f"""
            INSERT INTO {self.config.full_table_name}
            ({", ".join(columns)})
            VALUES ({placeholders})
            ON CONFLICT (event_id) DO NOTHING
        """

        try:
            async with self.acquire() as conn:
                await conn.executemany(insert_sql, values)

            if self.config.debug:
                logger.info(f"Inserted {len(events)} events to database")

            return len(events)
        except Exception as e:
            logger.error(f"Error inserting events: {e}")
            return 0

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            self._initialized = False

            if self.config.debug:
                logger.info("Closed asyncpg connection pool")

    @property
    def is_initialized(self) -> bool:
        """Check if the database is initialized."""
        return self._initialized

    @property
    def pool(self):
        """Get the underlying asyncpg pool."""
        return self._pool


# Global async database instance
_async_db: AsyncDatabase | None = None


async def get_async_db() -> AsyncDatabase:
    """Get or create the global async database instance."""
    global _async_db
    if _async_db is None:
        _async_db = AsyncDatabase()
    return _async_db


async def init_async_db(
    min_size: int = 2,
    max_size: int = 10,
    create_tables: bool = True,
) -> AsyncDatabase:
    """
    Initialize the global async database instance.

    Args:
        min_size: Minimum number of connections in the pool
        max_size: Maximum number of connections in the pool
        create_tables: Whether to create the events table if it doesn't exist

    Returns:
        The initialized AsyncDatabase instance
    """
    db = await get_async_db()
    await db.initialize(min_size=min_size, max_size=max_size, create_tables=create_tables)
    return db


async def close_async_db() -> None:
    """Close the global async database instance."""
    global _async_db
    if _async_db:
        await _async_db.close()
        _async_db = None
