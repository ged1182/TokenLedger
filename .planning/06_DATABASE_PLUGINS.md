# TokenLedger Database Plugin Architecture

**Version:** 1.0
**Date:** 2026-01-21
**Status:** Design Document

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Plugin Architecture Pattern](#plugin-architecture-pattern)
3. [Interface Definitions](#interface-definitions)
4. [Per-Database Implementation Notes](#per-database-implementation-notes)
5. [Entry Point Configuration](#entry-point-configuration)
6. [Installation Patterns](#installation-patterns)
7. [Migration/Schema Strategy](#migrationschema-strategy)
8. [Implementation Roadmap](#implementation-roadmap)

---

## Executive Summary

This document outlines the database plugin architecture for TokenLedger, enabling support for multiple analytics databases beyond PostgreSQL. The design follows patterns established by SQLAlchemy dialects and Apache Airflow providers, using Python entry points for plugin discovery and Protocol-based interfaces for maximum flexibility.

### Goals

- **Extensibility**: Allow third-party database backends without core modifications
- **Type Safety**: Full type hints with Protocol-based structural subtyping
- **Performance**: Backend-specific optimizations (batching, async, connection pooling)
- **Simplicity**: Easy installation via pip extras (`tokenledger[clickhouse]`)

### Target Databases

| Database | Primary Use Case | Write Pattern |
|----------|-----------------|---------------|
| PostgreSQL | Default, OLTP-friendly | Direct INSERT with batching |
| TimescaleDB | Time-series analytics | Hypertable with COPY |
| ClickHouse | High-volume analytics | Kafka buffer or async inserts |
| BigQuery | Cloud data warehouse | Storage Write API (streaming/batch) |
| Snowflake | Enterprise data warehouse | Snowpipe or COPY |

---

## Plugin Architecture Pattern

### Design Decision: Protocol + ABC Hybrid

Based on research into SQLAlchemy dialects and modern Python best practices, we recommend a **layered approach**:

1. **Protocol** for the public interface (consumed by core TokenLedger)
2. **ABC** for internal base class (shared implementation logic)

This provides:
- Maximum decoupling (domain depends only on Protocols)
- Shared logic via ABC (connection management, error handling)
- Runtime guarantees (ABC enforces method implementation)
- Static type checking (Protocol enables structural subtyping)

### Plugin Discovery Pattern

TokenLedger uses **entry points** for plugin discovery, following the pattern established by SQLAlchemy:

```
URL Pattern: tokenledger+<dialect>://<connection-string>
Example: tokenledger+clickhouse://localhost:9000/default
```

Plugins are discovered via `importlib.metadata.entry_points()` at runtime.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TokenLedger Core                             │
├─────────────────────────────────────────────────────────────────────┤
│  TokenTracker / AsyncTokenTracker                                   │
│       │                                                             │
│       ▼                                                             │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              DatabaseBackend (Protocol)                       │   │
│  │  - initialize()        - insert_events()                      │   │
│  │  - create_schema()     - close()                             │   │
│  │  - health_check()      - supports_async: bool                │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│              ┌───────────────┼───────────────┐                      │
│              ▼               ▼               ▼                      │
│  ┌────────────────┐ ┌────────────────┐ ┌────────────────┐          │
│  │ PostgresBackend│ │ClickHouseBackend│ │ BigQueryBackend│          │
│  └────────────────┘ └────────────────┘ └────────────────┘          │
└─────────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   Entry Point           Entry Point           Entry Point
   (built-in)          (pip install)         (pip install)
```

---

## Interface Definitions

### Core Protocol Definition

```python
# tokenledger/backends/protocol.py
"""
Database backend protocol for TokenLedger.
Defines the interface that all database backends must implement.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if TYPE_CHECKING:
    from tokenledger.config import TokenLedgerConfig


@runtime_checkable
class DatabaseBackend(Protocol):
    """
    Protocol defining the interface for TokenLedger database backends.

    All backends must implement these methods. The Protocol approach allows
    third-party implementations without requiring inheritance from a base class.

    Example:
        >>> class MyCustomBackend:
        ...     def initialize(self, config: TokenLedgerConfig) -> None: ...
        ...     def insert_events(self, events: list[dict[str, Any]]) -> int: ...
        ...     # ... other required methods
        >>>
        >>> assert isinstance(MyCustomBackend(), DatabaseBackend)  # True via structural subtyping
    """

    @property
    def name(self) -> str:
        """Human-readable name of the backend (e.g., 'PostgreSQL', 'ClickHouse')."""
        ...

    @property
    def supports_async(self) -> bool:
        """Whether this backend supports async operations."""
        ...

    @property
    def supports_batch_insert(self) -> bool:
        """Whether this backend supports efficient batch inserts."""
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
            ImportError: If required driver package is not installed
        """
        ...

    def create_schema(self) -> None:
        """
        Create the required schema (tables, indexes, views).

        This is idempotent - safe to call multiple times.
        """
        ...

    def insert_events(self, events: list[dict[str, Any]]) -> int:
        """
        Insert a batch of events into the database.

        Args:
            events: List of event dictionaries matching the LLMEvent schema

        Returns:
            Number of events successfully inserted

        Raises:
            ConnectionError: If the database connection is lost
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
        """Close the database connection and release resources."""
        ...

    def get_connection_info(self) -> dict[str, Any]:
        """
        Get information about the current connection.

        Returns:
            Dictionary with connection details (host, database, pool size, etc.)
        """
        ...


@runtime_checkable
class AsyncDatabaseBackend(Protocol):
    """
    Protocol for async-capable database backends.

    Extends the base functionality with async versions of key methods.
    """

    @property
    def name(self) -> str:
        """Human-readable name of the backend."""
        ...

    @property
    def supports_async(self) -> bool:
        """Always True for async backends."""
        ...

    @property
    def is_initialized(self) -> bool:
        """Whether the backend has been initialized."""
        ...

    async def initialize(
        self,
        config: TokenLedgerConfig,
        create_schema: bool = True,
        min_pool_size: int = 2,
        max_pool_size: int = 10,
    ) -> None:
        """Initialize the async database backend with connection pooling."""
        ...

    async def create_schema(self) -> None:
        """Create the required schema asynchronously."""
        ...

    async def insert_events(self, events: list[dict[str, Any]]) -> int:
        """Insert a batch of events asynchronously."""
        ...

    async def health_check(self) -> bool:
        """Check if the database connection is healthy."""
        ...

    async def close(self) -> None:
        """Close all connections in the pool."""
        ...


class BackendCapabilities:
    """
    Dataclass describing backend capabilities.

    Used for feature detection and query optimization.
    """

    def __init__(
        self,
        *,
        supports_jsonb: bool = False,
        supports_upsert: bool = True,
        supports_returning: bool = False,
        supports_window_functions: bool = True,
        supports_cte: bool = True,
        max_batch_size: int = 10000,
        recommended_batch_size: int = 1000,
        supports_streaming: bool = False,
        supports_compression: bool = False,
    ):
        self.supports_jsonb = supports_jsonb
        self.supports_upsert = supports_upsert
        self.supports_returning = supports_returning
        self.supports_window_functions = supports_window_functions
        self.supports_cte = supports_cte
        self.max_batch_size = max_batch_size
        self.recommended_batch_size = recommended_batch_size
        self.supports_streaming = supports_streaming
        self.supports_compression = supports_compression
```

### Abstract Base Class for Shared Implementation

```python
# tokenledger/backends/base.py
"""
Abstract base class providing shared implementation logic for database backends.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from .protocol import BackendCapabilities, DatabaseBackend

if TYPE_CHECKING:
    from tokenledger.config import TokenLedgerConfig

logger = logging.getLogger("tokenledger.backends")


class BaseBackend(ABC):
    """
    Abstract base class for TokenLedger database backends.

    Provides shared implementation logic while requiring subclasses
    to implement database-specific methods.

    Subclasses must implement:
        - _connect(): Establish database connection
        - _create_tables(): Create schema-specific tables
        - _insert_batch(): Database-specific batch insert logic
        - _health_check(): Connection health verification
        - _close(): Close connections
    """

    # Column definitions shared across all backends
    COLUMNS = [
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

    def __init__(self):
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
    def supports_async(self) -> bool:
        """Override in async-capable backends."""
        return False

    @property
    def supports_batch_insert(self) -> bool:
        """Most backends support batch inserts."""
        return True

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def initialize(
        self,
        config: TokenLedgerConfig,
        create_schema: bool = True,
    ) -> None:
        """Initialize the backend with configuration."""
        if self._initialized:
            logger.debug(f"{self.name} backend already initialized")
            return

        self._config = config
        self._connect()

        if create_schema:
            self.create_schema()

        self._initialized = True
        logger.info(f"{self.name} backend initialized")

    @abstractmethod
    def _connect(self) -> None:
        """Establish database connection. Implement in subclass."""
        ...

    def create_schema(self) -> None:
        """Create tables and indexes."""
        if not self._connection:
            raise RuntimeError("Backend not connected. Call initialize() first.")
        self._create_tables()
        logger.info(f"{self.name}: Schema created/verified")

    @abstractmethod
    def _create_tables(self) -> None:
        """Create database-specific tables. Implement in subclass."""
        ...

    def insert_events(self, events: list[dict[str, Any]]) -> int:
        """Insert events with error handling and logging."""
        if not events:
            return 0

        if not self._initialized:
            raise RuntimeError("Backend not initialized. Call initialize() first.")

        try:
            count = self._insert_batch(events)
            if self._config and self._config.debug:
                logger.debug(f"{self.name}: Inserted {count} events")
            return count
        except Exception as e:
            logger.error(f"{self.name}: Error inserting events: {e}")
            raise

    @abstractmethod
    def _insert_batch(self, events: list[dict[str, Any]]) -> int:
        """Database-specific batch insert. Implement in subclass."""
        ...

    def health_check(self) -> bool:
        """Check connection health with error handling."""
        try:
            return self._health_check()
        except Exception as e:
            logger.warning(f"{self.name}: Health check failed: {e}")
            return False

    @abstractmethod
    def _health_check(self) -> bool:
        """Database-specific health check. Implement in subclass."""
        ...

    def close(self) -> None:
        """Close connection with cleanup."""
        if self._connection:
            self._close()
            self._connection = None
            self._initialized = False
            logger.info(f"{self.name} backend closed")

    @abstractmethod
    def _close(self) -> None:
        """Database-specific close logic. Implement in subclass."""
        ...

    def get_connection_info(self) -> dict[str, Any]:
        """Return connection information for debugging."""
        return {
            "backend": self.name,
            "initialized": self._initialized,
            "supports_async": self.supports_async,
            "capabilities": {
                "jsonb": self.capabilities.supports_jsonb,
                "upsert": self.capabilities.supports_upsert,
                "max_batch_size": self.capabilities.max_batch_size,
            },
        }

    def _prepare_event_values(self, event: dict[str, Any]) -> tuple[Any, ...]:
        """Convert event dict to tuple of values in column order."""
        import json

        values = []
        for col in self.COLUMNS:
            val = event.get(col)
            # Serialize metadata to JSON string if needed
            if col == "metadata" and val is not None and not isinstance(val, str):
                val = json.dumps(val)
            values.append(val)
        return tuple(values)
```

### Backend Registry

```python
# tokenledger/backends/registry.py
"""
Backend registry for dynamic plugin loading.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any

if sys.version_info >= (3, 10):
    from importlib.metadata import entry_points
else:
    from importlib_metadata import entry_points

from .protocol import AsyncDatabaseBackend, DatabaseBackend

if TYPE_CHECKING:
    from tokenledger.config import TokenLedgerConfig

logger = logging.getLogger("tokenledger.backends")

# Built-in backends (always available)
_BUILTIN_BACKENDS: dict[str, str] = {
    "postgres": "tokenledger.backends.postgres:PostgresBackend",
    "postgresql": "tokenledger.backends.postgres:PostgresBackend",
    "asyncpg": "tokenledger.backends.postgres:AsyncPostgresBackend",
}

# Runtime-registered backends (for testing or custom usage)
_REGISTERED_BACKENDS: dict[str, str] = {}

# Entry point group name
ENTRY_POINT_GROUP = "tokenledger.backends"


def register(name: str, module_path: str, class_name: str) -> None:
    """
    Register a backend at runtime without setuptools.

    Useful for testing or embedding custom backends.

    Args:
        name: Backend identifier (e.g., "clickhouse")
        module_path: Full module path (e.g., "myapp.backends.clickhouse")
        class_name: Class name within the module (e.g., "ClickHouseBackend")

    Example:
        >>> from tokenledger.backends import registry
        >>> registry.register("mydb", "myapp.backends", "MyDBBackend")
    """
    _REGISTERED_BACKENDS[name] = f"{module_path}:{class_name}"
    logger.debug(f"Registered backend: {name} -> {module_path}:{class_name}")


def get_available_backends() -> dict[str, str]:
    """
    Get all available backends (built-in + installed + registered).

    Returns:
        Dictionary mapping backend names to their module:class paths
    """
    backends = dict(_BUILTIN_BACKENDS)
    backends.update(_REGISTERED_BACKENDS)

    # Discover installed plugins via entry points
    eps = entry_points(group=ENTRY_POINT_GROUP)
    for ep in eps:
        backends[ep.name] = ep.value

    return backends


def load_backend(name: str) -> type[DatabaseBackend] | type[AsyncDatabaseBackend]:
    """
    Load a backend class by name.

    Args:
        name: Backend identifier (e.g., "postgres", "clickhouse")

    Returns:
        The backend class (not an instance)

    Raises:
        KeyError: If backend is not found
        ImportError: If backend module cannot be imported

    Example:
        >>> BackendClass = load_backend("postgres")
        >>> backend = BackendClass()
        >>> backend.initialize(config)
    """
    available = get_available_backends()

    if name not in available:
        raise KeyError(
            f"Unknown backend: {name}. "
            f"Available backends: {', '.join(sorted(available.keys()))}"
        )

    module_class = available[name]
    module_path, class_name = module_class.rsplit(":", 1)

    # Import the module
    import importlib
    module = importlib.import_module(module_path)

    # Get the class
    backend_class = getattr(module, class_name)

    return backend_class


def create_backend(
    name: str,
    config: TokenLedgerConfig,
    **kwargs: Any,
) -> DatabaseBackend:
    """
    Create and initialize a backend instance.

    Args:
        name: Backend identifier
        config: TokenLedger configuration
        **kwargs: Additional arguments passed to initialize()

    Returns:
        Initialized backend instance

    Example:
        >>> from tokenledger.config import TokenLedgerConfig
        >>> config = TokenLedgerConfig(database_url="postgresql://...")
        >>> backend = create_backend("postgres", config)
    """
    BackendClass = load_backend(name)
    backend = BackendClass()
    backend.initialize(config, **kwargs)
    return backend


async def create_async_backend(
    name: str,
    config: TokenLedgerConfig,
    **kwargs: Any,
) -> AsyncDatabaseBackend:
    """
    Create and initialize an async backend instance.

    Args:
        name: Backend identifier (must support async)
        config: TokenLedger configuration
        **kwargs: Additional arguments passed to initialize()

    Returns:
        Initialized async backend instance

    Raises:
        ValueError: If the backend doesn't support async
    """
    BackendClass = load_backend(name)
    backend = BackendClass()

    if not getattr(backend, "supports_async", False):
        raise ValueError(f"Backend {name} does not support async operations")

    await backend.initialize(config, **kwargs)
    return backend


def parse_url(url: str) -> tuple[str, str]:
    """
    Parse a TokenLedger database URL to extract backend name.

    URL format: tokenledger+<backend>://<connection-string>

    Args:
        url: Full database URL

    Returns:
        Tuple of (backend_name, connection_string)

    Example:
        >>> parse_url("tokenledger+clickhouse://localhost:9000/default")
        ('clickhouse', 'clickhouse://localhost:9000/default')
        >>> parse_url("postgresql://localhost/mydb")
        ('postgres', 'postgresql://localhost/mydb')
    """
    if url.startswith("tokenledger+"):
        # Extract backend from URL
        rest = url[len("tokenledger+"):]
        backend_name, _, connection = rest.partition("://")
        return backend_name, f"{backend_name}://{connection}"

    # Infer backend from URL scheme
    scheme = url.split("://")[0].split("+")[0]
    backend_map = {
        "postgresql": "postgres",
        "postgres": "postgres",
        "clickhouse": "clickhouse",
        "bigquery": "bigquery",
        "snowflake": "snowflake",
    }
    return backend_map.get(scheme, scheme), url
```

---

## Per-Database Implementation Notes

### PostgreSQL (Built-in)

PostgreSQL is the default backend, already implemented in TokenLedger.

**Key Characteristics:**
- Direct INSERT with batching (psycopg2 `execute_values` or psycopg3 `executemany`)
- JSONB for metadata (efficient querying)
- Conflict handling with `ON CONFLICT DO NOTHING`
- Connection pooling via asyncpg for async operations

**Implementation:**
```python
# tokenledger/backends/postgres.py (simplified)

from .base import BaseBackend
from .protocol import BackendCapabilities


class PostgresBackend(BaseBackend):
    """PostgreSQL backend using psycopg2 or psycopg3."""

    @property
    def name(self) -> str:
        return "PostgreSQL"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_jsonb=True,
            supports_upsert=True,
            supports_returning=True,
            max_batch_size=10000,
            recommended_batch_size=1000,
        )

    def _connect(self) -> None:
        try:
            import psycopg2
            from psycopg2.extras import execute_values
            self._connection = psycopg2.connect(self._config.database_url)
            self._execute_values = execute_values
            self._use_psycopg2 = True
        except ImportError:
            import psycopg
            self._connection = psycopg.connect(self._config.database_url)
            self._use_psycopg2 = False

    def _create_tables(self) -> None:
        # ... existing table creation logic ...
        pass

    def _insert_batch(self, events: list[dict]) -> int:
        values = [self._prepare_event_values(e) for e in events]

        with self._connection.cursor() as cur:
            if self._use_psycopg2:
                sql = f"INSERT INTO {self._config.full_table_name} (...) VALUES %s ON CONFLICT DO NOTHING"
                self._execute_values(cur, sql, values)
            else:
                sql = f"INSERT INTO {self._config.full_table_name} (...) VALUES (...) ON CONFLICT DO NOTHING"
                cur.executemany(sql, values)

        self._connection.commit()
        return len(events)

    def _health_check(self) -> bool:
        with self._connection.cursor() as cur:
            cur.execute("SELECT 1")
            return cur.fetchone()[0] == 1

    def _close(self) -> None:
        self._connection.close()
```

---

### ClickHouse

**Best Practices Research Summary:**
- Batch 10,000-100,000 rows per insert (avoid single-row inserts)
- Use Kafka/Redpanda as a buffer for high-throughput scenarios
- Enable async inserts with `async_insert=1, wait_for_async_insert=1`
- Consider MergeTree engine with appropriate partitioning

**Sources:**
- [ClickHouse Insert Strategy](https://clickhouse.com/docs/best-practices/selecting-an-insert-strategy)
- [Supercharging ClickHouse Data Loads](https://clickhouse.com/blog/supercharge-your-clickhouse-data-loads-part1)
- [asynch GitHub](https://github.com/long2ice/asynch) - Native async Python driver

**Implementation Notes:**

```python
# tokenledger_clickhouse/backend.py

from tokenledger.backends.base import BaseBackend
from tokenledger.backends.protocol import BackendCapabilities


class ClickHouseBackend(BaseBackend):
    """
    ClickHouse backend for high-volume analytics.

    Recommended for:
    - High write throughput (>10k events/second)
    - Analytics queries over large datasets
    - Time-series aggregations

    Requirements:
        pip install tokenledger[clickhouse]
        # or
        pip install clickhouse-driver asynch
    """

    @property
    def name(self) -> str:
        return "ClickHouse"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_jsonb=False,  # Uses JSON, not JSONB
            supports_upsert=False,  # ReplacingMergeTree instead
            supports_returning=False,
            max_batch_size=100000,
            recommended_batch_size=10000,
            supports_compression=True,
        )

    def _connect(self) -> None:
        try:
            from clickhouse_driver import Client
        except ImportError as e:
            raise ImportError(
                "clickhouse-driver is required. Install with: "
                "pip install tokenledger[clickhouse]"
            ) from e

        # Parse connection URL
        # clickhouse://user:password@host:9000/database
        self._client = Client.from_url(self._config.database_url)

    def _create_tables(self) -> None:
        """Create ClickHouse-optimized schema."""
        create_sql = """
        CREATE TABLE IF NOT EXISTS token_ledger_events (
            event_id UUID,
            trace_id Nullable(UUID),
            span_id Nullable(UUID),
            parent_span_id Nullable(UUID),

            timestamp DateTime64(3, 'UTC'),
            duration_ms Nullable(Float64),

            provider LowCardinality(String),
            model LowCardinality(String),

            input_tokens UInt32,
            output_tokens UInt32,
            total_tokens UInt32,
            cached_tokens UInt32 DEFAULT 0,

            cost_usd Decimal64(8),

            endpoint Nullable(String),
            request_type LowCardinality(String) DEFAULT 'chat',

            user_id Nullable(String),
            session_id Nullable(String),
            organization_id Nullable(String),

            app_name Nullable(LowCardinality(String)),
            environment Nullable(LowCardinality(String)),

            status LowCardinality(String) DEFAULT 'success',
            error_type Nullable(String),
            error_message Nullable(String),

            metadata Nullable(String),  -- JSON as string

            request_preview Nullable(String),
            response_preview Nullable(String)
        )
        ENGINE = MergeTree()
        PARTITION BY toYYYYMM(timestamp)
        ORDER BY (timestamp, provider, model)
        TTL timestamp + INTERVAL 90 DAY DELETE
        SETTINGS index_granularity = 8192
        """
        self._client.execute(create_sql)

    def _insert_batch(self, events: list[dict]) -> int:
        """
        Insert using native protocol for efficiency.

        ClickHouse native protocol is columnar, which is highly
        efficient for batch inserts.
        """
        # Prepare data in columnar format
        data = [self._prepare_event_values(e) for e in events]

        insert_sql = f"""
        INSERT INTO token_ledger_events
        ({', '.join(self.COLUMNS)}) VALUES
        """

        self._client.execute(insert_sql, data)
        return len(events)

    def _health_check(self) -> bool:
        result = self._client.execute("SELECT 1")
        return result[0][0] == 1

    def _close(self) -> None:
        self._client.disconnect()


class AsyncClickHouseBackend:
    """
    Async ClickHouse backend using asynch library.

    For high-concurrency async applications.
    """

    @property
    def name(self) -> str:
        return "ClickHouse (Async)"

    @property
    def supports_async(self) -> bool:
        return True

    async def _connect(self) -> None:
        try:
            from asynch import Connection
            from asynch import Pool
        except ImportError as e:
            raise ImportError(
                "asynch is required for async ClickHouse. Install with: "
                "pip install asynch"
            ) from e

        # Parse URL and create pool
        self._pool = Pool(
            host=self._host,
            port=self._port,
            database=self._database,
            user=self._user,
            password=self._password,
            minsize=self._config.pool_min_size,
            maxsize=self._config.pool_max_size,
        )

    async def _insert_batch(self, events: list[dict]) -> int:
        data = [self._prepare_event_values(e) for e in events]

        async with self._pool.connection() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    f"INSERT INTO token_ledger_events ({', '.join(self.COLUMNS)}) VALUES",
                    data
                )

        return len(events)
```

**Kafka Buffer Architecture (for extreme throughput):**

For applications generating >100k events/second, use Kafka as a buffer:

```
Application -> TokenLedger -> Kafka -> ClickHouse Kafka Engine -> MergeTree
```

```python
# Example Kafka producer integration
class ClickHouseKafkaBackend(BaseBackend):
    """
    ClickHouse backend with Kafka buffer for extreme throughput.

    Events are sent to Kafka, and ClickHouse consumes via Kafka table engine.
    This decouples producers from ClickHouse's merge pressure.
    """

    def _connect(self) -> None:
        from confluent_kafka import Producer

        self._producer = Producer({
            'bootstrap.servers': self._kafka_servers,
            'client.id': 'tokenledger',
            'linger.ms': 100,  # Batch for 100ms
            'batch.size': 65536,  # 64KB batches
        })

    def _insert_batch(self, events: list[dict]) -> int:
        import json

        for event in events:
            self._producer.produce(
                topic='tokenledger_events',
                value=json.dumps(event).encode('utf-8'),
                key=event['event_id'].encode('utf-8'),
            )

        self._producer.flush()
        return len(events)
```

---

### TimescaleDB

**Best Practices Research Summary:**
- Use hypertables for automatic time partitioning
- Chunk size should fit 25% of memory for indexes
- Use COPY for bulk inserts (pgcopy library)
- Enable compression for older chunks
- Connection pooling is critical (PgBouncer or asyncpg pool)

**Sources:**
- [TimescaleDB Best Practices](https://docs-dev.timescale.com/docs-tutorial-lambda-cd/timescaledb/tutorial-lambda-cd/how-to-guides/hypertables/best-practices/)
- [Asyncpg and PostgreSQL](https://www.tigerdata.com/blog/how-to-build-applications-with-asyncpg-and-postgresql)

**Implementation Notes:**

```python
# tokenledger_timescale/backend.py

from tokenledger.backends.postgres import PostgresBackend
from tokenledger.backends.protocol import BackendCapabilities


class TimescaleDBBackend(PostgresBackend):
    """
    TimescaleDB backend extending PostgreSQL with hypertables.

    Recommended for:
    - Time-series analytics with automatic partitioning
    - Compression of historical data
    - Continuous aggregates for dashboards

    Requirements:
        pip install tokenledger[timescale]
        # Requires TimescaleDB extension in PostgreSQL
    """

    @property
    def name(self) -> str:
        return "TimescaleDB"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_jsonb=True,
            supports_upsert=True,
            supports_returning=True,
            max_batch_size=50000,
            recommended_batch_size=5000,
            supports_compression=True,
        )

    def _create_tables(self) -> None:
        """Create hypertable with TimescaleDB-specific optimizations."""

        # First, create the base table
        create_sql = """
        CREATE TABLE IF NOT EXISTS token_ledger_events (
            event_id UUID NOT NULL,
            trace_id UUID,
            span_id UUID,
            parent_span_id UUID,

            timestamp TIMESTAMPTZ NOT NULL,
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
            response_preview TEXT,

            -- TimescaleDB requires unique constraint to include time column
            UNIQUE (event_id, timestamp)
        );
        """

        with self._connection.cursor() as cur:
            cur.execute(create_sql)

            # Convert to hypertable (idempotent with if_not_exists)
            cur.execute("""
                SELECT create_hypertable(
                    'token_ledger_events',
                    'timestamp',
                    chunk_time_interval => INTERVAL '1 day',
                    if_not_exists => TRUE
                );
            """)

            # Enable compression on chunks older than 7 days
            cur.execute("""
                ALTER TABLE token_ledger_events
                SET (
                    timescaledb.compress,
                    timescaledb.compress_segmentby = 'provider, model',
                    timescaledb.compress_orderby = 'timestamp DESC'
                );
            """)

            # Add compression policy
            cur.execute("""
                SELECT add_compression_policy(
                    'token_ledger_events',
                    INTERVAL '7 days',
                    if_not_exists => TRUE
                );
            """)

            # Create continuous aggregate for daily costs
            cur.execute("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS token_ledger_daily_costs_cagg
                WITH (timescaledb.continuous) AS
                SELECT
                    time_bucket('1 day', timestamp) AS bucket,
                    provider,
                    model,
                    COUNT(*) as request_count,
                    SUM(input_tokens) as total_input_tokens,
                    SUM(output_tokens) as total_output_tokens,
                    SUM(total_tokens) as total_tokens,
                    SUM(cost_usd) as total_cost,
                    AVG(duration_ms) as avg_latency_ms
                FROM token_ledger_events
                GROUP BY bucket, provider, model
                WITH NO DATA;
            """)

            # Refresh policy for continuous aggregate
            cur.execute("""
                SELECT add_continuous_aggregate_policy(
                    'token_ledger_daily_costs_cagg',
                    start_offset => INTERVAL '7 days',
                    end_offset => INTERVAL '1 hour',
                    schedule_interval => INTERVAL '1 hour',
                    if_not_exists => TRUE
                );
            """)

        self._connection.commit()

    def _insert_batch(self, events: list[dict]) -> int:
        """
        Use COPY for efficient bulk inserts.

        For high-volume ingestion, COPY is significantly faster than INSERT.
        """
        try:
            from pgcopy import CopyManager

            values = [self._prepare_event_values(e) for e in events]

            mgr = CopyManager(
                self._connection,
                'token_ledger_events',
                self.COLUMNS
            )
            mgr.copy(values)
            self._connection.commit()
            return len(events)

        except ImportError:
            # Fall back to regular INSERT if pgcopy not available
            return super()._insert_batch(events)
```

---

### BigQuery

**Best Practices Research Summary:**
- Use Storage Write API (not legacy streaming)
- Batch 500 rows per request (max 10,000)
- Committed stream for immediate availability
- Pending stream for atomic batch loads
- Handle quota limits and retries

**Sources:**
- [BigQuery Storage Write API](https://docs.cloud.google.com/bigquery/docs/write-api)
- [BigQuery Streaming vs Batch](https://medium.com/@sattarkars45/bigquery-streaming-vs-job-load-understanding-write-disposition-and-when-to-use-each-4f084fd4c202)

**Implementation Notes:**

```python
# tokenledger_bigquery/backend.py

from tokenledger.backends.base import BaseBackend
from tokenledger.backends.protocol import BackendCapabilities


class BigQueryBackend(BaseBackend):
    """
    BigQuery backend using Storage Write API.

    Recommended for:
    - Google Cloud environments
    - Large-scale analytics
    - Integration with other GCP services

    Requirements:
        pip install tokenledger[bigquery]
        # or
        pip install google-cloud-bigquery google-cloud-bigquery-storage
    """

    @property
    def name(self) -> str:
        return "BigQuery"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_jsonb=False,  # Uses JSON/STRUCT
            supports_upsert=False,  # Use MERGE statements instead
            supports_returning=False,
            max_batch_size=10000,
            recommended_batch_size=500,
            supports_streaming=True,
        )

    def __init__(self, project_id: str, dataset_id: str):
        super().__init__()
        self._project_id = project_id
        self._dataset_id = dataset_id
        self._table_id = "token_ledger_events"

    def _connect(self) -> None:
        try:
            from google.cloud import bigquery
        except ImportError as e:
            raise ImportError(
                "google-cloud-bigquery is required. Install with: "
                "pip install tokenledger[bigquery]"
            ) from e

        self._client = bigquery.Client(project=self._project_id)
        self._table_ref = f"{self._project_id}.{self._dataset_id}.{self._table_id}"

    def _create_tables(self) -> None:
        """Create BigQuery table with appropriate schema."""
        from google.cloud import bigquery

        schema = [
            bigquery.SchemaField("event_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("trace_id", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("span_id", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("parent_span_id", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("duration_ms", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("provider", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("model", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("input_tokens", "INT64", mode="REQUIRED"),
            bigquery.SchemaField("output_tokens", "INT64", mode="REQUIRED"),
            bigquery.SchemaField("total_tokens", "INT64", mode="REQUIRED"),
            bigquery.SchemaField("cached_tokens", "INT64", mode="NULLABLE"),
            bigquery.SchemaField("cost_usd", "NUMERIC", mode="NULLABLE"),
            bigquery.SchemaField("endpoint", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("request_type", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("user_id", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("session_id", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("organization_id", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("app_name", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("environment", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("status", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("error_type", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("error_message", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("metadata", "JSON", mode="NULLABLE"),
            bigquery.SchemaField("request_preview", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("response_preview", "STRING", mode="NULLABLE"),
        ]

        table = bigquery.Table(self._table_ref, schema=schema)

        # Partition by day for query efficiency
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="timestamp",
            expiration_ms=90 * 24 * 60 * 60 * 1000,  # 90 days
        )

        # Cluster for common query patterns
        table.clustering_fields = ["provider", "model", "user_id"]

        self._client.create_table(table, exists_ok=True)

    def _insert_batch(self, events: list[dict]) -> int:
        """
        Insert using streaming API (insert_rows_json).

        For very high volume, consider using Storage Write API
        with pending streams for better throughput.
        """
        # Convert events to BigQuery-compatible format
        rows = []
        for event in events:
            row = dict(event)
            # BigQuery expects ISO format strings for timestamps
            if 'timestamp' in row and not isinstance(row['timestamp'], str):
                row['timestamp'] = row['timestamp'].isoformat()
            # Convert metadata dict to JSON string if needed
            if 'metadata' in row and isinstance(row['metadata'], dict):
                import json
                row['metadata'] = json.dumps(row['metadata'])
            rows.append(row)

        errors = self._client.insert_rows_json(self._table_ref, rows)

        if errors:
            # Log errors but don't fail completely
            import logging
            logger = logging.getLogger("tokenledger.backends.bigquery")
            for error in errors:
                logger.error(f"BigQuery insert error: {error}")
            return len(rows) - len(errors)

        return len(rows)

    def _health_check(self) -> bool:
        try:
            query = "SELECT 1"
            result = self._client.query(query).result()
            return list(result)[0][0] == 1
        except Exception:
            return False

    def _close(self) -> None:
        self._client.close()


class BigQueryStorageWriteBackend(BigQueryBackend):
    """
    BigQuery backend using Storage Write API for higher throughput.

    Use this for:
    - Very high volume (>10k events/second)
    - Need for exactly-once semantics
    - Atomic batch commits
    """

    def _insert_batch(self, events: list[dict]) -> int:
        """Use Storage Write API with pending stream."""
        from google.cloud.bigquery_storage_v1 import BigQueryWriteClient
        from google.cloud.bigquery_storage_v1 import types
        from google.protobuf import descriptor_pb2

        # This is a simplified example - full implementation requires
        # protobuf schema definition
        write_client = BigQueryWriteClient()

        parent = write_client.table_path(
            self._project_id, self._dataset_id, self._table_id
        )

        # Create a pending write stream
        write_stream = types.WriteStream(type_=types.WriteStream.Type.PENDING)
        write_stream = write_client.create_write_stream(
            parent=parent, write_stream=write_stream
        )

        # ... append rows and commit ...
        # Full implementation requires protobuf serialization

        return len(events)
```

---

### Snowflake

**Best Practices Research Summary:**
- Snowpipe for continuous ingestion (median 5s latency)
- COPY for batch loads (use appropriately sized files)
- 10,000+ records per file for efficiency
- Use connection pooling and async queries
- Snowpipe Streaming for sub-second latency

**Sources:**
- [Snowflake Python Connector 2025](https://docs.snowflake.com/en/release-notes/clients-drivers/python-connector-2025)
- [Snowflake Ingestion Mechanisms](https://medium.com/@divyanshsaxenaofficial/2025-confused-between-copy-snowpipe-dynamic-tables-7c69475e0914)

**Implementation Notes:**

```python
# tokenledger_snowflake/backend.py

from tokenledger.backends.base import BaseBackend
from tokenledger.backends.protocol import BackendCapabilities


class SnowflakeBackend(BaseBackend):
    """
    Snowflake backend for enterprise data warehouse integration.

    Recommended for:
    - Enterprise environments with existing Snowflake
    - Cross-cloud deployments
    - Advanced analytics and ML features

    Requirements:
        pip install tokenledger[snowflake]
        # or
        pip install snowflake-connector-python[pandas]
    """

    @property
    def name(self) -> str:
        return "Snowflake"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_jsonb=False,  # Uses VARIANT type
            supports_upsert=True,  # MERGE statement
            supports_returning=False,
            max_batch_size=16384,  # Batch bind limit
            recommended_batch_size=10000,
            supports_compression=True,
        )

    def __init__(
        self,
        account: str,
        user: str,
        password: str = None,
        private_key: str = None,
        warehouse: str = "COMPUTE_WH",
        database: str = "TOKENLEDGER",
        schema: str = "PUBLIC",
    ):
        super().__init__()
        self._account = account
        self._user = user
        self._password = password
        self._private_key = private_key
        self._warehouse = warehouse
        self._database = database
        self._schema = schema

    def _connect(self) -> None:
        try:
            import snowflake.connector
        except ImportError as e:
            raise ImportError(
                "snowflake-connector-python is required. Install with: "
                "pip install tokenledger[snowflake]"
            ) from e

        connect_kwargs = {
            'account': self._account,
            'user': self._user,
            'warehouse': self._warehouse,
            'database': self._database,
            'schema': self._schema,
        }

        if self._password:
            connect_kwargs['password'] = self._password
        elif self._private_key:
            connect_kwargs['private_key'] = self._private_key

        self._connection = snowflake.connector.connect(**connect_kwargs)

    def _create_tables(self) -> None:
        """Create Snowflake-optimized table schema."""

        create_sql = """
        CREATE TABLE IF NOT EXISTS token_ledger_events (
            event_id VARCHAR(36) NOT NULL,
            trace_id VARCHAR(36),
            span_id VARCHAR(36),
            parent_span_id VARCHAR(36),

            timestamp TIMESTAMP_NTZ NOT NULL,
            duration_ms FLOAT,

            provider VARCHAR(50) NOT NULL,
            model VARCHAR(100) NOT NULL,

            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            total_tokens INTEGER NOT NULL DEFAULT 0,
            cached_tokens INTEGER DEFAULT 0,

            cost_usd NUMBER(12, 8),

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

            metadata VARIANT,  -- Snowflake's semi-structured type

            request_preview TEXT,
            response_preview TEXT,

            PRIMARY KEY (event_id)
        )
        CLUSTER BY (timestamp::DATE, provider, model);
        """

        with self._connection.cursor() as cur:
            cur.execute(create_sql)

            # Create a stream for change tracking (useful for downstream)
            cur.execute("""
                CREATE STREAM IF NOT EXISTS token_ledger_events_stream
                ON TABLE token_ledger_events
                APPEND_ONLY = TRUE;
            """)

    def _insert_batch(self, events: list[dict]) -> int:
        """
        Insert using batch binding for efficiency.

        Snowflake supports up to 16,384 rows per batch bind.
        """
        import json

        insert_sql = """
        INSERT INTO token_ledger_events (
            event_id, trace_id, span_id, parent_span_id,
            timestamp, duration_ms,
            provider, model,
            input_tokens, output_tokens, total_tokens, cached_tokens,
            cost_usd, endpoint, request_type,
            user_id, session_id, organization_id,
            app_name, environment,
            status, error_type, error_message,
            metadata, request_preview, response_preview
        )
        VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, PARSE_JSON(%s), %s, %s
        )
        """

        rows = []
        for event in events:
            row = list(self._prepare_event_values(event))
            # Snowflake needs PARSE_JSON for VARIANT, so keep metadata as string
            rows.append(row)

        with self._connection.cursor() as cur:
            cur.executemany(insert_sql, rows)

        return len(events)

    def _health_check(self) -> bool:
        with self._connection.cursor() as cur:
            cur.execute("SELECT 1")
            return cur.fetchone()[0] == 1

    def _close(self) -> None:
        self._connection.close()


class SnowpipeStreamingBackend(SnowflakeBackend):
    """
    Snowflake backend using Snowpipe Streaming for sub-second latency.

    Requires Snowflake Enterprise Edition or higher.
    """

    def _insert_batch(self, events: list[dict]) -> int:
        """
        Use Snowpipe Streaming API for near real-time ingestion.

        This provides sub-second latency compared to regular Snowpipe's
        1-minute default batching.
        """
        from snowflake.ingest import SnowflakeStreamingIngestClient

        # Create streaming client
        client = SnowflakeStreamingIngestClient(
            account=self._account,
            user=self._user,
            private_key=self._private_key,
            role="TOKENLEDGER_ROLE",
        )

        # Open a channel
        channel = client.open_channel(
            database=self._database,
            schema=self._schema,
            table="token_ledger_events",
            channel_name="tokenledger_channel",
        )

        # Insert rows
        for event in events:
            channel.insert_row(event)

        # Commit the batch
        channel.close()

        return len(events)
```

---

## Entry Point Configuration

### pyproject.toml Configuration

**Core Package (tokenledger):**

```toml
# pyproject.toml

[project]
name = "tokenledger"
version = "0.2.0"
# ... other fields ...

[project.optional-dependencies]
# Database backends
postgres = ["psycopg2-binary>=2.9.0"]
asyncpg = ["asyncpg>=0.29.0"]
timescale = [
    "psycopg2-binary>=2.9.0",
    "asyncpg>=0.29.0",
    "pgcopy>=1.5.0",
]
clickhouse = [
    "clickhouse-driver>=0.2.6",
]
clickhouse-async = [
    "asynch>=0.2.5",
]
bigquery = [
    "google-cloud-bigquery>=3.0.0",
    "google-cloud-bigquery-storage>=2.0.0",
]
snowflake = [
    "snowflake-connector-python>=3.0.0",
]

# Bundle all backends
all-backends = [
    "tokenledger[postgres]",
    "tokenledger[asyncpg]",
    "tokenledger[clickhouse]",
    "tokenledger[bigquery]",
    "tokenledger[snowflake]",
]

# ... existing optional deps ...

[project.entry-points."tokenledger.backends"]
# Built-in backends
postgres = "tokenledger.backends.postgres:PostgresBackend"
postgresql = "tokenledger.backends.postgres:PostgresBackend"
asyncpg = "tokenledger.backends.postgres:AsyncPostgresBackend"
timescale = "tokenledger.backends.timescale:TimescaleDBBackend"
```

**External Plugin Package (tokenledger-clickhouse):**

```toml
# tokenledger-clickhouse/pyproject.toml

[project]
name = "tokenledger-clickhouse"
version = "0.1.0"
description = "ClickHouse backend for TokenLedger"
dependencies = [
    "tokenledger>=0.2.0",
    "clickhouse-driver>=0.2.6",
]

[project.optional-dependencies]
async = ["asynch>=0.2.5"]
kafka = ["confluent-kafka>=2.0.0"]

[project.entry-points."tokenledger.backends"]
clickhouse = "tokenledger_clickhouse:ClickHouseBackend"
clickhouse-async = "tokenledger_clickhouse:AsyncClickHouseBackend"
clickhouse-kafka = "tokenledger_clickhouse:ClickHouseKafkaBackend"
```

**External Plugin Package (tokenledger-bigquery):**

```toml
# tokenledger-bigquery/pyproject.toml

[project]
name = "tokenledger-bigquery"
version = "0.1.0"
description = "BigQuery backend for TokenLedger"
dependencies = [
    "tokenledger>=0.2.0",
    "google-cloud-bigquery>=3.0.0",
]

[project.optional-dependencies]
storage = ["google-cloud-bigquery-storage>=2.0.0"]

[project.entry-points."tokenledger.backends"]
bigquery = "tokenledger_bigquery:BigQueryBackend"
bigquery-storage = "tokenledger_bigquery:BigQueryStorageWriteBackend"
```

---

## Installation Patterns

### User Installation Examples

```bash
# PostgreSQL only (default)
pip install tokenledger

# PostgreSQL with async support
pip install "tokenledger[postgres,asyncpg]"

# TimescaleDB (includes pgcopy for COPY support)
pip install "tokenledger[timescale]"

# ClickHouse
pip install tokenledger tokenledger-clickhouse
# or with async
pip install tokenledger "tokenledger-clickhouse[async]"

# BigQuery
pip install tokenledger tokenledger-bigquery
# or with Storage Write API
pip install tokenledger "tokenledger-bigquery[storage]"

# Snowflake
pip install tokenledger tokenledger-snowflake

# Everything (for development)
pip install "tokenledger[all-backends]"
pip install tokenledger-clickhouse tokenledger-bigquery tokenledger-snowflake
```

### Usage Examples

```python
# Auto-detect backend from URL
import tokenledger

# PostgreSQL (default)
tokenledger.configure(database_url="postgresql://localhost/mydb")

# ClickHouse
tokenledger.configure(database_url="tokenledger+clickhouse://localhost:9000/default")

# BigQuery (requires explicit configuration)
from tokenledger.backends import registry

backend = registry.create_backend(
    "bigquery",
    config=tokenledger.get_config(),
    project_id="my-project",
    dataset_id="analytics",
)

# Explicit backend selection
from tokenledger.backends import registry

BackendClass = registry.load_backend("timescale")
backend = BackendClass()
backend.initialize(config)

# List available backends
print(registry.get_available_backends())
# {'postgres': '...', 'asyncpg': '...', 'clickhouse': '...', ...}
```

---

## Migration/Schema Strategy

### Per-Backend Migration Approach

Each backend has different schema management requirements:

| Backend | Migration Strategy | Tool |
|---------|-------------------|------|
| PostgreSQL | SQL migrations | Alembic or manual |
| TimescaleDB | SQL + hypertable DDL | Alembic with TimescaleDB support |
| ClickHouse | ALTER TABLE (limited) | Manual SQL scripts |
| BigQuery | Schema evolution | BigQuery schema updates |
| Snowflake | ALTER TABLE | Snowflake migrations |

### Migration Interface

```python
# tokenledger/backends/migrations.py

from abc import ABC, abstractmethod
from typing import Any


class MigrationManager(ABC):
    """
    Abstract base for backend-specific migrations.

    Each backend implements its own migration strategy based on
    the database's schema evolution capabilities.
    """

    @abstractmethod
    def get_current_version(self) -> int:
        """Get the current schema version."""
        ...

    @abstractmethod
    def get_available_migrations(self) -> list[tuple[int, str]]:
        """Get list of (version, description) for available migrations."""
        ...

    @abstractmethod
    def migrate(self, target_version: int | None = None) -> None:
        """
        Run migrations up to target_version.

        Args:
            target_version: Target version, or None for latest
        """
        ...

    @abstractmethod
    def rollback(self, target_version: int) -> None:
        """
        Rollback to target_version.

        Note: Not all backends support rollback.
        """
        ...


class PostgresMigrationManager(MigrationManager):
    """PostgreSQL migration manager using version table."""

    MIGRATIONS = [
        (1, "Initial schema", """
            CREATE TABLE IF NOT EXISTS token_ledger_events (...);
            CREATE INDEX IF NOT EXISTS idx_token_ledger_timestamp ON token_ledger_events (timestamp DESC);
        """),
        (2, "Add cached_tokens column", """
            ALTER TABLE token_ledger_events ADD COLUMN IF NOT EXISTS cached_tokens INTEGER DEFAULT 0;
        """),
        (3, "Add composite dashboard index", """
            CREATE INDEX IF NOT EXISTS idx_token_ledger_dashboard
            ON token_ledger_events (timestamp DESC, model, user_id)
            INCLUDE (cost_usd, total_tokens);
        """),
    ]

    def __init__(self, connection):
        self._conn = connection
        self._ensure_version_table()

    def _ensure_version_table(self):
        with self._conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS token_ledger_schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TIMESTAMPTZ DEFAULT NOW(),
                    description TEXT
                );
            """)
        self._conn.commit()

    def get_current_version(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute("SELECT COALESCE(MAX(version), 0) FROM token_ledger_schema_version")
            return cur.fetchone()[0]

    def migrate(self, target_version: int | None = None) -> None:
        current = self.get_current_version()
        target = target_version or max(v for v, _, _ in self.MIGRATIONS)

        for version, description, sql in self.MIGRATIONS:
            if current < version <= target:
                with self._conn.cursor() as cur:
                    cur.execute(sql)
                    cur.execute(
                        "INSERT INTO token_ledger_schema_version (version, description) VALUES (%s, %s)",
                        (version, description)
                    )
                self._conn.commit()
                print(f"Applied migration {version}: {description}")


class ClickHouseMigrationManager(MigrationManager):
    """
    ClickHouse migration manager.

    Note: ClickHouse has limited ALTER TABLE support.
    Adding columns is easy, but modifying/removing requires table recreation.
    """

    def migrate(self, target_version: int | None = None) -> None:
        # ClickHouse approach: use ReplacingMergeTree for schema versioning
        # New columns can be added with ALTER TABLE ADD COLUMN
        # Incompatible changes require creating new table and migrating data
        pass
```

### Schema Versioning Strategy

```python
# tokenledger/backends/schema.py
"""
Schema versioning and compatibility checking.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class SchemaVersion:
    """Represents a schema version with compatibility info."""
    version: int
    min_compatible_version: int
    description: str
    changes: list[str]


# Current schema version
CURRENT_SCHEMA = SchemaVersion(
    version=3,
    min_compatible_version=1,
    description="TokenLedger schema v3 with cached_tokens and dashboard index",
    changes=[
        "v1: Initial schema with token_ledger_events table",
        "v2: Added cached_tokens column",
        "v3: Added composite dashboard index",
    ]
)


def check_schema_compatibility(backend_version: int) -> bool:
    """Check if backend schema is compatible with current code."""
    return backend_version >= CURRENT_SCHEMA.min_compatible_version


def get_required_migrations(current_version: int) -> list[int]:
    """Get list of migrations needed to reach current schema."""
    return [
        v for v in range(current_version + 1, CURRENT_SCHEMA.version + 1)
    ]
```

---

## Implementation Roadmap

### Phase 1: Core Plugin Infrastructure (Week 1-2)

1. **Create Protocol and Base Classes**
   - `tokenledger/backends/protocol.py`
   - `tokenledger/backends/base.py`
   - `tokenledger/backends/registry.py`

2. **Refactor PostgreSQL Backend**
   - Extract from `tracker.py` and `async_db.py`
   - Implement as first backend using new architecture
   - Maintain backward compatibility

3. **Update Configuration**
   - Add backend selection to `TokenLedgerConfig`
   - Support URL-based backend detection

### Phase 2: TimescaleDB Backend (Week 3)

1. **Built-in TimescaleDB Support**
   - Extend PostgreSQL backend
   - Add hypertable creation
   - Implement continuous aggregates
   - COPY-based bulk inserts

### Phase 3: ClickHouse Plugin (Week 4-5)

1. **Create `tokenledger-clickhouse` Package**
   - Sync backend with clickhouse-driver
   - Async backend with asynch
   - MergeTree schema optimization
   - Optional Kafka buffer support

### Phase 4: Cloud Backends (Week 6-8)

1. **Create `tokenledger-bigquery` Package**
   - Storage Write API integration
   - Streaming and batch modes
   - Partition and clustering

2. **Create `tokenledger-snowflake` Package**
   - Standard connector integration
   - Snowpipe Streaming (optional)
   - VARIANT type for metadata

### Phase 5: Documentation and Testing (Week 9-10)

1. **Comprehensive Documentation**
   - Backend selection guide
   - Performance tuning per backend
   - Migration guides

2. **Test Suite**
   - Backend compliance tests
   - Performance benchmarks
   - Integration tests per backend

---

## References

### Python Plugin Patterns

- [Python Packaging Guide: Creating and Discovering Plugins](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/)
- [Setuptools Entry Points](https://setuptools.pypa.io/en/latest/userguide/entry_point.html)
- [PEP 544 - Protocols: Structural Subtyping](https://peps.python.org/pep-0544/)
- [ABC vs Protocol Guide](https://jellis18.github.io/post/2022-01-11-abc-vs-protocol/)

### SQLAlchemy Architecture

- [SQLAlchemy Dialects](https://docs.sqlalchemy.org/en/20/dialects/)
- [SQLAlchemy README.dialects.rst](https://github.com/sqlalchemy/sqlalchemy/blob/main/README.dialects.rst)
- [SQLAlchemy Entry Points (Wheelodex)](https://www.wheelodex.org/entry-points/sqlalchemy.dialects/)

### Database-Specific Resources

**ClickHouse:**
- [ClickHouse Insert Strategy](https://clickhouse.com/docs/best-practices/selecting-an-insert-strategy)
- [ClickHouse Async Inserts](https://clickhouse.com/docs/optimize/asynchronous-inserts)
- [asynch - Async ClickHouse Driver](https://github.com/long2ice/asynch)

**TimescaleDB:**
- [TimescaleDB Best Practices](https://docs-dev.timescale.com/docs-tutorial-lambda-cd/timescaledb/tutorial-lambda-cd/how-to-guides/hypertables/best-practices/)
- [Building Apps with asyncpg](https://www.tigerdata.com/blog/how-to-build-applications-with-asyncpg-and-postgresql)

**BigQuery:**
- [BigQuery Storage Write API](https://docs.cloud.google.com/bigquery/docs/write-api)
- [BigQuery Streaming vs Job Load](https://medium.com/@sattarkars45/bigquery-streaming-vs-job-load-understanding-write-disposition-and-when-to-use-each-4f084fd4c202)

**Snowflake:**
- [Snowflake Python Connector](https://docs.snowflake.com/en/developer-guide/python-connector/python-connector-example)
- [Snowflake Ingestion Mechanisms 2025](https://medium.com/@divyanshsaxenaofficial/2025-confused-between-copy-snowpipe-dynamic-tables-7c69475e0914)

### Apache Airflow (Provider Architecture Reference)

- [Airflow Provider Packages](https://airflow.apache.org/docs/apache-airflow-providers-amazon/stable/index.html)
- [Airflow Standard Provider](https://pypi.org/project/apache-airflow-providers-standard/)
