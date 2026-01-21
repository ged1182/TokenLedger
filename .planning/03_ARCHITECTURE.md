# TokenLedger Architecture Analysis & Plugin Extensibility

**Author:** Architecture Review
**Date:** January 2026
**Status:** Technical Analysis Document

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current Architecture](#current-architecture)
3. [Coupling Analysis](#coupling-analysis)
4. [Plugin Architecture Proposal](#plugin-architecture-proposal)
5. [Interface Definitions](#interface-definitions)
6. [Refactoring Roadmap](#refactoring-roadmap)
7. [Backend-Specific Considerations](#backend-specific-considerations)
8. [Recommendations](#recommendations)

---

## Executive Summary

TokenLedger currently has a **tightly-coupled PostgreSQL architecture** with database-specific code embedded throughout the tracker, queries, and async modules. While functional for PostgreSQL deployments, this design prevents support for alternative backends such as ClickHouse, TimescaleDB, BigQuery, or Snowflake.

This document proposes a **plugin architecture** using Python Protocols (PEP 544) and entry points to enable:

- **Hot-swappable storage backends** without code changes
- **Async-first design** for streaming backends (Kafka/Redpanda)
- **Clean separation of concerns** between event capture and persistence
- **Zero-config backend discovery** via setuptools entry points

**Estimated refactoring effort:** 2-3 weeks for core abstractions, 1 week per additional backend.

---

## Current Architecture

### System Overview (Text-Based Diagram)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           User Application                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │ OpenAI SDK   │  │ Anthropic SDK│  │ @track_llm   │  │ track_cost()     │ │
│  │              │  │              │  │ decorator    │  │ function         │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘ │
└─────────┼────────────────┼────────────────┼─────────────────────┼───────────┘
          │                │                │                     │
          ▼                ▼                ▼                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Interceptor Layer                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  interceptors/openai.py  │  interceptors/anthropic.py                │   │
│  │  - Monkey patches SDK methods                                         │   │
│  │  - Extracts tokens, model, latency                                    │   │
│  │  - Creates LLMEvent objects                                           │   │
│  └──────────────────────────────┬───────────────────────────────────────┘   │
└─────────────────────────────────┼───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Core Layer                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     tracker.py                                        │   │
│  │  ┌────────────┐  ┌──────────────────┐  ┌────────────────────────┐    │   │
│  │  │ LLMEvent   │  │ TokenTracker     │  │ AsyncTokenTracker      │    │   │
│  │  │ dataclass  │  │ - Queue + Thread │  │ - asyncio.Lock         │    │   │
│  │  │            │  │ - Batching logic │  │ - Batch buffering      │    │   │
│  │  │            │  │ - Sampling       │  │ - Delegates to         │    │   │
│  │  │            │  │                  │  │   AsyncDatabase        │    │   │
│  │  └────────────┘  └────────┬─────────┘  └───────────┬────────────┘    │   │
│  └───────────────────────────┼────────────────────────┼─────────────────┘   │
│                              │                        │                      │
│  ┌───────────────────────────┼────────────────────────┼─────────────────┐   │
│  │                 Database Access (TIGHTLY COUPLED)                     │   │
│  │                              │                        │               │   │
│  │                              ▼                        ▼               │   │
│  │  ┌──────────────────────────────────────────────────────────────┐    │   │
│  │  │  PostgreSQL-Specific Code                                     │    │   │
│  │  │  - psycopg2/psycopg3 (sync)                                   │    │   │
│  │  │  - asyncpg (async)                                            │    │   │
│  │  │  - SQL with PostgreSQL-specific syntax                        │    │   │
│  │  │  - JSONB, TIMESTAMPTZ, UUID types                             │    │   │
│  │  │  - PERCENTILE_CONT, DATE_TRUNC functions                      │    │   │
│  │  └──────────────────────────────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PostgreSQL                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  token_ledger_events table                                            │   │
│  │  - UUID primary key                                                   │   │
│  │  - JSONB metadata                                                     │   │
│  │  - Composite indexes                                                  │   │
│  │  - Views: token_ledger_daily_costs, token_ledger_user_costs           │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Module Responsibilities

| Module | Responsibility | PostgreSQL Coupling |
|--------|---------------|---------------------|
| `tracker.py` | Event capture, batching, queue management | **HIGH** - Direct psycopg2/psycopg3 imports, SQL DDL |
| `async_db.py` | Async connection pooling, batch inserts | **VERY HIGH** - asyncpg-specific, SQL syntax |
| `queries.py` | Analytics queries (sync & async) | **VERY HIGH** - PostgreSQL-specific SQL functions |
| `config.py` | Configuration management | **LOW** - Just `database_url` string |
| `server.py` | FastAPI dashboard endpoints | **MEDIUM** - Uses queries.py |
| `interceptors/*` | SDK monkey-patching | **NONE** - Pure event creation |
| `decorators.py` | @track_llm, track_cost() | **NONE** - Uses tracker interface |
| `middleware.py` | FastAPI/Flask integration | **NONE** - Uses tracker interface |
| `pricing.py` | Cost calculation | **NONE** - Pure functions |

---

## Coupling Analysis

### Tightly Coupled Components

#### 1. `tracker.py` - TokenTracker._write_batch()

**Location:** Lines 303-374

**PostgreSQL Dependencies:**
- `psycopg2.connect()` / `psycopg.connect()` - Direct driver imports
- `execute_values()` - psycopg2-specific bulk insert
- SQL `INSERT ... ON CONFLICT (event_id) DO NOTHING` - PostgreSQL upsert syntax
- `conn.rollback()` / `conn.commit()` - Transaction management

```python
# Current tight coupling (tracker.py:119-136)
def _get_connection(self):
    if self._connection is None:
        try:
            import psycopg2
            from psycopg2.extras import execute_values
            self._connection = psycopg2.connect(self.config.database_url)
            # ... PostgreSQL-specific setup
```

**Problem:** The tracker directly instantiates PostgreSQL connections. No abstraction layer.

#### 2. `tracker.py` - TokenTracker._create_tables()

**Location:** Lines 159-217

**PostgreSQL Dependencies:**
- `UUID PRIMARY KEY` - PostgreSQL UUID type
- `TIMESTAMPTZ` - PostgreSQL timestamp with timezone
- `JSONB` - PostgreSQL binary JSON
- `CREATE INDEX IF NOT EXISTS` - PostgreSQL DDL

**Problem:** Schema creation is embedded in the tracker, not delegated to a storage adapter.

#### 3. `async_db.py` - AsyncDatabase

**Location:** Entire file (314 lines)

**PostgreSQL Dependencies:**
- `asyncpg.create_pool()` - asyncpg-specific pooling
- `$1, $2, ...` - asyncpg parameter style (vs `%s` for psycopg)
- `conn.execute()`, `conn.fetch()` - asyncpg API
- Duplicate schema DDL from tracker.py

**Problem:** Entire module is asyncpg-specific with no interface abstraction.

#### 4. `queries.py` - TokenLedgerQueries & AsyncTokenLedgerQueries

**Location:** Entire file (803 lines)

**PostgreSQL Dependencies:**
- `PERCENTILE_CONT() WITHIN GROUP` - PostgreSQL window function
- `DATE_TRUNC('hour', timestamp)` - PostgreSQL date function
- `NOW() - INTERVAL '%s days'` - PostgreSQL interval syntax
- Parameter binding: `%s` (psycopg) vs `$1` (asyncpg)

**Problem:** All analytics queries use PostgreSQL-specific SQL that won't work on other databases.

### Loosely Coupled Components (Good Examples)

#### 1. `LLMEvent` Dataclass

**Location:** `tracker.py:25-92`

**Strengths:**
- Pure Python dataclass with no database dependencies
- `to_dict()` method provides serialization interface
- Could be directly serialized to JSON for Kafka/BigQuery

#### 2. Interceptors (`interceptors/openai.py`, `interceptors/anthropic.py`)

**Strengths:**
- Create `LLMEvent` objects, don't interact with storage
- Call `tracker.track(event)` - clean interface
- Could work with any tracker implementation

#### 3. `pricing.py`

**Strengths:**
- Pure functions for cost calculation
- No I/O dependencies
- Reusable across all backends

---

## Plugin Architecture Proposal

### Design Principles

1. **Protocol-based interfaces** (PEP 544) for structural typing
2. **Entry points** for zero-config plugin discovery
3. **Async-first** with sync wrappers for compatibility
4. **Factory pattern** for backend instantiation
5. **Separate concerns**: Event capture vs. Persistence vs. Querying

### Proposed Architecture (Text-Based Diagram)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           User Application                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Interceptors & Decorators (unchanged)                    │
│                     Creates LLMEvent objects                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Core Tracker Interface                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  TrackerProtocol                                                      │   │
│  │  - track(event: LLMEvent) -> None                                     │   │
│  │  - flush() -> int                                                     │   │
│  │  - shutdown() -> None                                                 │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  BaseTracker (abstract)                                               │   │
│  │  - Batching logic                                                     │   │
│  │  - Sampling                                                           │   │
│  │  - Queue management                                                   │   │
│  │  - Delegates to StorageBackend                                        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Storage Backend Interface                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  StorageBackendProtocol                                               │   │
│  │  - async initialize() -> None                                         │   │
│  │  - async write_events(events: list[LLMEvent]) -> int                  │   │
│  │  - async close() -> None                                              │   │
│  │  - async health_check() -> bool                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│  ┌───────────────────────────┼─────────────────────────────────────────┐    │
│  │                           │                                          │    │
│  ▼                           ▼                           ▼              ▼    │
│  ┌────────────┐  ┌────────────────┐  ┌───────────┐  ┌──────────────┐        │
│  │ PostgreSQL │  │ ClickHouse     │  │ BigQuery  │  │ Snowflake    │        │
│  │ Backend    │  │ Backend        │  │ Backend   │  │ Backend      │        │
│  │            │  │ (via Kafka)    │  │           │  │              │        │
│  └────────────┘  └────────────────┘  └───────────┘  └──────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Query Interface (Separate)                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  QueryBackendProtocol                                                 │   │
│  │  - async get_cost_summary(...) -> CostSummary                         │   │
│  │  - async get_costs_by_model(...) -> list[ModelCost]                   │   │
│  │  - async get_daily_costs(...) -> list[DailyCost]                      │   │
│  │  - ... (other analytics methods)                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Plugin Discovery via Entry Points

```toml
# pyproject.toml
[project.entry-points."tokenledger.backends"]
postgresql = "tokenledger.backends.postgresql:PostgreSQLBackend"
timescaledb = "tokenledger.backends.timescaledb:TimescaleDBBackend"
clickhouse = "tokenledger.backends.clickhouse:ClickHouseBackend"
bigquery = "tokenledger.backends.bigquery:BigQueryBackend"
snowflake = "tokenledger.backends.snowflake:SnowflakeBackend"

[project.entry-points."tokenledger.query_backends"]
postgresql = "tokenledger.backends.postgresql:PostgreSQLQueryBackend"
# ... similar for other backends
```

---

## Interface Definitions

### Core Protocols

```python
# tokenledger/protocols.py
"""
TokenLedger Protocol Definitions (PEP 544)
These define the contracts for pluggable storage backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, runtime_checkable


@dataclass
class LLMEvent:
    """Event data structure - unchanged from current implementation"""
    event_id: str
    timestamp: datetime
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_tokens: int
    cost_usd: float | None
    duration_ms: float | None
    user_id: str | None
    session_id: str | None
    organization_id: str | None
    app_name: str | None
    environment: str | None
    status: str
    error_type: str | None
    error_message: str | None
    metadata: dict[str, Any]
    # ... additional fields


@runtime_checkable
class StorageBackendProtocol(Protocol):
    """
    Protocol for storage backends.

    All backends must implement async methods for write operations.
    Sync trackers wrap these in asyncio.run() or similar.
    """

    @property
    def backend_name(self) -> str:
        """Human-readable backend name for logging."""
        ...

    @property
    def supports_transactions(self) -> bool:
        """Whether the backend supports ACID transactions."""
        ...

    @property
    def supports_streaming(self) -> bool:
        """Whether the backend supports streaming writes (e.g., Kafka)."""
        ...

    async def initialize(self, create_schema: bool = True) -> None:
        """
        Initialize the backend (create connections, pools, tables).

        Args:
            create_schema: Whether to create tables/schema if not exists

        Raises:
            ConnectionError: If unable to connect to the backend
            SchemaError: If schema creation fails
        """
        ...

    async def write_events(self, events: list[LLMEvent]) -> int:
        """
        Write a batch of events to the storage backend.

        Args:
            events: List of LLMEvent objects to persist

        Returns:
            Number of events successfully written

        Raises:
            WriteError: If the write operation fails
        """
        ...

    async def health_check(self) -> bool:
        """
        Check if the backend is healthy and accepting writes.

        Returns:
            True if healthy, False otherwise
        """
        ...

    async def close(self) -> None:
        """
        Close connections and release resources.

        Should be idempotent (safe to call multiple times).
        """
        ...


@runtime_checkable
class QueryBackendProtocol(Protocol):
    """
    Protocol for query backends.

    Separated from StorageBackendProtocol because:
    1. Write and read paths may use different optimizations
    2. Some backends (Kafka) are write-only
    3. Query syntax varies significantly between databases
    """

    async def get_cost_summary(
        self,
        days: int = 30,
        user_id: str | None = None,
        model: str | None = None,
        app_name: str | None = None,
    ) -> CostSummary:
        """Get cost summary for a time period."""
        ...

    async def get_costs_by_model(
        self,
        days: int = 30,
        limit: int = 10,
    ) -> list[ModelCost]:
        """Get cost breakdown by model."""
        ...

    async def get_costs_by_user(
        self,
        days: int = 30,
        limit: int = 20,
    ) -> list[UserCost]:
        """Get cost breakdown by user."""
        ...

    async def get_daily_costs(
        self,
        days: int = 30,
        user_id: str | None = None,
    ) -> list[DailyCost]:
        """Get daily cost trends."""
        ...

    async def get_hourly_costs(
        self,
        hours: int = 24,
    ) -> list[HourlyCost]:
        """Get hourly cost trends."""
        ...

    async def get_error_rate(
        self,
        days: int = 7,
    ) -> dict[str, Any]:
        """Get error rate statistics."""
        ...

    async def get_latency_percentiles(
        self,
        days: int = 7,
    ) -> dict[str, float]:
        """Get latency percentiles (p50, p90, p95, p99)."""
        ...


@runtime_checkable
class StreamingBackendProtocol(Protocol):
    """
    Protocol for streaming backends (Kafka, Redpanda, Kinesis).

    Extends StorageBackendProtocol with streaming-specific methods.
    """

    async def initialize(self, create_schema: bool = True) -> None:
        ...

    async def write_events(self, events: list[LLMEvent]) -> int:
        ...

    async def health_check(self) -> bool:
        ...

    async def close(self) -> None:
        ...

    @property
    def topic_name(self) -> str:
        """Kafka/Redpanda topic name."""
        ...

    async def ensure_topic_exists(self) -> None:
        """Create the topic if it doesn't exist."""
        ...

    async def get_consumer(self) -> Any:
        """Get a consumer for reading events (for testing/debugging)."""
        ...


# Result dataclasses (unchanged from current implementation)
@dataclass
class CostSummary:
    total_cost: float
    total_tokens: int
    total_input_tokens: int
    total_output_tokens: int
    total_requests: int
    avg_cost_per_request: float
    avg_tokens_per_request: float


@dataclass
class ModelCost:
    model: str
    provider: str
    total_cost: float
    total_requests: int
    total_tokens: int
    avg_cost_per_request: float


@dataclass
class UserCost:
    user_id: str
    total_cost: float
    total_requests: int
    total_tokens: int


@dataclass
class DailyCost:
    date: datetime
    total_cost: float
    total_requests: int
    total_tokens: int


@dataclass
class HourlyCost:
    hour: datetime
    total_cost: float
    total_requests: int
```

### Backend Factory

```python
# tokenledger/backends/factory.py
"""
Backend factory with entry point discovery.
"""

from importlib.metadata import entry_points
from typing import Type

from ..config import TokenLedgerConfig
from ..protocols import StorageBackendProtocol, QueryBackendProtocol


class BackendRegistry:
    """Registry for storage and query backends."""

    _storage_backends: dict[str, Type[StorageBackendProtocol]] = {}
    _query_backends: dict[str, Type[QueryBackendProtocol]] = {}
    _discovered: bool = False

    @classmethod
    def discover_backends(cls) -> None:
        """Discover backends via entry points."""
        if cls._discovered:
            return

        # Python 3.10+ style
        eps = entry_points(group="tokenledger.backends")
        for ep in eps:
            try:
                backend_class = ep.load()
                cls._storage_backends[ep.name] = backend_class
            except Exception as e:
                import logging
                logging.warning(f"Failed to load backend {ep.name}: {e}")

        eps = entry_points(group="tokenledger.query_backends")
        for ep in eps:
            try:
                backend_class = ep.load()
                cls._query_backends[ep.name] = backend_class
            except Exception as e:
                import logging
                logging.warning(f"Failed to load query backend {ep.name}: {e}")

        cls._discovered = True

    @classmethod
    def get_storage_backend(
        cls,
        name: str,
        config: TokenLedgerConfig,
    ) -> StorageBackendProtocol:
        """Get a storage backend by name."""
        cls.discover_backends()

        if name not in cls._storage_backends:
            available = list(cls._storage_backends.keys())
            raise ValueError(
                f"Unknown storage backend: {name}. "
                f"Available: {available}"
            )

        backend_class = cls._storage_backends[name]
        return backend_class(config)

    @classmethod
    def get_query_backend(
        cls,
        name: str,
        config: TokenLedgerConfig,
    ) -> QueryBackendProtocol:
        """Get a query backend by name."""
        cls.discover_backends()

        if name not in cls._query_backends:
            available = list(cls._query_backends.keys())
            raise ValueError(
                f"Unknown query backend: {name}. "
                f"Available: {available}"
            )

        backend_class = cls._query_backends[name]
        return backend_class(config)

    @classmethod
    def list_storage_backends(cls) -> list[str]:
        """List available storage backend names."""
        cls.discover_backends()
        return list(cls._storage_backends.keys())

    @classmethod
    def list_query_backends(cls) -> list[str]:
        """List available query backend names."""
        cls.discover_backends()
        return list(cls._query_backends.keys())


def get_backend(
    config: TokenLedgerConfig | None = None,
    backend_name: str | None = None,
) -> StorageBackendProtocol:
    """
    Convenience function to get the configured storage backend.

    Args:
        config: Configuration object. Uses global config if not provided.
        backend_name: Backend name. Uses config.backend if not provided.

    Returns:
        Configured storage backend instance.
    """
    from ..config import get_config

    config = config or get_config()
    name = backend_name or getattr(config, "backend", "postgresql")

    return BackendRegistry.get_storage_backend(name, config)
```

### Abstract Base Tracker

```python
# tokenledger/tracker_base.py
"""
Abstract base tracker with common batching logic.
"""

import asyncio
import logging
import random
import threading
import time
from abc import ABC, abstractmethod
from queue import Empty, Queue
from typing import Any

from .config import TokenLedgerConfig, get_config
from .protocols import LLMEvent, StorageBackendProtocol

logger = logging.getLogger("tokenledger")


class BaseTracker(ABC):
    """
    Abstract base tracker with batching, sampling, and queue management.

    Subclasses must provide a storage backend.
    """

    def __init__(self, config: TokenLedgerConfig | None = None):
        self.config = config or get_config()
        self._backend: StorageBackendProtocol | None = None
        self._queue: Queue[LLMEvent] = Queue(maxsize=self.config.max_queue_size)
        self._batch: list[LLMEvent] = []
        self._lock = threading.Lock()
        self._flush_thread: threading.Thread | None = None
        self._running = False
        self._initialized = False
        self._context = threading.local()

    @abstractmethod
    def _get_backend(self) -> StorageBackendProtocol:
        """Get or create the storage backend. Implemented by subclasses."""
        ...

    def initialize(self, create_tables: bool = True) -> None:
        """Initialize the tracker and storage backend."""
        if self._initialized:
            return

        self._backend = self._get_backend()

        # Run async initialization synchronously
        asyncio.run(self._backend.initialize(create_schema=create_tables))

        if self.config.async_mode:
            self._start_flush_thread()

        self._initialized = True

    def track(self, event: LLMEvent) -> None:
        """Track an LLM event."""
        if not self._initialized:
            self.initialize()

        # Apply sampling
        if self.config.sample_rate < 1.0:
            if random.random() > self.config.sample_rate:
                return

        # Add default metadata
        if self.config.default_metadata:
            event.metadata = {**self.config.default_metadata, **event.metadata}

        # Add app info
        if not event.app_name:
            event.app_name = self.config.app_name
        if not event.environment:
            event.environment = self.config.environment

        if self.config.async_mode:
            try:
                self._queue.put_nowait(event)
            except Exception:
                logger.warning("Event queue full, dropping event")
        else:
            self._add_to_batch(event)
            if len(self._batch) >= self.config.batch_size:
                self.flush()

    def _add_to_batch(self, event: LLMEvent) -> None:
        """Add event to batch with thread safety."""
        with self._lock:
            self._batch.append(event)

    def _start_flush_thread(self) -> None:
        """Start background thread for flushing batches."""
        self._running = True
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()

    def _flush_loop(self) -> None:
        """Background loop for periodic flushing."""
        while self._running:
            try:
                time.sleep(self.config.flush_interval_seconds)
                self.flush()
            except Exception as e:
                logger.error(f"Error in flush loop: {e}")

    def flush(self) -> int:
        """Flush all pending events to the storage backend."""
        # Drain queue if in async mode
        if self.config.async_mode:
            while True:
                try:
                    event = self._queue.get_nowait()
                    self._add_to_batch(event)
                except Empty:
                    break

        with self._lock:
            if not self._batch:
                return 0

            events_to_flush = self._batch
            self._batch = []

        # Write via backend
        if self._backend:
            return asyncio.run(self._backend.write_events(events_to_flush))
        return 0

    def shutdown(self) -> None:
        """Shutdown the tracker, flushing pending events."""
        self._running = False
        if self._flush_thread:
            self._flush_thread.join(timeout=5.0)
        self.flush()
        if self._backend:
            asyncio.run(self._backend.close())
            self._backend = None


class AsyncBaseTracker(ABC):
    """
    Async abstract base tracker.

    Subclasses must provide a storage backend.
    """

    def __init__(self, config: TokenLedgerConfig | None = None):
        self.config = config or get_config()
        self._backend: StorageBackendProtocol | None = None
        self._batch: list[LLMEvent] = []
        self._lock: asyncio.Lock | None = None
        self._initialized = False

    @abstractmethod
    async def _get_backend(self) -> StorageBackendProtocol:
        """Get or create the storage backend. Implemented by subclasses."""
        ...

    async def _get_lock(self) -> asyncio.Lock:
        """Get or create the async lock."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def initialize(self, create_tables: bool = True) -> None:
        """Initialize the tracker and storage backend."""
        if self._initialized:
            return

        self._backend = await self._get_backend()
        await self._backend.initialize(create_schema=create_tables)
        self._initialized = True

    async def track(self, event: LLMEvent) -> None:
        """Track an LLM event asynchronously."""
        if not self._initialized:
            await self.initialize()

        # Apply sampling
        if self.config.sample_rate < 1.0:
            if random.random() > self.config.sample_rate:
                return

        # Add default metadata
        if self.config.default_metadata:
            event.metadata = {**self.config.default_metadata, **event.metadata}

        # Add app info
        if not event.app_name:
            event.app_name = self.config.app_name
        if not event.environment:
            event.environment = self.config.environment

        lock = await self._get_lock()
        async with lock:
            self._batch.append(event)
            if len(self._batch) >= self.config.batch_size:
                await self._flush_batch()

    async def _flush_batch(self) -> int:
        """Flush the current batch (assumes lock is held)."""
        if not self._batch:
            return 0

        events_to_flush = self._batch
        self._batch = []

        if self._backend:
            return await self._backend.write_events(events_to_flush)
        return 0

    async def flush(self) -> int:
        """Flush all pending events."""
        lock = await self._get_lock()
        async with lock:
            return await self._flush_batch()

    async def shutdown(self) -> None:
        """Shutdown the tracker."""
        await self.flush()
        if self._backend:
            await self._backend.close()
            self._backend = None
        self._initialized = False
```

---

## Refactoring Roadmap

### Phase 1: Interface Extraction (Week 1)

**Goal:** Define protocols without breaking existing functionality.

1. **Create `tokenledger/protocols.py`**
   - Define `StorageBackendProtocol`
   - Define `QueryBackendProtocol`
   - Define `StreamingBackendProtocol`
   - Move result dataclasses (`CostSummary`, etc.)

2. **Create `tokenledger/tracker_base.py`**
   - Extract batching logic from `tracker.py`
   - Create `BaseTracker` abstract class
   - Create `AsyncBaseTracker` abstract class

3. **Create `tokenledger/backends/__init__.py`**
   - Set up package structure
   - Create `factory.py` with `BackendRegistry`

**Deliverables:**
- New files created
- Existing code unchanged (backward compatible)
- Unit tests for protocols

### Phase 2: PostgreSQL Backend Extraction (Week 1-2)

**Goal:** Extract PostgreSQL code into a proper backend implementation.

1. **Create `tokenledger/backends/postgresql/`**
   ```
   backends/postgresql/
   ├── __init__.py
   ├── storage.py      # PostgreSQLBackend(StorageBackendProtocol)
   ├── queries.py      # PostgreSQLQueryBackend(QueryBackendProtocol)
   ├── schema.py       # DDL statements
   └── async_storage.py # AsyncPostgreSQLBackend
   ```

2. **Migrate code from:**
   - `tracker.py._create_tables()` -> `schema.py`
   - `tracker.py._write_batch()` -> `storage.py`
   - `async_db.py` -> `async_storage.py`
   - `queries.py` -> `postgresql/queries.py`

3. **Update entry points in `pyproject.toml`**

**Deliverables:**
- PostgreSQL backend as a proper plugin
- Old `tracker.py` delegates to backend
- All existing tests pass

### Phase 3: Update Trackers to Use Backends (Week 2)

**Goal:** Refactor trackers to use backend abstraction.

1. **Update `tracker.py`**
   - `TokenTracker` extends `BaseTracker`
   - `_get_backend()` returns PostgreSQL backend by default
   - Remove direct psycopg imports

2. **Update `async_db.py`**
   - Thin wrapper around `AsyncPostgreSQLBackend`
   - Deprecate direct use, prefer tracker

3. **Update `queries.py`**
   - Delegate to query backend
   - Keep backward-compatible API

4. **Update `config.py`**
   - Add `backend: str = "postgresql"` field
   - Add backend-specific config sections

**Deliverables:**
- Cleaner tracker implementations
- Backend configurable via config
- Deprecation warnings for old APIs

### Phase 4: TimescaleDB Backend (Week 3)

**Goal:** First alternative backend to validate architecture.

1. **Create `tokenledger/backends/timescaledb/`**
   - Extends PostgreSQL backend
   - Adds hypertable creation
   - Optimized time-series queries

2. **TimescaleDB-specific features:**
   - `CREATE EXTENSION IF NOT EXISTS timescaledb`
   - `SELECT create_hypertable(...)`
   - Continuous aggregates for rollups
   - Compression policies

**Deliverables:**
- Working TimescaleDB backend
- Documentation
- Integration tests

### Phase 5: ClickHouse Backend via Kafka (Week 3-4)

**Goal:** Streaming backend for high-volume analytics.

1. **Create `tokenledger/backends/kafka/`**
   - Implements `StreamingBackendProtocol`
   - Uses `aiokafka` for async production
   - JSON serialization with schema

2. **Create `tokenledger/backends/clickhouse/`**
   - Query backend for ClickHouse
   - Kafka consumer (separate process)
   - ClickHouse-specific SQL

3. **Architecture:**
   ```
   TokenTracker -> KafkaBackend -> Kafka Topic -> ClickHouse Kafka Engine -> ClickHouse Table
   ```

**Deliverables:**
- Kafka streaming backend
- ClickHouse query backend
- Docker Compose for local testing
- Consumer worker process

### Phase 6: BigQuery Backend (Week 4)

**Goal:** Serverless analytics backend.

1. **Create `tokenledger/backends/bigquery/`**
   - Uses `google-cloud-bigquery`
   - Streaming inserts or batch loading
   - BigQuery-specific SQL

2. **BigQuery considerations:**
   - Streaming inserts (real-time, more expensive)
   - Batch loading via GCS (cheaper, delayed)
   - Partitioned tables by timestamp
   - Clustered by model, user_id

**Deliverables:**
- Working BigQuery backend
- Cost optimization documentation
- GCS batch loading option

### Phase 7: Snowflake Backend (Week 5)

**Goal:** Enterprise data warehouse support.

1. **Create `tokenledger/backends/snowflake/`**
   - Uses `snowflake-connector-python`
   - Stage-based loading
   - Snowflake SQL dialect

2. **Snowflake considerations:**
   - Snowpipe for streaming
   - PUT/COPY for batch
   - Clustering keys
   - Warehouse sizing

**Deliverables:**
- Working Snowflake backend
- Documentation for Snowpipe setup
- Cost optimization guide

---

## Backend-Specific Considerations

### PostgreSQL (Current)

**Strengths:**
- ACID transactions
- JSONB for flexible metadata
- Rich analytics functions
- Existing implementation

**Optimizations:**
- Partitioning by timestamp (range)
- BRIN indexes for time-series
- Parallel query execution

**Schema:**
```sql
-- Current schema is well-designed
-- Add partitioning for scale:
CREATE TABLE token_ledger_events (
    ...
) PARTITION BY RANGE (timestamp);
```

### TimescaleDB

**Strengths:**
- Built on PostgreSQL (familiar)
- Automatic partitioning (chunks)
- Compression (10x reduction)
- Continuous aggregates

**Schema:**
```sql
-- Convert to hypertable
SELECT create_hypertable('token_ledger_events', 'timestamp');

-- Add compression
ALTER TABLE token_ledger_events SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'model, user_id'
);

-- Add compression policy
SELECT add_compression_policy('token_ledger_events', INTERVAL '7 days');

-- Continuous aggregate for daily costs
CREATE MATERIALIZED VIEW token_ledger_daily_agg
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', timestamp) AS bucket,
    model,
    provider,
    COUNT(*) as request_count,
    SUM(cost_usd) as total_cost
FROM token_ledger_events
GROUP BY bucket, model, provider;
```

### ClickHouse (via Kafka)

**Strengths:**
- Column-oriented (fast analytics)
- Extreme compression
- Horizontal scaling
- Real-time ingestion via Kafka

**Architecture:**
```
┌─────────────┐     ┌─────────┐     ┌────────────────────┐
│ TokenTracker│────▶│ Kafka   │────▶│ ClickHouse         │
│             │     │ Topic   │     │ Kafka Engine Table │
└─────────────┘     └─────────┘     └──────────┬─────────┘
                                               │
                                               ▼
                                    ┌────────────────────┐
                                    │ ClickHouse         │
                                    │ MergeTree Table    │
                                    └────────────────────┘
```

**Schema:**
```sql
-- Kafka engine table (ingestion)
CREATE TABLE token_ledger_events_queue (
    event_id String,
    timestamp DateTime64(3),
    provider LowCardinality(String),
    model LowCardinality(String),
    input_tokens UInt32,
    output_tokens UInt32,
    cost_usd Decimal64(8),
    user_id Nullable(String),
    metadata String  -- JSON string
) ENGINE = Kafka()
SETTINGS
    kafka_broker_list = 'kafka:9092',
    kafka_topic_list = 'tokenledger_events',
    kafka_group_name = 'clickhouse_consumer',
    kafka_format = 'JSONEachRow';

-- MergeTree table (storage)
CREATE TABLE token_ledger_events (
    event_id String,
    timestamp DateTime64(3),
    provider LowCardinality(String),
    model LowCardinality(String),
    input_tokens UInt32,
    output_tokens UInt32,
    total_tokens UInt32 MATERIALIZED input_tokens + output_tokens,
    cost_usd Decimal64(8),
    user_id Nullable(String),
    metadata String
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, provider, model)
TTL timestamp + INTERVAL 1 YEAR;

-- Materialized view for ingestion
CREATE MATERIALIZED VIEW token_ledger_events_mv TO token_ledger_events AS
SELECT * FROM token_ledger_events_queue;
```

### BigQuery

**Strengths:**
- Serverless (no infrastructure)
- Petabyte scale
- Standard SQL
- Integrates with GCP ecosystem

**Considerations:**
- Streaming inserts: $0.01 per 200 MB
- Batch loading: Free (from GCS)
- Storage: $0.02/GB/month
- Queries: $5/TB scanned

**Schema:**
```sql
-- BigQuery table
CREATE TABLE `project.dataset.token_ledger_events` (
    event_id STRING NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    provider STRING NOT NULL,
    model STRING NOT NULL,
    input_tokens INT64 NOT NULL,
    output_tokens INT64 NOT NULL,
    total_tokens INT64 NOT NULL,
    cached_tokens INT64,
    cost_usd NUMERIC,
    user_id STRING,
    session_id STRING,
    organization_id STRING,
    app_name STRING,
    environment STRING,
    status STRING,
    error_type STRING,
    error_message STRING,
    metadata JSON
)
PARTITION BY DATE(timestamp)
CLUSTER BY model, user_id;
```

### Snowflake

**Strengths:**
- Separation of storage and compute
- Time travel and cloning
- Semi-structured data support
- Enterprise security

**Schema:**
```sql
-- Snowflake table
CREATE TABLE token_ledger_events (
    event_id VARCHAR(36) NOT NULL,
    timestamp TIMESTAMP_TZ NOT NULL,
    provider VARCHAR(50) NOT NULL,
    model VARCHAR(100) NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,
    cached_tokens INTEGER,
    cost_usd DECIMAL(12, 8),
    user_id VARCHAR(255),
    session_id VARCHAR(255),
    organization_id VARCHAR(255),
    app_name VARCHAR(100),
    environment VARCHAR(50),
    status VARCHAR(20),
    error_type VARCHAR(100),
    error_message TEXT,
    metadata VARIANT  -- Semi-structured
)
CLUSTER BY (timestamp, model);

-- Snowpipe for streaming
CREATE PIPE token_ledger_pipe
AUTO_INGEST = TRUE
AS
COPY INTO token_ledger_events
FROM @token_ledger_stage
FILE_FORMAT = (TYPE = 'JSON');
```

---

## Recommendations

### Immediate Actions

1. **Start with Phase 1-2** - Extract interfaces without breaking changes
2. **Add `backend` config option** - Default to `"postgresql"` for backward compatibility
3. **Deprecate direct database access** - Guide users toward tracker interface

### Architecture Decisions

1. **Use Protocols over ABCs** - Better for structural typing, no inheritance required
2. **Async-first design** - Sync wrappers are simpler than async wrappers
3. **Separate read/write interfaces** - Different optimization strategies
4. **Entry points for discovery** - Zero-config plugin loading

### Testing Strategy

1. **Protocol compliance tests** - Verify backends implement protocols correctly
2. **Integration test suite** - Docker Compose with all backends
3. **Performance benchmarks** - Compare write throughput across backends

### Documentation Needs

1. **Backend selection guide** - Help users choose the right backend
2. **Migration guides** - PostgreSQL to TimescaleDB, etc.
3. **Plugin development guide** - How to create custom backends

### Future Considerations

1. **Multi-backend support** - Write to multiple backends simultaneously
2. **Backend migration tooling** - Export/import between backends
3. **Aggregation pipelines** - Pre-computed rollups for dashboards
4. **Retention policies** - Automatic data lifecycle management

---

## Appendix: File Changes Summary

### New Files to Create

```
tokenledger/
├── protocols.py           # Protocol definitions
├── tracker_base.py        # Abstract base trackers
├── backends/
│   ├── __init__.py
│   ├── factory.py         # Backend registry & factory
│   ├── postgresql/
│   │   ├── __init__.py
│   │   ├── storage.py     # PostgreSQLBackend
│   │   ├── queries.py     # PostgreSQLQueryBackend
│   │   ├── schema.py      # DDL statements
│   │   └── async_storage.py
│   ├── timescaledb/
│   │   └── ...
│   ├── clickhouse/
│   │   └── ...
│   ├── kafka/
│   │   └── ...
│   ├── bigquery/
│   │   └── ...
│   └── snowflake/
│       └── ...
```

### Files to Modify

| File | Changes |
|------|---------|
| `config.py` | Add `backend` field, backend-specific config |
| `tracker.py` | Extend `BaseTracker`, delegate to backend |
| `async_db.py` | Thin wrapper, deprecation warnings |
| `queries.py` | Delegate to query backend |
| `server.py` | Use query backend from config |
| `pyproject.toml` | Entry points, optional dependencies |

### Files Unchanged

| File | Reason |
|------|--------|
| `pricing.py` | Pure functions, no database dependency |
| `decorators.py` | Uses tracker interface |
| `middleware.py` | Uses tracker interface |
| `interceptors/*` | Uses tracker interface |

---

*This document provides a comprehensive roadmap for evolving TokenLedger from a PostgreSQL-only solution to a pluggable analytics platform supporting multiple storage backends. The phased approach ensures backward compatibility while enabling future extensibility.*
