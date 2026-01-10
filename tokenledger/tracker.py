"""
TokenLedger Core Tracker
Handles event logging to PostgreSQL with batching and async support.
"""

import atexit
import json
import logging
import random
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from queue import Empty, Queue
from typing import Any

from .config import TokenLedgerConfig, get_config
from .pricing import calculate_cost

logger = logging.getLogger("tokenledger")


@dataclass
class LLMEvent:
    """Represents a single LLM API call event"""

    # Identifiers
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str | None = None
    span_id: str | None = None
    parent_span_id: str | None = None

    # Timing
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    duration_ms: float | None = None

    # Provider & Model
    provider: str = ""
    model: str = ""

    # Token counts
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0

    # Cost
    cost_usd: float | None = None

    # Request details
    endpoint: str | None = None
    request_type: str = "chat"  # chat, completion, embedding, etc.

    # User & context
    user_id: str | None = None
    session_id: str | None = None
    organization_id: str | None = None

    # Application context
    app_name: str | None = None
    environment: str | None = None

    # Status
    status: str = "success"  # success, error, timeout
    error_type: str | None = None
    error_message: str | None = None

    # Custom metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Request/Response (optional, for debugging)
    request_preview: str | None = None  # First N chars of prompt
    response_preview: str | None = None  # First N chars of response

    def __post_init__(self):
        self.total_tokens = self.input_tokens + self.output_tokens

        # Calculate cost if not provided
        if self.cost_usd is None and self.model:
            self.cost_usd = calculate_cost(
                self.model, self.input_tokens, self.output_tokens, self.cached_tokens, self.provider
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for database insertion"""
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        d["metadata"] = json.dumps(self.metadata) if self.metadata else None
        return d


class TokenTracker:
    """
    Main tracker class for logging LLM events to PostgreSQL.
    Supports batching, async logging, and sampling.
    """

    def __init__(self, config: TokenLedgerConfig | None = None):
        self.config = config or get_config()
        self._connection = None
        self._async_connection = None
        self._queue: Queue = Queue(maxsize=self.config.max_queue_size)
        self._batch: list[LLMEvent] = []
        self._lock = threading.Lock()
        self._flush_thread: threading.Thread | None = None
        self._running = False
        self._initialized = False

        # Context for tracking
        self._context = threading.local()

    def _get_connection(self):
        """Get or create database connection"""
        if self._connection is None:
            try:
                import psycopg2
                from psycopg2.extras import execute_values

                self._connection = psycopg2.connect(self.config.database_url)
                self._execute_values = execute_values
            except ImportError:
                try:
                    import psycopg

                    self._connection = psycopg.connect(self.config.database_url)
                except ImportError as err:
                    raise ImportError(
                        "No PostgreSQL driver found. Install psycopg2 or psycopg: "
                        "pip install psycopg2-binary"
                    ) from err
        return self._connection

    def initialize(self, create_tables: bool = True) -> None:
        """
        Initialize the tracker and optionally create tables.

        Args:
            create_tables: Whether to create the events table if it doesn't exist
        """
        if self._initialized:
            return

        conn = self._get_connection()

        if create_tables:
            self._create_tables(conn)

        if self.config.async_mode:
            self._start_flush_thread()

        self._initialized = True
        atexit.register(self.shutdown)

    def _create_tables(self, conn) -> None:
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

        with conn.cursor() as cur:
            cur.execute(create_sql)
        conn.commit()

        if self.config.debug:
            logger.info(f"Created table {self.config.full_table_name}")

    def _start_flush_thread(self) -> None:
        """Start background thread for flushing batches"""
        self._running = True
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()

    def _flush_loop(self) -> None:
        """Background loop for periodic flushing"""
        while self._running:
            try:
                time.sleep(self.config.flush_interval_seconds)
                self.flush()
            except Exception as e:
                logger.error(f"Error in flush loop: {e}")

    def track(self, event: LLMEvent) -> None:
        """
        Track an LLM event.

        Args:
            event: The LLMEvent to track
        """
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

        # Add trace context if available
        if hasattr(self._context, "trace_id") and not event.trace_id:
            event.trace_id = self._context.trace_id

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
        """Add event to batch with thread safety"""
        with self._lock:
            self._batch.append(event)

    def flush(self) -> int:
        """
        Flush all pending events to the database.

        Returns:
            Number of events flushed
        """
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

        return self._write_batch(events_to_flush)

    def _write_batch(self, events: list[LLMEvent]) -> int:
        """Write a batch of events to the database"""
        if not events:
            return 0

        conn = self._get_connection()

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

        values = []
        for event in events:
            d = event.to_dict()
            values.append(tuple(d.get(col) for col in columns))

        insert_sql = f"""
            INSERT INTO {self.config.full_table_name}
            ({", ".join(columns)})
            VALUES %s
            ON CONFLICT (event_id) DO NOTHING
        """

        try:
            with conn.cursor() as cur:
                self._execute_values(cur, insert_sql, values)
            conn.commit()

            if self.config.debug:
                logger.info(f"Flushed {len(events)} events to database")

            return len(events)
        except Exception as e:
            logger.error(f"Error writing batch: {e}")
            conn.rollback()
            return 0

    @contextmanager
    def trace(self, trace_id: str | None = None):
        """
        Context manager for tracing a group of related LLM calls.

        Args:
            trace_id: Optional trace ID, will be generated if not provided

        Example:
            >>> with tracker.trace() as trace_id:
            ...     # All LLM calls in this block will share the trace_id
            ...     response = openai.chat(...)
        """
        self._context.trace_id = trace_id or str(uuid.uuid4())
        try:
            yield self._context.trace_id
        finally:
            delattr(self._context, "trace_id")

    def shutdown(self) -> None:
        """Shutdown the tracker, flushing any pending events"""
        self._running = False
        if self._flush_thread:
            self._flush_thread.join(timeout=5.0)
        self.flush()
        if self._connection:
            self._connection.close()
            self._connection = None


# Global tracker instance
_tracker: TokenTracker | None = None


def get_tracker() -> TokenTracker:
    """Get or create the global tracker instance"""
    global _tracker
    if _tracker is None:
        _tracker = TokenTracker()
    return _tracker


def track_event(event: LLMEvent) -> None:
    """Track an event using the global tracker"""
    get_tracker().track(event)
