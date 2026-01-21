# TokenLedger Streaming Architecture: Kafka/Redpanda

**Status:** Research & Design
**Last Updated:** January 2026

## Table of Contents

1. [Overview](#overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Library Recommendations](#library-recommendations)
4. [Schema Design](#schema-design)
5. [Producer Implementation](#producer-implementation)
6. [Consumer Implementation](#consumer-implementation)
7. [Delivery Guarantees](#delivery-guarantees)
8. [Backpressure & Flow Control](#backpressure--flow-control)
9. [Error Handling & Dead Letter Queue](#error-handling--dead-letter-queue)
10. [ClickHouse Integration](#clickhouse-integration)
11. [Observability & Metrics](#observability--metrics)
12. [Configuration Reference](#configuration-reference)
13. [Operational Runbook](#operational-runbook)

---

## Overview

This document defines the streaming architecture for TokenLedger's high-throughput LLM event pipeline. The system ingests LLM API usage events from Python applications, streams them through Kafka/Redpanda, and sinks to ClickHouse for analytics.

### Design Goals

1. **Async-first**: Native asyncio integration for Python applications
2. **High throughput**: Handle 100K+ events/second at scale
3. **At-least-once delivery**: No data loss with deduplication at sink
4. **Schema evolution**: Forward/backward compatible event schemas
5. **Operational simplicity**: Redpanda-compatible for simpler operations

### Key Decision Summary

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| Message Broker | Redpanda (Kafka-compatible) | No JVM, simpler ops, same API |
| Python Client | aiokafka (primary) | Native asyncio, better throughput |
| Serialization | JSON Schema (initial) -> Avro (v2) | Human-readable, Schema Registry support |
| Delivery | Idempotent producer + at-least-once | Simplicity over exactly-once |
| ClickHouse Sink | External Python consumer | Flexibility, separation of concerns |

---

## Architecture Diagram

```
                                     TokenLedger Streaming Architecture

+-------------------+     +-------------------+     +-------------------+
|   LLM API Call    |     |   LLM API Call    |     |   LLM API Call    |
|   (Application)   |     |   (Application)   |     |   (Application)   |
+--------+----------+     +--------+----------+     +--------+----------+
         |                         |                         |
         v                         v                         v
+--------+-------------------------+-------------------------+----------+
|                        TokenLedger SDK (Python)                       |
|  +------------------+  +------------------+  +------------------+      |
|  | OpenAI          |  | Anthropic        |  | Custom           |      |
|  | Interceptor     |  | Interceptor      |  | track() calls    |      |
|  +--------+--------+  +--------+---------+  +--------+---------+      |
|           |                    |                     |                |
|           +--------------------+---------------------+                |
|                                |                                      |
|  +-----------------------------v-----------------------------+        |
|  |              StreamingTokenTracker                        |        |
|  |  +------------------+  +------------------+               |        |
|  |  | Event Buffer     |  | Batch Builder    |               |        |
|  |  | (asyncio.Queue)  |  | (linger_ms=100)  |               |        |
|  |  +--------+---------+  +--------+---------+               |        |
|  |           |                     |                         |        |
|  |  +--------v---------------------v---------+               |        |
|  |  |        AIOKafkaProducer                |               |        |
|  |  |  - enable_idempotence=True             |               |        |
|  |  |  - acks="all"                          |               |        |
|  |  |  - compression_type="lz4"              |               |        |
|  |  +--------+-------------------------------+               |        |
|  +-----------|-------------------------------------------+   |        |
+--------------|-------------------------------------------------+
               |
               | LLM Events (JSON/Avro)
               v
+------------------------------------------------------------------------------------+
|                        Kafka/Redpanda Cluster                                       |
|                                                                                     |
|  +---------------------------+  +---------------------------+                       |
|  | Topic: llm_events         |  | Topic: llm_events_dlq     |                       |
|  | Partitions: 12            |  | Partitions: 3             |                       |
|  | Replication: 3            |  | (Dead Letter Queue)       |                       |
|  | Retention: 7 days         |  +---------------------------+                       |
|  +-------------+-------------+                                                      |
|                |                                                                    |
|  +-------------v-------------+                                                      |
|  | Schema Registry           |  (Redpanda built-in or Confluent)                   |
|  | - JSON Schema / Avro      |                                                      |
|  | - Compatibility: BACKWARD |                                                      |
|  +---------------------------+                                                      |
+---------------------------+--------------------------------------------------------+
                            |
                            | Consumer Group: tokenledger-clickhouse-sink
                            v
+---------------------------+--------------------------------------------------------+
|                     ClickHouse Sink Consumer                                        |
|                                                                                     |
|  +-----------------------------+  +-----------------------------+                   |
|  |    AIOKafkaConsumer         |  |    ClickHouse Batch Writer  |                   |
|  |  - group_id: ch-sink        |  |  - Batch size: 10,000       |                   |
|  |  - auto_commit: False       |  |  - Flush interval: 5s       |                   |
|  |  - isolation: read_committed|  |  - Async insert             |                   |
|  +-------------+---------------+  +-------------+---------------+                   |
|                |                                |                                   |
|  +-------------v--------------------------------v---------------+                   |
|  |              Offset Commit (after ClickHouse ACK)            |                   |
|  +--------------------------------------------------------------+                   |
+------------------------------------------------------------------------------------+
                            |
                            v
+------------------------------------------------------------------------------------+
|                           ClickHouse Cluster                                        |
|                                                                                     |
|  +---------------------------+  +---------------------------+                       |
|  | Table: llm_events         |  | Materialized Views        |                       |
|  | Engine: ReplicatedMerge-  |  | - Daily aggregations      |                       |
|  |         Tree              |  | - User cost summaries     |                       |
|  | Partition: toYYYYMMDD()   |  | - Model usage stats       |                       |
|  +---------------------------+  +---------------------------+                       |
+------------------------------------------------------------------------------------+
```

---

## Library Recommendations

### Primary Choice: aiokafka

**Version:** 0.13.0+ (January 2026)

| Pros | Cons |
|------|------|
| Native asyncio support | Depends on kafka-python (maintenance concerns) |
| Higher throughput than confluent-kafka in benchmarks (45K vs 22K msg/sec) | No built-in Schema Registry client |
| Background heartbeating (Java-like behavior) | Fewer enterprise features |
| Excellent Redpanda compatibility | |
| Active development (Python 3.14 support) | |

### Alternative: confluent-kafka-python

**Version:** 2.13.0+ (with new AIOProducer/AIOConsumer)

| Pros | Cons |
|------|------|
| Built-in Schema Registry support | AsyncIO support is newer/experimental |
| Commercial support from Confluent | librdkafka dependency (C library) |
| More enterprise features | Slightly lower async throughput |
| Transactional API support | |

### Recommendation

**Use aiokafka** for TokenLedger because:

1. TokenLedger is async-first (FastAPI, asyncpg)
2. Higher throughput for the producer workload
3. Simpler dependency chain for self-hosted users
4. Schema Registry can be added separately via `python-schema-registry-client`

For Schema Registry integration with aiokafka, use:
```bash
pip install python-schema-registry-client
```

### Redpanda Compatibility

Both libraries are fully compatible with Redpanda. Redpanda supports Kafka API versions 0.11+, and both aiokafka and confluent-kafka work seamlessly.

**Note:** Redpanda includes a built-in Schema Registry that is API-compatible with Confluent Schema Registry.

---

## Schema Design

### Event Schema (JSON Schema)

For initial implementation, use JSON Schema for human readability and debugging ease.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://tokenledger.io/schemas/llm-event-v1.json",
  "title": "LLMEvent",
  "description": "An LLM API call event for cost tracking",
  "type": "object",
  "required": ["event_id", "timestamp", "provider", "model", "input_tokens", "output_tokens"],
  "properties": {
    "event_id": {
      "type": "string",
      "format": "uuid",
      "description": "Unique event identifier"
    },
    "trace_id": {
      "type": ["string", "null"],
      "format": "uuid",
      "description": "Distributed trace identifier"
    },
    "span_id": {
      "type": ["string", "null"],
      "format": "uuid"
    },
    "parent_span_id": {
      "type": ["string", "null"],
      "format": "uuid"
    },
    "timestamp": {
      "type": "string",
      "format": "date-time",
      "description": "ISO 8601 timestamp with timezone"
    },
    "duration_ms": {
      "type": ["number", "null"],
      "minimum": 0
    },
    "provider": {
      "type": "string",
      "enum": ["openai", "anthropic", "google", "mistral", "azure", "custom"],
      "description": "LLM provider name"
    },
    "model": {
      "type": "string",
      "minLength": 1,
      "maxLength": 100,
      "description": "Model identifier (e.g., gpt-4o, claude-3-opus)"
    },
    "input_tokens": {
      "type": "integer",
      "minimum": 0
    },
    "output_tokens": {
      "type": "integer",
      "minimum": 0
    },
    "total_tokens": {
      "type": "integer",
      "minimum": 0
    },
    "cached_tokens": {
      "type": "integer",
      "minimum": 0,
      "default": 0
    },
    "cost_usd": {
      "type": ["number", "null"],
      "minimum": 0,
      "description": "Calculated cost in USD"
    },
    "endpoint": {
      "type": ["string", "null"],
      "maxLength": 255
    },
    "request_type": {
      "type": "string",
      "enum": ["chat", "completion", "embedding", "image", "audio", "moderation"],
      "default": "chat"
    },
    "user_id": {
      "type": ["string", "null"],
      "maxLength": 255
    },
    "session_id": {
      "type": ["string", "null"],
      "maxLength": 255
    },
    "organization_id": {
      "type": ["string", "null"],
      "maxLength": 255
    },
    "app_name": {
      "type": ["string", "null"],
      "maxLength": 100
    },
    "environment": {
      "type": ["string", "null"],
      "maxLength": 50
    },
    "status": {
      "type": "string",
      "enum": ["success", "error", "timeout"],
      "default": "success"
    },
    "error_type": {
      "type": ["string", "null"],
      "maxLength": 100
    },
    "error_message": {
      "type": ["string", "null"],
      "maxLength": 1000
    },
    "metadata": {
      "type": ["object", "null"],
      "additionalProperties": true,
      "description": "Custom key-value metadata"
    },
    "request_preview": {
      "type": ["string", "null"],
      "maxLength": 500
    },
    "response_preview": {
      "type": ["string", "null"],
      "maxLength": 500
    }
  },
  "additionalProperties": false
}
```

### Avro Schema (Future v2)

For higher throughput at scale, migrate to Avro:

```avro
{
  "type": "record",
  "name": "LLMEvent",
  "namespace": "io.tokenledger.events",
  "doc": "An LLM API call event for cost tracking",
  "fields": [
    {"name": "event_id", "type": "string", "doc": "UUID"},
    {"name": "trace_id", "type": ["null", "string"], "default": null},
    {"name": "span_id", "type": ["null", "string"], "default": null},
    {"name": "parent_span_id", "type": ["null", "string"], "default": null},
    {"name": "timestamp", "type": "long", "logicalType": "timestamp-millis"},
    {"name": "duration_ms", "type": ["null", "double"], "default": null},
    {"name": "provider", "type": {"type": "enum", "name": "Provider", "symbols": ["openai", "anthropic", "google", "mistral", "azure", "custom"]}},
    {"name": "model", "type": "string"},
    {"name": "input_tokens", "type": "int"},
    {"name": "output_tokens", "type": "int"},
    {"name": "total_tokens", "type": "int"},
    {"name": "cached_tokens", "type": "int", "default": 0},
    {"name": "cost_usd", "type": ["null", "double"], "default": null},
    {"name": "endpoint", "type": ["null", "string"], "default": null},
    {"name": "request_type", "type": {"type": "enum", "name": "RequestType", "symbols": ["chat", "completion", "embedding", "image", "audio", "moderation"]}, "default": "chat"},
    {"name": "user_id", "type": ["null", "string"], "default": null},
    {"name": "session_id", "type": ["null", "string"], "default": null},
    {"name": "organization_id", "type": ["null", "string"], "default": null},
    {"name": "app_name", "type": ["null", "string"], "default": null},
    {"name": "environment", "type": ["null", "string"], "default": null},
    {"name": "status", "type": {"type": "enum", "name": "Status", "symbols": ["success", "error", "timeout"]}, "default": "success"},
    {"name": "error_type", "type": ["null", "string"], "default": null},
    {"name": "error_message", "type": ["null", "string"], "default": null},
    {"name": "metadata", "type": ["null", {"type": "map", "values": "string"}], "default": null},
    {"name": "request_preview", "type": ["null", "string"], "default": null},
    {"name": "response_preview", "type": ["null", "string"], "default": null}
  ]
}
```

### Schema Evolution Strategy

Use **BACKWARD** compatibility mode:

- New optional fields can be added with defaults
- Fields can be removed (consumers ignore unknown fields)
- Required fields cannot be removed
- Field types cannot change

---

## Producer Implementation

### StreamingTokenTracker Class

```python
"""
tokenledger/streaming/producer.py
Async Kafka/Redpanda producer for LLM events
"""

import asyncio
import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaError, KafkaTimeoutError

from ..config import StreamingConfig
from ..tracker import LLMEvent

logger = logging.getLogger("tokenledger.streaming")


class StreamingTokenTracker:
    """
    Async-first Kafka/Redpanda producer for LLM events.

    Features:
    - Idempotent production (no duplicates on retry)
    - Automatic batching with configurable linger
    - Backpressure via bounded asyncio.Queue
    - Graceful shutdown with flush

    Example:
        >>> tracker = StreamingTokenTracker(config)
        >>> await tracker.start()
        >>> await tracker.track(event)
        >>> await tracker.stop()
    """

    def __init__(self, config: "StreamingConfig"):
        self.config = config
        self._producer: AIOKafkaProducer | None = None
        self._queue: asyncio.Queue[LLMEvent] = asyncio.Queue(
            maxsize=config.max_queue_size
        )
        self._flush_task: asyncio.Task | None = None
        self._running = False
        self._metrics = ProducerMetrics()

    async def start(self) -> None:
        """Initialize the Kafka producer and start background tasks."""
        if self._producer is not None:
            return

        self._producer = AIOKafkaProducer(
            bootstrap_servers=self.config.bootstrap_servers,
            # Idempotent producer settings
            enable_idempotence=True,
            acks="all",
            # Batching settings
            linger_ms=self.config.linger_ms,  # Default: 100ms
            max_batch_size=self.config.max_batch_size,  # Default: 1MB
            # Compression
            compression_type=self.config.compression_type,  # Default: "lz4"
            # Timeouts and retries
            request_timeout_ms=self.config.request_timeout_ms,
            retry_backoff_ms=100,
            # Memory limits
            max_request_size=self.config.max_request_size,
            # Serialization
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            value_serializer=self._serialize_event,
        )

        await self._producer.start()
        self._running = True

        # Start background flush task
        self._flush_task = asyncio.create_task(self._flush_loop())

        logger.info(
            "StreamingTokenTracker started",
            extra={
                "bootstrap_servers": self.config.bootstrap_servers,
                "topic": self.config.topic,
            }
        )

    def _serialize_event(self, event: LLMEvent) -> bytes:
        """Serialize LLMEvent to JSON bytes."""
        data = asdict(event)
        # Convert datetime to ISO format
        if isinstance(data.get("timestamp"), datetime):
            data["timestamp"] = data["timestamp"].isoformat()
        return json.dumps(data, default=str).encode("utf-8")

    async def track(self, event: LLMEvent) -> None:
        """
        Queue an event for async production to Kafka.

        This method is non-blocking. Events are queued and sent
        in batches by the background task.

        Raises:
            asyncio.QueueFull: If backpressure limit is reached
        """
        if not self._running:
            raise RuntimeError("StreamingTokenTracker not started")

        try:
            self._queue.put_nowait(event)
            self._metrics.events_queued += 1
        except asyncio.QueueFull:
            self._metrics.events_dropped += 1
            logger.warning(
                "Event queue full, dropping event",
                extra={"event_id": event.event_id}
            )
            raise

    async def track_blocking(self, event: LLMEvent, timeout: float = 5.0) -> None:
        """
        Queue an event with blocking backpressure.

        Waits up to `timeout` seconds for queue space.
        """
        await asyncio.wait_for(
            self._queue.put(event),
            timeout=timeout
        )
        self._metrics.events_queued += 1

    async def send_immediate(self, event: LLMEvent) -> None:
        """
        Send an event immediately, bypassing the queue.

        Use for critical events that must be sent synchronously.
        """
        if not self._producer:
            raise RuntimeError("StreamingTokenTracker not started")

        try:
            # Partition by user_id for ordering, or None for round-robin
            partition_key = event.user_id or event.organization_id

            await self._producer.send_and_wait(
                self.config.topic,
                value=event,
                key=partition_key,
            )
            self._metrics.events_sent += 1
        except KafkaError as e:
            self._metrics.events_failed += 1
            logger.error(f"Failed to send event: {e}", extra={"event_id": event.event_id})
            raise

    async def _flush_loop(self) -> None:
        """Background task to drain queue and send batches."""
        while self._running:
            try:
                await self._flush_batch()
                await asyncio.sleep(0.01)  # Yield to event loop
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in flush loop: {e}")
                await asyncio.sleep(1.0)  # Back off on error

    async def _flush_batch(self) -> int:
        """Drain queue and send events to Kafka."""
        if not self._producer or self._queue.empty():
            return 0

        batch: list[LLMEvent] = []
        batch_size = 0
        max_batch = self.config.producer_batch_size  # Default: 1000 events

        # Drain queue up to batch size
        while not self._queue.empty() and batch_size < max_batch:
            try:
                event = self._queue.get_nowait()
                batch.append(event)
                batch_size += 1
            except asyncio.QueueEmpty:
                break

        if not batch:
            return 0

        # Send batch (aiokafka handles batching internally)
        futures = []
        for event in batch:
            partition_key = event.user_id or event.organization_id
            fut = await self._producer.send(
                self.config.topic,
                value=event,
                key=partition_key,
            )
            futures.append(fut)

        # Wait for all to complete
        try:
            await asyncio.gather(*[f for f in futures])
            self._metrics.events_sent += len(batch)
            self._metrics.batches_sent += 1
        except KafkaError as e:
            self._metrics.events_failed += len(batch)
            logger.error(f"Batch send failed: {e}")
            # Events are lost - consider DLQ here

        return len(batch)

    async def flush(self) -> None:
        """Flush all pending events to Kafka."""
        if self._producer:
            # Drain queue
            while not self._queue.empty():
                await self._flush_batch()
            # Flush producer buffer
            await self._producer.flush()

    async def stop(self) -> None:
        """Gracefully stop the producer, flushing pending events."""
        self._running = False

        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        await self.flush()

        if self._producer:
            await self._producer.stop()
            self._producer = None

        logger.info(
            "StreamingTokenTracker stopped",
            extra=self._metrics.to_dict()
        )

    def get_metrics(self) -> "ProducerMetrics":
        """Get producer metrics for monitoring."""
        return self._metrics

    async def health_check(self) -> dict[str, Any]:
        """Check producer health status."""
        return {
            "healthy": self._producer is not None and self._running,
            "queue_size": self._queue.qsize(),
            "queue_capacity": self.config.max_queue_size,
            "queue_utilization": self._queue.qsize() / self.config.max_queue_size,
            "metrics": self._metrics.to_dict(),
        }


class ProducerMetrics:
    """Metrics for the streaming producer."""

    def __init__(self):
        self.events_queued: int = 0
        self.events_sent: int = 0
        self.events_failed: int = 0
        self.events_dropped: int = 0
        self.batches_sent: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "events_queued": self.events_queued,
            "events_sent": self.events_sent,
            "events_failed": self.events_failed,
            "events_dropped": self.events_dropped,
            "batches_sent": self.batches_sent,
        }
```

### Producer Configuration

```python
"""
tokenledger/streaming/config.py
Configuration for Kafka/Redpanda streaming
"""

from dataclasses import dataclass, field
import os


@dataclass
class StreamingConfig:
    """Configuration for Kafka/Redpanda streaming."""

    # Connection
    bootstrap_servers: str = ""
    topic: str = "llm_events"
    dlq_topic: str = "llm_events_dlq"

    # Producer settings
    enable_idempotence: bool = True
    acks: str = "all"
    compression_type: str = "lz4"  # Options: none, gzip, snappy, lz4, zstd
    linger_ms: int = 100  # Batch delay in ms
    max_batch_size: int = 1048576  # 1MB
    max_request_size: int = 1048576  # 1MB
    request_timeout_ms: int = 30000  # 30 seconds

    # Application-level batching
    producer_batch_size: int = 1000  # Events per batch
    max_queue_size: int = 100000  # Backpressure limit

    # Consumer settings (for sink)
    consumer_group_id: str = "tokenledger-clickhouse-sink"
    auto_offset_reset: str = "earliest"
    enable_auto_commit: bool = False
    max_poll_records: int = 1000
    session_timeout_ms: int = 45000
    heartbeat_interval_ms: int = 15000

    # Schema Registry (optional)
    schema_registry_url: str | None = None

    # Security (optional)
    security_protocol: str = "PLAINTEXT"  # PLAINTEXT, SSL, SASL_PLAINTEXT, SASL_SSL
    sasl_mechanism: str | None = None
    sasl_username: str | None = None
    sasl_password: str | None = None
    ssl_cafile: str | None = None
    ssl_certfile: str | None = None
    ssl_keyfile: str | None = None

    def __post_init__(self):
        if not self.bootstrap_servers:
            self.bootstrap_servers = os.getenv(
                "TOKENLEDGER_KAFKA_BOOTSTRAP_SERVERS",
                os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
            )

        if os.getenv("TOKENLEDGER_KAFKA_TOPIC"):
            self.topic = os.getenv("TOKENLEDGER_KAFKA_TOPIC")

        if os.getenv("TOKENLEDGER_SCHEMA_REGISTRY_URL"):
            self.schema_registry_url = os.getenv("TOKENLEDGER_SCHEMA_REGISTRY_URL")

    def get_producer_config(self) -> dict:
        """Get aiokafka producer configuration dict."""
        config = {
            "bootstrap_servers": self.bootstrap_servers,
            "enable_idempotence": self.enable_idempotence,
            "acks": self.acks,
            "compression_type": self.compression_type,
            "linger_ms": self.linger_ms,
            "max_batch_size": self.max_batch_size,
            "max_request_size": self.max_request_size,
            "request_timeout_ms": self.request_timeout_ms,
        }

        # Add security settings if configured
        if self.security_protocol != "PLAINTEXT":
            config["security_protocol"] = self.security_protocol
            if self.sasl_mechanism:
                config["sasl_mechanism"] = self.sasl_mechanism
                config["sasl_plain_username"] = self.sasl_username
                config["sasl_plain_password"] = self.sasl_password
            if self.ssl_cafile:
                config["ssl_context"] = self._create_ssl_context()

        return config

    def get_consumer_config(self) -> dict:
        """Get aiokafka consumer configuration dict."""
        config = {
            "bootstrap_servers": self.bootstrap_servers,
            "group_id": self.consumer_group_id,
            "auto_offset_reset": self.auto_offset_reset,
            "enable_auto_commit": self.enable_auto_commit,
            "max_poll_records": self.max_poll_records,
            "session_timeout_ms": self.session_timeout_ms,
            "heartbeat_interval_ms": self.heartbeat_interval_ms,
        }

        if self.security_protocol != "PLAINTEXT":
            config["security_protocol"] = self.security_protocol
            # Add SASL/SSL as above

        return config

    def _create_ssl_context(self):
        """Create SSL context for secure connections."""
        import ssl
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        if self.ssl_cafile:
            context.load_verify_locations(self.ssl_cafile)
        if self.ssl_certfile and self.ssl_keyfile:
            context.load_cert_chain(self.ssl_certfile, self.ssl_keyfile)
        return context
```

---

## Consumer Implementation

### ClickHouse Sink Consumer

```python
"""
tokenledger/streaming/consumer.py
Async Kafka consumer with ClickHouse sink
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any

from aiokafka import AIOKafkaConsumer, TopicPartition
from aiokafka.errors import KafkaError
from aiokafka.structs import ConsumerRecord

from .config import StreamingConfig

logger = logging.getLogger("tokenledger.streaming.consumer")


class ClickHouseSinkConsumer:
    """
    Kafka consumer that sinks LLM events to ClickHouse.

    Features:
    - Manual offset commits (at-least-once delivery)
    - Batch insertion to ClickHouse
    - Dead letter queue for failed events
    - Graceful rebalancing

    Example:
        >>> consumer = ClickHouseSinkConsumer(config, clickhouse_client)
        >>> await consumer.start()
        >>> await consumer.run()  # Runs until stopped
        >>> await consumer.stop()
    """

    def __init__(
        self,
        config: StreamingConfig,
        clickhouse_client: Any,  # ClickHouse async client
    ):
        self.config = config
        self.clickhouse = clickhouse_client
        self._consumer: AIOKafkaConsumer | None = None
        self._dlq_producer: Any = None  # For sending to DLQ
        self._running = False
        self._batch: list[dict] = []
        self._batch_offsets: dict[TopicPartition, int] = {}
        self._metrics = ConsumerMetrics()

    async def start(self) -> None:
        """Initialize the Kafka consumer."""
        self._consumer = AIOKafkaConsumer(
            self.config.topic,
            **self.config.get_consumer_config(),
            value_deserializer=self._deserialize_event,
            # Rebalance listener for graceful handling
            on_partitions_revoked=self._on_partitions_revoked,
            on_partitions_assigned=self._on_partitions_assigned,
        )

        await self._consumer.start()
        self._running = True

        logger.info(
            "ClickHouseSinkConsumer started",
            extra={
                "topic": self.config.topic,
                "group_id": self.config.consumer_group_id,
            }
        )

    def _deserialize_event(self, data: bytes) -> dict:
        """Deserialize JSON event from Kafka."""
        return json.loads(data.decode("utf-8"))

    async def _on_partitions_revoked(self, revoked: set[TopicPartition]) -> None:
        """Handle partition revocation during rebalance."""
        if revoked:
            logger.info(f"Partitions revoked: {revoked}")
            # Flush pending batch before giving up partitions
            await self._flush_batch()

    async def _on_partitions_assigned(self, assigned: set[TopicPartition]) -> None:
        """Handle partition assignment during rebalance."""
        if assigned:
            logger.info(f"Partitions assigned: {assigned}")

    async def run(self) -> None:
        """
        Main consumption loop.

        Processes messages in batches and commits offsets
        after successful ClickHouse insertion.
        """
        if not self._consumer:
            raise RuntimeError("Consumer not started")

        while self._running:
            try:
                # Fetch batch of messages
                messages = await self._consumer.getmany(
                    timeout_ms=self.config.consumer_poll_timeout_ms,
                    max_records=self.config.max_poll_records,
                )

                if not messages:
                    continue

                # Process messages by partition
                for tp, records in messages.items():
                    await self._process_records(tp, records)

                # Flush if batch is large enough or timeout
                if len(self._batch) >= self.config.sink_batch_size:
                    await self._flush_batch()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in consumption loop: {e}")
                self._metrics.errors += 1
                await asyncio.sleep(1.0)

    async def _process_records(
        self,
        tp: TopicPartition,
        records: list[ConsumerRecord]
    ) -> None:
        """Process a batch of records from a partition."""
        for record in records:
            try:
                event_data = record.value
                self._batch.append(event_data)
                self._batch_offsets[tp] = record.offset + 1
                self._metrics.events_processed += 1
            except Exception as e:
                logger.error(
                    f"Failed to process record: {e}",
                    extra={"offset": record.offset, "partition": tp.partition}
                )
                # Send to DLQ
                await self._send_to_dlq(record, str(e))
                self._metrics.events_dlq += 1

    async def _flush_batch(self) -> None:
        """Flush batch to ClickHouse and commit offsets."""
        if not self._batch:
            return

        try:
            # Insert to ClickHouse
            await self._insert_to_clickhouse(self._batch)

            # Commit offsets after successful insert
            if self._batch_offsets:
                await self._consumer.commit(self._batch_offsets)
                self._metrics.commits += 1

            self._metrics.batches_flushed += 1
            logger.debug(f"Flushed {len(self._batch)} events to ClickHouse")

        except Exception as e:
            logger.error(f"Failed to flush batch: {e}")
            self._metrics.errors += 1
            # Don't clear batch - will retry on next flush
            raise
        finally:
            self._batch = []
            self._batch_offsets = {}

    async def _insert_to_clickhouse(self, events: list[dict]) -> None:
        """Insert events to ClickHouse."""
        # Transform events to ClickHouse format
        rows = []
        for event in events:
            rows.append({
                "event_id": event["event_id"],
                "trace_id": event.get("trace_id"),
                "span_id": event.get("span_id"),
                "parent_span_id": event.get("parent_span_id"),
                "timestamp": datetime.fromisoformat(event["timestamp"]),
                "duration_ms": event.get("duration_ms"),
                "provider": event["provider"],
                "model": event["model"],
                "input_tokens": event["input_tokens"],
                "output_tokens": event["output_tokens"],
                "total_tokens": event.get("total_tokens", 0),
                "cached_tokens": event.get("cached_tokens", 0),
                "cost_usd": event.get("cost_usd"),
                "endpoint": event.get("endpoint"),
                "request_type": event.get("request_type", "chat"),
                "user_id": event.get("user_id"),
                "session_id": event.get("session_id"),
                "organization_id": event.get("organization_id"),
                "app_name": event.get("app_name"),
                "environment": event.get("environment"),
                "status": event.get("status", "success"),
                "error_type": event.get("error_type"),
                "error_message": event.get("error_message"),
                "metadata": json.dumps(event.get("metadata", {})),
                "request_preview": event.get("request_preview"),
                "response_preview": event.get("response_preview"),
            })

        # Async insert to ClickHouse
        await self.clickhouse.insert(
            "llm_events",
            rows,
            column_names=list(rows[0].keys()) if rows else [],
        )

    async def _send_to_dlq(self, record: ConsumerRecord, error: str) -> None:
        """Send failed record to dead letter queue."""
        if not self._dlq_producer:
            # Initialize DLQ producer lazily
            from aiokafka import AIOKafkaProducer
            self._dlq_producer = AIOKafkaProducer(
                bootstrap_servers=self.config.bootstrap_servers,
            )
            await self._dlq_producer.start()

        # Add error metadata to the message
        dlq_value = {
            "original_topic": record.topic,
            "original_partition": record.partition,
            "original_offset": record.offset,
            "original_timestamp": record.timestamp,
            "original_value": record.value,
            "error": error,
            "failed_at": datetime.utcnow().isoformat(),
        }

        await self._dlq_producer.send(
            self.config.dlq_topic,
            value=json.dumps(dlq_value).encode("utf-8"),
            key=record.key,
        )

    async def stop(self) -> None:
        """Gracefully stop the consumer."""
        self._running = False

        # Flush remaining batch
        try:
            await self._flush_batch()
        except Exception as e:
            logger.error(f"Error flushing final batch: {e}")

        if self._consumer:
            await self._consumer.stop()
            self._consumer = None

        if self._dlq_producer:
            await self._dlq_producer.stop()
            self._dlq_producer = None

        logger.info(
            "ClickHouseSinkConsumer stopped",
            extra=self._metrics.to_dict()
        )

    async def health_check(self) -> dict[str, Any]:
        """Check consumer health status."""
        lag = await self._calculate_lag()
        return {
            "healthy": self._consumer is not None and self._running,
            "consumer_lag": lag,
            "batch_size": len(self._batch),
            "metrics": self._metrics.to_dict(),
        }

    async def _calculate_lag(self) -> dict[str, int]:
        """Calculate consumer lag per partition."""
        if not self._consumer:
            return {}

        lag = {}
        for tp in self._consumer.assignment():
            try:
                position = await self._consumer.position(tp)
                highwater = self._consumer.highwater(tp)
                if highwater is not None:
                    lag[f"{tp.topic}-{tp.partition}"] = highwater - position
            except Exception:
                pass
        return lag


class ConsumerMetrics:
    """Metrics for the sink consumer."""

    def __init__(self):
        self.events_processed: int = 0
        self.events_dlq: int = 0
        self.batches_flushed: int = 0
        self.commits: int = 0
        self.errors: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "events_processed": self.events_processed,
            "events_dlq": self.events_dlq,
            "batches_flushed": self.batches_flushed,
            "commits": self.commits,
            "errors": self.errors,
        }
```

---

## Delivery Guarantees

### At-Least-Once Delivery

TokenLedger uses **at-least-once delivery** with deduplication at the ClickHouse sink:

```
Producer                     Kafka/Redpanda                Consumer                    ClickHouse
   |                              |                            |                            |
   |--send(event)--------------->|                            |                            |
   |                              |--store event------------->|                            |
   |<--ack (idempotent)----------|                            |                            |
   |                              |                            |--poll()------------------>|
   |                              |<--------------------------fetch events-----------------|
   |                              |                            |                            |
   |                              |                            |--insert (with event_id)-->|
   |                              |                            |                            |
   |                              |                            |<--ACK (or ReplacingMerge)-|
   |                              |<--commit offset------------|                            |
```

### Why Not Exactly-Once?

1. **Complexity**: Requires Kafka transactions + two-phase commit to ClickHouse
2. **Performance**: Transaction overhead for every batch
3. **Sufficiency**: LLM events are idempotent by `event_id` - duplicates are cheap to handle

### Deduplication at ClickHouse

```sql
-- ClickHouse table with ReplacingMergeTree for deduplication
CREATE TABLE llm_events
(
    event_id UUID,
    timestamp DateTime64(3),
    -- ... other columns
)
ENGINE = ReplacingMergeTree(timestamp)
ORDER BY (event_id)
PARTITION BY toYYYYMM(timestamp);

-- Query with FINAL to get deduplicated results
SELECT * FROM llm_events FINAL WHERE timestamp > now() - INTERVAL 1 HOUR;
```

### Idempotent Producer Configuration

```python
# aiokafka producer settings for idempotence
producer = AIOKafkaProducer(
    bootstrap_servers="localhost:9092",
    enable_idempotence=True,  # Prevents duplicates on retry
    acks="all",               # Required for idempotence
    max_in_flight_requests_per_connection=5,  # Default, works with idempotence
    retries=2147483647,       # Infinite retries (bounded by timeout)
    request_timeout_ms=30000,
)
```

---

## Backpressure & Flow Control

### Producer Backpressure

```python
class StreamingTokenTracker:
    """Producer with backpressure handling."""

    async def track(self, event: LLMEvent) -> None:
        """Non-blocking track with drop on backpressure."""
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            # Backpressure: queue is full
            self._metrics.events_dropped += 1
            logger.warning("Backpressure: dropping event")

    async def track_blocking(
        self,
        event: LLMEvent,
        timeout: float = 5.0
    ) -> None:
        """Blocking track that waits for queue space."""
        try:
            await asyncio.wait_for(
                self._queue.put(event),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            self._metrics.events_dropped += 1
            raise BackpressureError("Queue full, timeout waiting for space")

    async def track_with_callback(
        self,
        event: LLMEvent,
        on_success: Callable | None = None,
        on_failure: Callable | None = None,
    ) -> None:
        """Track with delivery callbacks."""
        if self._queue.qsize() > self.config.max_queue_size * 0.8:
            # Soft backpressure warning at 80%
            logger.warning("Queue at 80% capacity")

        await self._queue.put(event)
        # Callbacks handled in flush loop
```

### Consumer Backpressure

```python
class ClickHouseSinkConsumer:
    """Consumer with flow control."""

    async def run(self) -> None:
        """Consumption loop with pause/resume for backpressure."""
        while self._running:
            # Check ClickHouse health
            if not await self._clickhouse_healthy():
                logger.warning("ClickHouse unhealthy, pausing consumption")
                self._consumer.pause(*self._consumer.assignment())
                await asyncio.sleep(5.0)
                self._consumer.resume(*self._consumer.assignment())
                continue

            # Check batch size - pause if too large
            if len(self._batch) > self.config.max_batch_buffer:
                logger.warning("Batch buffer full, pausing consumption")
                self._consumer.pause(*self._consumer.assignment())
                await self._flush_batch()
                self._consumer.resume(*self._consumer.assignment())

            # Normal consumption
            messages = await self._consumer.getmany(
                timeout_ms=1000,
                max_records=self.config.max_poll_records
            )
            # ... process messages

    async def _clickhouse_healthy(self) -> bool:
        """Check if ClickHouse can accept writes."""
        try:
            await self.clickhouse.execute("SELECT 1")
            return True
        except Exception:
            return False
```

### Bounded Queue Configuration

```python
@dataclass
class StreamingConfig:
    # Producer backpressure
    max_queue_size: int = 100000  # Events in memory
    backpressure_threshold: float = 0.8  # 80% warning

    # Consumer backpressure
    max_batch_buffer: int = 50000  # Events before pause
    consumer_poll_timeout_ms: int = 1000
    max_poll_records: int = 1000
```

---

## Error Handling & Dead Letter Queue

### Error Categories

| Error Type | Handling | Example |
|------------|----------|---------|
| Transient | Retry with backoff | Network timeout, broker unavailable |
| Serialization | Send to DLQ | Malformed JSON, schema mismatch |
| Processing | Send to DLQ | ClickHouse constraint violation |
| Fatal | Stop consumer | Authentication failure |

### Retry Pattern

```python
import asyncio
from functools import wraps

def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retryable_exceptions: tuple = (KafkaTimeoutError, KafkaConnectionError),
):
    """Decorator for retry with exponential backoff."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        delay = min(
                            base_delay * (exponential_base ** attempt),
                            max_delay
                        )
                        logger.warning(
                            f"Retry {attempt + 1}/{max_attempts} after {delay}s: {e}"
                        )
                        await asyncio.sleep(delay)
            raise last_exception
        return wrapper
    return decorator


class ClickHouseSinkConsumer:
    @with_retry(max_attempts=3, base_delay=1.0)
    async def _insert_to_clickhouse(self, events: list[dict]) -> None:
        """Insert with retry on transient errors."""
        await self.clickhouse.insert("llm_events", events)
```

### Dead Letter Queue Implementation

```python
@dataclass
class DLQMessage:
    """Dead letter queue message structure."""
    original_topic: str
    original_partition: int
    original_offset: int
    original_timestamp: int
    original_key: bytes | None
    original_value: bytes
    error_type: str
    error_message: str
    retry_count: int
    failed_at: str
    consumer_group: str

    def to_json(self) -> bytes:
        return json.dumps(asdict(self)).encode("utf-8")


class DLQHandler:
    """Handles sending failed messages to DLQ."""

    def __init__(self, config: StreamingConfig):
        self.config = config
        self._producer: AIOKafkaProducer | None = None

    async def start(self) -> None:
        self._producer = AIOKafkaProducer(
            bootstrap_servers=self.config.bootstrap_servers,
        )
        await self._producer.start()

    async def send(
        self,
        record: ConsumerRecord,
        error: Exception,
        retry_count: int = 0,
    ) -> None:
        """Send failed record to DLQ with error context."""
        dlq_msg = DLQMessage(
            original_topic=record.topic,
            original_partition=record.partition,
            original_offset=record.offset,
            original_timestamp=record.timestamp,
            original_key=record.key,
            original_value=record.value,
            error_type=type(error).__name__,
            error_message=str(error),
            retry_count=retry_count,
            failed_at=datetime.utcnow().isoformat(),
            consumer_group=self.config.consumer_group_id,
        )

        await self._producer.send(
            self.config.dlq_topic,
            value=dlq_msg.to_json(),
            key=record.key,
            headers=[
                ("error_type", type(error).__name__.encode()),
                ("original_topic", record.topic.encode()),
            ]
        )

        logger.info(
            "Sent to DLQ",
            extra={
                "original_offset": record.offset,
                "error": str(error),
            }
        )

    async def stop(self) -> None:
        if self._producer:
            await self._producer.stop()
```

### DLQ Reprocessing

```python
class DLQReprocessor:
    """Reprocess messages from DLQ."""

    async def reprocess(
        self,
        filter_error_type: str | None = None,
        max_messages: int = 1000,
    ) -> int:
        """
        Reprocess DLQ messages back to main topic.

        Args:
            filter_error_type: Only reprocess specific error types
            max_messages: Maximum messages to reprocess

        Returns:
            Number of messages reprocessed
        """
        consumer = AIOKafkaConsumer(
            self.config.dlq_topic,
            bootstrap_servers=self.config.bootstrap_servers,
            group_id=f"{self.config.consumer_group_id}-dlq-reprocess",
            auto_offset_reset="earliest",
            enable_auto_commit=False,
        )

        producer = AIOKafkaProducer(
            bootstrap_servers=self.config.bootstrap_servers,
        )

        await consumer.start()
        await producer.start()

        reprocessed = 0
        try:
            async for msg in consumer:
                if reprocessed >= max_messages:
                    break

                dlq_data = json.loads(msg.value)

                # Filter by error type if specified
                if filter_error_type and dlq_data["error_type"] != filter_error_type:
                    continue

                # Send back to original topic
                await producer.send(
                    dlq_data["original_topic"],
                    value=dlq_data["original_value"].encode()
                        if isinstance(dlq_data["original_value"], str)
                        else dlq_data["original_value"],
                    key=dlq_data["original_key"],
                )

                await consumer.commit()
                reprocessed += 1

        finally:
            await consumer.stop()
            await producer.stop()

        return reprocessed
```

---

## ClickHouse Integration

### Why External Consumer vs Kafka Engine?

| Aspect | Kafka Engine | External Consumer |
|--------|--------------|-------------------|
| Complexity | Simpler setup | More code |
| Control | Limited | Full control |
| Transformations | SQL only | Python (rich) |
| Error handling | Limited | DLQ support |
| Scaling | ClickHouse-bound | Independent |
| Monitoring | ClickHouse metrics | Custom metrics |

**Decision:** Use external Python consumer for:
- Complex event transformation
- DLQ support for failed events
- Independent scaling of ingestion
- Rich error handling

### ClickHouse Schema

```sql
-- ClickHouse table optimized for time-series analytics
CREATE TABLE llm_events
(
    -- Identifiers
    event_id UUID,
    trace_id Nullable(UUID),
    span_id Nullable(UUID),
    parent_span_id Nullable(UUID),

    -- Timing
    timestamp DateTime64(3, 'UTC'),
    duration_ms Nullable(Float64),

    -- Provider & Model
    provider LowCardinality(String),
    model LowCardinality(String),

    -- Token counts
    input_tokens UInt32,
    output_tokens UInt32,
    total_tokens UInt32,
    cached_tokens UInt32 DEFAULT 0,

    -- Cost
    cost_usd Nullable(Decimal64(8)),

    -- Request details
    endpoint Nullable(String),
    request_type LowCardinality(String) DEFAULT 'chat',

    -- User & context
    user_id Nullable(String),
    session_id Nullable(String),
    organization_id Nullable(String),

    -- Application context
    app_name Nullable(LowCardinality(String)),
    environment Nullable(LowCardinality(String)),

    -- Status
    status LowCardinality(String) DEFAULT 'success',
    error_type Nullable(String),
    error_message Nullable(String),

    -- Metadata (JSON)
    metadata String DEFAULT '{}',

    -- Previews
    request_preview Nullable(String),
    response_preview Nullable(String),

    -- Kafka metadata (for debugging)
    _kafka_partition UInt16,
    _kafka_offset UInt64,
    _inserted_at DateTime64(3, 'UTC') DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(_inserted_at)
PARTITION BY toYYYYMM(timestamp)
ORDER BY (organization_id, app_name, timestamp, event_id)
TTL timestamp + INTERVAL 90 DAY
SETTINGS index_granularity = 8192;

-- Indexes for common queries
ALTER TABLE llm_events ADD INDEX idx_user_id user_id TYPE bloom_filter GRANULARITY 4;
ALTER TABLE llm_events ADD INDEX idx_model model TYPE set(100) GRANULARITY 4;
ALTER TABLE llm_events ADD INDEX idx_status status TYPE set(10) GRANULARITY 4;
```

### Materialized Views for Aggregations

```sql
-- Daily cost summary (real-time aggregation)
CREATE MATERIALIZED VIEW llm_events_daily_mv
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (organization_id, app_name, date, provider, model)
AS SELECT
    toDate(timestamp) AS date,
    organization_id,
    app_name,
    provider,
    model,
    count() AS request_count,
    sum(input_tokens) AS total_input_tokens,
    sum(output_tokens) AS total_output_tokens,
    sum(total_tokens) AS total_tokens,
    sum(cost_usd) AS total_cost_usd,
    countIf(status = 'error') AS error_count
FROM llm_events
GROUP BY date, organization_id, app_name, provider, model;

-- User cost summary
CREATE MATERIALIZED VIEW llm_events_user_mv
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (organization_id, user_id, date)
AS SELECT
    toDate(timestamp) AS date,
    organization_id,
    user_id,
    count() AS request_count,
    sum(total_tokens) AS total_tokens,
    sum(cost_usd) AS total_cost_usd
FROM llm_events
WHERE user_id IS NOT NULL
GROUP BY date, organization_id, user_id;
```

### ClickHouse Async Client

```python
"""
tokenledger/streaming/clickhouse.py
Async ClickHouse client for event insertion
"""

from typing import Any
import asyncio
from clickhouse_connect import get_async_client
from clickhouse_connect.driver.asyncclient import AsyncClient


class ClickHouseEventSink:
    """Async ClickHouse client optimized for event ingestion."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8123,
        database: str = "default",
        username: str = "default",
        password: str = "",
        **kwargs
    ):
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password
        self.kwargs = kwargs
        self._client: AsyncClient | None = None

    async def connect(self) -> None:
        """Initialize async connection."""
        self._client = await get_async_client(
            host=self.host,
            port=self.port,
            database=self.database,
            username=self.username,
            password=self.password,
            **self.kwargs
        )

    async def insert(
        self,
        table: str,
        rows: list[dict],
        column_names: list[str] | None = None,
    ) -> None:
        """Insert rows to ClickHouse."""
        if not self._client:
            raise RuntimeError("Client not connected")

        if not rows:
            return

        # Convert dicts to column-oriented format
        if column_names is None:
            column_names = list(rows[0].keys())

        data = [[row.get(col) for col in column_names] for row in rows]

        await self._client.insert(
            table,
            data,
            column_names=column_names,
        )

    async def execute(self, query: str) -> Any:
        """Execute a query."""
        if not self._client:
            raise RuntimeError("Client not connected")
        return await self._client.query(query)

    async def close(self) -> None:
        """Close the connection."""
        if self._client:
            await self._client.close()
            self._client = None
```

---

## Observability & Metrics

### Prometheus Metrics

```python
"""
tokenledger/streaming/metrics.py
Prometheus metrics for streaming components
"""

from prometheus_client import Counter, Gauge, Histogram, Info


# Producer metrics
PRODUCER_EVENTS_TOTAL = Counter(
    "tokenledger_producer_events_total",
    "Total events processed by producer",
    ["status"]  # queued, sent, dropped, failed
)

PRODUCER_QUEUE_SIZE = Gauge(
    "tokenledger_producer_queue_size",
    "Current producer queue size"
)

PRODUCER_BATCH_SIZE = Histogram(
    "tokenledger_producer_batch_size",
    "Producer batch sizes",
    buckets=[10, 50, 100, 500, 1000, 5000, 10000]
)

PRODUCER_LATENCY = Histogram(
    "tokenledger_producer_latency_seconds",
    "Producer send latency",
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
)

# Consumer metrics
CONSUMER_EVENTS_TOTAL = Counter(
    "tokenledger_consumer_events_total",
    "Total events processed by consumer",
    ["status"]  # processed, dlq, failed
)

CONSUMER_LAG = Gauge(
    "tokenledger_consumer_lag",
    "Consumer lag per partition",
    ["partition"]
)

CONSUMER_BATCH_DURATION = Histogram(
    "tokenledger_consumer_batch_duration_seconds",
    "Time to process and flush a batch",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

CONSUMER_COMMITS_TOTAL = Counter(
    "tokenledger_consumer_commits_total",
    "Total offset commits"
)

# ClickHouse metrics
CLICKHOUSE_INSERT_LATENCY = Histogram(
    "tokenledger_clickhouse_insert_latency_seconds",
    "ClickHouse insert latency",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

CLICKHOUSE_INSERT_ROWS = Counter(
    "tokenledger_clickhouse_insert_rows_total",
    "Total rows inserted to ClickHouse"
)

# Component info
STREAMING_INFO = Info(
    "tokenledger_streaming",
    "Streaming component information"
)
```

### Health Check Endpoint

```python
"""
tokenledger/streaming/health.py
Health check endpoint for streaming components
"""

from fastapi import APIRouter, Response
from enum import Enum

router = APIRouter(prefix="/health", tags=["health"])


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@router.get("/live")
async def liveness():
    """Kubernetes liveness probe."""
    return {"status": "ok"}


@router.get("/ready")
async def readiness(
    producer: StreamingTokenTracker,
    consumer: ClickHouseSinkConsumer,
):
    """Kubernetes readiness probe."""
    producer_health = await producer.health_check()
    consumer_health = await consumer.health_check()

    # Determine overall status
    if not producer_health["healthy"] or not consumer_health["healthy"]:
        return Response(
            content='{"status": "unhealthy"}',
            status_code=503,
            media_type="application/json"
        )

    # Check for degraded state (high lag, high queue utilization)
    degraded = False
    if producer_health["queue_utilization"] > 0.8:
        degraded = True
    if sum(consumer_health["consumer_lag"].values()) > 100000:
        degraded = True

    status = HealthStatus.DEGRADED if degraded else HealthStatus.HEALTHY

    return {
        "status": status,
        "producer": producer_health,
        "consumer": consumer_health,
    }


@router.get("/metrics")
async def metrics(
    producer: StreamingTokenTracker,
    consumer: ClickHouseSinkConsumer,
):
    """Detailed metrics endpoint."""
    return {
        "producer": producer.get_metrics().to_dict(),
        "consumer": consumer._metrics.to_dict(),
        "consumer_lag": await consumer._calculate_lag(),
    }
```

### Logging Configuration

```python
"""
tokenledger/streaming/logging.py
Structured logging for streaming components
"""

import logging
import json
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields
        if hasattr(record, "__dict__"):
            for key, value in record.__dict__.items():
                if key not in (
                    "name", "msg", "args", "created", "filename",
                    "funcName", "levelname", "levelno", "lineno",
                    "module", "msecs", "pathname", "process",
                    "processName", "relativeCreated", "stack_info",
                    "exc_info", "exc_text", "thread", "threadName",
                    "message", "asctime"
                ):
                    log_data[key] = value

        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def configure_logging(level: str = "INFO") -> None:
    """Configure structured logging for streaming components."""
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())

    logger = logging.getLogger("tokenledger.streaming")
    logger.setLevel(getattr(logging, level.upper()))
    logger.addHandler(handler)
    logger.propagate = False
```

---

## Configuration Reference

### Environment Variables

```bash
# Kafka/Redpanda Connection
TOKENLEDGER_KAFKA_BOOTSTRAP_SERVERS=localhost:9092
TOKENLEDGER_KAFKA_TOPIC=llm_events
TOKENLEDGER_KAFKA_DLQ_TOPIC=llm_events_dlq

# Schema Registry (optional)
TOKENLEDGER_SCHEMA_REGISTRY_URL=http://localhost:8081

# Producer Settings
TOKENLEDGER_KAFKA_COMPRESSION=lz4
TOKENLEDGER_KAFKA_LINGER_MS=100
TOKENLEDGER_KAFKA_BATCH_SIZE=1000
TOKENLEDGER_KAFKA_MAX_QUEUE_SIZE=100000

# Consumer Settings
TOKENLEDGER_KAFKA_CONSUMER_GROUP=tokenledger-clickhouse-sink
TOKENLEDGER_KAFKA_AUTO_OFFSET_RESET=earliest
TOKENLEDGER_KAFKA_MAX_POLL_RECORDS=1000

# ClickHouse Connection
TOKENLEDGER_CLICKHOUSE_HOST=localhost
TOKENLEDGER_CLICKHOUSE_PORT=8123
TOKENLEDGER_CLICKHOUSE_DATABASE=default
TOKENLEDGER_CLICKHOUSE_USER=default
TOKENLEDGER_CLICKHOUSE_PASSWORD=

# Security (optional)
TOKENLEDGER_KAFKA_SECURITY_PROTOCOL=PLAINTEXT
TOKENLEDGER_KAFKA_SASL_MECHANISM=
TOKENLEDGER_KAFKA_SASL_USERNAME=
TOKENLEDGER_KAFKA_SASL_PASSWORD=

# Observability
TOKENLEDGER_LOG_LEVEL=INFO
TOKENLEDGER_METRICS_PORT=9090
```

### Docker Compose Example

```yaml
version: "3.9"

services:
  redpanda:
    image: redpandadata/redpanda:v24.3.1
    command:
      - redpanda
      - start
      - --kafka-addr internal://0.0.0.0:9092,external://0.0.0.0:19092
      - --advertise-kafka-addr internal://redpanda:9092,external://localhost:19092
      - --schema-registry-addr internal://0.0.0.0:8081,external://0.0.0.0:18081
      - --pandaproxy-addr internal://0.0.0.0:8082,external://0.0.0.0:18082
      - --smp 1
      - --memory 1G
      - --overprovisioned
    ports:
      - "19092:19092"  # Kafka API
      - "18081:18081"  # Schema Registry
      - "18082:18082"  # HTTP Proxy
    volumes:
      - redpanda-data:/var/lib/redpanda/data

  clickhouse:
    image: clickhouse/clickhouse-server:24.12
    ports:
      - "8123:8123"  # HTTP
      - "9000:9000"  # Native
    volumes:
      - clickhouse-data:/var/lib/clickhouse
      - ./migrations/clickhouse:/docker-entrypoint-initdb.d
    environment:
      CLICKHOUSE_DB: tokenledger
      CLICKHOUSE_USER: default
      CLICKHOUSE_PASSWORD: ""

  tokenledger-producer:
    build: .
    environment:
      TOKENLEDGER_KAFKA_BOOTSTRAP_SERVERS: redpanda:9092
      TOKENLEDGER_KAFKA_TOPIC: llm_events
    depends_on:
      - redpanda

  tokenledger-consumer:
    build: .
    command: python -m tokenledger.streaming.consumer
    environment:
      TOKENLEDGER_KAFKA_BOOTSTRAP_SERVERS: redpanda:9092
      TOKENLEDGER_CLICKHOUSE_HOST: clickhouse
      TOKENLEDGER_CLICKHOUSE_DATABASE: tokenledger
    depends_on:
      - redpanda
      - clickhouse

volumes:
  redpanda-data:
  clickhouse-data:
```

---

## Operational Runbook

### Topic Creation

```bash
# Create main topic (12 partitions for parallelism)
rpk topic create llm_events \
  --partitions 12 \
  --replicas 3 \
  --config retention.ms=604800000 \
  --config cleanup.policy=delete

# Create DLQ topic
rpk topic create llm_events_dlq \
  --partitions 3 \
  --replicas 3 \
  --config retention.ms=2592000000  # 30 days

# Verify topics
rpk topic list
rpk topic describe llm_events
```

### Consumer Group Management

```bash
# List consumer groups
rpk group list

# Describe consumer group (check lag)
rpk group describe tokenledger-clickhouse-sink

# Reset consumer offset (use with caution!)
rpk group seek tokenledger-clickhouse-sink --to start
rpk group seek tokenledger-clickhouse-sink --to end
rpk group seek tokenledger-clickhouse-sink --to 1704067200000  # Timestamp

# Delete consumer group (all consumers must be stopped)
rpk group delete tokenledger-clickhouse-sink
```

### Monitoring Commands

```bash
# Check consumer lag
rpk group describe tokenledger-clickhouse-sink | grep -E "LAG|PARTITION"

# Check topic throughput
rpk topic consume llm_events --num 10 --format json

# Check broker health
rpk cluster health

# Schema Registry
curl http://localhost:18081/subjects
curl http://localhost:18081/subjects/llm_events-value/versions
```

### Troubleshooting

| Symptom | Possible Cause | Resolution |
|---------|----------------|------------|
| High producer queue | Broker slow/unavailable | Check broker health, increase batch size |
| High consumer lag | Slow ClickHouse inserts | Increase batch size, add partitions |
| Events in DLQ | Schema mismatch, CH errors | Check DLQ messages, fix schema/data |
| Duplicate events | Consumer restart mid-batch | ClickHouse FINAL query deduplicates |
| Producer timeout | Network issues | Increase timeout, check connectivity |

### Scaling Guidelines

1. **Partitions**: 1 partition per 10K events/sec expected throughput
2. **Consumers**: Match consumer count to partition count
3. **ClickHouse**: Add replicas for read scaling, shards for write scaling
4. **Producer instances**: Scale horizontally (each has own queue)

---

## References

### Documentation

- [aiokafka Documentation](https://aiokafka.readthedocs.io/en/stable/)
- [Redpanda Kafka Compatibility](https://docs.redpanda.com/current/develop/kafka-clients/)
- [Confluent Schema Registry](https://docs.confluent.io/platform/current/schema-registry/index.html)
- [ClickHouse Kafka Engine](https://clickhouse.com/docs/integrations/kafka/kafka-table-engine)

### Articles

- [Choosing a Python Kafka Client](https://quix.io/blog/choosing-python-kafka-client-comparative-analysis)
- [Avro vs JSON Schema vs Protobuf](https://www.automq.com/blog/avro-vs-json-schema-vs-protobuf-kafka-data-formats)
- [Kafka Dead Letter Queue Best Practices](https://www.superstream.ai/blog/kafka-dead-letter-queue)
- [Exactly-Once Semantics in Kafka](https://www.confluent.io/blog/exactly-once-semantics-are-possible-heres-how-apache-kafka-does-it/)

### Library Versions (January 2026)

| Library | Version | Notes |
|---------|---------|-------|
| aiokafka | 0.13.0+ | Python 3.14 support |
| confluent-kafka | 2.13.0+ | AIOProducer/AIOConsumer |
| clickhouse-connect | 0.8.x | Async client |
| python-schema-registry-client | 2.6.x | Schema Registry |
