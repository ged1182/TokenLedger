"""Tests for the tracker module."""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tokenledger.config import TokenLedgerConfig
from tokenledger.tracker import (
    AsyncTokenTracker,
    LLMEvent,
    TokenTracker,
    get_async_tracker,
    get_tracker,
    track_event,
    track_event_async,
)


class TestLLMEvent:
    """Tests for LLMEvent dataclass."""

    def test_create_basic_event(self) -> None:
        """Test creating a basic LLMEvent."""
        event = LLMEvent(
            provider="openai",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
        )

        assert event.provider == "openai"
        assert event.model == "gpt-4o"
        assert event.input_tokens == 100
        assert event.output_tokens == 50
        assert event.total_tokens == 150  # Auto-calculated
        assert event.status == "success"
        assert event.request_type == "chat"

    def test_event_id_auto_generated(self) -> None:
        """Test that event_id is auto-generated as UUID."""
        event = LLMEvent(provider="openai", model="gpt-4o")
        assert event.event_id is not None
        # Verify it's a valid UUID
        uuid.UUID(event.event_id)

    def test_timestamp_auto_generated(self) -> None:
        """Test that timestamp is auto-generated."""
        event = LLMEvent(provider="openai", model="gpt-4o")
        assert event.timestamp is not None
        assert event.timestamp.tzinfo is not None

    def test_total_tokens_calculated(self) -> None:
        """Test that total_tokens is calculated from input and output."""
        event = LLMEvent(
            provider="openai",
            model="gpt-4o",
            input_tokens=500,
            output_tokens=200,
        )
        assert event.total_tokens == 700

    def test_cost_auto_calculated(self) -> None:
        """Test that cost is auto-calculated when model is known."""
        event = LLMEvent(
            provider="openai",
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
        )
        # gpt-4o: $2.50/1M input, $10.00/1M output
        expected_cost = (1000 / 1_000_000 * 2.50) + (500 / 1_000_000 * 10.00)
        assert event.cost_usd is not None
        assert abs(event.cost_usd - expected_cost) < 0.0001

    def test_cost_not_overwritten_if_provided(self) -> None:
        """Test that provided cost is not overwritten."""
        event = LLMEvent(
            provider="openai",
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
            cost_usd=0.99,
        )
        assert event.cost_usd == 0.99

    def test_to_dict_conversion(self) -> None:
        """Test conversion to dictionary."""
        event = LLMEvent(
            provider="openai",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            user_id="test-user",
            metadata={"key": "value"},
        )

        d = event.to_dict()

        assert d["provider"] == "openai"
        assert d["model"] == "gpt-4o"
        assert d["input_tokens"] == 100
        assert d["output_tokens"] == 50
        assert d["user_id"] == "test-user"
        assert isinstance(d["timestamp"], str)  # Should be ISO format
        assert d["metadata"] == json.dumps({"key": "value"})

    def test_to_dict_with_empty_metadata(self) -> None:
        """Test to_dict with empty metadata returns None."""
        event = LLMEvent(provider="openai", model="gpt-4o")
        d = event.to_dict()
        assert d["metadata"] is None

    def test_event_with_error_status(self) -> None:
        """Test creating an event with error status."""
        event = LLMEvent(
            provider="openai",
            model="gpt-4o",
            status="error",
            error_type="APIError",
            error_message="Rate limit exceeded",
        )

        assert event.status == "error"
        assert event.error_type == "APIError"
        assert event.error_message == "Rate limit exceeded"

    def test_event_with_trace_ids(self) -> None:
        """Test creating an event with trace IDs."""
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())

        event = LLMEvent(
            provider="openai",
            model="gpt-4o",
            trace_id=trace_id,
            span_id=span_id,
        )

        assert event.trace_id == trace_id
        assert event.span_id == span_id

    def test_event_with_cached_tokens(self) -> None:
        """Test event with cached tokens."""
        event = LLMEvent(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            input_tokens=1000,
            output_tokens=500,
            cached_tokens=500,
        )

        assert event.cached_tokens == 500
        # Cost should account for cached tokens
        assert event.cost_usd is not None


class TestTokenTracker:
    """Tests for TokenTracker class."""

    def test_tracker_initialization(self, mock_config: TokenLedgerConfig) -> None:
        """Test TokenTracker initialization with config."""
        tracker = TokenTracker(config=mock_config)

        assert tracker.config == mock_config
        assert tracker._initialized is False
        assert tracker._connection is None
        assert len(tracker._batch) == 0

    def test_tracker_uses_global_config_if_none_provided(self) -> None:
        """Test that tracker uses global config when none provided."""
        from tokenledger.config import configure

        configure(
            database_url="postgresql://test:test@localhost/test",
            app_name="global-app",
        )

        tracker = TokenTracker()
        assert tracker.config.app_name == "global-app"

    @patch("tokenledger.tracker.TokenTracker._get_connection")
    def test_track_adds_event_to_batch(
        self, mock_get_conn: MagicMock, mock_config: TokenLedgerConfig
    ) -> None:
        """Test that track() adds events to batch in sync mode."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_conn.return_value = mock_conn

        tracker = TokenTracker(config=mock_config)
        event = LLMEvent(provider="openai", model="gpt-4o", input_tokens=100)

        tracker.track(event)

        assert len(tracker._batch) == 1
        assert tracker._batch[0] == event

    @patch("tokenledger.tracker.TokenTracker._get_connection")
    def test_track_applies_default_metadata(self, mock_get_conn: MagicMock) -> None:
        """Test that default metadata is applied to events."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_conn.return_value = mock_conn

        config = TokenLedgerConfig(
            database_url="postgresql://test:test@localhost/test",
            async_mode=False,
            default_metadata={"team": "platform"},
        )
        tracker = TokenTracker(config=config)
        event = LLMEvent(provider="openai", model="gpt-4o", metadata={"request_id": "123"})

        tracker.track(event)

        assert "team" in tracker._batch[0].metadata
        assert tracker._batch[0].metadata["team"] == "platform"
        assert tracker._batch[0].metadata["request_id"] == "123"

    @patch("tokenledger.tracker.TokenTracker._get_connection")
    def test_track_applies_app_info(self, mock_get_conn: MagicMock) -> None:
        """Test that app_name and environment are applied from config."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_conn.return_value = mock_conn

        config = TokenLedgerConfig(
            database_url="postgresql://test:test@localhost/test",
            async_mode=False,
            app_name="my-app",
            environment="production",
        )
        tracker = TokenTracker(config=config)
        event = LLMEvent(provider="openai", model="gpt-4o")

        tracker.track(event)

        assert tracker._batch[0].app_name == "my-app"
        assert tracker._batch[0].environment == "production"

    @patch("tokenledger.tracker.TokenTracker._get_connection")
    def test_track_respects_sampling(self, mock_get_conn: MagicMock) -> None:
        """Test that sampling rate is respected."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_conn.return_value = mock_conn

        config = TokenLedgerConfig(
            database_url="postgresql://test:test@localhost/test",
            async_mode=False,
            sample_rate=0.0,  # Sample nothing
        )
        tracker = TokenTracker(config=config)
        event = LLMEvent(provider="openai", model="gpt-4o")

        tracker.track(event)

        assert len(tracker._batch) == 0  # Event was not added due to sampling

    @patch("tokenledger.tracker.TokenTracker._get_connection")
    @patch("tokenledger.tracker.TokenTracker._write_batch")
    def test_flush_triggers_when_batch_full(
        self, mock_write: MagicMock, mock_get_conn: MagicMock
    ) -> None:
        """Test that flush is triggered when batch reaches max size."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_conn.return_value = mock_conn
        mock_write.return_value = 5

        config = TokenLedgerConfig(
            database_url="postgresql://test:test@localhost/test",
            async_mode=False,
            batch_size=5,
        )
        tracker = TokenTracker(config=config)

        # Add batch_size events to trigger flush
        for i in range(5):
            event = LLMEvent(provider="openai", model="gpt-4o", input_tokens=i)
            tracker.track(event)

        # _write_batch should have been called
        mock_write.assert_called()

    @patch("tokenledger.tracker.TokenTracker._get_connection")
    def test_flush_returns_zero_for_empty_batch(
        self, mock_get_conn: MagicMock, mock_config: TokenLedgerConfig
    ) -> None:
        """Test that flush returns 0 when batch is empty."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_conn.return_value = mock_conn

        tracker = TokenTracker(config=mock_config)
        result = tracker.flush()

        assert result == 0

    def test_trace_context_manager(self, mock_config: TokenLedgerConfig) -> None:
        """Test the trace context manager."""
        tracker = TokenTracker(config=mock_config)

        with tracker.trace() as trace_id:
            assert trace_id is not None
            # Verify it's a valid UUID
            uuid.UUID(trace_id)
            assert tracker._context.trace_id == trace_id

        # Context should be cleared after exiting
        assert not hasattr(tracker._context, "trace_id")

    def test_trace_context_manager_with_provided_id(self, mock_config: TokenLedgerConfig) -> None:
        """Test the trace context manager with a provided trace ID."""
        tracker = TokenTracker(config=mock_config)
        custom_trace_id = "custom-trace-123"

        with tracker.trace(trace_id=custom_trace_id) as trace_id:
            assert trace_id == custom_trace_id

    def test_shutdown_clears_state(self, mock_config: TokenLedgerConfig) -> None:
        """Test that shutdown clears tracker state."""
        tracker = TokenTracker(config=mock_config)
        tracker._running = True

        tracker.shutdown()

        assert tracker._running is False
        assert tracker._connection is None


class TestAsyncTokenTracker:
    """Tests for AsyncTokenTracker class."""

    def test_async_tracker_initialization(self, async_mock_config: TokenLedgerConfig) -> None:
        """Test AsyncTokenTracker initialization."""
        tracker = AsyncTokenTracker(config=async_mock_config)

        assert tracker.config == async_mock_config
        assert tracker._initialized is False
        assert tracker._db is None
        assert len(tracker._batch) == 0

    @pytest.mark.asyncio
    async def test_async_tracker_uses_lock(self, async_mock_config: TokenLedgerConfig) -> None:
        """Test that async tracker uses async lock."""
        tracker = AsyncTokenTracker(config=async_mock_config)
        lock = await tracker._get_lock()

        import asyncio

        assert isinstance(lock, asyncio.Lock)

    @pytest.mark.asyncio
    async def test_async_track_adds_to_batch(
        self, async_mock_config: TokenLedgerConfig, mock_async_db: AsyncMock
    ) -> None:
        """Test that async track adds events to batch."""
        tracker = AsyncTokenTracker(config=async_mock_config)
        tracker._db = mock_async_db
        tracker._initialized = True

        event = LLMEvent(provider="openai", model="gpt-4o", input_tokens=100)
        await tracker.track(event)

        assert len(tracker._batch) == 1

    @pytest.mark.asyncio
    async def test_async_flush_returns_count(
        self, async_mock_config: TokenLedgerConfig, mock_async_db: AsyncMock
    ) -> None:
        """Test that async flush returns event count."""
        tracker = AsyncTokenTracker(config=async_mock_config)
        tracker._db = mock_async_db
        tracker._initialized = True

        # Add an event
        event = LLMEvent(provider="openai", model="gpt-4o", input_tokens=100)
        tracker._batch.append(event)

        mock_async_db.insert_events.return_value = 1
        result = await tracker.flush()

        assert result == 1
        assert len(tracker._batch) == 0

    @pytest.mark.asyncio
    async def test_async_shutdown(
        self, async_mock_config: TokenLedgerConfig, mock_async_db: AsyncMock
    ) -> None:
        """Test async tracker shutdown."""
        tracker = AsyncTokenTracker(config=async_mock_config)
        tracker._db = mock_async_db
        tracker._initialized = True

        await tracker.shutdown()

        mock_async_db.close.assert_called_once()
        assert tracker._initialized is False


class TestGlobalTrackers:
    """Tests for global tracker functions."""

    def test_get_tracker_creates_instance(self) -> None:
        """Test that get_tracker creates a global instance."""
        tracker1 = get_tracker()
        tracker2 = get_tracker()

        assert tracker1 is tracker2  # Same instance

    @pytest.mark.asyncio
    async def test_get_async_tracker_creates_instance(self) -> None:
        """Test that get_async_tracker creates a global instance."""
        tracker1 = await get_async_tracker()
        tracker2 = await get_async_tracker()

        assert tracker1 is tracker2  # Same instance

    @patch("tokenledger.tracker.get_tracker")
    def test_track_event_uses_global_tracker(self, mock_get_tracker: MagicMock) -> None:
        """Test that track_event uses the global tracker."""
        mock_tracker = MagicMock()
        mock_get_tracker.return_value = mock_tracker

        event = LLMEvent(provider="openai", model="gpt-4o")
        track_event(event)

        mock_tracker.track.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_track_event_async_uses_global_tracker(self) -> None:
        """Test that track_event_async uses the global async tracker."""
        with patch("tokenledger.tracker.get_async_tracker") as mock_get:
            mock_tracker = AsyncMock()
            mock_get.return_value = mock_tracker

            event = LLMEvent(provider="openai", model="gpt-4o")
            await track_event_async(event)

            mock_tracker.track.assert_called_once_with(event)


class TestTokenTrackerWithBackend:
    """Tests for TokenTracker with backend abstraction."""

    def test_tracker_with_use_backend_flag(self) -> None:
        """Test that use_backend flag is set correctly."""
        config = TokenLedgerConfig(
            database_url="postgresql://test:test@localhost/test",
            async_mode=False,
        )

        tracker = TokenTracker(config=config, use_backend=True)

        assert tracker._use_backend is True
        assert tracker._backend is None  # Backend created on initialize

    def test_tracker_with_custom_backend(self) -> None:
        """Test that a custom backend can be provided."""
        config = TokenLedgerConfig(
            database_url="postgresql://test:test@localhost/test",
            async_mode=False,
        )

        mock_backend = MagicMock()
        tracker = TokenTracker(config=config, backend=mock_backend)

        assert tracker._use_backend is True
        assert tracker._backend is mock_backend

    def test_initialize_with_backend(self) -> None:
        """Test that initialize creates and initializes the backend."""
        config = TokenLedgerConfig(
            database_url="postgresql://test:test@localhost/test",
            async_mode=False,
        )

        mock_backend = MagicMock()
        mock_backend.is_initialized = True

        tracker = TokenTracker(config=config, backend=mock_backend)
        tracker.initialize()

        mock_backend.initialize.assert_called_once_with(config, create_schema=True)

    def test_write_batch_uses_backend(self) -> None:
        """Test that _write_batch uses the backend when enabled."""
        config = TokenLedgerConfig(
            database_url="postgresql://test:test@localhost/test",
            async_mode=False,
        )

        mock_backend = MagicMock()
        mock_backend.is_initialized = True
        mock_backend.write_events.return_value = 2

        tracker = TokenTracker(config=config, backend=mock_backend)
        tracker._initialized = True

        events = [
            LLMEvent(provider="openai", model="gpt-4o"),
            LLMEvent(provider="anthropic", model="claude-3"),
        ]

        result = tracker._write_batch(events)

        assert result == 2
        mock_backend.write_events.assert_called_once()
        # Verify events were converted to dicts
        call_args = mock_backend.write_events.call_args[0][0]
        assert len(call_args) == 2
        assert all(isinstance(e, dict) for e in call_args)

    def test_shutdown_closes_backend(self) -> None:
        """Test that shutdown closes the backend."""
        config = TokenLedgerConfig(
            database_url="postgresql://test:test@localhost/test",
            async_mode=False,
        )

        mock_backend = MagicMock()
        tracker = TokenTracker(config=config, backend=mock_backend)
        tracker._initialized = True

        tracker.shutdown()

        mock_backend.close.assert_called_once()
        assert tracker._backend is None


class TestAsyncTokenTrackerWithBackend:
    """Tests for AsyncTokenTracker with backend abstraction."""

    def test_async_tracker_with_use_backend_flag(self) -> None:
        """Test that use_backend flag is set correctly."""
        config = TokenLedgerConfig(
            database_url="postgresql://test:test@localhost/test",
        )

        tracker = AsyncTokenTracker(config=config, use_backend=True)

        assert tracker._use_backend is True
        assert tracker._backend is None  # Backend created on initialize

    def test_async_tracker_with_custom_backend(self) -> None:
        """Test that a custom backend can be provided."""
        config = TokenLedgerConfig(
            database_url="postgresql://test:test@localhost/test",
        )

        mock_backend = AsyncMock()
        tracker = AsyncTokenTracker(config=config, backend=mock_backend)

        assert tracker._use_backend is True
        assert tracker._backend is mock_backend

    @pytest.mark.asyncio
    async def test_async_initialize_with_backend(self) -> None:
        """Test that initialize creates and initializes the backend."""
        config = TokenLedgerConfig(
            database_url="postgresql://test:test@localhost/test",
        )

        mock_backend = AsyncMock()
        mock_backend.is_initialized = True

        tracker = AsyncTokenTracker(config=config, backend=mock_backend)
        await tracker.initialize()

        mock_backend.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_write_batch_uses_backend(self) -> None:
        """Test that _write_batch uses the backend when enabled."""
        config = TokenLedgerConfig(
            database_url="postgresql://test:test@localhost/test",
        )

        mock_backend = AsyncMock()
        mock_backend.is_initialized = True
        mock_backend.write_events.return_value = 2

        tracker = AsyncTokenTracker(config=config, backend=mock_backend)
        tracker._initialized = True

        events = [
            LLMEvent(provider="openai", model="gpt-4o"),
            LLMEvent(provider="anthropic", model="claude-3"),
        ]

        result = await tracker._write_batch(events)

        assert result == 2
        mock_backend.write_events.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_shutdown_closes_backend(self) -> None:
        """Test that shutdown closes the backend."""
        config = TokenLedgerConfig(
            database_url="postgresql://test:test@localhost/test",
        )

        mock_backend = AsyncMock()
        tracker = AsyncTokenTracker(config=config, backend=mock_backend)
        tracker._initialized = True

        await tracker.shutdown()

        mock_backend.close.assert_called_once()
        assert tracker._backend is None


class TestTrackerAttributionContext:
    """Tests for attribution context integration in tracker."""

    @patch("tokenledger.tracker.TokenTracker._get_connection")
    def test_track_applies_attribution_context(self, mock_get_conn: MagicMock) -> None:
        """Test that attribution context is applied to events."""
        from tokenledger.context import attribution

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_conn.return_value = mock_conn

        config = TokenLedgerConfig(
            database_url="postgresql://test:test@localhost/test",
            async_mode=False,
        )
        tracker = TokenTracker(config=config)

        with attribution(
            user_id="ctx_user",
            session_id="ctx_session",
            feature="ctx_feature",
            team="ctx_team",
            project="ctx_project",
            cost_center="ctx_cost_center",
        ):
            event = LLMEvent(provider="openai", model="gpt-4o")
            tracker.track(event)

        # Check attribution was applied
        tracked_event = tracker._batch[0]
        assert tracked_event.user_id == "ctx_user"
        assert tracked_event.session_id == "ctx_session"
        assert tracked_event.feature == "ctx_feature"
        assert tracked_event.team == "ctx_team"
        assert tracked_event.project == "ctx_project"
        assert tracked_event.cost_center == "ctx_cost_center"

    @patch("tokenledger.tracker.TokenTracker._get_connection")
    def test_event_values_take_precedence_over_context(self, mock_get_conn: MagicMock) -> None:
        """Test that event-level values take precedence over context."""
        from tokenledger.context import attribution

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_conn.return_value = mock_conn

        config = TokenLedgerConfig(
            database_url="postgresql://test:test@localhost/test",
            async_mode=False,
        )
        tracker = TokenTracker(config=config)

        with attribution(user_id="ctx_user", feature="ctx_feature"):
            event = LLMEvent(
                provider="openai",
                model="gpt-4o",
                user_id="event_user",  # This should take precedence
            )
            tracker.track(event)

        tracked_event = tracker._batch[0]
        assert tracked_event.user_id == "event_user"  # Event value used
        assert tracked_event.feature == "ctx_feature"  # Context value used


class TestTrackerAsyncMode:
    """Tests for async mode behavior."""

    @patch("tokenledger.tracker.TokenTracker._get_connection")
    @patch("tokenledger.tracker.TokenTracker._start_flush_thread")
    def test_async_mode_uses_queue(
        self, mock_start_flush: MagicMock, mock_get_conn: MagicMock
    ) -> None:
        """Test that async mode uses queue for events."""
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn

        config = TokenLedgerConfig(
            database_url="postgresql://test:test@localhost/test",
            async_mode=True,
        )
        tracker = TokenTracker(config=config)
        tracker._initialized = True  # Skip initialization

        event = LLMEvent(provider="openai", model="gpt-4o")
        tracker.track(event)

        # Event should be in queue, not batch
        assert len(tracker._batch) == 0
        assert not tracker._queue.empty()

    def test_flush_drains_queue_in_async_mode(self) -> None:
        """Test that flush drains the queue in async mode."""
        config = TokenLedgerConfig(
            database_url="postgresql://test:test@localhost/test",
            async_mode=True,
        )
        tracker = TokenTracker(config=config)
        tracker._initialized = True  # Skip initialization

        # Add events to queue
        event1 = LLMEvent(provider="openai", model="gpt-4o")
        event2 = LLMEvent(provider="anthropic", model="claude-3")
        tracker._queue.put(event1)
        tracker._queue.put(event2)

        # Mock backend to prevent actual DB write
        mock_backend = MagicMock()
        mock_backend.write_events.return_value = 2
        tracker._use_backend = True
        tracker._backend = mock_backend

        result = tracker.flush()

        assert result == 2
        assert tracker._queue.empty()


class TestTrackerWriteBatch:
    """Tests for _write_batch method."""

    @patch("tokenledger.tracker.TokenTracker._get_connection")
    def test_write_batch_empty_returns_zero(self, mock_get_conn: MagicMock) -> None:
        """Test that writing empty batch returns 0."""
        config = TokenLedgerConfig(
            database_url="postgresql://test:test@localhost/test",
            async_mode=False,
        )
        tracker = TokenTracker(config=config)

        result = tracker._write_batch([])
        assert result == 0

    @patch("tokenledger.tracker.TokenTracker._get_connection")
    def test_write_batch_with_error(self, mock_get_conn: MagicMock) -> None:
        """Test that write batch handles errors gracefully."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.executemany.side_effect = Exception("DB Error")
        mock_get_conn.return_value = mock_conn

        config = TokenLedgerConfig(
            database_url="postgresql://test:test@localhost/test",
            async_mode=False,
        )
        tracker = TokenTracker(config=config)
        tracker._use_psycopg2 = False  # Use psycopg3 path

        events = [LLMEvent(provider="openai", model="gpt-4o")]

        result = tracker._write_batch(events)

        assert result == 0
        mock_conn.rollback.assert_called_once()

    @patch("tokenledger.tracker.TokenTracker._get_connection")
    def test_write_batch_psycopg2_path(self, mock_get_conn: MagicMock) -> None:
        """Test write batch using psycopg2 execute_values."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        config = TokenLedgerConfig(
            database_url="postgresql://test:test@localhost/test",
            async_mode=False,
        )
        tracker = TokenTracker(config=config)
        tracker._use_psycopg2 = True
        mock_execute_values = MagicMock()
        tracker._execute_values = mock_execute_values

        events = [LLMEvent(provider="openai", model="gpt-4o")]

        result = tracker._write_batch(events)

        assert result == 1
        mock_execute_values.assert_called_once()
        mock_conn.commit.assert_called_once()

    @patch("tokenledger.tracker.TokenTracker._get_connection")
    def test_write_batch_logs_in_debug_mode(self, mock_get_conn: MagicMock) -> None:
        """Test write batch logs when debug is enabled."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn

        config = TokenLedgerConfig(
            database_url="postgresql://test:test@localhost/test",
            debug=True,
        )
        tracker = TokenTracker(config=config)
        tracker._use_psycopg2 = False

        events = [LLMEvent(provider="openai", model="gpt-4o")]

        with patch("tokenledger.tracker.logger") as mock_logger:
            result = tracker._write_batch(events)

            assert result == 1
            mock_logger.info.assert_called()


class TestTrackerFlushThread:
    """Tests for flush thread behavior."""

    @patch("tokenledger.tracker.TokenTracker._get_connection")
    def test_start_flush_thread(self, mock_get_conn: MagicMock) -> None:
        """Test that _start_flush_thread starts a background thread."""
        config = TokenLedgerConfig(
            database_url="postgresql://test:test@localhost/test",
            async_mode=True,
        )
        tracker = TokenTracker(config=config)
        tracker._running = False  # Ensure it's stopped initially

        tracker._start_flush_thread()

        assert tracker._running is True
        assert tracker._flush_thread is not None
        assert tracker._flush_thread.daemon is True

        # Clean up
        tracker._running = False
        tracker._flush_thread.join(timeout=1.0)

    @patch("tokenledger.tracker.TokenTracker._get_connection")
    @patch("tokenledger.tracker.time.sleep")
    def test_flush_loop_calls_flush(
        self, mock_sleep: MagicMock, mock_get_conn: MagicMock
    ) -> None:
        """Test that _flush_loop periodically calls flush."""
        config = TokenLedgerConfig(
            database_url="postgresql://test:test@localhost/test",
            flush_interval_seconds=1,
        )
        tracker = TokenTracker(config=config)
        tracker._running = True

        # Make sleep raise exception after first call to stop the loop
        call_count = 0

        def side_effect(_):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                tracker._running = False
            return None

        mock_sleep.side_effect = side_effect
        tracker.flush = MagicMock()

        tracker._flush_loop()

        # Should have called flush at least once
        assert tracker.flush.call_count >= 1

    @patch("tokenledger.tracker.TokenTracker._get_connection")
    @patch("tokenledger.tracker.time.sleep")
    def test_flush_loop_handles_exception(
        self, mock_sleep: MagicMock, mock_get_conn: MagicMock
    ) -> None:
        """Test that _flush_loop handles flush exceptions."""
        config = TokenLedgerConfig(
            database_url="postgresql://test:test@localhost/test",
            flush_interval_seconds=1,
        )
        tracker = TokenTracker(config=config)
        tracker._running = True

        call_count = 0

        def sleep_side_effect(_):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                tracker._running = False
            return None

        mock_sleep.side_effect = sleep_side_effect
        tracker.flush = MagicMock(side_effect=Exception("Flush error"))

        with patch("tokenledger.tracker.logger") as mock_logger:
            tracker._flush_loop()

            # Should have logged the error
            mock_logger.error.assert_called()


class TestTrackerShutdown:
    """Tests for tracker shutdown."""

    @patch("tokenledger.tracker.TokenTracker._get_connection")
    def test_shutdown_closes_connection(self, mock_get_conn: MagicMock) -> None:
        """Test that shutdown closes the database connection."""
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn

        config = TokenLedgerConfig(
            database_url="postgresql://test:test@localhost/test",
        )
        tracker = TokenTracker(config=config)
        tracker._connection = mock_conn
        tracker._initialized = True

        # Mock flush to avoid actual DB operations
        tracker.flush = MagicMock(return_value=0)

        tracker.shutdown()

        mock_conn.close.assert_called_once()
        assert tracker._connection is None

    @patch("tokenledger.tracker.TokenTracker._get_connection")
    def test_shutdown_joins_flush_thread(self, mock_get_conn: MagicMock) -> None:
        """Test that shutdown waits for flush thread to finish."""
        config = TokenLedgerConfig(
            database_url="postgresql://test:test@localhost/test",
        )
        tracker = TokenTracker(config=config)
        tracker._running = True

        mock_thread = MagicMock()
        tracker._flush_thread = mock_thread

        tracker.flush = MagicMock(return_value=0)

        tracker.shutdown()

        assert tracker._running is False
        mock_thread.join.assert_called_once_with(timeout=5.0)


class TestTrackerAttributionContextComplete:
    """Tests for complete attribution context coverage."""

    @patch("tokenledger.tracker.TokenTracker._get_connection")
    def test_track_applies_all_attribution_fields(self, mock_get_conn: MagicMock) -> None:
        """Test that all attribution context fields are applied."""
        from tokenledger.context import attribution

        config = TokenLedgerConfig(
            database_url="postgresql://test:test@localhost/test",
            async_mode=False,  # Disable async mode to use batch directly
        )
        tracker = TokenTracker(config=config)
        tracker._initialized = True

        # Note: metadata_extra kwargs are passed via **kwargs syntax
        with attribution(
            user_id="ctx_user",
            session_id="ctx_session",
            organization_id="ctx_org",
            feature="ctx_feature",
            page="/ctx/page",
            component="ctx_component",
            team="ctx_team",
            project="ctx_project",
            cost_center="ctx_cost_center",
            ctx_key="ctx_value",  # This goes into metadata_extra via **kwargs
        ):
            event = LLMEvent(provider="openai", model="gpt-4o")
            tracker.track(event)

        tracked_event = tracker._batch[0]
        assert tracked_event.user_id == "ctx_user"
        assert tracked_event.session_id == "ctx_session"
        assert tracked_event.organization_id == "ctx_org"
        assert tracked_event.feature == "ctx_feature"
        assert tracked_event.page == "/ctx/page"
        assert tracked_event.component == "ctx_component"
        assert tracked_event.team == "ctx_team"
        assert tracked_event.project == "ctx_project"
        assert tracked_event.cost_center == "ctx_cost_center"
        assert tracked_event.metadata_extra == {"ctx_key": "ctx_value"}

    @patch("tokenledger.tracker.TokenTracker._get_connection")
    def test_track_metadata_extra_merges_with_existing(
        self, mock_get_conn: MagicMock
    ) -> None:
        """Test that metadata_extra from context merges with event metadata_extra."""
        from tokenledger.context import attribution

        config = TokenLedgerConfig(
            database_url="postgresql://test:test@localhost/test",
            async_mode=False,  # Disable async mode to use batch directly
        )
        tracker = TokenTracker(config=config)
        tracker._initialized = True

        # Pass metadata via **kwargs
        with attribution(ctx_key="ctx_value"):
            event = LLMEvent(
                provider="openai",
                model="gpt-4o",
                metadata_extra={"event_key": "event_value"},
            )
            tracker.track(event)

        tracked_event = tracker._batch[0]
        # Event values should take precedence, but context values should be included
        assert tracked_event.metadata_extra["ctx_key"] == "ctx_value"
        assert tracked_event.metadata_extra["event_key"] == "event_value"

    @patch("tokenledger.tracker.TokenTracker._get_connection")
    def test_async_mode_queue_full_warning(self, mock_get_conn: MagicMock) -> None:
        """Test that warning is logged when queue is full."""
        config = TokenLedgerConfig(
            database_url="postgresql://test:test@localhost/test",
            async_mode=True,
        )
        tracker = TokenTracker(config=config)
        tracker._initialized = True

        # Make queue.put_nowait raise an exception
        tracker._queue.put_nowait = MagicMock(side_effect=Exception("Queue full"))

        with patch("tokenledger.tracker.logger") as mock_logger:
            event = LLMEvent(provider="openai", model="gpt-4o")
            tracker.track(event)

            mock_logger.warning.assert_called_once()
