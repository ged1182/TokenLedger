"""Tests for Anthropic SDK interceptor in TokenLedger."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from tokenledger.models import LLMEvent


# =============================================================================
# Mock Classes for Anthropic Responses
# =============================================================================


class MockAnthropicUsage:
    """Mock usage object for Anthropic responses."""

    def __init__(
        self,
        input_tokens: int = 100,
        output_tokens: int = 50,
        cache_read_input_tokens: int = 0,
    ):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cache_read_input_tokens


class MockContentBlock:
    """Mock content block for Anthropic responses."""

    def __init__(self, text: str = "Hello!", block_type: str = "text"):
        self.type = block_type
        self.text = text


class MockAnthropicResponse:
    """Mock Anthropic API response."""

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        input_tokens: int = 100,
        output_tokens: int = 50,
        cache_read_input_tokens: int = 0,
        content: str = "Hello!",
        stop_reason: str = "end_turn",
    ):
        self.model = model
        self.usage = MockAnthropicUsage(input_tokens, output_tokens, cache_read_input_tokens)
        self.content = [MockContentBlock(text=content)]
        self.stop_reason = stop_reason


class MockDelta:
    """Mock delta for streaming content blocks."""

    def __init__(self, text: str = ""):
        self.text = text


class MockStreamChunk:
    """Mock streaming chunk from Anthropic."""

    def __init__(
        self,
        chunk_type: str,
        text: str | None = None,
        model: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ):
        self.type = chunk_type

        if chunk_type == "content_block_delta" and text is not None:
            self.delta = MockDelta(text)

        if chunk_type == "message_start":
            self.message = MagicMock()
            self.message.model = model
            self.message.usage = MagicMock()
            self.message.usage.input_tokens = input_tokens

        if chunk_type == "message_delta":
            self.usage = MagicMock()
            self.usage.output_tokens = output_tokens


class MockSyncStreamIterator:
    """Mock synchronous stream iterator for Anthropic."""

    def __init__(self, chunks: list[MockStreamChunk]):
        self._chunks = iter(chunks)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._chunks)


class MockSyncStreamContext:
    """Mock synchronous stream context manager for Anthropic."""

    def __init__(self, chunks: list[MockStreamChunk]):
        self._chunks = chunks
        self._stream = None

    def __enter__(self):
        self._stream = MockSyncStreamIterator(self._chunks)
        return self._stream

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class MockAsyncStreamIterator:
    """Mock asynchronous stream iterator for Anthropic."""

    def __init__(self, chunks: list[MockStreamChunk]):
        self._chunks = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration from None


class MockAsyncStreamContext:
    """Mock asynchronous stream context manager for Anthropic."""

    def __init__(self, chunks: list[MockStreamChunk]):
        self._chunks = chunks
        self._stream = None

    async def __aenter__(self):
        self._stream = MockAsyncStreamIterator(self._chunks)
        return self._stream

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False


class MockBatchResponse:
    """Mock Anthropic batch API response."""

    def __init__(
        self,
        batch_id: str = "batch_123",
        processing_status: str = "in_progress",
    ):
        self.id = batch_id
        self.processing_status = processing_status


# =============================================================================
# Tests for Helper Functions
# =============================================================================


class TestExtractTokensFromResponse:
    """Tests for _extract_tokens_from_response helper."""

    def test_extracts_basic_tokens(self) -> None:
        """Test basic token extraction from response."""
        from tokenledger.interceptors.anthropic import _extract_tokens_from_response

        response = MockAnthropicResponse(input_tokens=100, output_tokens=50)
        tokens = _extract_tokens_from_response(response)

        assert tokens["input_tokens"] == 100
        assert tokens["output_tokens"] == 50
        assert tokens["cached_tokens"] == 0

    def test_extracts_cached_tokens(self) -> None:
        """Test extraction of cache_read_input_tokens."""
        from tokenledger.interceptors.anthropic import _extract_tokens_from_response

        response = MockAnthropicResponse(
            input_tokens=100,
            output_tokens=50,
            cache_read_input_tokens=25,
        )
        tokens = _extract_tokens_from_response(response)

        assert tokens["input_tokens"] == 100
        assert tokens["output_tokens"] == 50
        assert tokens["cached_tokens"] == 25

    def test_handles_missing_usage(self) -> None:
        """Test handling when usage attribute is missing."""
        from tokenledger.interceptors.anthropic import _extract_tokens_from_response

        response = MagicMock(spec=[])  # No usage attribute
        tokens = _extract_tokens_from_response(response)

        assert tokens["input_tokens"] == 0
        assert tokens["output_tokens"] == 0
        assert tokens["cached_tokens"] == 0

    def test_handles_none_token_values(self) -> None:
        """Test handling when token values are None."""
        from tokenledger.interceptors.anthropic import _extract_tokens_from_response

        response = MagicMock()
        response.usage = MagicMock()
        response.usage.input_tokens = None
        response.usage.output_tokens = None
        response.usage.cache_read_input_tokens = None

        tokens = _extract_tokens_from_response(response)

        assert tokens["input_tokens"] == 0
        assert tokens["output_tokens"] == 0
        assert tokens["cached_tokens"] == 0


class TestGetRequestPreview:
    """Tests for _get_request_preview helper."""

    def test_extracts_text_content(self) -> None:
        """Test extraction of text content from messages."""
        from tokenledger.interceptors.anthropic import _get_request_preview

        messages = [{"role": "user", "content": "Hello, Claude!"}]
        preview = _get_request_preview(messages)

        assert preview == "Hello, Claude!"

    def test_handles_empty_messages(self) -> None:
        """Test handling of empty messages list."""
        from tokenledger.interceptors.anthropic import _get_request_preview

        assert _get_request_preview([]) is None
        assert _get_request_preview(None) is None

    def test_truncates_long_content(self) -> None:
        """Test that long content is truncated."""
        from tokenledger.interceptors.anthropic import _get_request_preview

        long_content = "A" * 600
        messages = [{"role": "user", "content": long_content}]
        preview = _get_request_preview(messages, max_length=500)

        assert len(preview) == 500

    def test_handles_content_blocks(self) -> None:
        """Test extraction from content block format."""
        from tokenledger.interceptors.anthropic import _get_request_preview

        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello from content block!"}],
            }
        ]
        preview = _get_request_preview(messages)

        assert preview == "Hello from content block!"

    def test_extracts_from_last_message(self) -> None:
        """Test that it extracts from the last message."""
        from tokenledger.interceptors.anthropic import _get_request_preview

        messages = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "Assistant response"},
            {"role": "user", "content": "Last message"},
        ]
        preview = _get_request_preview(messages)

        assert preview == "Last message"


class TestGetResponsePreview:
    """Tests for _get_response_preview helper."""

    def test_extracts_text_content(self) -> None:
        """Test extraction of text content from response."""
        from tokenledger.interceptors.anthropic import _get_response_preview

        response = MockAnthropicResponse(content="Hello from Claude!")
        preview = _get_response_preview(response)

        assert preview == "Hello from Claude!"

    def test_handles_empty_content(self) -> None:
        """Test handling of empty content."""
        from tokenledger.interceptors.anthropic import _get_response_preview

        response = MagicMock()
        response.content = []
        preview = _get_response_preview(response)

        assert preview is None

    def test_truncates_long_content(self) -> None:
        """Test that long content is truncated."""
        from tokenledger.interceptors.anthropic import _get_response_preview

        long_content = "B" * 600
        response = MockAnthropicResponse(content=long_content)
        preview = _get_response_preview(response, max_length=500)

        assert len(preview) == 500


# =============================================================================
# Tests for Sync Messages Create Wrapper
# =============================================================================


class TestWrapMessagesCreate:
    """Tests for _wrap_messages_create wrapper."""

    def test_tracks_successful_call(self) -> None:
        """Test that successful API calls are tracked."""
        from tokenledger.interceptors.anthropic import _wrap_messages_create

        response = MockAnthropicResponse(
            model="claude-3-5-sonnet-20241022",
            input_tokens=100,
            output_tokens=50,
        )

        def mock_create(*args, **kwargs):
            return response

        tracked_event = None

        def capture_track(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.anthropic.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_track
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_messages_create(mock_create)
            result = wrapped(
                model="claude-3-5-sonnet-20241022",
                messages=[{"role": "user", "content": "Hello!"}],
            )

            assert result is response
            assert tracked_event is not None
            assert tracked_event.provider == "anthropic"
            assert tracked_event.model == "claude-3-5-sonnet-20241022"
            assert tracked_event.input_tokens == 100
            assert tracked_event.output_tokens == 50
            assert tracked_event.status == "success"

    def test_tracks_error(self) -> None:
        """Test that API errors are tracked."""
        from tokenledger.interceptors.anthropic import _wrap_messages_create

        def mock_create(*args, **kwargs):
            raise ValueError("API Error")

        tracked_event = None

        def capture_track(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.anthropic.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_track
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_messages_create(mock_create)

            with pytest.raises(ValueError, match="API Error"):
                wrapped(
                    model="claude-3-5-sonnet-20241022",
                    messages=[{"role": "user", "content": "Hello!"}],
                )

            assert tracked_event is not None
            assert tracked_event.status == "error"
            assert tracked_event.error_type == "ValueError"
            assert "API Error" in tracked_event.error_message

    def test_extracts_user_id_from_metadata(self) -> None:
        """Test extraction of user_id from metadata."""
        from tokenledger.interceptors.anthropic import _wrap_messages_create

        response = MockAnthropicResponse()

        def mock_create(*args, **kwargs):
            return response

        tracked_event = None

        def capture_track(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.anthropic.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_track
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_messages_create(mock_create)
            wrapped(
                model="claude-3-5-sonnet-20241022",
                messages=[{"role": "user", "content": "Hello!"}],
                metadata={"user_id": "test_user_123"},
            )

            assert tracked_event.user_id == "test_user_123"

    def test_handles_error_stop_reason(self) -> None:
        """Test that error stop reason is handled."""
        from tokenledger.interceptors.anthropic import _wrap_messages_create

        response = MockAnthropicResponse(stop_reason="error")

        def mock_create(*args, **kwargs):
            return response

        tracked_event = None

        def capture_track(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.anthropic.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_track
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_messages_create(mock_create)
            wrapped(
                model="claude-3-5-sonnet-20241022",
                messages=[{"role": "user", "content": "Hello!"}],
            )

            assert tracked_event.status == "error"

    def test_calculates_cost(self) -> None:
        """Test that cost is calculated."""
        from tokenledger.interceptors.anthropic import _wrap_messages_create

        response = MockAnthropicResponse(
            model="claude-3-5-sonnet-20241022",
            input_tokens=1000,
            output_tokens=500,
        )

        def mock_create(*args, **kwargs):
            return response

        tracked_event = None

        def capture_track(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.anthropic.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_track
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_messages_create(mock_create)
            wrapped(
                model="claude-3-5-sonnet-20241022",
                messages=[{"role": "user", "content": "Hello!"}],
            )

            # Cost should be calculated based on pricing.py
            assert tracked_event.cost_usd is not None
            assert tracked_event.cost_usd > 0

    def test_applies_attribution_context(self) -> None:
        """Test that attribution context is applied."""
        from tokenledger.context import attribution
        from tokenledger.interceptors.anthropic import _wrap_messages_create

        response = MockAnthropicResponse()

        def mock_create(*args, **kwargs):
            return response

        tracked_event = None

        def capture_track(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.anthropic.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_track
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_messages_create(mock_create)

            with attribution(user_id="ctx_user", feature="chat", team="ml"):
                wrapped(
                    model="claude-3-5-sonnet-20241022",
                    messages=[{"role": "user", "content": "Hello!"}],
                )

            assert tracked_event.user_id == "ctx_user"
            assert tracked_event.feature == "chat"
            assert tracked_event.team == "ml"


# =============================================================================
# Tests for Async Messages Create Wrapper
# =============================================================================


class TestWrapAsyncMessagesCreate:
    """Tests for _wrap_async_messages_create wrapper."""

    def test_tracks_successful_async_call(self) -> None:
        """Test that successful async API calls are tracked."""
        from tokenledger.interceptors.anthropic import _wrap_async_messages_create

        async def run_test():
            response = MockAnthropicResponse(
                model="claude-3-5-sonnet-20241022",
                input_tokens=200,
                output_tokens=100,
            )

            async def mock_create(*args, **kwargs):
                return response

            tracked_event = None

            def capture_track(event: Any) -> None:
                nonlocal tracked_event
                tracked_event = event

            with patch("tokenledger.interceptors.anthropic.get_tracker") as mock_get_tracker:
                mock_tracker = MagicMock()
                mock_tracker.track = capture_track
                mock_get_tracker.return_value = mock_tracker

                wrapped = _wrap_async_messages_create(mock_create)
                result = await wrapped(
                    model="claude-3-5-sonnet-20241022",
                    messages=[{"role": "user", "content": "Hello async!"}],
                )

                assert result is response
                assert tracked_event is not None
                assert tracked_event.provider == "anthropic"
                assert tracked_event.input_tokens == 200
                assert tracked_event.output_tokens == 100
                assert tracked_event.status == "success"

        asyncio.run(run_test())

    def test_tracks_async_error(self) -> None:
        """Test that async API errors are tracked."""
        from tokenledger.interceptors.anthropic import _wrap_async_messages_create

        async def run_test():
            async def mock_create(*args, **kwargs):
                raise RuntimeError("Async API Error")

            tracked_event = None

            def capture_track(event: Any) -> None:
                nonlocal tracked_event
                tracked_event = event

            with patch("tokenledger.interceptors.anthropic.get_tracker") as mock_get_tracker:
                mock_tracker = MagicMock()
                mock_tracker.track = capture_track
                mock_get_tracker.return_value = mock_tracker

                wrapped = _wrap_async_messages_create(mock_create)

                with pytest.raises(RuntimeError, match="Async API Error"):
                    await wrapped(
                        model="claude-3-5-sonnet-20241022",
                        messages=[{"role": "user", "content": "Hello!"}],
                    )

                assert tracked_event is not None
                assert tracked_event.status == "error"
                assert tracked_event.error_type == "RuntimeError"

        asyncio.run(run_test())


# =============================================================================
# Tests for Streaming Messages Wrapper
# =============================================================================


class TestWrapStreamingMessages:
    """Tests for _wrap_streaming_messages wrapper."""

    def test_streaming_tracks_tokens(self) -> None:
        """Test that streaming tracks token usage."""
        from tokenledger.interceptors.anthropic import _wrap_streaming_messages

        chunks = [
            MockStreamChunk("message_start", model="claude-3-5-sonnet-20241022", input_tokens=100),
            MockStreamChunk("content_block_delta", text="Hello"),
            MockStreamChunk("content_block_delta", text=" world"),
            MockStreamChunk("content_block_delta", text="!"),
            MockStreamChunk("message_delta", output_tokens=50),
        ]

        def mock_stream(*args, **kwargs):
            return MockSyncStreamContext(chunks)

        tracked_event = None

        def capture_track(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.anthropic.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_track
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_streaming_messages(mock_stream)
            stream_ctx = wrapped(
                model="claude-3-5-sonnet-20241022",
                messages=[{"role": "user", "content": "Hello!"}],
            )

            with stream_ctx as stream:
                for _ in stream:
                    pass

            assert tracked_event is not None
            assert tracked_event.input_tokens == 100
            assert tracked_event.output_tokens == 50
            assert tracked_event.request_type == "chat_stream"
            assert tracked_event.status == "success"

    def test_streaming_accumulates_response(self) -> None:
        """Test that streaming accumulates response text."""
        from tokenledger.interceptors.anthropic import _wrap_streaming_messages

        chunks = [
            MockStreamChunk("message_start", model="claude-3-5-sonnet-20241022", input_tokens=100),
            MockStreamChunk("content_block_delta", text="Hello"),
            MockStreamChunk("content_block_delta", text=" there"),
            MockStreamChunk("message_delta", output_tokens=10),
        ]

        def mock_stream(*args, **kwargs):
            return MockSyncStreamContext(chunks)

        tracked_event = None

        def capture_track(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.anthropic.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_track
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_streaming_messages(mock_stream)
            stream_ctx = wrapped(
                model="claude-3-5-sonnet-20241022",
                messages=[{"role": "user", "content": "Hello!"}],
            )

            with stream_ctx as stream:
                for _ in stream:
                    pass

            assert tracked_event.response_preview == "Hello there"

    def test_streaming_handles_error(self) -> None:
        """Test that streaming errors are tracked."""
        from tokenledger.interceptors.anthropic import _wrap_streaming_messages

        class ErrorStreamIterator:
            def __iter__(self):
                return self

            def __next__(self):
                raise ValueError("Stream error")

        class ErrorStreamContext:
            def __enter__(self):
                return ErrorStreamIterator()

            def __exit__(self, exc_type, exc_val, exc_tb):
                return False

        def mock_stream(*args, **kwargs):
            return ErrorStreamContext()

        tracked_event = None

        def capture_track(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.anthropic.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_track
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_streaming_messages(mock_stream)
            stream_ctx = wrapped(
                model="claude-3-5-sonnet-20241022",
                messages=[{"role": "user", "content": "Hello!"}],
            )

            with pytest.raises(ValueError, match="Stream error"):
                with stream_ctx as stream:
                    for _ in stream:
                        pass

            assert tracked_event is not None
            assert tracked_event.status == "error"
            assert tracked_event.error_type == "ValueError"

    def test_streaming_truncates_long_response(self) -> None:
        """Test that long streaming responses are truncated."""
        from tokenledger.interceptors.anthropic import _wrap_streaming_messages

        # Create chunks that result in >500 chars
        long_text = "A" * 600
        chunks = [
            MockStreamChunk("message_start", model="claude-3-5-sonnet-20241022", input_tokens=100),
            MockStreamChunk("content_block_delta", text=long_text),
            MockStreamChunk("message_delta", output_tokens=100),
        ]

        def mock_stream(*args, **kwargs):
            return MockSyncStreamContext(chunks)

        tracked_event = None

        def capture_track(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.anthropic.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_track
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_streaming_messages(mock_stream)
            stream_ctx = wrapped(
                model="claude-3-5-sonnet-20241022",
                messages=[{"role": "user", "content": "Hello!"}],
            )

            with stream_ctx as stream:
                for _ in stream:
                    pass

            assert len(tracked_event.response_preview) == 500


# =============================================================================
# Tests for Async Streaming Messages Wrapper
# =============================================================================


class TestWrapAsyncStreamingMessages:
    """Tests for _wrap_async_streaming_messages wrapper."""

    def test_async_streaming_tracks_tokens(self) -> None:
        """Test that async streaming tracks token usage."""
        from tokenledger.interceptors.anthropic import _wrap_async_streaming_messages

        async def run_test():
            chunks = [
                MockStreamChunk("message_start", model="claude-3-5-sonnet-20241022", input_tokens=150),
                MockStreamChunk("content_block_delta", text="Async"),
                MockStreamChunk("content_block_delta", text=" response"),
                MockStreamChunk("message_delta", output_tokens=75),
            ]

            # The original method returns the async context manager directly (not awaited)
            def mock_stream(*args, **kwargs):
                return MockAsyncStreamContext(chunks)

            tracked_event = None

            def capture_track(event: Any) -> None:
                nonlocal tracked_event
                tracked_event = event

            with patch("tokenledger.interceptors.anthropic.get_tracker") as mock_get_tracker:
                mock_tracker = MagicMock()
                mock_tracker.track = capture_track
                mock_get_tracker.return_value = mock_tracker

                wrapped = _wrap_async_streaming_messages(mock_stream)
                stream_ctx = await wrapped(
                    model="claude-3-5-sonnet-20241022",
                    messages=[{"role": "user", "content": "Hello async!"}],
                )

                async with stream_ctx as stream:
                    async for _ in stream:
                        pass

                assert tracked_event is not None
                assert tracked_event.input_tokens == 150
                assert tracked_event.output_tokens == 75
                assert tracked_event.response_preview == "Async response"

        asyncio.run(run_test())


# =============================================================================
# Tests for Batch API Wrappers
# =============================================================================


class TestWrapBatchesCreate:
    """Tests for _wrap_batches_create wrapper."""

    def test_batch_create_tracks_metadata(self) -> None:
        """Test that batch creation tracks metadata."""
        from tokenledger.interceptors.anthropic import _wrap_batches_create

        response = MockBatchResponse(batch_id="batch_abc123", processing_status="in_progress")

        def mock_create(*args, **kwargs):
            return response

        tracked_event = None

        def capture_track(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.anthropic.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_track
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_batches_create(mock_create)
            result = wrapped(requests=[{}, {}, {}])  # 3 requests

            assert result is response
            assert tracked_event is not None
            assert tracked_event.request_type == "batch_create"
            assert tracked_event.model == "batch"
            assert tracked_event.status == "success"
            assert tracked_event.metadata_extra["batch_id"] == "batch_abc123"
            assert tracked_event.metadata_extra["request_count"] == 3
            assert tracked_event.cost_usd is None  # Cost is calculated when batch completes

    def test_batch_create_tracks_error(self) -> None:
        """Test that batch creation errors are tracked."""
        from tokenledger.interceptors.anthropic import _wrap_batches_create

        def mock_create(*args, **kwargs):
            raise RuntimeError("Batch creation failed")

        tracked_event = None

        def capture_track(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.anthropic.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_track
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_batches_create(mock_create)

            with pytest.raises(RuntimeError, match="Batch creation failed"):
                wrapped(requests=[{}])

            assert tracked_event is not None
            assert tracked_event.status == "error"
            assert tracked_event.error_type == "RuntimeError"


class TestWrapAsyncBatchesCreate:
    """Tests for _wrap_async_batches_create wrapper."""

    def test_async_batch_create_tracks_metadata(self) -> None:
        """Test that async batch creation tracks metadata."""
        from tokenledger.interceptors.anthropic import _wrap_async_batches_create

        async def run_test():
            response = MockBatchResponse(batch_id="async_batch_123")

            async def mock_create(*args, **kwargs):
                return response

            tracked_event = None

            def capture_track(event: Any) -> None:
                nonlocal tracked_event
                tracked_event = event

            with patch("tokenledger.interceptors.anthropic.get_tracker") as mock_get_tracker:
                mock_tracker = MagicMock()
                mock_tracker.track = capture_track
                mock_get_tracker.return_value = mock_tracker

                wrapped = _wrap_async_batches_create(mock_create)
                result = await wrapped(requests=[{}, {}])

                assert result is response
                assert tracked_event is not None
                assert tracked_event.metadata_extra["batch_id"] == "async_batch_123"
                assert tracked_event.metadata_extra["request_count"] == 2

        asyncio.run(run_test())


# =============================================================================
# Tests for Cost Calculation
# =============================================================================


class TestCostCalculation:
    """Tests for cost calculation in the interceptor."""

    def test_cost_calculation_with_cache(self) -> None:
        """Test that cost is correctly calculated with cached tokens."""
        from tokenledger.interceptors.anthropic import _wrap_messages_create
        from tokenledger.pricing import calculate_cost

        response = MockAnthropicResponse(
            model="claude-3-5-sonnet-20241022",
            input_tokens=1000,
            output_tokens=500,
            cache_read_input_tokens=200,
        )

        def mock_create(*args, **kwargs):
            return response

        tracked_event = None

        def capture_track(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.anthropic.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_track
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_messages_create(mock_create)
            wrapped(
                model="claude-3-5-sonnet-20241022",
                messages=[{"role": "user", "content": "Hello!"}],
            )

            # Calculate expected cost
            expected_cost = calculate_cost(
                "claude-3-5-sonnet-20241022",
                1000,
                500,
                200,
                "anthropic",
            )

            assert tracked_event.cost_usd == expected_cost
            assert tracked_event.cached_tokens == 200

    def test_cost_with_unknown_model(self) -> None:
        """Test handling of unknown model pricing."""
        from tokenledger.interceptors.anthropic import _wrap_messages_create

        response = MockAnthropicResponse(
            model="claude-unknown-model",
            input_tokens=100,
            output_tokens=50,
        )

        def mock_create(*args, **kwargs):
            return response

        tracked_event = None

        def capture_track(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.anthropic.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_track
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_messages_create(mock_create)
            wrapped(
                model="claude-unknown-model",
                messages=[{"role": "user", "content": "Hello!"}],
            )

            # Unknown models should have None cost
            assert tracked_event.cost_usd is None


# =============================================================================
# Tests for Patching and Unpatching
# =============================================================================


class TestPatchingFunctions:
    """Tests for patch_anthropic and unpatch_anthropic functions."""

    def test_patch_requires_anthropic_sdk(self) -> None:
        """Test that patching requires Anthropic SDK."""
        from tokenledger.interceptors.anthropic import patch_anthropic, unpatch_anthropic, _patched
        import tokenledger.interceptors.anthropic as anthropic_module

        # Reset patched state
        anthropic_module._patched = False
        anthropic_module._original_methods.clear()

        with patch.dict("sys.modules", {"anthropic": None}):
            with patch("tokenledger.interceptors.anthropic.get_tracker"):
                # This should raise ImportError when anthropic is not installed
                with pytest.raises(ImportError, match="Anthropic SDK not installed"):
                    patch_anthropic()

    def test_patch_warns_when_already_patched(self) -> None:
        """Test that patching warns when already patched."""
        import tokenledger.interceptors.anthropic as anthropic_module

        # Set patched state
        anthropic_module._patched = True

        with patch("tokenledger.interceptors.anthropic.logger") as mock_logger:
            from tokenledger.interceptors.anthropic import patch_anthropic

            patch_anthropic()
            mock_logger.warning.assert_called_with("Anthropic is already patched")

        # Reset
        anthropic_module._patched = False


# =============================================================================
# Tests for Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_handles_missing_content(self) -> None:
        """Test handling of response with no content."""
        from tokenledger.interceptors.anthropic import _get_response_preview

        response = MagicMock()
        response.content = None

        preview = _get_response_preview(response)
        assert preview is None

    def test_handles_non_text_content_blocks(self) -> None:
        """Test handling of non-text content blocks."""
        from tokenledger.interceptors.anthropic import _get_response_preview

        response = MagicMock()
        response.content = [MagicMock(type="image", text=None)]

        preview = _get_response_preview(response)
        assert preview is None

    def test_duration_tracking(self) -> None:
        """Test that duration is tracked correctly."""
        from tokenledger.interceptors.anthropic import _wrap_messages_create
        import time

        response = MockAnthropicResponse()

        def mock_create(*args, **kwargs):
            time.sleep(0.01)  # Small delay to ensure measurable duration
            return response

        tracked_event = None

        def capture_track(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.anthropic.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_track
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_messages_create(mock_create)
            wrapped(
                model="claude-3-5-sonnet-20241022",
                messages=[{"role": "user", "content": "Hello!"}],
            )

            assert tracked_event.duration_ms is not None
            assert tracked_event.duration_ms >= 10  # At least 10ms

    def test_error_message_truncation(self) -> None:
        """Test that long error messages are truncated."""
        from tokenledger.interceptors.anthropic import _wrap_messages_create

        long_error = "E" * 2000

        def mock_create(*args, **kwargs):
            raise ValueError(long_error)

        tracked_event = None

        def capture_track(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.anthropic.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_track
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_messages_create(mock_create)

            with pytest.raises(ValueError):
                wrapped(
                    model="claude-3-5-sonnet-20241022",
                    messages=[{"role": "user", "content": "Hello!"}],
                )

            assert len(tracked_event.error_message) == 1000

    def test_uses_model_from_response(self) -> None:
        """Test that actual model from response is used instead of requested model."""
        from tokenledger.interceptors.anthropic import _wrap_messages_create

        # Response has different model than requested (e.g., alias resolution)
        response = MockAnthropicResponse(model="claude-3-5-sonnet-20241022")

        def mock_create(*args, **kwargs):
            return response

        tracked_event = None

        def capture_track(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.anthropic.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_track
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_messages_create(mock_create)
            wrapped(
                model="claude-3-5-sonnet-latest",  # Request with alias
                messages=[{"role": "user", "content": "Hello!"}],
            )

            # Should use the actual model from response
            assert tracked_event.model == "claude-3-5-sonnet-20241022"


# =============================================================================
# Tests for Different Model Names
# =============================================================================


class TestDifferentModels:
    """Tests for different Anthropic model names."""

    @pytest.mark.parametrize(
        "model_name",
        [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-opus-4-5-20251101",
            "claude-sonnet-4-5-20250929",
        ],
    )
    def test_tracks_various_models(self, model_name: str) -> None:
        """Test tracking with various Anthropic model names."""
        from tokenledger.interceptors.anthropic import _wrap_messages_create

        response = MockAnthropicResponse(model=model_name)

        def mock_create(*args, **kwargs):
            return response

        tracked_event = None

        def capture_track(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.anthropic.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_track
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_messages_create(mock_create)
            wrapped(
                model=model_name,
                messages=[{"role": "user", "content": "Test"}],
            )

            assert tracked_event.model == model_name
            assert tracked_event.provider == "anthropic"
