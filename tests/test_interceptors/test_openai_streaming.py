"""Tests for OpenAI streaming support in TokenLedger."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


class MockUsage:
    """Mock usage object for OpenAI responses."""

    def __init__(
        self,
        prompt_tokens: int = 100,
        completion_tokens: int = 50,
        cached_tokens: int = 0,
    ):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.prompt_tokens_details = MagicMock()
        self.prompt_tokens_details.cached_tokens = cached_tokens


class MockDelta:
    """Mock delta object for streaming chunks."""

    def __init__(self, content: str | None = None):
        self.content = content


class MockChoice:
    """Mock choice object for streaming chunks."""

    def __init__(self, delta: MockDelta | None = None):
        self.delta = delta or MockDelta()


class MockStreamChunk:
    """Mock streaming chunk from OpenAI."""

    def __init__(
        self,
        content: str | None = None,
        model: str | None = "gpt-4o",
        usage: MockUsage | None = None,
    ):
        self.choices = [MockChoice(MockDelta(content))] if content is not None else []
        self.model = model
        self.usage = usage


class MockSyncStream:
    """Mock synchronous stream iterator."""

    def __init__(self, chunks: list[MockStreamChunk]):
        self._chunks = iter(chunks)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._chunks)


class MockAsyncStream:
    """Mock asynchronous stream iterator."""

    def __init__(self, chunks: list[MockStreamChunk]):
        self._chunks = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration from None


class TestTrackedStreamIterator:
    """Tests for sync streaming wrapper."""

    def test_sync_stream_basic(self) -> None:
        """Test basic sync streaming with content accumulation."""
        from tokenledger.interceptors.openai import TrackedStreamIterator
        from tokenledger.models import LLMEvent

        chunks = [
            MockStreamChunk(content="Hello"),
            MockStreamChunk(content=" world"),
            MockStreamChunk(content="!"),
            MockStreamChunk(content=None, usage=MockUsage(100, 50)),
        ]
        mock_stream = MockSyncStream(chunks)

        event = LLMEvent.fast_construct(
            provider="openai",
            model="gpt-4o",
            request_type="chat_stream",
        )

        mock_tracker = MagicMock()
        mock_tracker.track = MagicMock()

        tracked = TrackedStreamIterator(
            stream=mock_stream,
            event=event,
            start_time=0.0,
            tracker=mock_tracker,
            model="gpt-4o",
        )

        # Consume the stream
        collected = []
        for chunk in tracked:
            collected.append(chunk)

        # Verify all chunks were returned
        assert len(collected) == 4

        # Verify event was tracked
        mock_tracker.track.assert_called_once()

        # Verify event has correct data
        assert event.status == "success"
        assert event.input_tokens == 100
        assert event.output_tokens == 50
        assert event.response_preview == "Hello world!"

    def test_sync_stream_model_extraction(self) -> None:
        """Test that model is extracted from stream chunks."""
        from tokenledger.interceptors.openai import TrackedStreamIterator
        from tokenledger.models import LLMEvent

        chunks = [
            MockStreamChunk(content="Hi", model="gpt-4o-2024-01-15"),
            # Second chunk has model=None to not overwrite the extracted model
            MockStreamChunk(content=None, model=None, usage=MockUsage(10, 5)),
        ]
        mock_stream = MockSyncStream(chunks)

        event = LLMEvent.fast_construct(
            provider="openai",
            model="gpt-4o",  # Initial model
            request_type="chat_stream",
        )

        mock_tracker = MagicMock()

        tracked = TrackedStreamIterator(
            stream=mock_stream,
            event=event,
            start_time=0.0,
            tracker=mock_tracker,
            model="gpt-4o",
        )

        # Consume the stream
        list(tracked)

        # Model should be updated from chunk
        assert event.model == "gpt-4o-2024-01-15"

    def test_sync_stream_error_handling(self) -> None:
        """Test that errors are properly tracked."""
        from tokenledger.interceptors.openai import TrackedStreamIterator
        from tokenledger.models import LLMEvent

        class ErrorStream:
            def __iter__(self):
                return self

            def __next__(self):
                raise ValueError("Test error")

        event = LLMEvent.fast_construct(
            provider="openai",
            model="gpt-4o",
            request_type="chat_stream",
        )

        mock_tracker = MagicMock()

        tracked = TrackedStreamIterator(
            stream=ErrorStream(),
            event=event,
            start_time=0.0,
            tracker=mock_tracker,
            model="gpt-4o",
        )

        with pytest.raises(ValueError, match="Test error"):
            list(tracked)

        # Verify error event was tracked
        mock_tracker.track.assert_called_once()
        assert event.status == "error"
        assert event.error_type == "ValueError"
        assert "Test error" in event.error_message

    def test_sync_stream_cached_tokens(self) -> None:
        """Test that cached tokens are extracted from usage."""
        from tokenledger.interceptors.openai import TrackedStreamIterator
        from tokenledger.models import LLMEvent

        chunks = [
            MockStreamChunk(content="Test"),
            MockStreamChunk(content=None, usage=MockUsage(100, 50, cached_tokens=25)),
        ]
        mock_stream = MockSyncStream(chunks)

        event = LLMEvent.fast_construct(
            provider="openai",
            model="gpt-4o",
            request_type="chat_stream",
        )

        mock_tracker = MagicMock()

        tracked = TrackedStreamIterator(
            stream=mock_stream,
            event=event,
            start_time=0.0,
            tracker=mock_tracker,
            model="gpt-4o",
        )

        list(tracked)

        assert event.cached_tokens == 25


class TestAsyncTrackedStreamIterator:
    """Tests for async streaming wrapper."""

    def test_async_stream_basic(self) -> None:
        """Test basic async streaming with content accumulation."""
        from tokenledger.interceptors.openai import AsyncTrackedStreamIterator
        from tokenledger.models import LLMEvent

        async def run_test():
            chunks = [
                MockStreamChunk(content="Hello"),
                MockStreamChunk(content=" async"),
                MockStreamChunk(content=" world!"),
                MockStreamChunk(content=None, usage=MockUsage(200, 100)),
            ]
            mock_stream = MockAsyncStream(chunks)

            event = LLMEvent.fast_construct(
                provider="openai",
                model="gpt-4o",
                request_type="chat_stream",
            )

            mock_tracker = MagicMock()

            tracked = AsyncTrackedStreamIterator(
                stream=mock_stream,
                event=event,
                start_time=0.0,
                tracker=mock_tracker,
                model="gpt-4o",
            )

            # Consume the stream
            collected = []
            async for chunk in tracked:
                collected.append(chunk)

            # Verify all chunks were returned
            assert len(collected) == 4

            # Verify event was tracked
            mock_tracker.track.assert_called_once()

            # Verify event has correct data
            assert event.status == "success"
            assert event.input_tokens == 200
            assert event.output_tokens == 100
            assert event.response_preview == "Hello async world!"

        asyncio.run(run_test())

    def test_async_stream_error_handling(self) -> None:
        """Test that async errors are properly tracked."""
        from tokenledger.interceptors.openai import AsyncTrackedStreamIterator
        from tokenledger.models import LLMEvent

        class AsyncErrorStream:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise RuntimeError("Async test error")

        async def run_test():
            event = LLMEvent.fast_construct(
                provider="openai",
                model="gpt-4o",
                request_type="chat_stream",
            )

            mock_tracker = MagicMock()

            tracked = AsyncTrackedStreamIterator(
                stream=AsyncErrorStream(),
                event=event,
                start_time=0.0,
                tracker=mock_tracker,
                model="gpt-4o",
            )

            with pytest.raises(RuntimeError, match="Async test error"):
                async for _ in tracked:
                    pass

            # Verify error event was tracked
            mock_tracker.track.assert_called_once()
            assert event.status == "error"
            assert event.error_type == "RuntimeError"
            assert "Async test error" in event.error_message

        asyncio.run(run_test())


class TestWrapChatCompletionsCreateStreaming:
    """Tests for the wrapped chat.completions.create with streaming."""

    def test_streaming_returns_tracked_iterator(self) -> None:
        """Test that streaming calls return TrackedStreamIterator."""
        from tokenledger.interceptors.openai import (
            TrackedStreamIterator,
            _wrap_chat_completions_create,
        )

        chunks = [
            MockStreamChunk(content="Test"),
            MockStreamChunk(content=None, usage=MockUsage(10, 5)),
        ]

        def mock_create(*args, **kwargs):
            return MockSyncStream(chunks)

        with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_chat_completions_create(mock_create, track_streaming=True)

            result = wrapped(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
            )

            assert isinstance(result, TrackedStreamIterator)

    def test_non_streaming_works_normally(self) -> None:
        """Test that non-streaming calls work as before."""
        from tokenledger.interceptors.openai import _wrap_chat_completions_create

        mock_response = MagicMock()
        mock_response.model = "gpt-4o"
        mock_response.usage = MockUsage(100, 50)
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello!"

        def mock_create(*args, **kwargs):
            return mock_response

        with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_chat_completions_create(mock_create, track_streaming=True)

            result = wrapped(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                stream=False,
            )

            # Should return original response, not iterator
            assert result is mock_response

            # Event should be tracked immediately
            mock_tracker.track.assert_called_once()

    def test_streaming_disabled(self) -> None:
        """Test that streaming tracking can be disabled."""
        from tokenledger.interceptors.openai import (
            TrackedStreamIterator,
            _wrap_chat_completions_create,
        )

        chunks = [MockStreamChunk(content="Test")]

        def mock_create(*args, **kwargs):
            return MockSyncStream(chunks)

        with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_get_tracker.return_value = mock_tracker

            # Disable streaming tracking
            wrapped = _wrap_chat_completions_create(mock_create, track_streaming=False)

            result = wrapped(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
            )

            # Should return raw stream, not tracked iterator
            assert not isinstance(result, TrackedStreamIterator)


class TestWrapAsyncChatCompletionsCreateStreaming:
    """Tests for the wrapped async chat.completions.create with streaming."""

    def test_async_streaming_returns_tracked_iterator(self) -> None:
        """Test that async streaming calls return AsyncTrackedStreamIterator."""
        from tokenledger.interceptors.openai import (
            AsyncTrackedStreamIterator,
            _wrap_async_chat_completions_create,
        )

        async def run_test():
            chunks = [
                MockStreamChunk(content="Test"),
                MockStreamChunk(content=None, usage=MockUsage(10, 5)),
            ]

            async def mock_create(*args, **kwargs):
                return MockAsyncStream(chunks)

            with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
                mock_tracker = MagicMock()
                mock_get_tracker.return_value = mock_tracker

                wrapped = _wrap_async_chat_completions_create(mock_create, track_streaming=True)

                result = await wrapped(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "Hello"}],
                    stream=True,
                )

                assert isinstance(result, AsyncTrackedStreamIterator)

        asyncio.run(run_test())


class TestStreamingIntegration:
    """Integration-style tests for streaming with attribution context."""

    def test_streaming_with_attribution_context(self) -> None:
        """Test that attribution context is applied to streaming events."""
        from tokenledger.context import attribution
        from tokenledger.interceptors.openai import (
            TrackedStreamIterator,
            _wrap_chat_completions_create,
        )

        chunks = [
            MockStreamChunk(content="Hello"),
            MockStreamChunk(content=None, usage=MockUsage(10, 5)),
        ]

        def mock_create(*args, **kwargs):
            return MockSyncStream(chunks)

        tracked_event = None

        def capture_tracker_track(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_tracker_track
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_chat_completions_create(mock_create, track_streaming=True)

            with attribution(user_id="test_user", feature="streaming_test"):
                result = wrapped(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "Hello"}],
                    stream=True,
                )

                assert isinstance(result, TrackedStreamIterator)

                # Consume the stream to trigger tracking
                list(result)

            # Verify attribution context was applied
            assert tracked_event is not None
            assert tracked_event.user_id == "test_user"
            assert tracked_event.feature == "streaming_test"

    def test_streaming_request_type(self) -> None:
        """Test that streaming requests have correct request_type."""
        from tokenledger.interceptors.openai import _wrap_chat_completions_create

        chunks = [
            MockStreamChunk(content="Test"),
            MockStreamChunk(content=None, usage=MockUsage(10, 5)),
        ]

        def mock_create(*args, **kwargs):
            return MockSyncStream(chunks)

        tracked_event = None

        def capture_tracker_track(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_tracker_track
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_chat_completions_create(mock_create, track_streaming=True)

            result = wrapped(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
            )

            # Consume the stream
            list(result)

            # Verify request type is chat_stream
            assert tracked_event is not None
            assert tracked_event.request_type == "chat_stream"

    def test_response_preview_truncation(self) -> None:
        """Test that long responses are truncated in preview."""
        from tokenledger.interceptors.openai import _wrap_chat_completions_create

        # Create a response that's longer than 500 chars
        long_content = "A" * 600
        chunks = [
            MockStreamChunk(content=long_content),
            MockStreamChunk(content=None, usage=MockUsage(100, 50)),
        ]

        def mock_create(*args, **kwargs):
            return MockSyncStream(chunks)

        tracked_event = None

        def capture_tracker_track(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_tracker_track
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_chat_completions_create(mock_create, track_streaming=True)

            result = wrapped(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
            )

            # Consume the stream
            list(result)

            # Verify response preview is truncated to 500 chars
            assert tracked_event is not None
            assert len(tracked_event.response_preview) == 500
