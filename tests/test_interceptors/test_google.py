"""Tests for Google GenAI (Gemini) SDK interceptor in TokenLedger."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# Mock Classes for Google GenAI SDK
# =============================================================================


class MockUsageMetadata:
    """Mock usage_metadata object for Google GenAI responses."""

    def __init__(
        self,
        prompt_tokens: int = 100,
        candidates_tokens: int = 50,
        cached_tokens: int = 0,
    ):
        self.prompt_token_count = prompt_tokens
        self.candidates_token_count = candidates_tokens
        self.cached_content_token_count = cached_tokens


class MockPart:
    """Mock part object in Google GenAI content."""

    def __init__(self, text: str | None = None):
        self.text = text


class MockContent:
    """Mock content object in Google GenAI response."""

    def __init__(self, parts: list[MockPart] | None = None):
        self.parts = parts or []


class MockCandidate:
    """Mock candidate object in Google GenAI response."""

    def __init__(self, content: MockContent | None = None):
        self.content = content or MockContent()


class MockGenerateContentResponse:
    """Mock response from Google GenAI generate_content."""

    def __init__(
        self,
        text: str = "Hello, world!",
        usage_metadata: MockUsageMetadata | None = None,
        candidates: list[MockCandidate] | None = None,
    ):
        self.text = text
        self.usage_metadata = usage_metadata or MockUsageMetadata()
        if candidates is None:
            self.candidates = [MockCandidate(MockContent([MockPart(text)]))]
        else:
            self.candidates = candidates


class MockEmbedContentResponse:
    """Mock response from Google GenAI embed_content."""

    def __init__(self, usage_metadata: MockUsageMetadata | None = None):
        self.usage_metadata = usage_metadata or MockUsageMetadata(prompt_tokens=50)


class MockSyncStreamIterator:
    """Mock synchronous stream iterator for generate_content_stream."""

    def __init__(self, chunks: list[MockGenerateContentResponse]):
        self._chunks = iter(chunks)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._chunks)


class MockAsyncStreamIterator:
    """Mock asynchronous stream iterator for generate_content_stream."""

    def __init__(self, chunks: list[MockGenerateContentResponse]):
        self._chunks = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration from None


# =============================================================================
# Tests for Token Extraction
# =============================================================================


class TestExtractTokensFromResponse:
    """Tests for the _extract_tokens_from_response function."""

    def test_extract_tokens_basic(self) -> None:
        """Test basic token extraction from response."""
        from tokenledger.interceptors.google import _extract_tokens_from_response

        response = MockGenerateContentResponse(
            usage_metadata=MockUsageMetadata(
                prompt_tokens=100,
                candidates_tokens=50,
                cached_tokens=25,
            )
        )

        tokens = _extract_tokens_from_response(response)

        assert tokens["input_tokens"] == 100
        assert tokens["output_tokens"] == 50
        assert tokens["cached_tokens"] == 25

    def test_extract_tokens_missing_usage_metadata(self) -> None:
        """Test token extraction when usage_metadata is missing."""
        from tokenledger.interceptors.google import _extract_tokens_from_response

        response = MagicMock(spec=[])  # No usage_metadata attribute

        tokens = _extract_tokens_from_response(response)

        assert tokens["input_tokens"] == 0
        assert tokens["output_tokens"] == 0
        assert tokens["cached_tokens"] == 0

    def test_extract_tokens_none_values(self) -> None:
        """Test token extraction when values are None."""
        from tokenledger.interceptors.google import _extract_tokens_from_response

        response = MagicMock()
        response.usage_metadata = MagicMock()
        response.usage_metadata.prompt_token_count = None
        response.usage_metadata.candidates_token_count = None
        response.usage_metadata.cached_content_token_count = None

        tokens = _extract_tokens_from_response(response)

        assert tokens["input_tokens"] == 0
        assert tokens["output_tokens"] == 0
        assert tokens["cached_tokens"] == 0


# =============================================================================
# Tests for Request/Response Preview
# =============================================================================


class TestGetRequestPreview:
    """Tests for the _get_request_preview function."""

    def test_string_content(self) -> None:
        """Test preview extraction from string content."""
        from tokenledger.interceptors.google import _get_request_preview

        result = _get_request_preview("Hello, world!")
        assert result == "Hello, world!"

    def test_string_content_truncation(self) -> None:
        """Test that long string content is truncated."""
        from tokenledger.interceptors.google import _get_request_preview

        long_content = "A" * 600
        result = _get_request_preview(long_content, max_length=500)
        assert len(result) == 500

    def test_list_of_strings(self) -> None:
        """Test preview extraction from list of strings."""
        from tokenledger.interceptors.google import _get_request_preview

        contents = ["First message", "Second message", "Last message"]
        result = _get_request_preview(contents)
        assert result == "Last message"

    def test_content_with_text_attribute(self) -> None:
        """Test preview extraction from content with text attribute."""
        from tokenledger.interceptors.google import _get_request_preview

        content_obj = MagicMock()
        content_obj.text = "Content text"
        contents = [content_obj]

        result = _get_request_preview(contents)
        assert result == "Content text"

    def test_content_with_parts(self) -> None:
        """Test preview extraction from content with parts."""
        from tokenledger.interceptors.google import _get_request_preview

        part = MagicMock()
        part.text = "Part text"
        content_obj = MagicMock(spec=["parts"])
        content_obj.parts = [part]
        contents = [content_obj]

        result = _get_request_preview(contents)
        assert result == "Part text"

    def test_empty_content(self) -> None:
        """Test preview extraction from empty content."""
        from tokenledger.interceptors.google import _get_request_preview

        assert _get_request_preview(None) is None
        assert _get_request_preview([]) is None


class TestGetResponsePreview:
    """Tests for the _get_response_preview function."""

    def test_response_from_candidates(self) -> None:
        """Test preview extraction from candidates."""
        from tokenledger.interceptors.google import _get_response_preview

        response = MockGenerateContentResponse(text="Response text")
        result = _get_response_preview(response)
        assert result == "Response text"

    def test_response_text_fallback(self) -> None:
        """Test fallback to text property."""
        from tokenledger.interceptors.google import _get_response_preview

        response = MagicMock(spec=["text", "candidates"])
        response.candidates = []
        response.text = "Fallback text"

        result = _get_response_preview(response)
        assert result == "Fallback text"

    def test_response_preview_truncation(self) -> None:
        """Test that long responses are truncated."""
        from tokenledger.interceptors.google import _get_response_preview

        long_text = "B" * 600
        response = MockGenerateContentResponse(text=long_text)

        result = _get_response_preview(response, max_length=500)
        assert len(result) == 500

    def test_response_empty_candidates(self) -> None:
        """Test preview extraction with empty candidates."""
        from tokenledger.interceptors.google import _get_response_preview

        response = MagicMock(spec=["candidates"])
        response.candidates = []

        result = _get_response_preview(response)
        assert result is None


# =============================================================================
# Tests for Sync generate_content Wrapper
# =============================================================================


class TestWrapGenerateContent:
    """Tests for the sync generate_content wrapper."""

    def test_basic_tracking(self) -> None:
        """Test that generate_content calls are tracked."""
        from tokenledger.interceptors.google import _wrap_generate_content
        from tokenledger.models import LLMEvent

        response = MockGenerateContentResponse(
            text="Hello!",
            usage_metadata=MockUsageMetadata(
                prompt_tokens=100,
                candidates_tokens=50,
                cached_tokens=10,
            ),
        )

        def mock_generate_content(self, *, model: str, contents: Any, **kwargs):
            return response

        tracked_event = None

        def capture_track(event: LLMEvent) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.google.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_track
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_generate_content(mock_generate_content)
            result = wrapped(
                MagicMock(),
                model="gemini-2.0-flash",
                contents="Hello",
            )

            assert result is response
            assert tracked_event is not None
            assert tracked_event.provider == "google"
            assert tracked_event.model == "gemini-2.0-flash"
            assert tracked_event.input_tokens == 100
            assert tracked_event.output_tokens == 50
            assert tracked_event.cached_tokens == 10
            assert tracked_event.total_tokens == 150
            assert tracked_event.status == "success"
            assert tracked_event.request_type == "chat"
            assert tracked_event.response_preview == "Hello!"

    def test_error_tracking(self) -> None:
        """Test that errors are tracked correctly."""
        from tokenledger.interceptors.google import _wrap_generate_content
        from tokenledger.models import LLMEvent

        def mock_generate_content_error(self, *, model: str, contents: Any, **kwargs):
            raise ValueError("API Error")

        tracked_event = None

        def capture_track(event: LLMEvent) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.google.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_track
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_generate_content(mock_generate_content_error)

            with pytest.raises(ValueError, match="API Error"):
                wrapped(MagicMock(), model="gemini-2.0-flash", contents="Hello")

            assert tracked_event is not None
            assert tracked_event.status == "error"
            assert tracked_event.error_type == "ValueError"
            assert "API Error" in tracked_event.error_message

    def test_cost_calculation(self) -> None:
        """Test that cost is calculated correctly."""
        from tokenledger.interceptors.google import _wrap_generate_content
        from tokenledger.models import LLMEvent

        response = MockGenerateContentResponse(
            usage_metadata=MockUsageMetadata(
                prompt_tokens=1000,
                candidates_tokens=500,
                cached_tokens=0,
            ),
        )

        def mock_generate_content(self, *, model: str, contents: Any, **kwargs):
            return response

        tracked_event = None

        def capture_track(event: LLMEvent) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.google.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_track
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_generate_content(mock_generate_content)
            wrapped(MagicMock(), model="gemini-2.0-flash", contents="Hello")

            assert tracked_event is not None
            assert tracked_event.cost_usd is not None
            # gemini-2.0-flash: $0.10/1M input, $0.40/1M output
            expected_cost = (1000 / 1_000_000) * 0.10 + (500 / 1_000_000) * 0.40
            assert abs(tracked_event.cost_usd - expected_cost) < 0.0001

    def test_duration_tracking(self) -> None:
        """Test that duration is tracked."""
        from tokenledger.interceptors.google import _wrap_generate_content
        from tokenledger.models import LLMEvent

        response = MockGenerateContentResponse()

        def mock_generate_content(self, *, model: str, contents: Any, **kwargs):
            return response

        tracked_event = None

        def capture_track(event: LLMEvent) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.google.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_track
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_generate_content(mock_generate_content)
            wrapped(MagicMock(), model="gemini-2.0-flash", contents="Hello")

            assert tracked_event is not None
            assert tracked_event.duration_ms is not None
            assert tracked_event.duration_ms >= 0


# =============================================================================
# Tests for Async generate_content Wrapper
# =============================================================================


class TestWrapAsyncGenerateContent:
    """Tests for the async generate_content wrapper."""

    def test_async_basic_tracking(self) -> None:
        """Test that async generate_content calls are tracked."""
        from tokenledger.interceptors.google import _wrap_async_generate_content
        from tokenledger.models import LLMEvent

        async def run_test():
            response = MockGenerateContentResponse(
                text="Async response!",
                usage_metadata=MockUsageMetadata(
                    prompt_tokens=200,
                    candidates_tokens=100,
                    cached_tokens=0,
                ),
            )

            async def mock_async_generate_content(self, *, model: str, contents: Any, **kwargs):
                return response

            tracked_event = None

            def capture_track(event: LLMEvent) -> None:
                nonlocal tracked_event
                tracked_event = event

            with patch("tokenledger.interceptors.google.get_tracker") as mock_get_tracker:
                mock_tracker = MagicMock()
                mock_tracker.track = capture_track
                mock_get_tracker.return_value = mock_tracker

                wrapped = _wrap_async_generate_content(mock_async_generate_content)
                result = await wrapped(
                    MagicMock(),
                    model="gemini-2.0-flash",
                    contents="Hello async",
                )

                assert result is response
                assert tracked_event is not None
                assert tracked_event.provider == "google"
                assert tracked_event.model == "gemini-2.0-flash"
                assert tracked_event.input_tokens == 200
                assert tracked_event.output_tokens == 100
                assert tracked_event.status == "success"
                assert tracked_event.response_preview == "Async response!"

        asyncio.run(run_test())

    def test_async_error_tracking(self) -> None:
        """Test that async errors are tracked correctly."""
        from tokenledger.interceptors.google import _wrap_async_generate_content
        from tokenledger.models import LLMEvent

        async def run_test():
            async def mock_async_generate_content_error(
                self, *, model: str, contents: Any, **kwargs
            ):
                raise RuntimeError("Async API Error")

            tracked_event = None

            def capture_track(event: LLMEvent) -> None:
                nonlocal tracked_event
                tracked_event = event

            with patch("tokenledger.interceptors.google.get_tracker") as mock_get_tracker:
                mock_tracker = MagicMock()
                mock_tracker.track = capture_track
                mock_get_tracker.return_value = mock_tracker

                wrapped = _wrap_async_generate_content(mock_async_generate_content_error)

                with pytest.raises(RuntimeError, match="Async API Error"):
                    await wrapped(MagicMock(), model="gemini-2.0-flash", contents="Hello")

                assert tracked_event is not None
                assert tracked_event.status == "error"
                assert tracked_event.error_type == "RuntimeError"
                assert "Async API Error" in tracked_event.error_message

        asyncio.run(run_test())


# =============================================================================
# Tests for Streaming
# =============================================================================


class TestSyncStreamIterator:
    """Tests for the sync streaming wrapper (TrackedStreamIterator)."""

    def test_sync_stream_basic(self) -> None:
        """Test basic sync streaming with content accumulation."""
        from tokenledger.interceptors.google import _wrap_generate_content_stream
        from tokenledger.models import LLMEvent

        chunks = [
            MockGenerateContentResponse(
                text="Hello",
                usage_metadata=MockUsageMetadata(
                    prompt_tokens=0, candidates_tokens=0, cached_tokens=0
                ),
            ),
            MockGenerateContentResponse(
                text=" world",
                usage_metadata=MockUsageMetadata(
                    prompt_tokens=0, candidates_tokens=0, cached_tokens=0
                ),
            ),
            MockGenerateContentResponse(
                text="!",
                usage_metadata=MockUsageMetadata(
                    prompt_tokens=100, candidates_tokens=50, cached_tokens=10
                ),
            ),
        ]

        def mock_stream(self, *, model: str, contents: Any, **kwargs):
            return MockSyncStreamIterator(chunks)

        tracked_event = None

        def capture_track(event: LLMEvent) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.google.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_track
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_generate_content_stream(mock_stream)
            stream = wrapped(MagicMock(), model="gemini-2.0-flash", contents="Hello")

            # Consume the stream
            collected = []
            for chunk in stream:
                collected.append(chunk)

            assert len(collected) == 3
            assert tracked_event is not None
            assert tracked_event.status == "success"
            assert tracked_event.input_tokens == 100
            assert tracked_event.output_tokens == 50
            assert tracked_event.cached_tokens == 10
            assert tracked_event.request_type == "chat_stream"
            assert "Hello world!" in tracked_event.response_preview

    def test_sync_stream_error_handling(self) -> None:
        """Test that sync stream errors are tracked."""
        from tokenledger.interceptors.google import _wrap_generate_content_stream
        from tokenledger.models import LLMEvent

        class ErrorStream:
            def __iter__(self):
                return self

            def __next__(self):
                raise ValueError("Stream error")

        def mock_stream(self, *, model: str, contents: Any, **kwargs):
            return ErrorStream()

        tracked_event = None

        def capture_track(event: LLMEvent) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.google.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_track
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_generate_content_stream(mock_stream)
            stream = wrapped(MagicMock(), model="gemini-2.0-flash", contents="Hello")

            with pytest.raises(ValueError, match="Stream error"):
                list(stream)

            assert tracked_event is not None
            assert tracked_event.status == "error"
            assert tracked_event.error_type == "ValueError"
            assert "Stream error" in tracked_event.error_message


class TestAsyncStreamIterator:
    """Tests for the async streaming wrapper (AsyncTrackedStreamIterator)."""

    def test_async_stream_basic(self) -> None:
        """Test basic async streaming with content accumulation."""
        from tokenledger.interceptors.google import _wrap_async_generate_content_stream
        from tokenledger.models import LLMEvent

        async def run_test():
            chunks = [
                MockGenerateContentResponse(
                    text="Async",
                    usage_metadata=MockUsageMetadata(
                        prompt_tokens=0, candidates_tokens=0, cached_tokens=0
                    ),
                ),
                MockGenerateContentResponse(
                    text=" stream!",
                    usage_metadata=MockUsageMetadata(
                        prompt_tokens=200, candidates_tokens=100, cached_tokens=0
                    ),
                ),
            ]

            async def mock_async_stream(self, *, model: str, contents: Any, **kwargs):
                return MockAsyncStreamIterator(chunks)

            tracked_event = None

            def capture_track(event: LLMEvent) -> None:
                nonlocal tracked_event
                tracked_event = event

            with patch("tokenledger.interceptors.google.get_tracker") as mock_get_tracker:
                mock_tracker = MagicMock()
                mock_tracker.track = capture_track
                mock_get_tracker.return_value = mock_tracker

                wrapped = _wrap_async_generate_content_stream(mock_async_stream)
                stream = await wrapped(MagicMock(), model="gemini-2.0-flash", contents="Hello")

                # Consume the stream
                collected = []
                async for chunk in stream:
                    collected.append(chunk)

                assert len(collected) == 2
                assert tracked_event is not None
                assert tracked_event.status == "success"
                assert tracked_event.input_tokens == 200
                assert tracked_event.output_tokens == 100
                assert tracked_event.request_type == "chat_stream"

        asyncio.run(run_test())

    def test_async_stream_error_handling(self) -> None:
        """Test that async stream errors are tracked."""
        from tokenledger.interceptors.google import _wrap_async_generate_content_stream
        from tokenledger.models import LLMEvent

        async def run_test():
            class AsyncErrorStream:
                def __aiter__(self):
                    return self

                async def __anext__(self):
                    raise RuntimeError("Async stream error")

            async def mock_async_stream(self, *, model: str, contents: Any, **kwargs):
                return AsyncErrorStream()

            tracked_event = None

            def capture_track(event: LLMEvent) -> None:
                nonlocal tracked_event
                tracked_event = event

            with patch("tokenledger.interceptors.google.get_tracker") as mock_get_tracker:
                mock_tracker = MagicMock()
                mock_tracker.track = capture_track
                mock_get_tracker.return_value = mock_tracker

                wrapped = _wrap_async_generate_content_stream(mock_async_stream)
                stream = await wrapped(MagicMock(), model="gemini-2.0-flash", contents="Hello")

                with pytest.raises(RuntimeError, match="Async stream error"):
                    async for _ in stream:
                        pass

                assert tracked_event is not None
                assert tracked_event.status == "error"
                assert tracked_event.error_type == "RuntimeError"
                assert "Async stream error" in tracked_event.error_message

        asyncio.run(run_test())


# =============================================================================
# Tests for Embedding API
# =============================================================================


class TestExtractEmbeddingTokens:
    """Tests for the _extract_embedding_tokens function."""

    def test_extract_embedding_tokens_basic(self) -> None:
        """Test basic embedding token extraction."""
        from tokenledger.interceptors.google import _extract_embedding_tokens

        response = MockEmbedContentResponse(usage_metadata=MockUsageMetadata(prompt_tokens=75))

        tokens = _extract_embedding_tokens(response)
        assert tokens == 75

    def test_extract_embedding_tokens_missing(self) -> None:
        """Test extraction when usage_metadata is missing."""
        from tokenledger.interceptors.google import _extract_embedding_tokens

        response = MagicMock(spec=[])
        tokens = _extract_embedding_tokens(response)
        assert tokens == 0


class TestGetEmbeddingContentPreview:
    """Tests for the _get_embedding_content_preview function."""

    def test_string_content(self) -> None:
        """Test preview extraction from string content."""
        from tokenledger.interceptors.google import _get_embedding_content_preview

        result = _get_embedding_content_preview("Embed this text")
        assert result == "Embed this text"

    def test_list_content(self) -> None:
        """Test preview extraction from list content."""
        from tokenledger.interceptors.google import _get_embedding_content_preview

        result = _get_embedding_content_preview(["First text", "Second text"])
        assert result == "First text"

    def test_empty_content(self) -> None:
        """Test preview extraction from empty content."""
        from tokenledger.interceptors.google import _get_embedding_content_preview

        assert _get_embedding_content_preview(None) is None
        assert _get_embedding_content_preview([]) is None


class TestWrapEmbedContent:
    """Tests for the embed_content wrapper."""

    def test_embed_content_tracking(self) -> None:
        """Test that embed_content calls are tracked."""
        from tokenledger.interceptors.google import _wrap_embed_content
        from tokenledger.models import LLMEvent

        response = MockEmbedContentResponse(usage_metadata=MockUsageMetadata(prompt_tokens=100))

        def mock_embed_content(self, *, model: str, contents: Any, **kwargs):
            return response

        tracked_event = None

        def capture_track(event: LLMEvent) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.google.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_track
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_embed_content(mock_embed_content)
            result = wrapped(
                MagicMock(),
                model="text-embedding-004",
                contents="Embed this",
            )

            assert result is response
            assert tracked_event is not None
            assert tracked_event.provider == "google"
            assert tracked_event.model == "text-embedding-004"
            assert tracked_event.request_type == "embedding"
            assert tracked_event.input_tokens == 100
            assert tracked_event.total_tokens == 100
            assert tracked_event.status == "success"

    def test_embed_content_error_tracking(self) -> None:
        """Test that embed_content errors are tracked."""
        from tokenledger.interceptors.google import _wrap_embed_content
        from tokenledger.models import LLMEvent

        def mock_embed_content_error(self, *, model: str, contents: Any, **kwargs):
            raise RuntimeError("Embedding error")

        tracked_event = None

        def capture_track(event: LLMEvent) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.google.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_track
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_embed_content(mock_embed_content_error)

            with pytest.raises(RuntimeError, match="Embedding error"):
                wrapped(MagicMock(), model="text-embedding-004", contents="Embed this")

            assert tracked_event is not None
            assert tracked_event.status == "error"
            assert tracked_event.error_type == "RuntimeError"


class TestWrapAsyncEmbedContent:
    """Tests for the async embed_content wrapper."""

    def test_async_embed_content_tracking(self) -> None:
        """Test that async embed_content calls are tracked."""
        from tokenledger.interceptors.google import _wrap_async_embed_content
        from tokenledger.models import LLMEvent

        async def run_test():
            response = MockEmbedContentResponse(usage_metadata=MockUsageMetadata(prompt_tokens=150))

            async def mock_async_embed_content(self, *, model: str, contents: Any, **kwargs):
                return response

            tracked_event = None

            def capture_track(event: LLMEvent) -> None:
                nonlocal tracked_event
                tracked_event = event

            with patch("tokenledger.interceptors.google.get_tracker") as mock_get_tracker:
                mock_tracker = MagicMock()
                mock_tracker.track = capture_track
                mock_get_tracker.return_value = mock_tracker

                wrapped = _wrap_async_embed_content(mock_async_embed_content)
                result = await wrapped(
                    MagicMock(),
                    model="text-embedding-004",
                    contents="Async embed",
                )

                assert result is response
                assert tracked_event is not None
                assert tracked_event.provider == "google"
                assert tracked_event.model == "text-embedding-004"
                assert tracked_event.request_type == "embedding"
                assert tracked_event.input_tokens == 150
                assert tracked_event.status == "success"

        asyncio.run(run_test())


# =============================================================================
# Tests for Attribution Context
# =============================================================================


class TestAttributionContext:
    """Tests for attribution context integration."""

    def test_attribution_context_applied(self) -> None:
        """Test that attribution context is applied to events."""
        from tokenledger.context import attribution
        from tokenledger.interceptors.google import _wrap_generate_content
        from tokenledger.models import LLMEvent

        response = MockGenerateContentResponse()

        def mock_generate_content(self, *, model: str, contents: Any, **kwargs):
            return response

        tracked_event = None

        def capture_track(event: LLMEvent) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.google.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_track
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_generate_content(mock_generate_content)

            with attribution(
                user_id="test_user",
                feature="gemini_test",
                team="ml-team",
                cost_center="CC-001",
            ):
                wrapped(MagicMock(), model="gemini-2.0-flash", contents="Hello")

            assert tracked_event is not None
            assert tracked_event.user_id == "test_user"
            assert tracked_event.feature == "gemini_test"
            assert tracked_event.team == "ml-team"
            assert tracked_event.cost_center == "CC-001"

    def test_streaming_with_attribution_context(self) -> None:
        """Test that attribution context is applied to streaming events."""
        from tokenledger.context import attribution
        from tokenledger.interceptors.google import _wrap_generate_content_stream
        from tokenledger.models import LLMEvent

        chunks = [
            MockGenerateContentResponse(
                text="Test",
                usage_metadata=MockUsageMetadata(
                    prompt_tokens=100, candidates_tokens=50, cached_tokens=0
                ),
            ),
        ]

        def mock_stream(self, *, model: str, contents: Any, **kwargs):
            return MockSyncStreamIterator(chunks)

        tracked_event = None

        def capture_track(event: LLMEvent) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.google.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_track
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_generate_content_stream(mock_stream)

            with attribution(user_id="stream_user", feature="streaming_test"):
                stream = wrapped(MagicMock(), model="gemini-2.0-flash", contents="Hello")
                list(stream)  # Consume the stream

            assert tracked_event is not None
            assert tracked_event.user_id == "stream_user"
            assert tracked_event.feature == "streaming_test"


# =============================================================================
# Tests for Different Model Names
# =============================================================================


class TestDifferentModels:
    """Tests for different Google model names and cost calculation."""

    @pytest.mark.parametrize(
        "model_name,expected_input_price,expected_output_price",
        [
            ("gemini-2.0-flash", 0.10, 0.40),
            ("gemini-2.5-pro", 1.25, 10.00),
            ("gemini-2.5-flash", 0.30, 2.50),
            ("text-embedding-004", 0.01, 0.0),
        ],
    )
    def test_model_cost_calculation(
        self, model_name: str, expected_input_price: float, expected_output_price: float
    ) -> None:
        """Test cost calculation for different Google models."""
        from tokenledger.interceptors.google import _wrap_generate_content
        from tokenledger.models import LLMEvent

        response = MockGenerateContentResponse(
            usage_metadata=MockUsageMetadata(
                prompt_tokens=1_000_000,  # 1M tokens
                candidates_tokens=1_000_000,  # 1M tokens
                cached_tokens=0,
            ),
        )

        def mock_generate_content(self, *, model: str, contents: Any, **kwargs):
            return response

        tracked_event = None

        def capture_track(event: LLMEvent) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.google.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_track
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_generate_content(mock_generate_content)
            wrapped(MagicMock(), model=model_name, contents="Hello")

            assert tracked_event is not None
            expected_cost = expected_input_price + expected_output_price
            assert tracked_event.cost_usd is not None
            assert abs(tracked_event.cost_usd - expected_cost) < 0.01


# =============================================================================
# Tests for Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_empty_response(self) -> None:
        """Test handling of empty response."""
        from tokenledger.interceptors.google import _wrap_generate_content
        from tokenledger.models import LLMEvent

        response = MagicMock(spec=["usage_metadata", "candidates", "text"])
        response.usage_metadata = None
        response.candidates = []
        response.text = None

        def mock_generate_content(self, *, model: str, contents: Any, **kwargs):
            return response

        tracked_event = None

        def capture_track(event: LLMEvent) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.google.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_track
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_generate_content(mock_generate_content)
            result = wrapped(MagicMock(), model="gemini-2.0-flash", contents="Hello")

            assert result is response
            assert tracked_event is not None
            assert tracked_event.input_tokens == 0
            assert tracked_event.output_tokens == 0
            assert tracked_event.status == "success"

    def test_unknown_model_no_cost(self) -> None:
        """Test that unknown models result in None cost."""
        from tokenledger.interceptors.google import _wrap_generate_content
        from tokenledger.models import LLMEvent

        response = MockGenerateContentResponse(
            usage_metadata=MockUsageMetadata(
                prompt_tokens=100, candidates_tokens=50, cached_tokens=0
            ),
        )

        def mock_generate_content(self, *, model: str, contents: Any, **kwargs):
            return response

        tracked_event = None

        def capture_track(event: LLMEvent) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.google.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_track
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_generate_content(mock_generate_content)
            wrapped(MagicMock(), model="unknown-model-xyz", contents="Hello")

            assert tracked_event is not None
            # Unknown model should have None cost
            assert tracked_event.cost_usd is None

    def test_long_error_message_truncation(self) -> None:
        """Test that long error messages are truncated to 1000 chars."""
        from tokenledger.interceptors.google import _wrap_generate_content
        from tokenledger.models import LLMEvent

        long_error = "E" * 2000

        def mock_generate_content_error(self, *, model: str, contents: Any, **kwargs):
            raise ValueError(long_error)

        tracked_event = None

        def capture_track(event: LLMEvent) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.google.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_track
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_generate_content(mock_generate_content_error)

            with pytest.raises(ValueError):
                wrapped(MagicMock(), model="gemini-2.0-flash", contents="Hello")

            assert tracked_event is not None
            assert len(tracked_event.error_message) == 1000

    def test_stream_not_tracked_twice(self) -> None:
        """Test that stream events are not tracked multiple times."""
        from tokenledger.interceptors.google import _wrap_generate_content_stream
        from tokenledger.models import LLMEvent

        chunks = [
            MockGenerateContentResponse(
                text="Test",
                usage_metadata=MockUsageMetadata(
                    prompt_tokens=100, candidates_tokens=50, cached_tokens=0
                ),
            ),
        ]

        def mock_stream(self, *, model: str, contents: Any, **kwargs):
            return MockSyncStreamIterator(chunks)

        track_count = 0

        def capture_track(event: LLMEvent) -> None:
            nonlocal track_count
            track_count += 1

        with patch("tokenledger.interceptors.google.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_track
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_generate_content_stream(mock_stream)
            stream = wrapped(MagicMock(), model="gemini-2.0-flash", contents="Hello")

            # Consume the stream multiple times (if possible)
            list(stream)

            # Should only track once
            assert track_count == 1

    def test_response_preview_truncation(self) -> None:
        """Test that response preview is truncated to 500 chars."""
        from tokenledger.interceptors.google import _wrap_generate_content_stream
        from tokenledger.models import LLMEvent

        long_text = "A" * 600
        chunks = [
            MockGenerateContentResponse(
                text=long_text,
                usage_metadata=MockUsageMetadata(
                    prompt_tokens=100, candidates_tokens=50, cached_tokens=0
                ),
            ),
        ]

        def mock_stream(self, *, model: str, contents: Any, **kwargs):
            return MockSyncStreamIterator(chunks)

        tracked_event = None

        def capture_track(event: LLMEvent) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.google.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_track
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_generate_content_stream(mock_stream)
            stream = wrapped(MagicMock(), model="gemini-2.0-flash", contents="Hello")
            list(stream)

            assert tracked_event is not None
            assert len(tracked_event.response_preview) == 500
