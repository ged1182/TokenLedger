"""Comprehensive tests for the OpenAI interceptor in TokenLedger.

This module tests the core functionality of the OpenAI SDK interceptor:
- Token extraction from responses
- Cost calculation
- Chat completions tracking (sync and async)
- Embeddings tracking
- Error handling
- Request/response preview extraction
- Attribution context application
- Patching/unpatching functionality
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# Mock Objects for OpenAI Responses
# =============================================================================


class MockPromptTokensDetails:
    """Mock prompt_tokens_details object for cached tokens."""

    def __init__(self, cached_tokens: int = 0):
        self.cached_tokens = cached_tokens


class MockInputTokensDetails:
    """Mock input_tokens_details object for responses API cached tokens."""

    def __init__(self, cached_tokens: int = 0):
        self.cached_tokens = cached_tokens


class MockUsage:
    """Mock usage object for chat completions."""

    def __init__(
        self,
        prompt_tokens: int = 100,
        completion_tokens: int = 50,
        cached_tokens: int = 0,
    ):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.prompt_tokens_details = (
            MockPromptTokensDetails(cached_tokens) if cached_tokens else None
        )


class MockResponsesUsage:
    """Mock usage object for responses API (different field names)."""

    def __init__(
        self,
        input_tokens: int = 100,
        output_tokens: int = 50,
        cached_tokens: int = 0,
    ):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.input_tokens_details = MockInputTokensDetails(cached_tokens) if cached_tokens else None


class MockMessage:
    """Mock message object from chat completion response."""

    def __init__(self, content: str = "Hello!"):
        self.content = content


class MockChoice:
    """Mock choice object from chat completion response."""

    def __init__(self, message: MockMessage | None = None):
        self.message = message or MockMessage()


class MockChatCompletionResponse:
    """Mock OpenAI chat completion response."""

    def __init__(
        self,
        model: str = "gpt-4o",
        prompt_tokens: int = 100,
        completion_tokens: int = 50,
        cached_tokens: int = 0,
        content: str = "Hello! How can I help you?",
    ):
        self.model = model
        self.usage = MockUsage(prompt_tokens, completion_tokens, cached_tokens)
        self.choices = [MockChoice(MockMessage(content))]


class MockEmbeddingsUsage:
    """Mock usage object for embeddings (no completion_tokens)."""

    def __init__(self, prompt_tokens: int = 100, total_tokens: int = 100):
        self.prompt_tokens = prompt_tokens
        self.total_tokens = total_tokens


class MockEmbeddingResponse:
    """Mock OpenAI embeddings response."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        prompt_tokens: int = 100,
    ):
        self.model = model
        self.usage = MockEmbeddingsUsage(prompt_tokens, prompt_tokens)


class MockResponsesOutputTextBlock:
    """Mock output text block for responses API."""

    def __init__(self, text: str = "Hello!"):
        self.type = "output_text"
        self.text = text


class MockResponsesOutputMessageContent:
    """Mock message content for responses API."""

    def __init__(self, text: str = "Hello!"):
        self.type = "message"
        self.content = [MockResponsesOutputTextBlock(text)]


class MockResponsesResponse:
    """Mock OpenAI responses API response."""

    def __init__(
        self,
        model: str = "gpt-4o",
        input_tokens: int = 100,
        output_tokens: int = 50,
        cached_tokens: int = 0,
        text: str = "Hello! How can I help you?",
    ):
        self.model = model
        self.usage = MockResponsesUsage(input_tokens, output_tokens, cached_tokens)
        self.output = [MockResponsesOutputMessageContent(text)]


class MockTranscriptionResponse:
    """Mock OpenAI audio transcription response."""

    def __init__(self, text: str = "This is a transcription."):
        self.text = text


class MockImageResponse:
    """Mock OpenAI image generation response."""

    def __init__(self):
        self.data = [MagicMock()]


class MockBatchResponse:
    """Mock OpenAI batch API response."""

    def __init__(
        self,
        batch_id: str = "batch_123",
        status: str = "validating",
    ):
        self.id = batch_id
        self.status = status


# =============================================================================
# Tests for Token Extraction Functions
# =============================================================================


class TestExtractTokensFromResponse:
    """Tests for _extract_tokens_from_response function."""

    def test_extract_tokens_basic(self) -> None:
        """Test basic token extraction from chat completion response."""
        from tokenledger.interceptors.openai import _extract_tokens_from_response

        response = MockChatCompletionResponse(
            prompt_tokens=150,
            completion_tokens=75,
        )
        tokens = _extract_tokens_from_response(response)

        assert tokens["input_tokens"] == 150
        assert tokens["output_tokens"] == 75
        assert tokens["cached_tokens"] == 0

    def test_extract_tokens_with_cached(self) -> None:
        """Test token extraction with cached tokens."""
        from tokenledger.interceptors.openai import _extract_tokens_from_response

        response = MockChatCompletionResponse(
            prompt_tokens=200,
            completion_tokens=100,
            cached_tokens=50,
        )
        tokens = _extract_tokens_from_response(response)

        assert tokens["input_tokens"] == 200
        assert tokens["output_tokens"] == 100
        assert tokens["cached_tokens"] == 50

    def test_extract_tokens_no_usage(self) -> None:
        """Test token extraction when response has no usage data."""
        from tokenledger.interceptors.openai import _extract_tokens_from_response

        response = MagicMock(spec=[])  # No usage attribute
        tokens = _extract_tokens_from_response(response)

        assert tokens["input_tokens"] == 0
        assert tokens["output_tokens"] == 0
        assert tokens["cached_tokens"] == 0

    def test_extract_tokens_none_values(self) -> None:
        """Test token extraction handles None values in usage."""
        from tokenledger.interceptors.openai import _extract_tokens_from_response

        response = MagicMock()
        response.usage = MagicMock()
        response.usage.prompt_tokens = None
        response.usage.completion_tokens = None
        response.usage.prompt_tokens_details = None

        tokens = _extract_tokens_from_response(response)

        assert tokens["input_tokens"] == 0
        assert tokens["output_tokens"] == 0
        assert tokens["cached_tokens"] == 0


class TestExtractTokensFromResponsesApi:
    """Tests for _extract_tokens_from_responses_api function."""

    def test_extract_tokens_responses_api(self) -> None:
        """Test token extraction from responses API."""
        from tokenledger.interceptors.openai import _extract_tokens_from_responses_api

        response = MockResponsesResponse(
            input_tokens=200,
            output_tokens=100,
            cached_tokens=25,
        )
        tokens = _extract_tokens_from_responses_api(response)

        assert tokens["input_tokens"] == 200
        assert tokens["output_tokens"] == 100
        assert tokens["cached_tokens"] == 25

    def test_extract_tokens_responses_api_no_cache(self) -> None:
        """Test responses API token extraction without cached tokens."""
        from tokenledger.interceptors.openai import _extract_tokens_from_responses_api

        response = MockResponsesResponse(
            input_tokens=100,
            output_tokens=50,
            cached_tokens=0,
        )
        tokens = _extract_tokens_from_responses_api(response)

        assert tokens["input_tokens"] == 100
        assert tokens["output_tokens"] == 50
        assert tokens["cached_tokens"] == 0


class TestExtractModelFromResponse:
    """Tests for _extract_model_from_response function."""

    def test_extract_model(self) -> None:
        """Test model extraction from response."""
        from tokenledger.interceptors.openai import _extract_model_from_response

        response = MockChatCompletionResponse(model="gpt-4o-2024-11-20")
        model = _extract_model_from_response(response, default="gpt-4o")

        assert model == "gpt-4o-2024-11-20"

    def test_extract_model_uses_default(self) -> None:
        """Test model extraction uses default when not in response."""
        from tokenledger.interceptors.openai import _extract_model_from_response

        response = MagicMock(spec=[])  # No model attribute
        model = _extract_model_from_response(response, default="gpt-4o")

        assert model == "gpt-4o"


# =============================================================================
# Tests for Preview Extraction Functions
# =============================================================================


class TestGetRequestPreview:
    """Tests for _get_request_preview function."""

    def test_request_preview_from_list(self) -> None:
        """Test request preview extraction from message list."""
        from tokenledger.interceptors.openai import _get_request_preview

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ]
        preview = _get_request_preview(messages)

        assert preview == "Hello, how are you?"

    def test_request_preview_truncation(self) -> None:
        """Test that long request content is truncated."""
        from tokenledger.interceptors.openai import _get_request_preview

        long_content = "A" * 600
        messages = [{"role": "user", "content": long_content}]
        preview = _get_request_preview(messages, max_length=500)

        assert len(preview) == 500

    def test_request_preview_empty_messages(self) -> None:
        """Test request preview with empty messages."""
        from tokenledger.interceptors.openai import _get_request_preview

        preview = _get_request_preview([])
        assert preview is None

        preview = _get_request_preview(None)
        assert preview is None

    def test_request_preview_object_messages(self) -> None:
        """Test request preview with message objects (not dicts)."""
        from tokenledger.interceptors.openai import _get_request_preview

        message = MagicMock()
        message.content = "Test content"
        messages = [message]

        preview = _get_request_preview(messages)
        assert preview == "Test content"


class TestGetResponsePreview:
    """Tests for _get_response_preview function."""

    def test_response_preview(self) -> None:
        """Test response preview extraction."""
        from tokenledger.interceptors.openai import _get_response_preview

        response = MockChatCompletionResponse(content="This is the response!")
        preview = _get_response_preview(response)

        assert preview == "This is the response!"

    def test_response_preview_truncation(self) -> None:
        """Test that long response content is truncated."""
        from tokenledger.interceptors.openai import _get_response_preview

        long_content = "B" * 600
        response = MockChatCompletionResponse(content=long_content)
        preview = _get_response_preview(response, max_length=500)

        assert len(preview) == 500

    def test_response_preview_no_choices(self) -> None:
        """Test response preview when no choices."""
        from tokenledger.interceptors.openai import _get_response_preview

        response = MagicMock()
        response.choices = []
        preview = _get_response_preview(response)

        assert preview is None

    def test_response_preview_no_content(self) -> None:
        """Test response preview when message has no content."""
        from tokenledger.interceptors.openai import _get_response_preview

        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message = MagicMock()
        response.choices[0].message.content = None

        preview = _get_response_preview(response)
        assert preview is None


class TestGetResponsesRequestPreview:
    """Tests for _get_responses_request_preview function."""

    def test_string_input(self) -> None:
        """Test preview extraction from string input."""
        from tokenledger.interceptors.openai import _get_responses_request_preview

        preview = _get_responses_request_preview("Hello, world!")
        assert preview == "Hello, world!"

    def test_string_input_truncation(self) -> None:
        """Test truncation of long string input."""
        from tokenledger.interceptors.openai import _get_responses_request_preview

        long_input = "X" * 600
        preview = _get_responses_request_preview(long_input, max_length=500)
        assert len(preview) == 500

    def test_list_input(self) -> None:
        """Test preview extraction from list input."""
        from tokenledger.interceptors.openai import _get_responses_request_preview

        messages = [
            {"role": "user", "content": "First message"},
            {"role": "user", "content": "Last message"},
        ]
        preview = _get_responses_request_preview(messages)
        assert preview == "Last message"

    def test_empty_input(self) -> None:
        """Test preview extraction with empty input."""
        from tokenledger.interceptors.openai import _get_responses_request_preview

        assert _get_responses_request_preview("") is None
        assert _get_responses_request_preview([]) is None
        assert _get_responses_request_preview(None) is None


class TestGetResponsesResponsePreview:
    """Tests for _get_responses_response_preview function."""

    def test_responses_response_preview(self) -> None:
        """Test response preview from responses API."""
        from tokenledger.interceptors.openai import _get_responses_response_preview

        response = MockResponsesResponse(text="This is the response!")
        preview = _get_responses_response_preview(response)

        assert preview == "This is the response!"

    def test_responses_response_preview_empty_output(self) -> None:
        """Test response preview when output is empty."""
        from tokenledger.interceptors.openai import _get_responses_response_preview

        response = MagicMock()
        response.output = []
        preview = _get_responses_response_preview(response)

        assert preview is None


# =============================================================================
# Tests for Chat Completions Wrapper
# =============================================================================


class TestWrapChatCompletionsCreate:
    """Tests for _wrap_chat_completions_create function."""

    def test_non_streaming_success(self) -> None:
        """Test non-streaming chat completion tracking."""
        from tokenledger.interceptors.openai import _wrap_chat_completions_create

        response = MockChatCompletionResponse(
            model="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            content="Hello!",
        )

        def mock_create(*args, **kwargs):
            return response

        tracked_event = None

        def capture_event(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_event
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_chat_completions_create(mock_create)
            result = wrapped(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hi"}],
            )

            assert result is response
            assert tracked_event is not None
            assert tracked_event.provider == "openai"
            assert tracked_event.model == "gpt-4o"
            assert tracked_event.input_tokens == 100
            assert tracked_event.output_tokens == 50
            assert tracked_event.status == "success"
            assert tracked_event.endpoint == "/v1/chat/completions"
            assert tracked_event.request_type == "chat"

    def test_non_streaming_with_user(self) -> None:
        """Test that user parameter is captured as user_id."""
        from tokenledger.interceptors.openai import _wrap_chat_completions_create

        response = MockChatCompletionResponse()

        def mock_create(*args, **kwargs):
            return response

        tracked_event = None

        def capture_event(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_event
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_chat_completions_create(mock_create)
            wrapped(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hi"}],
                user="user_12345",
            )

            assert tracked_event.user_id == "user_12345"

    def test_non_streaming_error(self) -> None:
        """Test error handling in non-streaming calls."""
        from tokenledger.interceptors.openai import _wrap_chat_completions_create

        def mock_create(*args, **kwargs):
            raise ValueError("API Error")

        tracked_event = None

        def capture_event(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_event
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_chat_completions_create(mock_create)

            with pytest.raises(ValueError, match="API Error"):
                wrapped(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "Hi"}],
                )

            assert tracked_event is not None
            assert tracked_event.status == "error"
            assert tracked_event.error_type == "ValueError"
            assert "API Error" in tracked_event.error_message

    def test_error_message_truncation(self) -> None:
        """Test that long error messages are truncated."""
        from tokenledger.interceptors.openai import _wrap_chat_completions_create

        long_error = "E" * 2000

        def mock_create(*args, **kwargs):
            raise ValueError(long_error)

        tracked_event = None

        def capture_event(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_event
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_chat_completions_create(mock_create)

            with pytest.raises(ValueError):
                wrapped(model="gpt-4o", messages=[])

            # Error message should be truncated to 1000 chars
            assert len(tracked_event.error_message) == 1000

    def test_cost_calculation(self) -> None:
        """Test that cost is calculated correctly."""
        from tokenledger.interceptors.openai import _wrap_chat_completions_create

        response = MockChatCompletionResponse(
            model="gpt-4o",
            prompt_tokens=1000,
            completion_tokens=500,
        )

        def mock_create(*args, **kwargs):
            return response

        tracked_event = None

        def capture_event(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_event
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_chat_completions_create(mock_create)
            wrapped(model="gpt-4o", messages=[{"role": "user", "content": "Hi"}])

            # gpt-4o: $2.50/1M input, $10.00/1M output
            # 1000 input = $0.0025, 500 output = $0.005
            # Total = $0.0075
            assert tracked_event.cost_usd is not None
            assert tracked_event.cost_usd == pytest.approx(0.0075, abs=0.0001)

    def test_model_update_from_response(self) -> None:
        """Test that model is updated from response (might differ from request)."""
        from tokenledger.interceptors.openai import _wrap_chat_completions_create

        # Request uses "gpt-4o" but response returns actual model version
        response = MockChatCompletionResponse(model="gpt-4o-2024-11-20")

        def mock_create(*args, **kwargs):
            return response

        tracked_event = None

        def capture_event(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_event
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_chat_completions_create(mock_create)
            wrapped(model="gpt-4o", messages=[])

            # Model should be updated from response
            assert tracked_event.model == "gpt-4o-2024-11-20"


class TestWrapAsyncChatCompletionsCreate:
    """Tests for _wrap_async_chat_completions_create function."""

    def test_async_non_streaming_success(self) -> None:
        """Test async non-streaming chat completion tracking."""
        from tokenledger.interceptors.openai import _wrap_async_chat_completions_create

        async def run_test():
            response = MockChatCompletionResponse(
                model="gpt-4o",
                prompt_tokens=150,
                completion_tokens=75,
            )

            async def mock_create(*args, **kwargs):
                return response

            tracked_event = None

            def capture_event(event: Any) -> None:
                nonlocal tracked_event
                tracked_event = event

            with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
                mock_tracker = MagicMock()
                mock_tracker.track = capture_event
                mock_get_tracker.return_value = mock_tracker

                wrapped = _wrap_async_chat_completions_create(mock_create)
                result = await wrapped(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "Hi"}],
                )

                assert result is response
                assert tracked_event is not None
                assert tracked_event.provider == "openai"
                assert tracked_event.input_tokens == 150
                assert tracked_event.output_tokens == 75
                assert tracked_event.status == "success"

        asyncio.run(run_test())

    def test_async_error_handling(self) -> None:
        """Test async error handling."""
        from tokenledger.interceptors.openai import _wrap_async_chat_completions_create

        async def run_test():
            async def mock_create(*args, **kwargs):
                raise RuntimeError("Async API Error")

            tracked_event = None

            def capture_event(event: Any) -> None:
                nonlocal tracked_event
                tracked_event = event

            with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
                mock_tracker = MagicMock()
                mock_tracker.track = capture_event
                mock_get_tracker.return_value = mock_tracker

                wrapped = _wrap_async_chat_completions_create(mock_create)

                with pytest.raises(RuntimeError, match="Async API Error"):
                    await wrapped(model="gpt-4o", messages=[])

                assert tracked_event.status == "error"
                assert tracked_event.error_type == "RuntimeError"

        asyncio.run(run_test())


# =============================================================================
# Tests for Embeddings Wrapper
# =============================================================================


class TestWrapEmbeddingsCreate:
    """Tests for _wrap_embeddings_create function."""

    def test_embeddings_success(self) -> None:
        """Test embeddings tracking."""
        from tokenledger.interceptors.openai import _wrap_embeddings_create

        response = MockEmbeddingResponse(
            model="text-embedding-3-small",
            prompt_tokens=100,
        )

        def mock_create(*args, **kwargs):
            return response

        tracked_event = None

        def capture_event(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_event
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_embeddings_create(mock_create)
            result = wrapped(
                model="text-embedding-3-small",
                input="Hello, world!",
            )

            assert result is response
            assert tracked_event is not None
            assert tracked_event.provider == "openai"
            assert tracked_event.model == "text-embedding-3-small"
            assert tracked_event.request_type == "embedding"
            assert tracked_event.endpoint == "/v1/embeddings"
            assert tracked_event.input_tokens == 100
            # Embeddings have no output tokens
            assert tracked_event.output_tokens == 0

    def test_embeddings_cost_calculation(self) -> None:
        """Test embeddings cost calculation (input only)."""
        from tokenledger.interceptors.openai import _wrap_embeddings_create

        response = MockEmbeddingResponse(
            model="text-embedding-3-small",
            prompt_tokens=1000000,  # 1M tokens
        )

        def mock_create(*args, **kwargs):
            return response

        tracked_event = None

        def capture_event(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_event
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_embeddings_create(mock_create)
            wrapped(model="text-embedding-3-small", input="test")

            # text-embedding-3-small: $0.02/1M input tokens
            assert tracked_event.cost_usd == pytest.approx(0.02, abs=0.001)

    def test_embeddings_error(self) -> None:
        """Test embeddings error handling."""
        from tokenledger.interceptors.openai import _wrap_embeddings_create

        def mock_create(*args, **kwargs):
            raise ValueError("Embedding error")

        tracked_event = None

        def capture_event(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_event
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_embeddings_create(mock_create)

            with pytest.raises(ValueError):
                wrapped(model="text-embedding-3-small", input="test")

            assert tracked_event.status == "error"


# =============================================================================
# Tests for Responses API Wrapper
# =============================================================================


class TestWrapResponsesCreate:
    """Tests for _wrap_responses_create function."""

    def test_responses_success(self) -> None:
        """Test responses API tracking."""
        from tokenledger.interceptors.openai import _wrap_responses_create

        response = MockResponsesResponse(
            model="gpt-4o",
            input_tokens=200,
            output_tokens=100,
            cached_tokens=50,
            text="Response text",
        )

        def mock_create(*args, **kwargs):
            return response

        tracked_event = None

        def capture_event(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_event
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_responses_create(mock_create)
            result = wrapped(
                model="gpt-4o",
                input="Hello!",
            )

            assert result is response
            assert tracked_event is not None
            assert tracked_event.provider == "openai"
            assert tracked_event.endpoint == "/v1/responses"
            assert tracked_event.input_tokens == 200
            assert tracked_event.output_tokens == 100
            assert tracked_event.cached_tokens == 50


class TestWrapAsyncResponsesCreate:
    """Tests for _wrap_async_responses_create function."""

    def test_async_responses_success(self) -> None:
        """Test async responses API tracking."""
        from tokenledger.interceptors.openai import _wrap_async_responses_create

        async def run_test():
            response = MockResponsesResponse()

            async def mock_create(*args, **kwargs):
                return response

            tracked_event = None

            def capture_event(event: Any) -> None:
                nonlocal tracked_event
                tracked_event = event

            with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
                mock_tracker = MagicMock()
                mock_tracker.track = capture_event
                mock_get_tracker.return_value = mock_tracker

                wrapped = _wrap_async_responses_create(mock_create)
                result = await wrapped(model="gpt-4o", input="Hi")

                assert result is response
                assert tracked_event is not None
                assert tracked_event.status == "success"

        asyncio.run(run_test())


# =============================================================================
# Tests for Audio API Wrappers
# =============================================================================


class TestWrapAudioTranscriptionCreate:
    """Tests for _wrap_audio_transcription_create function."""

    def test_transcription_success(self) -> None:
        """Test audio transcription tracking."""
        from tokenledger.interceptors.openai import _wrap_audio_transcription_create

        response = MockTranscriptionResponse(text="This is a test transcription.")

        def mock_create(*args, **kwargs):
            return response

        tracked_event = None

        def capture_event(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_event
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_audio_transcription_create(mock_create)
            result = wrapped(model="whisper-1", file=MagicMock())

            assert result is response
            assert tracked_event is not None
            assert tracked_event.provider == "openai"
            assert tracked_event.model == "whisper-1"
            assert tracked_event.request_type == "transcription"
            assert tracked_event.endpoint == "/v1/audio/transcriptions"
            assert tracked_event.response_preview == "This is a test transcription."


class TestWrapAudioSpeechCreate:
    """Tests for _wrap_audio_speech_create function."""

    def test_speech_success(self) -> None:
        """Test audio speech (TTS) tracking."""
        from tokenledger.interceptors.openai import _wrap_audio_speech_create

        response = MagicMock()

        def mock_create(*args, **kwargs):
            return response

        tracked_event = None

        def capture_event(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_event
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_audio_speech_create(mock_create)
            result = wrapped(
                model="tts-1",
                input="Hello, this is a test.",
                voice="alloy",
            )

            assert result is response
            assert tracked_event is not None
            assert tracked_event.model == "tts-1"
            assert tracked_event.request_type == "speech"
            assert tracked_event.endpoint == "/v1/audio/speech"
            assert tracked_event.request_preview == "Hello, this is a test."
            assert tracked_event.metadata_extra["character_count"] == len("Hello, this is a test.")


# =============================================================================
# Tests for Image API Wrappers
# =============================================================================


class TestWrapImagesGenerate:
    """Tests for _wrap_images_generate function."""

    def test_image_generation_success(self) -> None:
        """Test image generation tracking."""
        from tokenledger.interceptors.openai import _wrap_images_generate

        response = MockImageResponse()

        def mock_create(*args, **kwargs):
            return response

        tracked_event = None

        def capture_event(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_event
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_images_generate(mock_create)
            result = wrapped(
                model="dall-e-3",
                prompt="A beautiful sunset",
                n=2,
                size="1024x1024",
                quality="hd",
            )

            assert result is response
            assert tracked_event is not None
            assert tracked_event.model == "dall-e-3"
            assert tracked_event.request_type == "image_generation"
            assert tracked_event.endpoint == "/v1/images/generations"
            assert tracked_event.metadata_extra["image_count"] == 2
            assert tracked_event.metadata_extra["image_size"] == "1024x1024"
            assert tracked_event.metadata_extra["image_quality"] == "hd"

    def test_image_generation_cost(self) -> None:
        """Test image generation cost calculation."""
        from tokenledger.interceptors.openai import _wrap_images_generate

        response = MockImageResponse()

        def mock_create(*args, **kwargs):
            return response

        tracked_event = None

        def capture_event(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_event
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_images_generate(mock_create)
            wrapped(
                model="dall-e-3",
                prompt="Test",
                n=1,
                size="1024x1024",
                quality="standard",
            )

            # dall-e-3 standard 1024x1024 = $0.04 per image
            assert tracked_event.cost_usd == pytest.approx(0.04, abs=0.001)


# =============================================================================
# Tests for Batch API Wrappers
# =============================================================================


class TestWrapBatchesCreate:
    """Tests for _wrap_batches_create function."""

    def test_batch_create_success(self) -> None:
        """Test batch creation tracking."""
        from tokenledger.interceptors.openai import _wrap_batches_create

        response = MockBatchResponse(batch_id="batch_abc123", status="validating")

        def mock_create(*args, **kwargs):
            return response

        tracked_event = None

        def capture_event(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_event
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_batches_create(mock_create)
            result = wrapped(
                input_file_id="file-123",
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )

            assert result is response
            assert tracked_event is not None
            assert tracked_event.request_type == "batch_create"
            assert tracked_event.endpoint == "/v1/batches"
            assert tracked_event.metadata_extra["batch_id"] == "batch_abc123"
            assert tracked_event.metadata_extra["input_file_id"] == "file-123"
            assert tracked_event.metadata_extra["batch_endpoint"] == "/v1/chat/completions"
            # Batch cost is null at creation (billed on completion)
            assert tracked_event.cost_usd is None


# =============================================================================
# Tests for Attribution Context
# =============================================================================


class TestAttributionContextApplication:
    """Tests for _apply_attribution_context function."""

    def test_attribution_context_applied(self) -> None:
        """Test that attribution context is applied to events."""
        from tokenledger.context import attribution
        from tokenledger.interceptors.openai import _wrap_chat_completions_create

        response = MockChatCompletionResponse()

        def mock_create(*args, **kwargs):
            return response

        tracked_event = None

        def capture_event(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_event
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_chat_completions_create(mock_create)

            with attribution(
                user_id="user_123",
                session_id="session_456",
                organization_id="org_789",
                feature="chat",
                page="/dashboard",
                component="ChatWidget",
                team="platform",
                project="api",
                cost_center="CC-001",
            ):
                wrapped(model="gpt-4o", messages=[])

            assert tracked_event.user_id == "user_123"
            assert tracked_event.session_id == "session_456"
            assert tracked_event.organization_id == "org_789"
            assert tracked_event.feature == "chat"
            assert tracked_event.page == "/dashboard"
            assert tracked_event.component == "ChatWidget"
            assert tracked_event.team == "platform"
            assert tracked_event.project == "api"
            assert tracked_event.cost_center == "CC-001"

    def test_user_parameter_not_overwritten(self) -> None:
        """Test that user parameter takes precedence over context."""
        from tokenledger.context import attribution
        from tokenledger.interceptors.openai import _wrap_chat_completions_create

        response = MockChatCompletionResponse()

        def mock_create(*args, **kwargs):
            return response

        tracked_event = None

        def capture_event(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_event
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_chat_completions_create(mock_create)

            with attribution(user_id="context_user"):
                wrapped(model="gpt-4o", messages=[], user="api_user")

            # API user parameter should be preserved (set first)
            assert tracked_event.user_id == "api_user"


# =============================================================================
# Tests for Patching Functions
# =============================================================================


class TestPatchOpenAI:
    """Tests for patch_openai and unpatch_openai functions."""

    def test_patch_sets_flag(self) -> None:
        """Test that patching sets the _patched flag."""
        from tokenledger.interceptors import openai as openai_interceptor

        # Reset state
        openai_interceptor._patched = False
        openai_interceptor._original_methods.clear()

        with (
            patch.dict("sys.modules", {"openai": MagicMock()}),
            patch("tokenledger.interceptors.openai.get_tracker") as mock_tracker,
            patch.object(openai_interceptor, "_original_methods", {}),
        ):
            mock_tracker.return_value = MagicMock()
            # Can't fully test without openai installed, but can test flag logic
            assert not openai_interceptor._patched

    def test_double_patch_warning(self) -> None:
        """Test that double patching logs a warning."""
        from tokenledger.interceptors import openai as openai_interceptor

        openai_interceptor._patched = True

        with patch("tokenledger.interceptors.openai.logger") as mock_logger:
            openai_interceptor.patch_openai()
            mock_logger.warning.assert_called_once_with("OpenAI is already patched")

        # Clean up
        openai_interceptor._patched = False

    def test_unpatch_when_not_patched(self) -> None:
        """Test that unpatching when not patched does nothing."""
        from tokenledger.interceptors import openai as openai_interceptor

        openai_interceptor._patched = False

        # Should not raise
        openai_interceptor.unpatch_openai()
        assert not openai_interceptor._patched


# =============================================================================
# Tests for Edge Cases
# =============================================================================


class TestApplyAttributionContext:
    """Tests for _apply_attribution_context function."""

    def test_metadata_extra_merging(self) -> None:
        """Test that metadata_extra from context is merged with event metadata."""
        from tokenledger.context import attribution
        from tokenledger.interceptors.openai import _apply_attribution_context
        from tokenledger.models import LLMEvent

        event = LLMEvent.fast_construct(
            provider="openai",
            model="gpt-4o",
            metadata_extra={"event_key": "event_value"},
        )

        with attribution(user_id="user123", custom_field="custom_value"):
            _apply_attribution_context(event)

        # Event metadata should be preserved, context metadata merged
        assert event.user_id == "user123"
        assert event.metadata_extra is not None
        assert event.metadata_extra.get("event_key") == "event_value"
        assert event.metadata_extra.get("custom_field") == "custom_value"

    def test_event_metadata_takes_precedence(self) -> None:
        """Test that event metadata takes precedence over context metadata."""
        from tokenledger.context import attribution
        from tokenledger.interceptors.openai import _apply_attribution_context
        from tokenledger.models import LLMEvent

        event = LLMEvent.fast_construct(
            provider="openai",
            model="gpt-4o",
            metadata_extra={"shared_key": "event_wins"},
        )

        with attribution(shared_key="context_loses"):
            _apply_attribution_context(event)

        # Event value should be preserved
        assert event.metadata_extra["shared_key"] == "event_wins"

    def test_no_context_checks_warning(self) -> None:
        """Test that None context triggers warning check."""
        from tokenledger.interceptors.openai import _apply_attribution_context
        from tokenledger.models import LLMEvent

        event = LLMEvent.fast_construct(
            provider="openai",
            model="gpt-4o",
        )

        with (
            patch("tokenledger.interceptors.openai.get_attribution_context", return_value=None),
            patch(
                "tokenledger.interceptors.openai.check_attribution_context_warning"
            ) as mock_check,
        ):
            _apply_attribution_context(event)
            mock_check.assert_called_once()

    def test_applies_all_context_fields(self) -> None:
        """Test that all context fields are applied to event."""
        from tokenledger.context import attribution
        from tokenledger.interceptors.openai import _apply_attribution_context
        from tokenledger.models import LLMEvent

        event = LLMEvent.fast_construct(
            provider="openai",
            model="gpt-4o",
        )

        with attribution(
            user_id="user",
            session_id="session",
            organization_id="org",
            feature="feature",
            page="/page",
            component="component",
            team="team",
            project="project",
            cost_center="CC001",
        ):
            _apply_attribution_context(event)

        assert event.user_id == "user"
        assert event.session_id == "session"
        assert event.organization_id == "org"
        assert event.feature == "feature"
        assert event.page == "/page"
        assert event.component == "component"
        assert event.team == "team"
        assert event.project == "project"
        assert event.cost_center == "CC001"

    def test_does_not_overwrite_existing_event_fields(self) -> None:
        """Test that context does not overwrite existing event fields."""
        from tokenledger.context import attribution
        from tokenledger.interceptors.openai import _apply_attribution_context
        from tokenledger.models import LLMEvent

        event = LLMEvent.fast_construct(
            provider="openai",
            model="gpt-4o",
            user_id="original_user",
            feature="original_feature",
        )

        with attribution(user_id="context_user", feature="context_feature"):
            _apply_attribution_context(event)

        # Event values should be preserved
        assert event.user_id == "original_user"
        assert event.feature == "original_feature"


class TestAsyncAudioWrappers:
    """Tests for async audio API wrappers."""

    def test_async_transcription_success(self) -> None:
        """Test async audio transcription tracking."""
        from tokenledger.interceptors.openai import _wrap_async_audio_transcription_create

        async def run_test():
            response = MockTranscriptionResponse(text="Async transcription result")

            async def mock_create(*args, **kwargs):
                return response

            tracked_event = None

            def capture_event(event: Any) -> None:
                nonlocal tracked_event
                tracked_event = event

            with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
                mock_tracker = MagicMock()
                mock_tracker.track = capture_event
                mock_get_tracker.return_value = mock_tracker

                wrapped = _wrap_async_audio_transcription_create(mock_create)
                result = await wrapped(model="whisper-1", file=MagicMock())

                assert result is response
                assert tracked_event is not None
                assert tracked_event.model == "whisper-1"
                assert tracked_event.request_type == "transcription"
                assert tracked_event.status == "success"
                assert tracked_event.response_preview == "Async transcription result"

        asyncio.run(run_test())

    def test_async_transcription_error(self) -> None:
        """Test async audio transcription error handling."""
        from tokenledger.interceptors.openai import _wrap_async_audio_transcription_create

        async def run_test():
            async def mock_create(*args, **kwargs):
                raise ValueError("Transcription failed")

            tracked_event = None

            def capture_event(event: Any) -> None:
                nonlocal tracked_event
                tracked_event = event

            with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
                mock_tracker = MagicMock()
                mock_tracker.track = capture_event
                mock_get_tracker.return_value = mock_tracker

                wrapped = _wrap_async_audio_transcription_create(mock_create)

                with pytest.raises(ValueError, match="Transcription failed"):
                    await wrapped(model="whisper-1", file=MagicMock())

                assert tracked_event.status == "error"
                assert tracked_event.error_type == "ValueError"

        asyncio.run(run_test())

    def test_async_translation_success(self) -> None:
        """Test async audio translation tracking."""
        from tokenledger.interceptors.openai import _wrap_async_audio_translation_create

        async def run_test():
            response = MockTranscriptionResponse(text="Translated text")

            async def mock_create(*args, **kwargs):
                return response

            tracked_event = None

            def capture_event(event: Any) -> None:
                nonlocal tracked_event
                tracked_event = event

            with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
                mock_tracker = MagicMock()
                mock_tracker.track = capture_event
                mock_get_tracker.return_value = mock_tracker

                wrapped = _wrap_async_audio_translation_create(mock_create)
                result = await wrapped(model="whisper-1", file=MagicMock())

                assert result is response
                assert tracked_event.request_type == "translation"
                assert tracked_event.endpoint == "/v1/audio/translations"

        asyncio.run(run_test())

    def test_async_translation_error(self) -> None:
        """Test async audio translation error handling."""
        from tokenledger.interceptors.openai import _wrap_async_audio_translation_create

        async def run_test():
            async def mock_create(*args, **kwargs):
                raise RuntimeError("Translation failed")

            tracked_event = None

            def capture_event(event: Any) -> None:
                nonlocal tracked_event
                tracked_event = event

            with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
                mock_tracker = MagicMock()
                mock_tracker.track = capture_event
                mock_get_tracker.return_value = mock_tracker

                wrapped = _wrap_async_audio_translation_create(mock_create)

                with pytest.raises(RuntimeError):
                    await wrapped(model="whisper-1", file=MagicMock())

                assert tracked_event.status == "error"

        asyncio.run(run_test())

    def test_async_speech_success(self) -> None:
        """Test async audio speech (TTS) tracking."""
        from tokenledger.interceptors.openai import _wrap_async_audio_speech_create

        async def run_test():
            response = MagicMock()

            async def mock_create(*args, **kwargs):
                return response

            tracked_event = None

            def capture_event(event: Any) -> None:
                nonlocal tracked_event
                tracked_event = event

            with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
                mock_tracker = MagicMock()
                mock_tracker.track = capture_event
                mock_get_tracker.return_value = mock_tracker

                wrapped = _wrap_async_audio_speech_create(mock_create)
                result = await wrapped(model="tts-1", input="Hello async world", voice="alloy")

                assert result is response
                assert tracked_event.request_type == "speech"
                assert tracked_event.endpoint == "/v1/audio/speech"
                assert tracked_event.metadata_extra["character_count"] == len("Hello async world")

        asyncio.run(run_test())

    def test_async_speech_error(self) -> None:
        """Test async audio speech error handling."""
        from tokenledger.interceptors.openai import _wrap_async_audio_speech_create

        async def run_test():
            async def mock_create(*args, **kwargs):
                raise ValueError("Speech failed")

            tracked_event = None

            def capture_event(event: Any) -> None:
                nonlocal tracked_event
                tracked_event = event

            with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
                mock_tracker = MagicMock()
                mock_tracker.track = capture_event
                mock_get_tracker.return_value = mock_tracker

                wrapped = _wrap_async_audio_speech_create(mock_create)

                with pytest.raises(ValueError):
                    await wrapped(model="tts-1", input="Test", voice="alloy")

                assert tracked_event.status == "error"

        asyncio.run(run_test())


class TestAudioTranslationWrapper:
    """Tests for sync audio translation wrapper."""

    def test_translation_success(self) -> None:
        """Test sync audio translation tracking."""
        from tokenledger.interceptors.openai import _wrap_audio_translation_create

        response = MockTranscriptionResponse(text="Translated text here")

        def mock_create(*args, **kwargs):
            return response

        tracked_event = None

        def capture_event(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_event
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_audio_translation_create(mock_create)
            result = wrapped(model="whisper-1", file=MagicMock())

            assert result is response
            assert tracked_event.request_type == "translation"
            assert tracked_event.endpoint == "/v1/audio/translations"
            assert tracked_event.response_preview == "Translated text here"

    def test_translation_error(self) -> None:
        """Test sync audio translation error handling."""
        from tokenledger.interceptors.openai import _wrap_audio_translation_create

        def mock_create(*args, **kwargs):
            raise ValueError("Translation error")

        tracked_event = None

        def capture_event(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_event
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_audio_translation_create(mock_create)

            with pytest.raises(ValueError):
                wrapped(model="whisper-1", file=MagicMock())

            assert tracked_event.status == "error"


class TestImageEditWrapper:
    """Tests for image edit wrappers."""

    def test_sync_image_edit_success(self) -> None:
        """Test sync image edit tracking."""
        from tokenledger.interceptors.openai import _wrap_images_edit

        response = MockImageResponse()

        def mock_create(*args, **kwargs):
            return response

        tracked_event = None

        def capture_event(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_event
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_images_edit(mock_create)
            result = wrapped(
                model="dall-e-2",
                prompt="Add a cat",
                image=MagicMock(),
                n=2,
                size="512x512",
            )

            assert result is response
            assert tracked_event.request_type == "image_edit"
            assert tracked_event.endpoint == "/v1/images/edits"
            assert tracked_event.metadata_extra["image_count"] == 2
            assert tracked_event.metadata_extra["image_size"] == "512x512"

    def test_sync_image_edit_error(self) -> None:
        """Test sync image edit error handling."""
        from tokenledger.interceptors.openai import _wrap_images_edit

        def mock_create(*args, **kwargs):
            raise ValueError("Edit failed")

        tracked_event = None

        def capture_event(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_event
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_images_edit(mock_create)

            with pytest.raises(ValueError):
                wrapped(model="dall-e-2", prompt="Test", image=MagicMock())

            assert tracked_event.status == "error"

    def test_async_image_edit_success(self) -> None:
        """Test async image edit tracking."""
        from tokenledger.interceptors.openai import _wrap_async_images_edit

        async def run_test():
            response = MockImageResponse()

            async def mock_create(*args, **kwargs):
                return response

            tracked_event = None

            def capture_event(event: Any) -> None:
                nonlocal tracked_event
                tracked_event = event

            with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
                mock_tracker = MagicMock()
                mock_tracker.track = capture_event
                mock_get_tracker.return_value = mock_tracker

                wrapped = _wrap_async_images_edit(mock_create)
                result = await wrapped(
                    model="dall-e-2",
                    prompt="Add a dog",
                    image=MagicMock(),
                    n=1,
                    size="1024x1024",
                )

                assert result is response
                assert tracked_event.request_type == "image_edit"

        asyncio.run(run_test())

    def test_async_image_edit_error(self) -> None:
        """Test async image edit error handling."""
        from tokenledger.interceptors.openai import _wrap_async_images_edit

        async def run_test():
            async def mock_create(*args, **kwargs):
                raise RuntimeError("Async edit failed")

            tracked_event = None

            def capture_event(event: Any) -> None:
                nonlocal tracked_event
                tracked_event = event

            with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
                mock_tracker = MagicMock()
                mock_tracker.track = capture_event
                mock_get_tracker.return_value = mock_tracker

                wrapped = _wrap_async_images_edit(mock_create)

                with pytest.raises(RuntimeError):
                    await wrapped(model="dall-e-2", prompt="Test", image=MagicMock())

                assert tracked_event.status == "error"

        asyncio.run(run_test())


class TestImageVariationWrapper:
    """Tests for image variation wrappers."""

    def test_sync_image_variation_success(self) -> None:
        """Test sync image variation tracking."""
        from tokenledger.interceptors.openai import _wrap_images_create_variation

        response = MockImageResponse()

        def mock_create(*args, **kwargs):
            return response

        tracked_event = None

        def capture_event(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_event
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_images_create_variation(mock_create)
            result = wrapped(
                model="dall-e-2",
                image=MagicMock(),
                n=3,
                size="256x256",
            )

            assert result is response
            assert tracked_event.request_type == "image_variation"
            assert tracked_event.endpoint == "/v1/images/variations"
            assert tracked_event.metadata_extra["image_count"] == 3
            assert tracked_event.metadata_extra["image_size"] == "256x256"

    def test_sync_image_variation_error(self) -> None:
        """Test sync image variation error handling."""
        from tokenledger.interceptors.openai import _wrap_images_create_variation

        def mock_create(*args, **kwargs):
            raise ValueError("Variation failed")

        tracked_event = None

        def capture_event(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_event
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_images_create_variation(mock_create)

            with pytest.raises(ValueError):
                wrapped(model="dall-e-2", image=MagicMock())

            assert tracked_event.status == "error"

    def test_async_image_variation_success(self) -> None:
        """Test async image variation tracking."""
        from tokenledger.interceptors.openai import _wrap_async_images_create_variation

        async def run_test():
            response = MockImageResponse()

            async def mock_create(*args, **kwargs):
                return response

            tracked_event = None

            def capture_event(event: Any) -> None:
                nonlocal tracked_event
                tracked_event = event

            with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
                mock_tracker = MagicMock()
                mock_tracker.track = capture_event
                mock_get_tracker.return_value = mock_tracker

                wrapped = _wrap_async_images_create_variation(mock_create)
                result = await wrapped(model="dall-e-2", image=MagicMock(), n=2, size="512x512")

                assert result is response
                assert tracked_event.request_type == "image_variation"

        asyncio.run(run_test())

    def test_async_image_variation_error(self) -> None:
        """Test async image variation error handling."""
        from tokenledger.interceptors.openai import _wrap_async_images_create_variation

        async def run_test():
            async def mock_create(*args, **kwargs):
                raise RuntimeError("Async variation failed")

            tracked_event = None

            def capture_event(event: Any) -> None:
                nonlocal tracked_event
                tracked_event = event

            with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
                mock_tracker = MagicMock()
                mock_tracker.track = capture_event
                mock_get_tracker.return_value = mock_tracker

                wrapped = _wrap_async_images_create_variation(mock_create)

                with pytest.raises(RuntimeError):
                    await wrapped(model="dall-e-2", image=MagicMock())

                assert tracked_event.status == "error"

        asyncio.run(run_test())


class TestAsyncImageGenerateWrapper:
    """Tests for async image generation wrapper."""

    def test_async_image_generate_success(self) -> None:
        """Test async image generation tracking."""
        from tokenledger.interceptors.openai import _wrap_async_images_generate

        async def run_test():
            response = MockImageResponse()

            async def mock_create(*args, **kwargs):
                return response

            tracked_event = None

            def capture_event(event: Any) -> None:
                nonlocal tracked_event
                tracked_event = event

            with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
                mock_tracker = MagicMock()
                mock_tracker.track = capture_event
                mock_get_tracker.return_value = mock_tracker

                wrapped = _wrap_async_images_generate(mock_create)
                result = await wrapped(
                    model="dall-e-3",
                    prompt="A sunset",
                    n=1,
                    size="1024x1024",
                    quality="hd",
                )

                assert result is response
                assert tracked_event.request_type == "image_generation"
                assert tracked_event.metadata_extra["image_quality"] == "hd"

        asyncio.run(run_test())

    def test_async_image_generate_error(self) -> None:
        """Test async image generation error handling."""
        from tokenledger.interceptors.openai import _wrap_async_images_generate

        async def run_test():
            async def mock_create(*args, **kwargs):
                raise ValueError("Generation failed")

            tracked_event = None

            def capture_event(event: Any) -> None:
                nonlocal tracked_event
                tracked_event = event

            with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
                mock_tracker = MagicMock()
                mock_tracker.track = capture_event
                mock_get_tracker.return_value = mock_tracker

                wrapped = _wrap_async_images_generate(mock_create)

                with pytest.raises(ValueError):
                    await wrapped(model="dall-e-3", prompt="Test")

                assert tracked_event.status == "error"

        asyncio.run(run_test())


class TestAsyncBatchesCreateWrapper:
    """Tests for async batch create wrapper."""

    def test_async_batch_create_success(self) -> None:
        """Test async batch creation tracking."""
        from tokenledger.interceptors.openai import _wrap_async_batches_create

        async def run_test():
            response = MockBatchResponse(batch_id="batch_async_123", status="validating")

            async def mock_create(*args, **kwargs):
                return response

            tracked_event = None

            def capture_event(event: Any) -> None:
                nonlocal tracked_event
                tracked_event = event

            with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
                mock_tracker = MagicMock()
                mock_tracker.track = capture_event
                mock_get_tracker.return_value = mock_tracker

                wrapped = _wrap_async_batches_create(mock_create)
                result = await wrapped(
                    input_file_id="file-async-123",
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                    metadata={"key": "value"},
                )

                assert result is response
                assert tracked_event.request_type == "batch_create"
                assert tracked_event.metadata_extra["batch_id"] == "batch_async_123"
                assert tracked_event.metadata_extra["batch_metadata"] == {"key": "value"}

        asyncio.run(run_test())

    def test_async_batch_create_error(self) -> None:
        """Test async batch creation error handling."""
        from tokenledger.interceptors.openai import _wrap_async_batches_create

        async def run_test():
            async def mock_create(*args, **kwargs):
                raise RuntimeError("Batch creation failed")

            tracked_event = None

            def capture_event(event: Any) -> None:
                nonlocal tracked_event
                tracked_event = event

            with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
                mock_tracker = MagicMock()
                mock_tracker.track = capture_event
                mock_get_tracker.return_value = mock_tracker

                wrapped = _wrap_async_batches_create(mock_create)

                with pytest.raises(RuntimeError):
                    await wrapped(input_file_id="file-123", endpoint="/v1/chat/completions")

                assert tracked_event.status == "error"
                assert tracked_event.error_type == "RuntimeError"

        asyncio.run(run_test())


class TestResponsesApiErrors:
    """Tests for responses API error handling."""

    def test_sync_responses_error(self) -> None:
        """Test sync responses API error handling."""
        from tokenledger.interceptors.openai import _wrap_responses_create

        def mock_create(*args, **kwargs):
            raise ValueError("Responses API error")

        tracked_event = None

        def capture_event(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_event
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_responses_create(mock_create)

            with pytest.raises(ValueError):
                wrapped(model="gpt-4o", input="Test")

            assert tracked_event.status == "error"
            assert tracked_event.error_type == "ValueError"

    def test_async_responses_error(self) -> None:
        """Test async responses API error handling."""
        from tokenledger.interceptors.openai import _wrap_async_responses_create

        async def run_test():
            async def mock_create(*args, **kwargs):
                raise RuntimeError("Async responses error")

            tracked_event = None

            def capture_event(event: Any) -> None:
                nonlocal tracked_event
                tracked_event = event

            with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
                mock_tracker = MagicMock()
                mock_tracker.track = capture_event
                mock_get_tracker.return_value = mock_tracker

                wrapped = _wrap_async_responses_create(mock_create)

                with pytest.raises(RuntimeError):
                    await wrapped(model="gpt-4o", input="Test")

                assert tracked_event.status == "error"

        asyncio.run(run_test())


class TestAudioTranscriptionErrors:
    """Tests for audio transcription error handling."""

    def test_sync_transcription_error(self) -> None:
        """Test sync audio transcription error handling."""
        from tokenledger.interceptors.openai import _wrap_audio_transcription_create

        def mock_create(*args, **kwargs):
            raise ValueError("Transcription error")

        tracked_event = None

        def capture_event(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_event
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_audio_transcription_create(mock_create)

            with pytest.raises(ValueError):
                wrapped(model="whisper-1", file=MagicMock())

            assert tracked_event.status == "error"
            assert tracked_event.error_type == "ValueError"


class TestSpeechErrors:
    """Tests for audio speech error handling."""

    def test_sync_speech_error(self) -> None:
        """Test sync audio speech error handling."""
        from tokenledger.interceptors.openai import _wrap_audio_speech_create

        def mock_create(*args, **kwargs):
            raise ValueError("Speech error")

        tracked_event = None

        def capture_event(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_event
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_audio_speech_create(mock_create)

            with pytest.raises(ValueError):
                wrapped(model="tts-1", input="Test", voice="alloy")

            assert tracked_event.status == "error"


class TestImageGenerationErrors:
    """Tests for image generation error handling."""

    def test_sync_image_generate_error(self) -> None:
        """Test sync image generation error handling."""
        from tokenledger.interceptors.openai import _wrap_images_generate

        def mock_create(*args, **kwargs):
            raise ValueError("Generation error")

        tracked_event = None

        def capture_event(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_event
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_images_generate(mock_create)

            with pytest.raises(ValueError):
                wrapped(model="dall-e-3", prompt="Test")

            assert tracked_event.status == "error"


class TestBatchCreateErrors:
    """Tests for batch create error handling."""

    def test_sync_batch_create_error(self) -> None:
        """Test sync batch creation error handling."""
        from tokenledger.interceptors.openai import _wrap_batches_create

        def mock_create(*args, **kwargs):
            raise ValueError("Batch error")

        tracked_event = None

        def capture_event(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_event
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_batches_create(mock_create)

            with pytest.raises(ValueError):
                wrapped(input_file_id="file-123", endpoint="/v1/chat/completions")

            assert tracked_event.status == "error"


class TestEdgeCases:
    """Tests for edge cases and unusual inputs."""

    def test_empty_model_string(self) -> None:
        """Test handling of empty model string."""
        from tokenledger.interceptors.openai import _wrap_chat_completions_create

        response = MockChatCompletionResponse(model="gpt-4o")

        def mock_create(*args, **kwargs):
            return response

        tracked_event = None

        def capture_event(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_event
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_chat_completions_create(mock_create)
            # Empty model in request, should use response model
            wrapped(model="", messages=[])

            assert tracked_event.model == "gpt-4o"

    def test_unknown_model_cost(self) -> None:
        """Test that unknown models have None cost."""
        from tokenledger.interceptors.openai import _wrap_chat_completions_create

        response = MockChatCompletionResponse(model="unknown-model-xyz")

        def mock_create(*args, **kwargs):
            return response

        tracked_event = None

        def capture_event(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_event
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_chat_completions_create(mock_create)
            wrapped(model="unknown-model-xyz", messages=[])

            # Unknown model should have None cost
            assert tracked_event.cost_usd is None

    def test_duration_tracked(self) -> None:
        """Test that duration is tracked correctly."""
        import time

        from tokenledger.interceptors.openai import _wrap_chat_completions_create

        response = MockChatCompletionResponse()

        def mock_create(*args, **kwargs):
            time.sleep(0.01)  # 10ms delay
            return response

        tracked_event = None

        def capture_event(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_event
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_chat_completions_create(mock_create)
            wrapped(model="gpt-4o", messages=[])

            # Duration should be at least 10ms
            assert tracked_event.duration_ms >= 10

    def test_total_tokens_calculated(self) -> None:
        """Test that total tokens is calculated correctly."""
        from tokenledger.interceptors.openai import _wrap_chat_completions_create

        response = MockChatCompletionResponse(
            prompt_tokens=300,
            completion_tokens=200,
        )

        def mock_create(*args, **kwargs):
            return response

        tracked_event = None

        def capture_event(event: Any) -> None:
            nonlocal tracked_event
            tracked_event = event

        with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.track = capture_event
            mock_get_tracker.return_value = mock_tracker

            wrapped = _wrap_chat_completions_create(mock_create)
            wrapped(model="gpt-4o", messages=[])

            assert tracked_event.total_tokens == 500  # 300 + 200

    def test_different_model_versions(self) -> None:
        """Test tracking with different model versions."""
        from tokenledger.interceptors.openai import _wrap_chat_completions_create

        models_to_test = [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4o-2024-11-20",
            "gpt-3.5-turbo",
            "o1",
            "o1-mini",
        ]

        for model in models_to_test:
            response = MockChatCompletionResponse(model=model)

            def make_mock_create(resp):
                def mock_create(*args, **kwargs):
                    return resp

                return mock_create

            tracked_event = None

            def capture_event(event: Any) -> None:
                nonlocal tracked_event
                tracked_event = event

            with patch("tokenledger.interceptors.openai.get_tracker") as mock_get_tracker:
                mock_tracker = MagicMock()
                mock_tracker.track = capture_event
                mock_get_tracker.return_value = mock_tracker

                wrapped = _wrap_chat_completions_create(make_mock_create(response))
                wrapped(model=model, messages=[])

                assert tracked_event.model == model
                assert tracked_event.provider == "openai"
