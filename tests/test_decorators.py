"""Tests for the decorators module."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from tokenledger.decorators import _detect_provider, _extract_from_response, track_cost, track_llm
from tokenledger.tracker import LLMEvent


class TestTrackLlmDecorator:
    """Tests for @track_llm decorator."""

    @patch("tokenledger.decorators.get_tracker")
    def test_sync_function_tracking(self, mock_get_tracker: MagicMock) -> None:
        """Test decorator tracks sync function calls."""
        mock_tracker = MagicMock()
        mock_get_tracker.return_value = mock_tracker

        @track_llm(model="gpt-4o", provider="openai")
        def my_func():
            return {"result": "test"}

        result = my_func()

        assert result == {"result": "test"}
        mock_tracker.track.assert_called_once()
        tracked_event = mock_tracker.track.call_args[0][0]
        assert tracked_event.model == "gpt-4o"
        assert tracked_event.provider == "openai"
        assert tracked_event.status == "success"

    @patch("tokenledger.decorators.get_tracker")
    def test_sync_function_tracks_duration(self, mock_get_tracker: MagicMock) -> None:
        """Test decorator tracks function duration."""
        mock_tracker = MagicMock()
        mock_get_tracker.return_value = mock_tracker

        @track_llm(model="gpt-4o", provider="openai")
        def slow_func():
            time.sleep(0.1)
            return "done"

        slow_func()

        tracked_event = mock_tracker.track.call_args[0][0]
        assert tracked_event.duration_ms is not None
        assert tracked_event.duration_ms >= 100  # At least 100ms

    @patch("tokenledger.decorators.get_tracker")
    def test_sync_function_tracks_errors(self, mock_get_tracker: MagicMock) -> None:
        """Test decorator tracks errors from sync functions."""
        mock_tracker = MagicMock()
        mock_get_tracker.return_value = mock_tracker

        @track_llm(model="gpt-4o", provider="openai")
        def error_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            error_func()

        mock_tracker.track.assert_called_once()
        tracked_event = mock_tracker.track.call_args[0][0]
        assert tracked_event.status == "error"
        assert tracked_event.error_type == "ValueError"
        assert tracked_event.error_message == "Test error"

    @patch("tokenledger.decorators.get_tracker")
    def test_decorator_with_metadata(self, mock_get_tracker: MagicMock) -> None:
        """Test decorator with custom metadata."""
        mock_tracker = MagicMock()
        mock_get_tracker.return_value = mock_tracker

        @track_llm(model="gpt-4o", provider="openai", metadata={"custom_key": "custom_value"})
        def my_func():
            return "result"

        my_func()

        tracked_event = mock_tracker.track.call_args[0][0]
        assert tracked_event.metadata == {"custom_key": "custom_value"}

    @patch("tokenledger.decorators.get_tracker")
    def test_decorator_with_user_id(self, mock_get_tracker: MagicMock) -> None:
        """Test decorator with user_id."""
        mock_tracker = MagicMock()
        mock_get_tracker.return_value = mock_tracker

        @track_llm(model="gpt-4o", provider="openai", user_id="user-123")
        def my_func():
            return "result"

        my_func()

        tracked_event = mock_tracker.track.call_args[0][0]
        assert tracked_event.user_id == "user-123"

    @patch("tokenledger.decorators.get_tracker")
    def test_decorator_with_endpoint(self, mock_get_tracker: MagicMock) -> None:
        """Test decorator with endpoint."""
        mock_tracker = MagicMock()
        mock_get_tracker.return_value = mock_tracker

        @track_llm(model="gpt-4o", provider="openai", endpoint="/v1/chat/completions")
        def my_func():
            return "result"

        my_func()

        tracked_event = mock_tracker.track.call_args[0][0]
        assert tracked_event.endpoint == "/v1/chat/completions"

    @patch("tokenledger.decorators.get_tracker")
    def test_decorator_extracts_from_openai_response(
        self, mock_get_tracker: MagicMock, mock_openai_response: MagicMock
    ) -> None:
        """Test decorator extracts token info from OpenAI response."""
        mock_tracker = MagicMock()
        mock_get_tracker.return_value = mock_tracker

        @track_llm()
        def my_func():
            return mock_openai_response

        my_func()

        tracked_event = mock_tracker.track.call_args[0][0]
        assert tracked_event.input_tokens == 100
        assert tracked_event.output_tokens == 50
        assert tracked_event.model == "gpt-4o"

    @patch("tokenledger.decorators.get_tracker")
    def test_decorator_extracts_from_anthropic_response(
        self, mock_get_tracker: MagicMock, mock_anthropic_response: MagicMock
    ) -> None:
        """Test decorator extracts token info from Anthropic response."""
        mock_tracker = MagicMock()
        mock_get_tracker.return_value = mock_tracker

        @track_llm()
        def my_func():
            return mock_anthropic_response

        my_func()

        tracked_event = mock_tracker.track.call_args[0][0]
        assert tracked_event.input_tokens == 100
        assert tracked_event.output_tokens == 50
        assert tracked_event.cached_tokens == 20
        assert tracked_event.model == "claude-3-5-sonnet-20241022"

    @patch("tokenledger.decorators.get_tracker")
    def test_decorator_defaults_to_unknown(self, mock_get_tracker: MagicMock) -> None:
        """Test decorator defaults to unknown model/provider."""
        mock_tracker = MagicMock()
        mock_get_tracker.return_value = mock_tracker

        @track_llm()
        def my_func():
            return "plain result"

        my_func()

        tracked_event = mock_tracker.track.call_args[0][0]
        # Model stays unknown since response has no model attribute
        assert tracked_event.model == "unknown"

    @patch("tokenledger.decorators.get_tracker")
    def test_decorator_preserves_function_metadata(self, mock_get_tracker: MagicMock) -> None:
        """Test decorator preserves original function metadata."""
        mock_tracker = MagicMock()
        mock_get_tracker.return_value = mock_tracker

        @track_llm(model="gpt-4o")
        def my_documented_func():
            """This is my docstring."""
            return "result"

        assert my_documented_func.__name__ == "my_documented_func"
        assert my_documented_func.__doc__ == "This is my docstring."


class TestTrackLlmDecoratorAsync:
    """Tests for @track_llm decorator with async functions."""

    @pytest.mark.asyncio
    @patch("tokenledger.decorators.get_tracker")
    async def test_async_function_tracking(self, mock_get_tracker: MagicMock) -> None:
        """Test decorator tracks async function calls."""
        mock_tracker = MagicMock()
        mock_get_tracker.return_value = mock_tracker

        @track_llm(model="claude-3-5-sonnet-20241022", provider="anthropic")
        async def my_async_func():
            return {"result": "async_test"}

        result = await my_async_func()

        assert result == {"result": "async_test"}
        mock_tracker.track.assert_called_once()
        tracked_event = mock_tracker.track.call_args[0][0]
        assert tracked_event.model == "claude-3-5-sonnet-20241022"
        assert tracked_event.provider == "anthropic"
        assert tracked_event.status == "success"

    @pytest.mark.asyncio
    @patch("tokenledger.decorators.get_tracker")
    async def test_async_function_tracks_errors(self, mock_get_tracker: MagicMock) -> None:
        """Test decorator tracks errors from async functions."""
        mock_tracker = MagicMock()
        mock_get_tracker.return_value = mock_tracker

        @track_llm(model="gpt-4o", provider="openai")
        async def async_error_func():
            raise RuntimeError("Async error")

        with pytest.raises(RuntimeError, match="Async error"):
            await async_error_func()

        mock_tracker.track.assert_called_once()
        tracked_event = mock_tracker.track.call_args[0][0]
        assert tracked_event.status == "error"
        assert tracked_event.error_type == "RuntimeError"
        assert tracked_event.error_message == "Async error"

    @pytest.mark.asyncio
    @patch("tokenledger.decorators.get_tracker")
    async def test_async_function_tracks_duration(self, mock_get_tracker: MagicMock) -> None:
        """Test decorator tracks async function duration."""
        import asyncio

        mock_tracker = MagicMock()
        mock_get_tracker.return_value = mock_tracker

        @track_llm(model="gpt-4o", provider="openai")
        async def slow_async_func():
            await asyncio.sleep(0.1)
            return "done"

        await slow_async_func()

        tracked_event = mock_tracker.track.call_args[0][0]
        assert tracked_event.duration_ms is not None
        assert tracked_event.duration_ms >= 100  # At least 100ms


class TestTrackCost:
    """Tests for track_cost function."""

    @patch("tokenledger.decorators.get_tracker")
    def test_track_cost_basic(self, mock_get_tracker: MagicMock) -> None:
        """Test basic track_cost call."""
        mock_tracker = MagicMock()
        mock_get_tracker.return_value = mock_tracker

        track_cost(
            input_tokens=1000,
            output_tokens=500,
            model="gpt-4o",
            provider="openai",
        )

        mock_tracker.track.assert_called_once()
        tracked_event = mock_tracker.track.call_args[0][0]
        assert tracked_event.input_tokens == 1000
        assert tracked_event.output_tokens == 500
        assert tracked_event.model == "gpt-4o"
        assert tracked_event.provider == "openai"

    @patch("tokenledger.decorators.get_tracker")
    def test_track_cost_with_user_id(self, mock_get_tracker: MagicMock) -> None:
        """Test track_cost with user_id."""
        mock_tracker = MagicMock()
        mock_get_tracker.return_value = mock_tracker

        track_cost(
            input_tokens=100,
            output_tokens=50,
            model="gpt-4o",
            user_id="user-456",
        )

        tracked_event = mock_tracker.track.call_args[0][0]
        assert tracked_event.user_id == "user-456"

    @patch("tokenledger.decorators.get_tracker")
    def test_track_cost_with_metadata(self, mock_get_tracker: MagicMock) -> None:
        """Test track_cost with custom metadata."""
        mock_tracker = MagicMock()
        mock_get_tracker.return_value = mock_tracker

        track_cost(
            input_tokens=100,
            output_tokens=50,
            model="gpt-4o",
            metadata={"session": "abc123"},
        )

        tracked_event = mock_tracker.track.call_args[0][0]
        assert tracked_event.metadata == {"session": "abc123"}

    @patch("tokenledger.decorators.get_tracker")
    def test_track_cost_auto_detects_provider(self, mock_get_tracker: MagicMock) -> None:
        """Test track_cost auto-detects provider from model name."""
        mock_tracker = MagicMock()
        mock_get_tracker.return_value = mock_tracker

        track_cost(
            input_tokens=100,
            output_tokens=50,
            model="claude-3-5-sonnet-20241022",
            provider="auto",
        )

        tracked_event = mock_tracker.track.call_args[0][0]
        assert tracked_event.provider == "anthropic"


class TestDetectProvider:
    """Tests for _detect_provider function."""

    def test_detect_openai_gpt_models(self) -> None:
        """Test detection of OpenAI GPT models."""
        assert _detect_provider("gpt-4o") == "openai"
        assert _detect_provider("gpt-4") == "openai"
        assert _detect_provider("gpt-3.5-turbo") == "openai"
        assert _detect_provider("gpt-4o-mini") == "openai"

    def test_detect_openai_o_models(self) -> None:
        """Test detection of OpenAI o-series models."""
        assert _detect_provider("o1") == "openai"
        assert _detect_provider("o1-mini") == "openai"
        assert _detect_provider("o1-preview") == "openai"

    def test_detect_openai_embeddings(self) -> None:
        """Test detection of OpenAI embedding models."""
        assert _detect_provider("text-embedding-3-small") == "openai"
        assert _detect_provider("text-embedding-3-large") == "openai"
        assert _detect_provider("text-embedding-ada-002") == "openai"

    def test_detect_anthropic_models(self) -> None:
        """Test detection of Anthropic Claude models."""
        assert _detect_provider("claude-3-5-sonnet-20241022") == "anthropic"
        assert _detect_provider("claude-3-opus-20240229") == "anthropic"
        assert _detect_provider("claude-3-haiku-20240307") == "anthropic"

    def test_detect_google_models(self) -> None:
        """Test detection of Google Gemini models."""
        assert _detect_provider("gemini-1.5-pro") == "google"
        assert _detect_provider("gemini-1.5-flash") == "google"

    def test_detect_mistral_models(self) -> None:
        """Test detection of Mistral models."""
        assert _detect_provider("mistral-large-latest") == "mistral"
        assert _detect_provider("open-mistral-7b") == "mistral"
        assert _detect_provider("open-mixtral-8x7b") == "mistral"

    def test_detect_unknown_model(self) -> None:
        """Test detection returns unknown for unrecognized models."""
        assert _detect_provider("some-unknown-model") == "unknown"
        assert _detect_provider("llama-2-70b") == "unknown"


class TestExtractFromResponse:
    """Tests for _extract_from_response function."""

    def test_extract_openai_format(self, mock_openai_response: MagicMock) -> None:
        """Test extraction from OpenAI response format."""
        event = LLMEvent(provider="unknown", model="unknown")

        _extract_from_response(event, mock_openai_response, None, None)

        assert event.input_tokens == 100
        assert event.output_tokens == 50
        assert event.total_tokens == 150
        assert event.model == "gpt-4o"
        assert event.provider == "openai"

    def test_extract_anthropic_format(self, mock_anthropic_response: MagicMock) -> None:
        """Test extraction from Anthropic response format."""
        event = LLMEvent(provider="unknown", model="unknown")

        _extract_from_response(event, mock_anthropic_response, None, None)

        assert event.input_tokens == 100
        assert event.output_tokens == 50
        assert event.cached_tokens == 20
        assert event.total_tokens == 150
        assert event.model == "claude-3-5-sonnet-20241022"
        assert event.provider == "anthropic"

    def test_extract_uses_defaults(self) -> None:
        """Test extraction uses defaults when response has no data."""
        event = LLMEvent(provider="unknown", model="unknown")
        plain_response = {"text": "Hello"}

        _extract_from_response(event, plain_response, "gpt-4o", "openai")

        assert event.model == "gpt-4o"
        assert event.provider == "openai"

    def test_extract_calculates_cost(self, mock_openai_response: MagicMock) -> None:
        """Test that extraction calculates cost."""
        event = LLMEvent(provider="unknown", model="unknown")

        _extract_from_response(event, mock_openai_response, None, None)

        assert event.cost_usd is not None
        # gpt-4o: $2.50/1M input, $10.00/1M output
        expected = (100 / 1_000_000 * 2.50) + (50 / 1_000_000 * 10.00)
        assert abs(event.cost_usd - expected) < 0.0001

    def test_extract_handles_none_usage(self) -> None:
        """Test extraction handles response with no usage."""
        event = LLMEvent(provider="openai", model="gpt-4o")
        response = MagicMock(spec=[])  # No usage attribute

        _extract_from_response(event, response, "gpt-4o", "openai")

        # Should not raise, tokens stay at 0
        assert event.input_tokens == 0
        assert event.output_tokens == 0
