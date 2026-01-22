"""
OpenAI SDK Interceptor
Automatically tracks all OpenAI API calls with zero code changes.
"""

import functools
import logging
import time
from collections.abc import Callable
from typing import Any

from ..context import get_attribution_context
from ..models import LLMEvent
from ..tracker import get_tracker

logger = logging.getLogger("tokenledger.openai")

# Store original methods for unpatching
_original_methods: dict[str, Callable] = {}
_patched = False


def _apply_attribution_context(event: LLMEvent) -> None:
    """Apply current attribution context to an event."""
    ctx = get_attribution_context()
    if ctx is None:
        return

    if ctx.user_id is not None and event.user_id is None:
        event.user_id = ctx.user_id
    if ctx.session_id is not None and event.session_id is None:
        event.session_id = ctx.session_id
    if ctx.organization_id is not None and event.organization_id is None:
        event.organization_id = ctx.organization_id
    if ctx.feature is not None and event.feature is None:
        event.feature = ctx.feature
    if ctx.page is not None and event.page is None:
        event.page = ctx.page
    if ctx.component is not None and event.component is None:
        event.component = ctx.component
    if ctx.team is not None and event.team is None:
        event.team = ctx.team
    if ctx.project is not None and event.project is None:
        event.project = ctx.project
    if ctx.cost_center is not None and event.cost_center is None:
        event.cost_center = ctx.cost_center
    if ctx.metadata_extra:
        existing_extra = event.metadata_extra or {}
        event.metadata_extra = {**ctx.metadata_extra, **existing_extra}


def _extract_tokens_from_response(response: Any) -> dict[str, int]:
    """Extract token counts from OpenAI response (chat completions)"""
    tokens = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
    }

    usage = getattr(response, "usage", None)
    if usage:
        tokens["input_tokens"] = getattr(usage, "prompt_tokens", 0) or 0
        tokens["output_tokens"] = getattr(usage, "completion_tokens", 0) or 0

        # Handle cached tokens (prompt caching)
        prompt_tokens_details = getattr(usage, "prompt_tokens_details", None)
        if prompt_tokens_details:
            tokens["cached_tokens"] = getattr(prompt_tokens_details, "cached_tokens", 0) or 0

    return tokens


def _extract_tokens_from_responses_api(response: Any) -> dict[str, int]:
    """Extract token counts from OpenAI responses API (uses different field names)"""
    tokens = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
    }

    usage = getattr(response, "usage", None)
    if usage:
        # Responses API uses input_tokens/output_tokens directly
        tokens["input_tokens"] = getattr(usage, "input_tokens", 0) or 0
        tokens["output_tokens"] = getattr(usage, "output_tokens", 0) or 0

        # Handle cached tokens if present
        input_tokens_details = getattr(usage, "input_tokens_details", None)
        if input_tokens_details:
            tokens["cached_tokens"] = getattr(input_tokens_details, "cached_tokens", 0) or 0

    return tokens


def _extract_model_from_response(response: Any, default: str = "") -> str:
    """Extract model name from response"""
    return getattr(response, "model", default)


def _get_request_preview(messages: Any, max_length: int = 500) -> str | None:
    """Get a preview of the request messages"""
    if not messages:
        return None

    try:
        if isinstance(messages, list) and len(messages) > 0:
            last_msg = messages[-1]
            if isinstance(last_msg, dict):
                content = last_msg.get("content", "")
            else:
                content = getattr(last_msg, "content", "")

            if isinstance(content, str):
                return content[:max_length] if len(content) > max_length else content
    except Exception:
        pass

    return None


def _get_response_preview(response: Any, max_length: int = 500) -> str | None:
    """Get a preview of the response content"""
    try:
        choices = getattr(response, "choices", [])
        if choices:
            message = getattr(choices[0], "message", None)
            if message:
                content = getattr(message, "content", "")
                if content:
                    return content[:max_length] if len(content) > max_length else content
    except Exception:
        pass

    return None


def _wrap_chat_completions_create(original_method: Callable) -> Callable:
    """Wrap the chat.completions.create method"""

    @functools.wraps(original_method)
    def wrapper(*args, **kwargs):
        tracker = get_tracker()
        start_time = time.perf_counter()

        # Extract request info
        model = kwargs.get("model", "")
        messages = kwargs.get("messages", [])
        user = kwargs.get("user")

        event = LLMEvent.fast_construct(
            provider="openai",
            model=model,
            request_type="chat",
            endpoint="/v1/chat/completions",
            user_id=user,
            request_preview=_get_request_preview(messages),
        )

        # Apply attribution context
        _apply_attribution_context(event)

        try:
            response = original_method(*args, **kwargs)

            # Calculate duration
            event.duration_ms = (time.perf_counter() - start_time) * 1000

            # Extract token info
            tokens = _extract_tokens_from_response(response)
            event.input_tokens = tokens["input_tokens"]
            event.output_tokens = tokens["output_tokens"]
            event.cached_tokens = tokens["cached_tokens"]
            event.total_tokens = event.input_tokens + event.output_tokens

            # Get actual model used
            event.model = _extract_model_from_response(response, model)

            # Get response preview
            event.response_preview = _get_response_preview(response)

            event.status = "success"

            # Recalculate cost with actual model
            from ..pricing import calculate_cost

            event.cost_usd = calculate_cost(
                event.model, event.input_tokens, event.output_tokens, event.cached_tokens, "openai"
            )

            tracker.track(event)
            return response

        except Exception as e:
            event.duration_ms = (time.perf_counter() - start_time) * 1000
            event.status = "error"
            event.error_type = type(e).__name__
            event.error_message = str(e)[:1000]
            tracker.track(event)
            raise

    return wrapper


def _wrap_async_chat_completions_create(original_method: Callable) -> Callable:
    """Wrap the async chat.completions.create method"""

    @functools.wraps(original_method)
    async def wrapper(*args, **kwargs):
        tracker = get_tracker()
        start_time = time.perf_counter()

        model = kwargs.get("model", "")
        messages = kwargs.get("messages", [])
        user = kwargs.get("user")

        event = LLMEvent.fast_construct(
            provider="openai",
            model=model,
            request_type="chat",
            endpoint="/v1/chat/completions",
            user_id=user,
            request_preview=_get_request_preview(messages),
        )

        # Apply attribution context
        _apply_attribution_context(event)

        try:
            response = await original_method(*args, **kwargs)

            event.duration_ms = (time.perf_counter() - start_time) * 1000

            tokens = _extract_tokens_from_response(response)
            event.input_tokens = tokens["input_tokens"]
            event.output_tokens = tokens["output_tokens"]
            event.cached_tokens = tokens["cached_tokens"]
            event.total_tokens = event.input_tokens + event.output_tokens

            event.model = _extract_model_from_response(response, model)
            event.response_preview = _get_response_preview(response)
            event.status = "success"

            from ..pricing import calculate_cost

            event.cost_usd = calculate_cost(
                event.model, event.input_tokens, event.output_tokens, event.cached_tokens, "openai"
            )

            tracker.track(event)
            return response

        except Exception as e:
            event.duration_ms = (time.perf_counter() - start_time) * 1000
            event.status = "error"
            event.error_type = type(e).__name__
            event.error_message = str(e)[:1000]
            tracker.track(event)
            raise

    return wrapper


def _wrap_embeddings_create(original_method: Callable) -> Callable:
    """Wrap the embeddings.create method"""

    @functools.wraps(original_method)
    def wrapper(*args, **kwargs):
        tracker = get_tracker()
        start_time = time.perf_counter()

        model = kwargs.get("model", "")
        user = kwargs.get("user")

        event = LLMEvent.fast_construct(
            provider="openai",
            model=model,
            request_type="embedding",
            endpoint="/v1/embeddings",
            user_id=user,
        )

        # Apply attribution context
        _apply_attribution_context(event)

        try:
            response = original_method(*args, **kwargs)

            event.duration_ms = (time.perf_counter() - start_time) * 1000

            usage = getattr(response, "usage", None)
            if usage:
                event.input_tokens = getattr(usage, "prompt_tokens", 0) or 0
                event.total_tokens = getattr(usage, "total_tokens", 0) or 0

            event.model = _extract_model_from_response(response, model)
            event.status = "success"

            from ..pricing import calculate_cost

            event.cost_usd = calculate_cost(event.model, event.input_tokens, 0, 0, "openai")

            tracker.track(event)
            return response

        except Exception as e:
            event.duration_ms = (time.perf_counter() - start_time) * 1000
            event.status = "error"
            event.error_type = type(e).__name__
            event.error_message = str(e)[:1000]
            tracker.track(event)
            raise

    return wrapper


def _get_responses_request_preview(input_data: Any, max_length: int = 500) -> str | None:
    """Get a preview of the responses API input"""
    if not input_data:
        return None

    try:
        # Handle string input
        if isinstance(input_data, str):
            return input_data[:max_length] if len(input_data) > max_length else input_data

        # Handle list of messages
        if isinstance(input_data, list) and len(input_data) > 0:
            last_msg = input_data[-1]
            if isinstance(last_msg, dict):
                content = last_msg.get("content", "")
            else:
                content = getattr(last_msg, "content", "")

            if isinstance(content, str):
                return content[:max_length] if len(content) > max_length else content
    except Exception:
        pass

    return None


def _get_responses_response_preview(response: Any, max_length: int = 500) -> str | None:
    """Get a preview of the responses API response content"""
    try:
        # Responses API has output list with different structure
        output = getattr(response, "output", [])
        if output:
            for item in output:
                # Check for message type content
                if getattr(item, "type", None) == "message":
                    content = getattr(item, "content", [])
                    for block in content:
                        if getattr(block, "type", None) == "output_text":
                            text = getattr(block, "text", "")
                            if text:
                                return text[:max_length] if len(text) > max_length else text
    except Exception:
        pass

    return None


def _wrap_responses_create(original_method: Callable) -> Callable:
    """Wrap the responses.create method"""

    @functools.wraps(original_method)
    def wrapper(*args, **kwargs):
        tracker = get_tracker()
        start_time = time.perf_counter()

        # Extract request info
        model = kwargs.get("model", "")
        input_data = kwargs.get("input", "")
        user = kwargs.get("user")

        event = LLMEvent.fast_construct(
            provider="openai",
            model=model,
            request_type="chat",
            endpoint="/v1/responses",
            user_id=user,
            request_preview=_get_responses_request_preview(input_data),
        )

        # Apply attribution context
        _apply_attribution_context(event)

        try:
            response = original_method(*args, **kwargs)

            # Calculate duration
            event.duration_ms = (time.perf_counter() - start_time) * 1000

            # Extract token info (responses API uses input_tokens/output_tokens)
            tokens = _extract_tokens_from_responses_api(response)
            event.input_tokens = tokens["input_tokens"]
            event.output_tokens = tokens["output_tokens"]
            event.cached_tokens = tokens["cached_tokens"]
            event.total_tokens = event.input_tokens + event.output_tokens

            # Get actual model used
            event.model = _extract_model_from_response(response, model)

            # Get response preview
            event.response_preview = _get_responses_response_preview(response)

            event.status = "success"

            # Recalculate cost with actual model
            from ..pricing import calculate_cost

            event.cost_usd = calculate_cost(
                event.model, event.input_tokens, event.output_tokens, event.cached_tokens, "openai"
            )

            tracker.track(event)
            return response

        except Exception as e:
            event.duration_ms = (time.perf_counter() - start_time) * 1000
            event.status = "error"
            event.error_type = type(e).__name__
            event.error_message = str(e)[:1000]
            tracker.track(event)
            raise

    return wrapper


def _wrap_async_responses_create(original_method: Callable) -> Callable:
    """Wrap the async responses.create method"""

    @functools.wraps(original_method)
    async def wrapper(*args, **kwargs):
        tracker = get_tracker()
        start_time = time.perf_counter()

        model = kwargs.get("model", "")
        input_data = kwargs.get("input", "")
        user = kwargs.get("user")

        event = LLMEvent.fast_construct(
            provider="openai",
            model=model,
            request_type="chat",
            endpoint="/v1/responses",
            user_id=user,
            request_preview=_get_responses_request_preview(input_data),
        )

        # Apply attribution context
        _apply_attribution_context(event)

        try:
            response = await original_method(*args, **kwargs)

            event.duration_ms = (time.perf_counter() - start_time) * 1000

            tokens = _extract_tokens_from_responses_api(response)
            event.input_tokens = tokens["input_tokens"]
            event.output_tokens = tokens["output_tokens"]
            event.cached_tokens = tokens["cached_tokens"]
            event.total_tokens = event.input_tokens + event.output_tokens

            event.model = _extract_model_from_response(response, model)
            event.response_preview = _get_responses_response_preview(response)
            event.status = "success"

            from ..pricing import calculate_cost

            event.cost_usd = calculate_cost(
                event.model, event.input_tokens, event.output_tokens, event.cached_tokens, "openai"
            )

            tracker.track(event)
            return response

        except Exception as e:
            event.duration_ms = (time.perf_counter() - start_time) * 1000
            event.status = "error"
            event.error_type = type(e).__name__
            event.error_message = str(e)[:1000]
            tracker.track(event)
            raise

    return wrapper


# =============================================================================
# Audio API Wrappers
# =============================================================================


def _wrap_audio_transcription_create(original_method: Callable) -> Callable:
    """Wrap the audio.transcriptions.create method (billed per minute)"""

    @functools.wraps(original_method)
    def wrapper(*args, **kwargs):
        tracker = get_tracker()
        start_time = time.perf_counter()

        model = kwargs.get("model", "whisper-1")
        user = kwargs.get("user")

        event = LLMEvent.fast_construct(
            provider="openai",
            model=model,
            request_type="transcription",
            endpoint="/v1/audio/transcriptions",
            user_id=user,
        )

        _apply_attribution_context(event)

        try:
            response = original_method(*args, **kwargs)

            event.duration_ms = (time.perf_counter() - start_time) * 1000
            event.status = "success"

            # Get transcribed text as response preview
            if hasattr(response, "text"):
                text = response.text
                event.response_preview = text[:500] if len(text) > 500 else text

            # Audio transcription is billed per minute, cost calculated separately
            from ..pricing import calculate_audio_cost

            event.cost_usd = calculate_audio_cost(model)

            tracker.track(event)
            return response

        except Exception as e:
            event.duration_ms = (time.perf_counter() - start_time) * 1000
            event.status = "error"
            event.error_type = type(e).__name__
            event.error_message = str(e)[:1000]
            tracker.track(event)
            raise

    return wrapper


def _wrap_async_audio_transcription_create(original_method: Callable) -> Callable:
    """Wrap the async audio.transcriptions.create method"""

    @functools.wraps(original_method)
    async def wrapper(*args, **kwargs):
        tracker = get_tracker()
        start_time = time.perf_counter()

        model = kwargs.get("model", "whisper-1")
        user = kwargs.get("user")

        event = LLMEvent.fast_construct(
            provider="openai",
            model=model,
            request_type="transcription",
            endpoint="/v1/audio/transcriptions",
            user_id=user,
        )

        _apply_attribution_context(event)

        try:
            response = await original_method(*args, **kwargs)

            event.duration_ms = (time.perf_counter() - start_time) * 1000
            event.status = "success"

            if hasattr(response, "text"):
                text = response.text
                event.response_preview = text[:500] if len(text) > 500 else text

            from ..pricing import calculate_audio_cost

            event.cost_usd = calculate_audio_cost(model)

            tracker.track(event)
            return response

        except Exception as e:
            event.duration_ms = (time.perf_counter() - start_time) * 1000
            event.status = "error"
            event.error_type = type(e).__name__
            event.error_message = str(e)[:1000]
            tracker.track(event)
            raise

    return wrapper


def _wrap_audio_translation_create(original_method: Callable) -> Callable:
    """Wrap the audio.translations.create method (billed per minute)"""

    @functools.wraps(original_method)
    def wrapper(*args, **kwargs):
        tracker = get_tracker()
        start_time = time.perf_counter()

        model = kwargs.get("model", "whisper-1")
        user = kwargs.get("user")

        event = LLMEvent.fast_construct(
            provider="openai",
            model=model,
            request_type="translation",
            endpoint="/v1/audio/translations",
            user_id=user,
        )

        _apply_attribution_context(event)

        try:
            response = original_method(*args, **kwargs)

            event.duration_ms = (time.perf_counter() - start_time) * 1000
            event.status = "success"

            if hasattr(response, "text"):
                text = response.text
                event.response_preview = text[:500] if len(text) > 500 else text

            from ..pricing import calculate_audio_cost

            event.cost_usd = calculate_audio_cost(model)

            tracker.track(event)
            return response

        except Exception as e:
            event.duration_ms = (time.perf_counter() - start_time) * 1000
            event.status = "error"
            event.error_type = type(e).__name__
            event.error_message = str(e)[:1000]
            tracker.track(event)
            raise

    return wrapper


def _wrap_async_audio_translation_create(original_method: Callable) -> Callable:
    """Wrap the async audio.translations.create method"""

    @functools.wraps(original_method)
    async def wrapper(*args, **kwargs):
        tracker = get_tracker()
        start_time = time.perf_counter()

        model = kwargs.get("model", "whisper-1")
        user = kwargs.get("user")

        event = LLMEvent.fast_construct(
            provider="openai",
            model=model,
            request_type="translation",
            endpoint="/v1/audio/translations",
            user_id=user,
        )

        _apply_attribution_context(event)

        try:
            response = await original_method(*args, **kwargs)

            event.duration_ms = (time.perf_counter() - start_time) * 1000
            event.status = "success"

            if hasattr(response, "text"):
                text = response.text
                event.response_preview = text[:500] if len(text) > 500 else text

            from ..pricing import calculate_audio_cost

            event.cost_usd = calculate_audio_cost(model)

            tracker.track(event)
            return response

        except Exception as e:
            event.duration_ms = (time.perf_counter() - start_time) * 1000
            event.status = "error"
            event.error_type = type(e).__name__
            event.error_message = str(e)[:1000]
            tracker.track(event)
            raise

    return wrapper


def _wrap_audio_speech_create(original_method: Callable) -> Callable:
    """Wrap the audio.speech.create method (billed per character)"""

    @functools.wraps(original_method)
    def wrapper(*args, **kwargs):
        tracker = get_tracker()
        start_time = time.perf_counter()

        model = kwargs.get("model", "tts-1")
        input_text = kwargs.get("input", "")
        user = kwargs.get("user")

        event = LLMEvent.fast_construct(
            provider="openai",
            model=model,
            request_type="speech",
            endpoint="/v1/audio/speech",
            user_id=user,
            request_preview=input_text[:500] if len(input_text) > 500 else input_text,
        )

        _apply_attribution_context(event)

        try:
            response = original_method(*args, **kwargs)

            event.duration_ms = (time.perf_counter() - start_time) * 1000
            event.status = "success"

            # TTS is billed per character
            from ..pricing import calculate_tts_cost

            event.cost_usd = calculate_tts_cost(model, len(input_text))

            # Store character count in metadata
            event.metadata_extra = event.metadata_extra or {}
            event.metadata_extra["character_count"] = len(input_text)

            tracker.track(event)
            return response

        except Exception as e:
            event.duration_ms = (time.perf_counter() - start_time) * 1000
            event.status = "error"
            event.error_type = type(e).__name__
            event.error_message = str(e)[:1000]
            tracker.track(event)
            raise

    return wrapper


def _wrap_async_audio_speech_create(original_method: Callable) -> Callable:
    """Wrap the async audio.speech.create method"""

    @functools.wraps(original_method)
    async def wrapper(*args, **kwargs):
        tracker = get_tracker()
        start_time = time.perf_counter()

        model = kwargs.get("model", "tts-1")
        input_text = kwargs.get("input", "")
        user = kwargs.get("user")

        event = LLMEvent.fast_construct(
            provider="openai",
            model=model,
            request_type="speech",
            endpoint="/v1/audio/speech",
            user_id=user,
            request_preview=input_text[:500] if len(input_text) > 500 else input_text,
        )

        _apply_attribution_context(event)

        try:
            response = await original_method(*args, **kwargs)

            event.duration_ms = (time.perf_counter() - start_time) * 1000
            event.status = "success"

            from ..pricing import calculate_tts_cost

            event.cost_usd = calculate_tts_cost(model, len(input_text))

            event.metadata_extra = event.metadata_extra or {}
            event.metadata_extra["character_count"] = len(input_text)

            tracker.track(event)
            return response

        except Exception as e:
            event.duration_ms = (time.perf_counter() - start_time) * 1000
            event.status = "error"
            event.error_type = type(e).__name__
            event.error_message = str(e)[:1000]
            tracker.track(event)
            raise

    return wrapper


# =============================================================================
# Image API Wrappers
# =============================================================================


def _wrap_images_generate(original_method: Callable) -> Callable:
    """Wrap the images.generate method (billed per image)"""

    @functools.wraps(original_method)
    def wrapper(*args, **kwargs):
        tracker = get_tracker()
        start_time = time.perf_counter()

        model = kwargs.get("model", "dall-e-2")
        prompt = kwargs.get("prompt", "")
        n = kwargs.get("n", 1)
        size = kwargs.get("size", "1024x1024")
        quality = kwargs.get("quality", "standard")
        user = kwargs.get("user")

        event = LLMEvent.fast_construct(
            provider="openai",
            model=model,
            request_type="image_generation",
            endpoint="/v1/images/generations",
            user_id=user,
            request_preview=prompt[:500] if len(prompt) > 500 else prompt,
        )

        _apply_attribution_context(event)

        try:
            response = original_method(*args, **kwargs)

            event.duration_ms = (time.perf_counter() - start_time) * 1000
            event.status = "success"

            # Calculate image cost
            from ..pricing import calculate_image_cost

            event.cost_usd = calculate_image_cost(model, n, size, quality)

            # Store image params in metadata
            event.metadata_extra = event.metadata_extra or {}
            event.metadata_extra["image_count"] = n
            event.metadata_extra["image_size"] = size
            event.metadata_extra["image_quality"] = quality

            tracker.track(event)
            return response

        except Exception as e:
            event.duration_ms = (time.perf_counter() - start_time) * 1000
            event.status = "error"
            event.error_type = type(e).__name__
            event.error_message = str(e)[:1000]
            tracker.track(event)
            raise

    return wrapper


def _wrap_async_images_generate(original_method: Callable) -> Callable:
    """Wrap the async images.generate method"""

    @functools.wraps(original_method)
    async def wrapper(*args, **kwargs):
        tracker = get_tracker()
        start_time = time.perf_counter()

        model = kwargs.get("model", "dall-e-2")
        prompt = kwargs.get("prompt", "")
        n = kwargs.get("n", 1)
        size = kwargs.get("size", "1024x1024")
        quality = kwargs.get("quality", "standard")
        user = kwargs.get("user")

        event = LLMEvent.fast_construct(
            provider="openai",
            model=model,
            request_type="image_generation",
            endpoint="/v1/images/generations",
            user_id=user,
            request_preview=prompt[:500] if len(prompt) > 500 else prompt,
        )

        _apply_attribution_context(event)

        try:
            response = await original_method(*args, **kwargs)

            event.duration_ms = (time.perf_counter() - start_time) * 1000
            event.status = "success"

            from ..pricing import calculate_image_cost

            event.cost_usd = calculate_image_cost(model, n, size, quality)

            event.metadata_extra = event.metadata_extra or {}
            event.metadata_extra["image_count"] = n
            event.metadata_extra["image_size"] = size
            event.metadata_extra["image_quality"] = quality

            tracker.track(event)
            return response

        except Exception as e:
            event.duration_ms = (time.perf_counter() - start_time) * 1000
            event.status = "error"
            event.error_type = type(e).__name__
            event.error_message = str(e)[:1000]
            tracker.track(event)
            raise

    return wrapper


def _wrap_images_edit(original_method: Callable) -> Callable:
    """Wrap the images.edit method (billed per image)"""

    @functools.wraps(original_method)
    def wrapper(*args, **kwargs):
        tracker = get_tracker()
        start_time = time.perf_counter()

        model = kwargs.get("model", "dall-e-2")
        prompt = kwargs.get("prompt", "")
        n = kwargs.get("n", 1)
        size = kwargs.get("size", "1024x1024")
        user = kwargs.get("user")

        event = LLMEvent.fast_construct(
            provider="openai",
            model=model,
            request_type="image_edit",
            endpoint="/v1/images/edits",
            user_id=user,
            request_preview=prompt[:500] if len(prompt) > 500 else prompt,
        )

        _apply_attribution_context(event)

        try:
            response = original_method(*args, **kwargs)

            event.duration_ms = (time.perf_counter() - start_time) * 1000
            event.status = "success"

            from ..pricing import calculate_image_cost

            event.cost_usd = calculate_image_cost(model, n, size, "standard")

            event.metadata_extra = event.metadata_extra or {}
            event.metadata_extra["image_count"] = n
            event.metadata_extra["image_size"] = size

            tracker.track(event)
            return response

        except Exception as e:
            event.duration_ms = (time.perf_counter() - start_time) * 1000
            event.status = "error"
            event.error_type = type(e).__name__
            event.error_message = str(e)[:1000]
            tracker.track(event)
            raise

    return wrapper


def _wrap_async_images_edit(original_method: Callable) -> Callable:
    """Wrap the async images.edit method"""

    @functools.wraps(original_method)
    async def wrapper(*args, **kwargs):
        tracker = get_tracker()
        start_time = time.perf_counter()

        model = kwargs.get("model", "dall-e-2")
        prompt = kwargs.get("prompt", "")
        n = kwargs.get("n", 1)
        size = kwargs.get("size", "1024x1024")
        user = kwargs.get("user")

        event = LLMEvent.fast_construct(
            provider="openai",
            model=model,
            request_type="image_edit",
            endpoint="/v1/images/edits",
            user_id=user,
            request_preview=prompt[:500] if len(prompt) > 500 else prompt,
        )

        _apply_attribution_context(event)

        try:
            response = await original_method(*args, **kwargs)

            event.duration_ms = (time.perf_counter() - start_time) * 1000
            event.status = "success"

            from ..pricing import calculate_image_cost

            event.cost_usd = calculate_image_cost(model, n, size, "standard")

            event.metadata_extra = event.metadata_extra or {}
            event.metadata_extra["image_count"] = n
            event.metadata_extra["image_size"] = size

            tracker.track(event)
            return response

        except Exception as e:
            event.duration_ms = (time.perf_counter() - start_time) * 1000
            event.status = "error"
            event.error_type = type(e).__name__
            event.error_message = str(e)[:1000]
            tracker.track(event)
            raise

    return wrapper


def _wrap_images_create_variation(original_method: Callable) -> Callable:
    """Wrap the images.create_variation method (billed per image)"""

    @functools.wraps(original_method)
    def wrapper(*args, **kwargs):
        tracker = get_tracker()
        start_time = time.perf_counter()

        model = kwargs.get("model", "dall-e-2")
        n = kwargs.get("n", 1)
        size = kwargs.get("size", "1024x1024")
        user = kwargs.get("user")

        event = LLMEvent.fast_construct(
            provider="openai",
            model=model,
            request_type="image_variation",
            endpoint="/v1/images/variations",
            user_id=user,
        )

        _apply_attribution_context(event)

        try:
            response = original_method(*args, **kwargs)

            event.duration_ms = (time.perf_counter() - start_time) * 1000
            event.status = "success"

            from ..pricing import calculate_image_cost

            event.cost_usd = calculate_image_cost(model, n, size, "standard")

            event.metadata_extra = event.metadata_extra or {}
            event.metadata_extra["image_count"] = n
            event.metadata_extra["image_size"] = size

            tracker.track(event)
            return response

        except Exception as e:
            event.duration_ms = (time.perf_counter() - start_time) * 1000
            event.status = "error"
            event.error_type = type(e).__name__
            event.error_message = str(e)[:1000]
            tracker.track(event)
            raise

    return wrapper


def _wrap_async_images_create_variation(original_method: Callable) -> Callable:
    """Wrap the async images.create_variation method"""

    @functools.wraps(original_method)
    async def wrapper(*args, **kwargs):
        tracker = get_tracker()
        start_time = time.perf_counter()

        model = kwargs.get("model", "dall-e-2")
        n = kwargs.get("n", 1)
        size = kwargs.get("size", "1024x1024")
        user = kwargs.get("user")

        event = LLMEvent.fast_construct(
            provider="openai",
            model=model,
            request_type="image_variation",
            endpoint="/v1/images/variations",
            user_id=user,
        )

        _apply_attribution_context(event)

        try:
            response = await original_method(*args, **kwargs)

            event.duration_ms = (time.perf_counter() - start_time) * 1000
            event.status = "success"

            from ..pricing import calculate_image_cost

            event.cost_usd = calculate_image_cost(model, n, size, "standard")

            event.metadata_extra = event.metadata_extra or {}
            event.metadata_extra["image_count"] = n
            event.metadata_extra["image_size"] = size

            tracker.track(event)
            return response

        except Exception as e:
            event.duration_ms = (time.perf_counter() - start_time) * 1000
            event.status = "error"
            event.error_type = type(e).__name__
            event.error_message = str(e)[:1000]
            tracker.track(event)
            raise

    return wrapper


# =============================================================================
# Batch API Wrappers
# =============================================================================


def _wrap_batches_create(original_method: Callable) -> Callable:
    """Wrap the batches.create method to track batch job creation."""

    @functools.wraps(original_method)
    def wrapper(*args, **kwargs):
        tracker = get_tracker()
        start_time = time.perf_counter()

        # Extract batch info
        input_file_id = kwargs.get("input_file_id", "")
        endpoint = kwargs.get("endpoint", "")
        completion_window = kwargs.get("completion_window", "24h")
        metadata = kwargs.get("metadata", {})

        event = LLMEvent.fast_construct(
            provider="openai",
            model="batch",  # Batch doesn't have a model until processing
            request_type="batch_create",
            endpoint="/v1/batches",
        )

        _apply_attribution_context(event)

        try:
            response = original_method(*args, **kwargs)

            event.duration_ms = (time.perf_counter() - start_time) * 1000
            event.status = "success"

            # Store batch details in metadata
            event.metadata_extra = event.metadata_extra or {}
            event.metadata_extra["batch_id"] = getattr(response, "id", None)
            event.metadata_extra["input_file_id"] = input_file_id
            event.metadata_extra["batch_endpoint"] = endpoint
            event.metadata_extra["completion_window"] = completion_window
            event.metadata_extra["batch_status"] = getattr(response, "status", None)
            if metadata:
                event.metadata_extra["batch_metadata"] = metadata

            # No cost at creation - cost is incurred when batch completes
            # Batch processing gets 50% discount
            event.cost_usd = None

            tracker.track(event)
            return response

        except Exception as e:
            event.duration_ms = (time.perf_counter() - start_time) * 1000
            event.status = "error"
            event.error_type = type(e).__name__
            event.error_message = str(e)[:1000]
            tracker.track(event)
            raise

    return wrapper


def _wrap_async_batches_create(original_method: Callable) -> Callable:
    """Wrap the async batches.create method."""

    @functools.wraps(original_method)
    async def wrapper(*args, **kwargs):
        tracker = get_tracker()
        start_time = time.perf_counter()

        input_file_id = kwargs.get("input_file_id", "")
        endpoint = kwargs.get("endpoint", "")
        completion_window = kwargs.get("completion_window", "24h")
        metadata = kwargs.get("metadata", {})

        event = LLMEvent.fast_construct(
            provider="openai",
            model="batch",
            request_type="batch_create",
            endpoint="/v1/batches",
        )

        _apply_attribution_context(event)

        try:
            response = await original_method(*args, **kwargs)

            event.duration_ms = (time.perf_counter() - start_time) * 1000
            event.status = "success"

            event.metadata_extra = event.metadata_extra or {}
            event.metadata_extra["batch_id"] = getattr(response, "id", None)
            event.metadata_extra["input_file_id"] = input_file_id
            event.metadata_extra["batch_endpoint"] = endpoint
            event.metadata_extra["completion_window"] = completion_window
            event.metadata_extra["batch_status"] = getattr(response, "status", None)
            if metadata:
                event.metadata_extra["batch_metadata"] = metadata

            event.cost_usd = None

            tracker.track(event)
            return response

        except Exception as e:
            event.duration_ms = (time.perf_counter() - start_time) * 1000
            event.status = "error"
            event.error_type = type(e).__name__
            event.error_message = str(e)[:1000]
            tracker.track(event)
            raise

    return wrapper


# =============================================================================
# Patching Functions
# =============================================================================


def _patch_batch_apis() -> None:
    """Patch OpenAI batch APIs."""
    try:
        from openai.resources import batches

        _original_methods["batches_create"] = batches.Batches.create
        batches.Batches.create = _wrap_batches_create(batches.Batches.create)

        _original_methods["async_batches_create"] = batches.AsyncBatches.create
        batches.AsyncBatches.create = _wrap_async_batches_create(batches.AsyncBatches.create)

        logger.debug("OpenAI batch APIs patched for tracking")
    except (ImportError, AttributeError) as e:
        logger.debug(f"Could not patch batch APIs: {e}")


def _unpatch_batch_apis() -> None:
    """Unpatch OpenAI batch APIs."""
    try:
        from openai.resources import batches

        if "batches_create" in _original_methods:
            batches.Batches.create = _original_methods["batches_create"]

        if "async_batches_create" in _original_methods:
            batches.AsyncBatches.create = _original_methods["async_batches_create"]

    except (ImportError, AttributeError):
        pass


def _patch_audio_apis() -> None:
    """Patch OpenAI audio APIs (transcription, translation, speech)."""
    try:
        from openai.resources.audio import speech, transcriptions, translations

        # Transcriptions
        _original_methods["audio_transcriptions_create"] = transcriptions.Transcriptions.create
        transcriptions.Transcriptions.create = _wrap_audio_transcription_create(
            transcriptions.Transcriptions.create
        )

        _original_methods["async_audio_transcriptions_create"] = (
            transcriptions.AsyncTranscriptions.create
        )
        transcriptions.AsyncTranscriptions.create = _wrap_async_audio_transcription_create(
            transcriptions.AsyncTranscriptions.create
        )

        # Translations
        _original_methods["audio_translations_create"] = translations.Translations.create
        translations.Translations.create = _wrap_audio_translation_create(
            translations.Translations.create
        )

        _original_methods["async_audio_translations_create"] = translations.AsyncTranslations.create
        translations.AsyncTranslations.create = _wrap_async_audio_translation_create(
            translations.AsyncTranslations.create
        )

        # Patch speech synthesis endpoint
        _original_methods["audio_speech_create"] = speech.Speech.create
        speech.Speech.create = _wrap_audio_speech_create(speech.Speech.create)

        _original_methods["async_audio_speech_create"] = speech.AsyncSpeech.create
        speech.AsyncSpeech.create = _wrap_async_audio_speech_create(speech.AsyncSpeech.create)

        logger.debug("OpenAI audio APIs patched for tracking")
    except (ImportError, AttributeError) as e:
        logger.debug(f"Could not patch audio APIs: {e}")


def _patch_image_apis() -> None:
    """Patch OpenAI image APIs (generate, edit, create_variation)."""
    try:
        from openai.resources import images

        # Generate
        _original_methods["images_generate"] = images.Images.generate
        images.Images.generate = _wrap_images_generate(images.Images.generate)

        _original_methods["async_images_generate"] = images.AsyncImages.generate
        images.AsyncImages.generate = _wrap_async_images_generate(images.AsyncImages.generate)

        # Edit
        _original_methods["images_edit"] = images.Images.edit
        images.Images.edit = _wrap_images_edit(images.Images.edit)

        _original_methods["async_images_edit"] = images.AsyncImages.edit
        images.AsyncImages.edit = _wrap_async_images_edit(images.AsyncImages.edit)

        # Create variation
        _original_methods["images_create_variation"] = images.Images.create_variation
        images.Images.create_variation = _wrap_images_create_variation(
            images.Images.create_variation
        )

        _original_methods["async_images_create_variation"] = images.AsyncImages.create_variation
        images.AsyncImages.create_variation = _wrap_async_images_create_variation(
            images.AsyncImages.create_variation
        )

        logger.debug("OpenAI image APIs patched for tracking")
    except (ImportError, AttributeError) as e:
        logger.debug(f"Could not patch image APIs: {e}")


def _unpatch_audio_apis() -> None:
    """Unpatch OpenAI audio APIs."""
    try:
        from openai.resources.audio import speech, transcriptions, translations

        if "audio_transcriptions_create" in _original_methods:
            transcriptions.Transcriptions.create = _original_methods["audio_transcriptions_create"]
        if "async_audio_transcriptions_create" in _original_methods:
            transcriptions.AsyncTranscriptions.create = _original_methods[
                "async_audio_transcriptions_create"
            ]
        if "audio_translations_create" in _original_methods:
            translations.Translations.create = _original_methods["audio_translations_create"]
        if "async_audio_translations_create" in _original_methods:
            translations.AsyncTranslations.create = _original_methods[
                "async_audio_translations_create"
            ]
        if "audio_speech_create" in _original_methods:
            speech.Speech.create = _original_methods["audio_speech_create"]
        if "async_audio_speech_create" in _original_methods:
            speech.AsyncSpeech.create = _original_methods["async_audio_speech_create"]

    except (ImportError, AttributeError):
        pass


def _unpatch_image_apis() -> None:
    """Unpatch OpenAI image APIs."""
    try:
        from openai.resources import images

        if "images_generate" in _original_methods:
            images.Images.generate = _original_methods["images_generate"]
        if "async_images_generate" in _original_methods:
            images.AsyncImages.generate = _original_methods["async_images_generate"]
        if "images_edit" in _original_methods:
            images.Images.edit = _original_methods["images_edit"]
        if "async_images_edit" in _original_methods:
            images.AsyncImages.edit = _original_methods["async_images_edit"]
        if "images_create_variation" in _original_methods:
            images.Images.create_variation = _original_methods["images_create_variation"]
        if "async_images_create_variation" in _original_methods:
            images.AsyncImages.create_variation = _original_methods["async_images_create_variation"]

    except (ImportError, AttributeError):
        pass


def patch_openai(
    client: Any | None = None,
    track_embeddings: bool = True,
    track_audio: bool = True,
    track_images: bool = True,
    track_batch: bool = True,
) -> None:
    """
    Patch the OpenAI SDK to automatically track all API calls.

    Args:
        client: Optional specific OpenAI client instance to patch.
                If None, patches the default client class.
        track_embeddings: Whether to also track embedding calls
        track_audio: Whether to track audio API calls (transcription, translation, TTS)
        track_images: Whether to track image API calls (generate, edit, variations)
        track_batch: Whether to track batch API calls (50% discount batch processing)

    Example:
        >>> import openai
        >>> import tokenledger
        >>>
        >>> tokenledger.configure(database_url="postgresql://...")
        >>> tokenledger.patch_openai()
        >>>
        >>> # Now all calls are automatically tracked
        >>> response = openai.chat.completions.create(
        ...     model="gpt-4o",
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )
    """
    global _patched

    if _patched:
        logger.warning("OpenAI is already patched")
        return

    try:
        import openai  # noqa: F401
    except ImportError as err:
        raise ImportError("OpenAI SDK not installed. Run: pip install openai") from err

    if client is not None:
        # Patch specific client instance
        _original_methods["instance_chat_create"] = client.chat.completions.create
        client.chat.completions.create = _wrap_chat_completions_create(
            client.chat.completions.create
        )

        if track_embeddings and hasattr(client, "embeddings"):
            _original_methods["instance_embeddings_create"] = client.embeddings.create
            client.embeddings.create = _wrap_embeddings_create(client.embeddings.create)

        # Patch responses API on client instance if available (used by pydantic-ai)
        if hasattr(client, "responses"):
            _original_methods["instance_responses_create"] = client.responses.create
            client.responses.create = _wrap_responses_create(client.responses.create)
    else:
        # Patch the class methods
        from openai.resources.chat import completions as chat_completions

        # Sync chat completions
        _original_methods["chat_create"] = chat_completions.Completions.create
        chat_completions.Completions.create = _wrap_chat_completions_create(
            chat_completions.Completions.create
        )

        # Async chat completions
        _original_methods["async_chat_create"] = chat_completions.AsyncCompletions.create
        chat_completions.AsyncCompletions.create = _wrap_async_chat_completions_create(
            chat_completions.AsyncCompletions.create
        )

        if track_embeddings:
            from openai.resources import embeddings

            _original_methods["embeddings_create"] = embeddings.Embeddings.create
            embeddings.Embeddings.create = _wrap_embeddings_create(embeddings.Embeddings.create)

        # Patch responses API (used by pydantic-ai and other frameworks)
        try:
            from openai.resources import responses

            # Sync responses
            _original_methods["responses_create"] = responses.Responses.create
            responses.Responses.create = _wrap_responses_create(responses.Responses.create)

            # Async responses
            _original_methods["async_responses_create"] = responses.AsyncResponses.create
            responses.AsyncResponses.create = _wrap_async_responses_create(
                responses.AsyncResponses.create
            )

            logger.debug("OpenAI responses API patched for tracking")
        except (ImportError, AttributeError) as e:
            logger.debug(f"Could not patch responses API: {e}")

        # Patch audio APIs (transcription, translation, speech)
        if track_audio:
            _patch_audio_apis()

        # Patch image APIs (generate, edit, variations)
        if track_images:
            _patch_image_apis()

        # Patch batch APIs (batch processing with 50% discount)
        if track_batch:
            _patch_batch_apis()

    _patched = True
    logger.info("OpenAI SDK patched for tracking")


def unpatch_openai() -> None:
    """Remove the OpenAI SDK patches"""
    global _patched

    if not _patched:
        return

    try:
        import openai  # noqa: F401
        from openai.resources import embeddings
        from openai.resources.chat import completions as chat_completions

        if "chat_create" in _original_methods:
            chat_completions.Completions.create = _original_methods["chat_create"]

        if "async_chat_create" in _original_methods:
            chat_completions.AsyncCompletions.create = _original_methods["async_chat_create"]

        if "embeddings_create" in _original_methods:
            embeddings.Embeddings.create = _original_methods["embeddings_create"]

        # Unpatch responses API
        try:
            from openai.resources import responses

            if "responses_create" in _original_methods:
                responses.Responses.create = _original_methods["responses_create"]

            if "async_responses_create" in _original_methods:
                responses.AsyncResponses.create = _original_methods["async_responses_create"]

        except (ImportError, AttributeError):
            pass

        # Unpatch audio APIs
        _unpatch_audio_apis()

        # Unpatch image APIs
        _unpatch_image_apis()

        # Unpatch batch APIs
        _unpatch_batch_apis()

        _original_methods.clear()
        _patched = False
        logger.info("OpenAI SDK unpatched")

    except ImportError:
        pass
