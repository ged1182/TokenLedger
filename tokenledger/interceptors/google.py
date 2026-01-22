"""
Google GenAI SDK Interceptor
Automatically tracks all Google Gemini API calls with zero code changes.
"""

import functools
import logging
import time
from collections.abc import Callable
from typing import Any

from ..context import check_attribution_context_warning, get_attribution_context
from ..models import LLMEvent
from ..tracker import get_tracker

logger = logging.getLogger("tokenledger.google")

# Store original methods for unpatching
_original_methods: dict[str, Callable] = {}
_patched = False


def _apply_attribution_context(event: LLMEvent) -> None:
    """Apply current attribution context to an event.

    This function reads the attribution context from the current async context
    and applies any set fields to the event. This is called at the start of
    each intercepted API call, before any async operations.

    Note: If attribution is not being captured, ensure that:
    1. You're using `with tokenledger.attribution(...)` or `async with tokenledger.attribution(...)`
    2. The LLM call happens within the same async task (not in a spawned task)
    3. No intermediate code is using thread pools or executors that don't propagate context
    4. For streaming/lazy responses, use `persistent=True` mode
    """
    ctx = get_attribution_context()
    if ctx is None:
        logger.debug("No attribution context found when applying to event")
        # Check if context was recently cleared (possible streaming issue)
        check_attribution_context_warning()
        return

    logger.debug(
        f"Applying attribution context: user_id={ctx.user_id}, "
        f"feature={ctx.feature}, page={ctx.page}"
    )

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
    """Extract token counts from Google GenAI response."""
    tokens = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
    }

    usage_metadata = getattr(response, "usage_metadata", None)
    if usage_metadata:
        tokens["input_tokens"] = getattr(usage_metadata, "prompt_token_count", 0) or 0
        tokens["output_tokens"] = getattr(usage_metadata, "candidates_token_count", 0) or 0
        tokens["cached_tokens"] = getattr(usage_metadata, "cached_content_token_count", 0) or 0

    return tokens


def _get_request_preview(contents: Any, max_length: int = 500) -> str | None:
    """Get a preview of the request contents."""
    if not contents:
        return None

    try:
        # Handle string input directly
        if isinstance(contents, str):
            return contents[:max_length] if len(contents) > max_length else contents

        # Handle list of content parts
        if isinstance(contents, list) and len(contents) > 0:
            last_content = contents[-1]

            # If it's a string, use directly
            if isinstance(last_content, str):
                return last_content[:max_length] if len(last_content) > max_length else last_content

            # If it has a text attribute (Content object)
            if hasattr(last_content, "text"):
                text = last_content.text
                return text[:max_length] if len(text) > max_length else text

            # If it has parts (Content with parts)
            parts = getattr(last_content, "parts", None)
            if parts and len(parts) > 0:
                for part in parts:
                    text = getattr(part, "text", None)
                    if text:
                        return text[:max_length] if len(text) > max_length else text

    except Exception:
        pass

    return None


def _get_response_preview(response: Any, max_length: int = 500) -> str | None:
    """Get a preview of the response content."""
    try:
        # Get text from candidates
        candidates = getattr(response, "candidates", [])
        if candidates:
            candidate = candidates[0]
            content = getattr(candidate, "content", None)
            if content:
                parts = getattr(content, "parts", [])
                for part in parts:
                    text = getattr(part, "text", None)
                    if text:
                        return text[:max_length] if len(text) > max_length else text

        # Fallback to text property if available
        text = getattr(response, "text", None)
        if text:
            return text[:max_length] if len(text) > max_length else text

    except Exception:
        pass

    return None


def _wrap_generate_content(original_method: Callable) -> Callable:
    """Wrap the models.generate_content method."""

    @functools.wraps(original_method)
    def wrapper(self, *, model: str, contents: Any, **kwargs):
        tracker = get_tracker()
        start_time = time.perf_counter()

        event = LLMEvent.fast_construct(
            provider="google",
            model=model,
            request_type="chat",
            endpoint="/v1/models/{model}:generateContent",
            request_preview=_get_request_preview(contents),
        )

        # Apply attribution context
        _apply_attribution_context(event)

        try:
            response = original_method(self, model=model, contents=contents, **kwargs)

            # Calculate duration
            event.duration_ms = (time.perf_counter() - start_time) * 1000

            # Extract token info
            tokens = _extract_tokens_from_response(response)
            event.input_tokens = tokens["input_tokens"]
            event.output_tokens = tokens["output_tokens"]
            event.cached_tokens = tokens["cached_tokens"]
            event.total_tokens = event.input_tokens + event.output_tokens

            # Get response preview
            event.response_preview = _get_response_preview(response)

            event.status = "success"

            # Calculate cost
            from ..pricing import calculate_cost

            event.cost_usd = calculate_cost(
                event.model,
                event.input_tokens,
                event.output_tokens,
                event.cached_tokens,
                "google",
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


def _wrap_async_generate_content(original_method: Callable) -> Callable:
    """Wrap the async models.generate_content method."""

    @functools.wraps(original_method)
    async def wrapper(self, *, model: str, contents: Any, **kwargs):
        tracker = get_tracker()
        start_time = time.perf_counter()

        event = LLMEvent.fast_construct(
            provider="google",
            model=model,
            request_type="chat",
            endpoint="/v1/models/{model}:generateContent",
            request_preview=_get_request_preview(contents),
        )

        # Apply attribution context
        _apply_attribution_context(event)

        try:
            response = await original_method(self, model=model, contents=contents, **kwargs)

            event.duration_ms = (time.perf_counter() - start_time) * 1000

            tokens = _extract_tokens_from_response(response)
            event.input_tokens = tokens["input_tokens"]
            event.output_tokens = tokens["output_tokens"]
            event.cached_tokens = tokens["cached_tokens"]
            event.total_tokens = event.input_tokens + event.output_tokens

            event.response_preview = _get_response_preview(response)
            event.status = "success"

            from ..pricing import calculate_cost

            event.cost_usd = calculate_cost(
                event.model,
                event.input_tokens,
                event.output_tokens,
                event.cached_tokens,
                "google",
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
# Streaming API Wrappers
# =============================================================================


def _wrap_generate_content_stream(original_method: Callable) -> Callable:
    """Wrap the models.generate_content_stream method for streaming responses."""

    @functools.wraps(original_method)
    def wrapper(self, *, model: str, contents: Any, **kwargs):
        tracker = get_tracker()
        start_time = time.perf_counter()

        event = LLMEvent.fast_construct(
            provider="google",
            model=model,
            request_type="chat_stream",
            endpoint="/v1/models/{model}:streamGenerateContent",
            request_preview=_get_request_preview(contents),
        )

        # Apply attribution context
        _apply_attribution_context(event)

        # Get the stream iterator
        stream_iterator = original_method(self, model=model, contents=contents, **kwargs)

        class TrackedStreamIterator:
            def __init__(self, stream, event, start_time, tracker):
                self._stream = stream
                self._event = event
                self._start_time = start_time
                self._tracker = tracker
                self._response_text: list[str] = []
                self._tracked = False

            def __iter__(self):
                return self

            def __next__(self):
                try:
                    chunk = next(self._stream)

                    # Extract text from chunk
                    text = _get_response_preview(chunk, max_length=10000)
                    if text:
                        self._response_text.append(text)

                    # Extract token counts from usage_metadata (usually in final chunk)
                    tokens = _extract_tokens_from_response(chunk)
                    if tokens["input_tokens"] > 0:
                        self._event.input_tokens = tokens["input_tokens"]
                    if tokens["output_tokens"] > 0:
                        self._event.output_tokens = tokens["output_tokens"]
                    if tokens["cached_tokens"] > 0:
                        self._event.cached_tokens = tokens["cached_tokens"]

                    return chunk
                except StopIteration:
                    self._finalize()
                    raise
                except Exception as e:
                    self._finalize_error(e)
                    raise

            def _finalize(self):
                """Finalize tracking when stream completes successfully."""
                if self._tracked:
                    return
                self._tracked = True

                self._event.duration_ms = (time.perf_counter() - self._start_time) * 1000
                self._event.total_tokens = (self._event.input_tokens or 0) + (
                    self._event.output_tokens or 0
                )
                self._event.status = "success"

                # Get response preview
                if self._response_text:
                    full_text = "".join(self._response_text)
                    self._event.response_preview = full_text[:500]

                # Calculate cost
                from ..pricing import calculate_cost

                self._event.cost_usd = calculate_cost(
                    self._event.model,
                    self._event.input_tokens or 0,
                    self._event.output_tokens or 0,
                    self._event.cached_tokens or 0,
                    "google",
                )

                self._tracker.track(self._event)

            def _finalize_error(self, error: Exception):
                """Finalize tracking when stream errors."""
                if self._tracked:
                    return
                self._tracked = True

                self._event.duration_ms = (time.perf_counter() - self._start_time) * 1000
                self._event.status = "error"
                self._event.error_type = type(error).__name__
                self._event.error_message = str(error)[:1000]
                self._tracker.track(self._event)

            # Proxy other methods to underlying stream
            def __getattr__(self, name):
                return getattr(self._stream, name)

        return TrackedStreamIterator(stream_iterator, event, start_time, tracker)

    return wrapper


def _wrap_async_generate_content_stream(original_method: Callable) -> Callable:
    """Wrap the async models.generate_content_stream method for streaming responses."""

    @functools.wraps(original_method)
    async def wrapper(self, *, model: str, contents: Any, **kwargs):
        tracker = get_tracker()
        start_time = time.perf_counter()

        event = LLMEvent.fast_construct(
            provider="google",
            model=model,
            request_type="chat_stream",
            endpoint="/v1/models/{model}:streamGenerateContent",
            request_preview=_get_request_preview(contents),
        )

        # Apply attribution context
        _apply_attribution_context(event)

        # Get the async stream iterator - must await since original_method is async
        stream_iterator = await original_method(self, model=model, contents=contents, **kwargs)

        class AsyncTrackedStreamIterator:
            def __init__(self, stream, event, start_time, tracker):
                self._stream = stream
                self._event = event
                self._start_time = start_time
                self._tracker = tracker
                self._response_text: list[str] = []
                self._tracked = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    chunk = await self._stream.__anext__()

                    # Extract text from chunk
                    text = _get_response_preview(chunk, max_length=10000)
                    if text:
                        self._response_text.append(text)

                    # Extract token counts from usage_metadata
                    tokens = _extract_tokens_from_response(chunk)
                    if tokens["input_tokens"] > 0:
                        self._event.input_tokens = tokens["input_tokens"]
                    if tokens["output_tokens"] > 0:
                        self._event.output_tokens = tokens["output_tokens"]
                    if tokens["cached_tokens"] > 0:
                        self._event.cached_tokens = tokens["cached_tokens"]

                    return chunk
                except StopAsyncIteration:
                    self._finalize()
                    raise
                except Exception as e:
                    self._finalize_error(e)
                    raise

            def _finalize(self):
                """Finalize tracking when stream completes successfully."""
                if self._tracked:
                    return
                self._tracked = True

                self._event.duration_ms = (time.perf_counter() - self._start_time) * 1000
                self._event.total_tokens = (self._event.input_tokens or 0) + (
                    self._event.output_tokens or 0
                )
                self._event.status = "success"

                if self._response_text:
                    full_text = "".join(self._response_text)
                    self._event.response_preview = full_text[:500]

                from ..pricing import calculate_cost

                self._event.cost_usd = calculate_cost(
                    self._event.model,
                    self._event.input_tokens or 0,
                    self._event.output_tokens or 0,
                    self._event.cached_tokens or 0,
                    "google",
                )

                self._tracker.track(self._event)

            def _finalize_error(self, error: Exception):
                """Finalize tracking when stream errors."""
                if self._tracked:
                    return
                self._tracked = True

                self._event.duration_ms = (time.perf_counter() - self._start_time) * 1000
                self._event.status = "error"
                self._event.error_type = type(error).__name__
                self._event.error_message = str(error)[:1000]
                self._tracker.track(self._event)

            # Proxy other methods to underlying stream
            def __getattr__(self, name):
                return getattr(self._stream, name)

        return AsyncTrackedStreamIterator(stream_iterator, event, start_time, tracker)

    return wrapper


# =============================================================================
# Embedding API Wrappers
# =============================================================================


def _extract_embedding_tokens(response: Any) -> int:
    """Extract token count from embedding response."""
    usage_metadata = getattr(response, "usage_metadata", None)
    if usage_metadata:
        return getattr(usage_metadata, "prompt_token_count", 0) or 0
    return 0


def _get_embedding_content_preview(content: Any, max_length: int = 500) -> str | None:
    """Get a preview of the embedding content."""
    if not content:
        return None

    try:
        if isinstance(content, str):
            return content[:max_length] if len(content) > max_length else content

        if isinstance(content, list) and len(content) > 0:
            first_item = content[0]
            if isinstance(first_item, str):
                return first_item[:max_length] if len(first_item) > max_length else first_item

    except Exception:
        pass

    return None


def _wrap_embed_content(original_method: Callable) -> Callable:
    """Wrap the models.embed_content method."""

    @functools.wraps(original_method)
    def wrapper(self, *, model: str, contents: Any, **kwargs):
        tracker = get_tracker()
        start_time = time.perf_counter()

        event = LLMEvent.fast_construct(
            provider="google",
            model=model,
            request_type="embedding",
            endpoint="/v1/models/{model}:embedContent",
            request_preview=_get_embedding_content_preview(contents),
        )

        # Apply attribution context
        _apply_attribution_context(event)

        try:
            response = original_method(self, model=model, contents=contents, **kwargs)

            event.duration_ms = (time.perf_counter() - start_time) * 1000
            event.input_tokens = _extract_embedding_tokens(response)
            event.total_tokens = event.input_tokens
            event.status = "success"

            from ..pricing import calculate_cost

            event.cost_usd = calculate_cost(event.model, event.input_tokens, 0, 0, "google")

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


def _wrap_async_embed_content(original_method: Callable) -> Callable:
    """Wrap the async models.embed_content method."""

    @functools.wraps(original_method)
    async def wrapper(self, *, model: str, contents: Any, **kwargs):
        tracker = get_tracker()
        start_time = time.perf_counter()

        event = LLMEvent.fast_construct(
            provider="google",
            model=model,
            request_type="embedding",
            endpoint="/v1/models/{model}:embedContent",
            request_preview=_get_embedding_content_preview(contents),
        )

        # Apply attribution context
        _apply_attribution_context(event)

        try:
            response = await original_method(self, model=model, contents=contents, **kwargs)

            event.duration_ms = (time.perf_counter() - start_time) * 1000
            event.input_tokens = _extract_embedding_tokens(response)
            event.total_tokens = event.input_tokens
            event.status = "success"

            from ..pricing import calculate_cost

            event.cost_usd = calculate_cost(event.model, event.input_tokens, 0, 0, "google")

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


def _patch_streaming_apis() -> None:
    """Patch Google streaming APIs (generate_content_stream)."""
    try:
        from google.genai import models

        # Sync generate_content_stream
        if hasattr(models.Models, "generate_content_stream"):
            _original_methods["generate_content_stream"] = models.Models.generate_content_stream
            models.Models.generate_content_stream = _wrap_generate_content_stream(
                models.Models.generate_content_stream
            )

        # Async generate_content_stream
        if hasattr(models.AsyncModels, "generate_content_stream"):
            _original_methods["async_generate_content_stream"] = (
                models.AsyncModels.generate_content_stream
            )
            models.AsyncModels.generate_content_stream = _wrap_async_generate_content_stream(
                models.AsyncModels.generate_content_stream
            )

        logger.debug("Google streaming APIs patched for tracking")
    except (ImportError, AttributeError) as e:
        logger.debug(f"Could not patch streaming APIs: {e}")


def _unpatch_streaming_apis() -> None:
    """Unpatch Google streaming APIs."""
    try:
        from google.genai import models

        if "generate_content_stream" in _original_methods:
            models.Models.generate_content_stream = _original_methods["generate_content_stream"]

        if "async_generate_content_stream" in _original_methods:
            models.AsyncModels.generate_content_stream = _original_methods[
                "async_generate_content_stream"
            ]

    except (ImportError, AttributeError):
        pass


def _patch_embedding_apis() -> None:
    """Patch Google embedding APIs (embed_content)."""
    try:
        from google.genai import models

        # Sync embed_content
        if hasattr(models.Models, "embed_content"):
            _original_methods["embed_content"] = models.Models.embed_content
            models.Models.embed_content = _wrap_embed_content(models.Models.embed_content)

        # Async embed_content
        if hasattr(models.AsyncModels, "embed_content"):
            _original_methods["async_embed_content"] = models.AsyncModels.embed_content
            models.AsyncModels.embed_content = _wrap_async_embed_content(
                models.AsyncModels.embed_content
            )

        logger.debug("Google embedding APIs patched for tracking")
    except (ImportError, AttributeError) as e:
        logger.debug(f"Could not patch embedding APIs: {e}")


def _unpatch_embedding_apis() -> None:
    """Unpatch Google embedding APIs."""
    try:
        from google.genai import models

        if "embed_content" in _original_methods:
            models.Models.embed_content = _original_methods["embed_content"]

        if "async_embed_content" in _original_methods:
            models.AsyncModels.embed_content = _original_methods["async_embed_content"]

    except (ImportError, AttributeError):
        pass


def patch_google(
    client: Any | None = None,
    track_embeddings: bool = True,
    track_streaming: bool = True,
) -> None:
    """
    Patch the Google GenAI SDK to automatically track all API calls.

    Args:
        client: Optional specific Google GenAI client instance to patch.
                If None, patches the default client class.
        track_embeddings: Whether to also track embedding calls
        track_streaming: Whether to track streaming calls (generate_content_stream)

    Example:
        >>> from google.genai import Client
        >>> import tokenledger
        >>>
        >>> tokenledger.configure(database_url="postgresql://...")
        >>> tokenledger.patch_google()
        >>>
        >>> # Now all calls are automatically tracked
        >>> client = Client(api_key="...")
        >>> response = client.models.generate_content(
        ...     model="gemini-2.0-flash",
        ...     contents="Hello!"
        ... )
    """
    global _patched

    if _patched:
        logger.warning("Google GenAI is already patched")
        return

    try:
        import google.genai  # noqa: F401
    except ImportError as err:
        raise ImportError("Google GenAI SDK not installed. Run: pip install google-genai") from err

    if client is not None:
        # Patch specific client instance
        _original_methods["instance_generate_content"] = client.models.generate_content
        client.models.generate_content = _wrap_generate_content(
            client.models.generate_content
        ).__get__(client.models, type(client.models))

        # Patch async client if available
        if hasattr(client, "aio") and hasattr(client.aio, "models"):
            _original_methods["instance_async_generate_content"] = (
                client.aio.models.generate_content
            )
            client.aio.models.generate_content = _wrap_async_generate_content(
                client.aio.models.generate_content
            ).__get__(client.aio.models, type(client.aio.models))

        # Patch streaming on client instance if available
        if track_streaming and hasattr(client.models, "generate_content_stream"):
            _original_methods["instance_generate_content_stream"] = (
                client.models.generate_content_stream
            )
            client.models.generate_content_stream = _wrap_generate_content_stream(
                client.models.generate_content_stream
            ).__get__(client.models, type(client.models))

            if hasattr(client, "aio") and hasattr(client.aio.models, "generate_content_stream"):
                _original_methods["instance_async_generate_content_stream"] = (
                    client.aio.models.generate_content_stream
                )
                client.aio.models.generate_content_stream = _wrap_async_generate_content_stream(
                    client.aio.models.generate_content_stream
                ).__get__(client.aio.models, type(client.aio.models))

        # Patch embeddings on client instance if available
        if track_embeddings and hasattr(client.models, "embed_content"):
            _original_methods["instance_embed_content"] = client.models.embed_content
            client.models.embed_content = _wrap_embed_content(client.models.embed_content).__get__(
                client.models, type(client.models)
            )

            if hasattr(client, "aio") and hasattr(client.aio.models, "embed_content"):
                _original_methods["instance_async_embed_content"] = client.aio.models.embed_content
                client.aio.models.embed_content = _wrap_async_embed_content(
                    client.aio.models.embed_content
                ).__get__(client.aio.models, type(client.aio.models))
    else:
        # Patch the class methods
        from google.genai import models

        # Sync generate_content
        _original_methods["generate_content"] = models.Models.generate_content
        models.Models.generate_content = _wrap_generate_content(models.Models.generate_content)

        # Async generate_content
        _original_methods["async_generate_content"] = models.AsyncModels.generate_content
        models.AsyncModels.generate_content = _wrap_async_generate_content(
            models.AsyncModels.generate_content
        )

        # Patch streaming APIs
        if track_streaming:
            _patch_streaming_apis()

        # Patch embedding APIs
        if track_embeddings:
            _patch_embedding_apis()

    _patched = True
    logger.info("Google GenAI SDK patched for tracking")


def unpatch_google() -> None:
    """Remove the Google GenAI SDK patches."""
    global _patched

    if not _patched:
        return

    try:
        from google.genai import models

        if "generate_content" in _original_methods:
            models.Models.generate_content = _original_methods["generate_content"]

        if "async_generate_content" in _original_methods:
            models.AsyncModels.generate_content = _original_methods["async_generate_content"]

        # Unpatch streaming APIs
        _unpatch_streaming_apis()

        # Unpatch embedding APIs
        _unpatch_embedding_apis()

        _original_methods.clear()
        _patched = False
        logger.info("Google GenAI SDK unpatched")

    except ImportError:
        pass
