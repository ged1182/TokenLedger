"""
Google GenAI SDK Interceptor
Automatically tracks all Google Gemini API calls with zero code changes.
"""

import functools
import logging
import time
from collections.abc import Callable
from typing import Any

from ..context import get_attribution_context
from ..models import LLMEvent
from ..tracker import get_tracker

logger = logging.getLogger("tokenledger.google")

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


def patch_google(
    client: Any | None = None,
) -> None:
    """
    Patch the Google GenAI SDK to automatically track all API calls.

    Args:
        client: Optional specific Google GenAI client instance to patch.
                If None, patches the default client class.

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

        _original_methods.clear()
        _patched = False
        logger.info("Google GenAI SDK unpatched")

    except ImportError:
        pass
