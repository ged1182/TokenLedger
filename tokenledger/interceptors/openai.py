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
    """Extract token counts from OpenAI response"""
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


def patch_openai(
    client: Any | None = None,
    track_embeddings: bool = True,
) -> None:
    """
    Patch the OpenAI SDK to automatically track all API calls.

    Args:
        client: Optional specific OpenAI client instance to patch.
                If None, patches the default client class.
        track_embeddings: Whether to also track embedding calls

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

        _original_methods.clear()
        _patched = False
        logger.info("OpenAI SDK unpatched")

    except ImportError:
        pass
