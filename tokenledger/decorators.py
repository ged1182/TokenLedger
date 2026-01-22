"""
TokenLedger Decorators
Easy-to-use decorators for tracking LLM calls.
"""

import functools
import inspect
import time
from collections.abc import Callable
from typing import Any

from .models import LLMEvent
from .tracker import get_tracker


def track_llm(
    model: str | None = None,
    provider: str | None = None,
    endpoint: str | None = None,
    user_id: str | None = None,
    metadata: dict[str, Any] | None = None,
):
    """
    Decorator to track LLM calls.

    Can be used on any function that makes LLM API calls.
    Automatically captures duration and any returned token counts.

    Args:
        model: Model name (can be extracted from response if not provided)
        provider: Provider name ('openai', 'anthropic', etc.)
        endpoint: API endpoint
        user_id: User identifier
        metadata: Additional metadata to include

    Example:
        >>> @track_llm(model="gpt-4o", provider="openai")
        ... def my_llm_function(prompt: str):
        ...     response = openai.chat.completions.create(...)
        ...     return response

        >>> @track_llm()  # Auto-detects from response
        ... async def my_async_function(prompt: str):
        ...     response = await client.messages.create(...)
        ...     return response
    """

    def decorator(func: Callable) -> Callable:
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                tracker = get_tracker()
                start_time = time.perf_counter()

                event = LLMEvent(
                    provider=provider or "unknown",
                    model=model or "unknown",
                    endpoint=endpoint,
                    user_id=user_id,
                    metadata=metadata or {},
                )

                try:
                    result = await func(*args, **kwargs)
                    event.duration_ms = (time.perf_counter() - start_time) * 1000

                    # Try to extract info from response
                    _extract_from_response(event, result, model, provider)

                    event.status = "success"
                    tracker.track(event)
                    return result

                except Exception as e:
                    event.duration_ms = (time.perf_counter() - start_time) * 1000
                    event.status = "error"
                    event.error_type = type(e).__name__
                    event.error_message = str(e)[:1000]
                    tracker.track(event)
                    raise

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                tracker = get_tracker()
                start_time = time.perf_counter()

                event = LLMEvent(
                    provider=provider or "unknown",
                    model=model or "unknown",
                    endpoint=endpoint,
                    user_id=user_id,
                    metadata=metadata or {},
                )

                try:
                    result = func(*args, **kwargs)
                    event.duration_ms = (time.perf_counter() - start_time) * 1000

                    _extract_from_response(event, result, model, provider)

                    event.status = "success"
                    tracker.track(event)
                    return result

                except Exception as e:
                    event.duration_ms = (time.perf_counter() - start_time) * 1000
                    event.status = "error"
                    event.error_type = type(e).__name__
                    event.error_message = str(e)[:1000]
                    tracker.track(event)
                    raise

            return sync_wrapper

    return decorator


def track_cost(
    input_tokens: int,
    output_tokens: int,
    model: str,
    provider: str = "auto",
    user_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Manually track a cost event.

    Use this when you need to log token usage that wasn't
    captured by automatic interception.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: Model name
        provider: Provider name
        user_id: User identifier
        metadata: Additional metadata

    Example:
        >>> # After a streaming response where you counted tokens
        >>> track_cost(
        ...     input_tokens=150,
        ...     output_tokens=500,
        ...     model="gpt-4o",
        ...     user_id="user_123"
        ... )
    """
    tracker = get_tracker()

    event = LLMEvent(
        provider=provider if provider != "auto" else _detect_provider(model),
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        user_id=user_id,
        metadata=metadata or {},
    )

    tracker.track(event)


def _detect_provider(model: str) -> str:
    """Detect provider from model name"""
    if model.startswith("gpt-") or model.startswith("o1") or model.startswith("text-embedding"):
        return "openai"
    elif model.startswith("claude"):
        return "anthropic"
    elif model.startswith("gemini"):
        return "google"
    elif "mistral" in model or "mixtral" in model:
        return "mistral"
    return "unknown"


def _extract_from_response(
    event: LLMEvent,
    response: Any,
    default_model: str | None,
    default_provider: str | None,
) -> None:
    """Try to extract token info from various response formats"""

    # Try OpenAI format
    usage = getattr(response, "usage", None)
    if usage:
        # OpenAI
        if hasattr(usage, "prompt_tokens"):
            event.input_tokens = getattr(usage, "prompt_tokens", 0) or 0
            event.output_tokens = getattr(usage, "completion_tokens", 0) or 0
        # Anthropic
        elif hasattr(usage, "input_tokens"):
            event.input_tokens = getattr(usage, "input_tokens", 0) or 0
            event.output_tokens = getattr(usage, "output_tokens", 0) or 0
            event.cached_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0

    event.total_tokens = event.input_tokens + event.output_tokens

    # Try to get model from response
    if hasattr(response, "model"):
        event.model = response.model
    elif default_model:
        event.model = default_model

    # Detect provider if not set
    if event.provider == "unknown" and event.model:
        event.provider = _detect_provider(event.model)
    elif default_provider:
        event.provider = default_provider

    # Calculate cost
    from .pricing import calculate_cost

    event.cost_usd = calculate_cost(
        event.model, event.input_tokens, event.output_tokens, event.cached_tokens, event.provider
    )
