"""
Anthropic SDK Interceptor
Automatically tracks all Anthropic API calls with zero code changes.
"""

import functools
import logging
import time
from collections.abc import Callable
from typing import Any

from ..context import get_attribution_context
from ..models import LLMEvent
from ..tracker import get_tracker

logger = logging.getLogger("tokenledger.anthropic")

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
    """Extract token counts from Anthropic response"""
    tokens = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
    }

    usage = getattr(response, "usage", None)
    if usage:
        tokens["input_tokens"] = getattr(usage, "input_tokens", 0) or 0
        tokens["output_tokens"] = getattr(usage, "output_tokens", 0) or 0

        # Handle cache read tokens (Anthropic prompt caching)
        tokens["cached_tokens"] = getattr(usage, "cache_read_input_tokens", 0) or 0

    return tokens


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
            elif isinstance(content, list):
                # Handle content blocks
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        return text[:max_length] if len(text) > max_length else text
    except Exception:
        pass

    return None


def _get_response_preview(response: Any, max_length: int = 500) -> str | None:
    """Get a preview of the response content"""
    try:
        content = getattr(response, "content", [])
        if content:
            for block in content:
                if getattr(block, "type", None) == "text":
                    text = getattr(block, "text", "")
                    if text:
                        return text[:max_length] if len(text) > max_length else text
    except Exception:
        pass

    return None


def _wrap_messages_create(original_method: Callable) -> Callable:
    """Wrap the messages.create method"""

    @functools.wraps(original_method)
    def wrapper(*args, **kwargs):
        tracker = get_tracker()
        start_time = time.perf_counter()

        # Extract request info
        model = kwargs.get("model", "")
        messages = kwargs.get("messages", [])

        # Extract metadata if passed
        metadata = kwargs.get("metadata", {})
        user_id = metadata.get("user_id") if isinstance(metadata, dict) else None

        event = LLMEvent.fast_construct(
            provider="anthropic",
            model=model,
            request_type="chat",
            endpoint="/v1/messages",
            user_id=user_id,
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
            event.model = getattr(response, "model", model)

            # Get response preview
            event.response_preview = _get_response_preview(response)

            # Check stop reason
            stop_reason = getattr(response, "stop_reason", None)
            if stop_reason == "error":
                event.status = "error"
            else:
                event.status = "success"

            # Recalculate cost with actual model
            from ..pricing import calculate_cost

            event.cost_usd = calculate_cost(
                event.model,
                event.input_tokens,
                event.output_tokens,
                event.cached_tokens,
                "anthropic",
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


def _wrap_async_messages_create(original_method: Callable) -> Callable:
    """Wrap the async messages.create method"""

    @functools.wraps(original_method)
    async def wrapper(*args, **kwargs):
        tracker = get_tracker()
        start_time = time.perf_counter()

        model = kwargs.get("model", "")
        messages = kwargs.get("messages", [])

        metadata = kwargs.get("metadata", {})
        user_id = metadata.get("user_id") if isinstance(metadata, dict) else None

        event = LLMEvent.fast_construct(
            provider="anthropic",
            model=model,
            request_type="chat",
            endpoint="/v1/messages",
            user_id=user_id,
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

            event.model = getattr(response, "model", model)
            event.response_preview = _get_response_preview(response)

            stop_reason = getattr(response, "stop_reason", None)
            event.status = "error" if stop_reason == "error" else "success"

            from ..pricing import calculate_cost

            event.cost_usd = calculate_cost(
                event.model,
                event.input_tokens,
                event.output_tokens,
                event.cached_tokens,
                "anthropic",
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


def _wrap_streaming_messages(original_method: Callable) -> Callable:
    """Wrap the messages.stream method for streaming responses"""

    @functools.wraps(original_method)
    def wrapper(*args, **kwargs):
        tracker = get_tracker()
        start_time = time.perf_counter()

        model = kwargs.get("model", "")
        messages = kwargs.get("messages", [])

        metadata = kwargs.get("metadata", {})
        user_id = metadata.get("user_id") if isinstance(metadata, dict) else None

        event = LLMEvent.fast_construct(
            provider="anthropic",
            model=model,
            request_type="chat_stream",
            endpoint="/v1/messages",
            user_id=user_id,
            request_preview=_get_request_preview(messages),
        )

        # Apply attribution context
        _apply_attribution_context(event)

        # Get the stream context manager
        stream_context = original_method(*args, **kwargs)

        class TrackedStream:
            def __init__(self, ctx, event, start_time, tracker):
                self._ctx = ctx
                self._event = event
                self._start_time = start_time
                self._tracker = tracker
                self._stream = None
                self._response_text = []

            def __enter__(self):
                self._stream = self._ctx.__enter__()
                return TrackedStreamIterator(
                    self._stream, self._event, self._start_time, self._tracker, self._response_text
                )

            def __exit__(self, exc_type, exc_val, exc_tb):
                result = self._ctx.__exit__(exc_type, exc_val, exc_tb)

                # Finalize event after stream completes
                self._event.duration_ms = (time.perf_counter() - self._start_time) * 1000

                if exc_type:
                    self._event.status = "error"
                    self._event.error_type = exc_type.__name__
                    self._event.error_message = str(exc_val)[:1000]
                else:
                    self._event.status = "success"

                # Get response preview
                if self._response_text:
                    full_text = "".join(self._response_text)
                    self._event.response_preview = full_text[:500]

                self._tracker.track(self._event)
                return result

        class TrackedStreamIterator:
            def __init__(self, stream, event, start_time, tracker, response_text):
                self._stream = stream
                self._event = event
                self._start_time = start_time
                self._tracker = tracker
                self._response_text = response_text

            def __iter__(self):
                return self

            def __next__(self):
                try:
                    chunk = next(self._stream)

                    # Extract text from chunk
                    if hasattr(chunk, "type"):
                        if chunk.type == "content_block_delta":
                            delta = getattr(chunk, "delta", None)
                            if delta and hasattr(delta, "text"):
                                self._response_text.append(delta.text)
                        elif chunk.type == "message_delta":
                            usage = getattr(chunk, "usage", None)
                            if usage:
                                self._event.output_tokens = getattr(usage, "output_tokens", 0)
                        elif chunk.type == "message_start":
                            message = getattr(chunk, "message", None)
                            if message:
                                usage = getattr(message, "usage", None)
                                if usage:
                                    self._event.input_tokens = getattr(usage, "input_tokens", 0)
                                self._event.model = getattr(message, "model", self._event.model)

                    return chunk
                except StopIteration:
                    # Calculate cost
                    from ..pricing import calculate_cost

                    self._event.cost_usd = calculate_cost(
                        self._event.model,
                        self._event.input_tokens,
                        self._event.output_tokens,
                        self._event.cached_tokens,
                        "anthropic",
                    )
                    raise

            # Proxy other methods to underlying stream
            def __getattr__(self, name):
                return getattr(self._stream, name)

        return TrackedStream(stream_context, event, start_time, tracker)

    return wrapper


def _wrap_async_streaming_messages(original_method: Callable) -> Callable:
    """Wrap the async messages.stream method for streaming responses"""

    @functools.wraps(original_method)
    async def wrapper(*args, **kwargs):
        tracker = get_tracker()
        start_time = time.perf_counter()

        model = kwargs.get("model", "")
        messages = kwargs.get("messages", [])

        metadata = kwargs.get("metadata", {})
        user_id = metadata.get("user_id") if isinstance(metadata, dict) else None

        event = LLMEvent.fast_construct(
            provider="anthropic",
            model=model,
            request_type="chat_stream",
            endpoint="/v1/messages",
            user_id=user_id,
            request_preview=_get_request_preview(messages),
        )

        # Apply attribution context
        _apply_attribution_context(event)

        # Get the async stream context manager
        stream_context = original_method(*args, **kwargs)

        class AsyncTrackedStream:
            def __init__(self, ctx, event, start_time, tracker):
                self._ctx = ctx
                self._event = event
                self._start_time = start_time
                self._tracker = tracker
                self._stream = None
                self._response_text = []

            async def __aenter__(self):
                self._stream = await self._ctx.__aenter__()
                return AsyncTrackedStreamIterator(
                    self._stream, self._event, self._start_time, self._tracker, self._response_text
                )

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                result = await self._ctx.__aexit__(exc_type, exc_val, exc_tb)

                # Finalize event after stream completes
                self._event.duration_ms = (time.perf_counter() - self._start_time) * 1000

                if exc_type:
                    self._event.status = "error"
                    self._event.error_type = exc_type.__name__
                    self._event.error_message = str(exc_val)[:1000]
                else:
                    self._event.status = "success"

                # Get response preview
                if self._response_text:
                    full_text = "".join(self._response_text)
                    self._event.response_preview = full_text[:500]

                self._tracker.track(self._event)
                return result

        class AsyncTrackedStreamIterator:
            def __init__(self, stream, event, start_time, tracker, response_text):
                self._stream = stream
                self._event = event
                self._start_time = start_time
                self._tracker = tracker
                self._response_text = response_text

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    chunk = await self._stream.__anext__()

                    # Extract text from chunk
                    if hasattr(chunk, "type"):
                        if chunk.type == "content_block_delta":
                            delta = getattr(chunk, "delta", None)
                            if delta and hasattr(delta, "text"):
                                self._response_text.append(delta.text)
                        elif chunk.type == "message_delta":
                            usage = getattr(chunk, "usage", None)
                            if usage:
                                self._event.output_tokens = getattr(usage, "output_tokens", 0)
                        elif chunk.type == "message_start":
                            message = getattr(chunk, "message", None)
                            if message:
                                usage = getattr(message, "usage", None)
                                if usage:
                                    self._event.input_tokens = getattr(usage, "input_tokens", 0)
                                self._event.model = getattr(message, "model", self._event.model)

                    return chunk
                except StopAsyncIteration:
                    # Calculate cost
                    from ..pricing import calculate_cost

                    self._event.cost_usd = calculate_cost(
                        self._event.model,
                        self._event.input_tokens,
                        self._event.output_tokens,
                        self._event.cached_tokens,
                        "anthropic",
                    )
                    raise

            # Proxy other methods to underlying stream
            def __getattr__(self, name):
                return getattr(self._stream, name)

        return AsyncTrackedStream(stream_context, event, start_time, tracker)

    return wrapper


def _patch_beta_messages(track_streaming: bool) -> None:
    """Patch beta.messages API (used by pydantic-ai and other frameworks)."""
    try:
        from anthropic.resources.beta import messages as beta_messages

        # Sync beta messages
        _original_methods["beta_messages_create"] = beta_messages.Messages.create
        beta_messages.Messages.create = _wrap_messages_create(beta_messages.Messages.create)

        # Async beta messages
        _original_methods["beta_async_messages_create"] = beta_messages.AsyncMessages.create
        beta_messages.AsyncMessages.create = _wrap_async_messages_create(
            beta_messages.AsyncMessages.create
        )

        if track_streaming:
            if hasattr(beta_messages.Messages, "stream"):
                _original_methods["beta_messages_stream"] = beta_messages.Messages.stream
                beta_messages.Messages.stream = _wrap_streaming_messages(
                    beta_messages.Messages.stream
                )
            if hasattr(beta_messages.AsyncMessages, "stream"):
                _original_methods["beta_async_messages_stream"] = beta_messages.AsyncMessages.stream
                beta_messages.AsyncMessages.stream = _wrap_async_streaming_messages(
                    beta_messages.AsyncMessages.stream
                )

        logger.debug("Anthropic beta.messages patched for tracking")
    except (ImportError, AttributeError) as e:
        logger.debug(f"Could not patch beta.messages: {e}")


def patch_anthropic(
    client: Any | None = None,
    track_streaming: bool = True,
) -> None:
    """
    Patch the Anthropic SDK to automatically track all API calls.

    Args:
        client: Optional specific Anthropic client instance to patch.
                If None, patches the default client class.
        track_streaming: Whether to also track streaming calls

    Example:
        >>> import anthropic
        >>> import tokenledger
        >>>
        >>> tokenledger.configure(database_url="postgresql://...")
        >>> tokenledger.patch_anthropic()
        >>>
        >>> # Now all calls are automatically tracked
        >>> client = anthropic.Anthropic()
        >>> response = client.messages.create(
        ...     model="claude-3-5-sonnet-20241022",
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )
    """
    global _patched

    if _patched:
        logger.warning("Anthropic is already patched")
        return

    try:
        import anthropic  # noqa: F401
    except ImportError as err:
        raise ImportError("Anthropic SDK not installed. Run: pip install anthropic") from err

    if client is not None:
        # Patch specific client instance
        _original_methods["instance_messages_create"] = client.messages.create
        client.messages.create = _wrap_messages_create(client.messages.create)

        if track_streaming and hasattr(client.messages, "stream"):
            _original_methods["instance_messages_stream"] = client.messages.stream
            client.messages.stream = _wrap_streaming_messages(client.messages.stream)

        # Patch beta.messages on client instance if available
        if hasattr(client, "beta") and hasattr(client.beta, "messages"):
            _original_methods["instance_beta_messages_create"] = client.beta.messages.create
            client.beta.messages.create = _wrap_messages_create(client.beta.messages.create)

            if track_streaming and hasattr(client.beta.messages, "stream"):
                _original_methods["instance_beta_messages_stream"] = client.beta.messages.stream
                client.beta.messages.stream = _wrap_streaming_messages(client.beta.messages.stream)
    else:
        # Patch the class methods
        from anthropic.resources import messages

        # Sync messages
        _original_methods["messages_create"] = messages.Messages.create
        messages.Messages.create = _wrap_messages_create(messages.Messages.create)

        # Async messages
        _original_methods["async_messages_create"] = messages.AsyncMessages.create
        messages.AsyncMessages.create = _wrap_async_messages_create(messages.AsyncMessages.create)

        if track_streaming and hasattr(messages.Messages, "stream"):
            _original_methods["messages_stream"] = messages.Messages.stream
            messages.Messages.stream = _wrap_streaming_messages(messages.Messages.stream)

        # Patch beta.messages (used by pydantic-ai and other frameworks)
        _patch_beta_messages(track_streaming)

    _patched = True
    logger.info("Anthropic SDK patched for tracking")


def unpatch_anthropic() -> None:
    """Remove the Anthropic SDK patches"""
    global _patched

    if not _patched:
        return

    try:
        from anthropic.resources import messages

        if "messages_create" in _original_methods:
            messages.Messages.create = _original_methods["messages_create"]

        if "async_messages_create" in _original_methods:
            messages.AsyncMessages.create = _original_methods["async_messages_create"]

        if "messages_stream" in _original_methods:
            messages.Messages.stream = _original_methods["messages_stream"]

        # Unpatch beta.messages
        try:
            from anthropic.resources.beta import messages as beta_messages

            if "beta_messages_create" in _original_methods:
                beta_messages.Messages.create = _original_methods["beta_messages_create"]

            if "beta_async_messages_create" in _original_methods:
                beta_messages.AsyncMessages.create = _original_methods["beta_async_messages_create"]

            if "beta_messages_stream" in _original_methods:
                beta_messages.Messages.stream = _original_methods["beta_messages_stream"]

            if "beta_async_messages_stream" in _original_methods:
                beta_messages.AsyncMessages.stream = _original_methods["beta_async_messages_stream"]

        except (ImportError, AttributeError):
            pass

        _original_methods.clear()
        _patched = False
        logger.info("Anthropic SDK unpatched")

    except ImportError:
        pass
