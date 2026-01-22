"""
TokenLedger Attribution Context

Provides context-based attribution for LLM cost tracking using Python contextvars.
Attribution context flows automatically through async code and thread boundaries.
"""

from __future__ import annotations

import logging
import time
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("tokenledger.context")

# Context variable for attribution
_attribution_context: ContextVar[AttributionContext | None] = ContextVar(
    "tokenledger_attribution", default=None
)

# Track when context was last set/cleared for debugging
_last_context_set_time: ContextVar[float | None] = ContextVar(
    "tokenledger_last_context_set", default=None
)
_last_context_cleared_time: ContextVar[float | None] = ContextVar(
    "tokenledger_last_context_cleared", default=None
)


@dataclass
class AttributionContext:
    """
    Attribution context for LLM cost tracking.

    Contains all attribution fields that can be automatically applied
    to LLM events. Fields are optional and only non-None values will
    be applied to events.
    """

    # User identification
    user_id: str | None = None
    session_id: str | None = None
    organization_id: str | None = None

    # Attribution fields
    feature: str | None = None  # Feature/capability (e.g., "summarize", "chat", "search")
    page: str | None = None  # Page/route (e.g., "/dashboard", "/api/chat")
    component: str | None = None  # UI component (e.g., "ChatWidget", "SearchBar")
    team: str | None = None  # Team responsible (e.g., "platform", "ml", "product")
    project: str | None = None  # Project name (e.g., "api", "web-app")
    cost_center: str | None = None  # Billing code (e.g., "CC-001", "ENG-ML")

    # Extra metadata
    metadata_extra: dict[str, Any] = field(default_factory=dict)

    def merge(self, other: AttributionContext) -> AttributionContext:
        """
        Merge another context into this one.

        Values from `other` take precedence (override) for non-None values.
        metadata_extra dicts are merged with `other` values taking precedence.
        """
        merged_extra = {**self.metadata_extra, **other.metadata_extra}

        return AttributionContext(
            user_id=other.user_id if other.user_id is not None else self.user_id,
            session_id=other.session_id if other.session_id is not None else self.session_id,
            organization_id=other.organization_id
            if other.organization_id is not None
            else self.organization_id,
            feature=other.feature if other.feature is not None else self.feature,
            page=other.page if other.page is not None else self.page,
            component=other.component if other.component is not None else self.component,
            team=other.team if other.team is not None else self.team,
            project=other.project if other.project is not None else self.project,
            cost_center=other.cost_center if other.cost_center is not None else self.cost_center,
            metadata_extra=merged_extra,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with only non-None values."""
        result: dict[str, Any] = {}

        if self.user_id is not None:
            result["user_id"] = self.user_id
        if self.session_id is not None:
            result["session_id"] = self.session_id
        if self.organization_id is not None:
            result["organization_id"] = self.organization_id
        if self.feature is not None:
            result["feature"] = self.feature
        if self.page is not None:
            result["page"] = self.page
        if self.component is not None:
            result["component"] = self.component
        if self.team is not None:
            result["team"] = self.team
        if self.project is not None:
            result["project"] = self.project
        if self.cost_center is not None:
            result["cost_center"] = self.cost_center
        if self.metadata_extra:
            result["metadata_extra"] = self.metadata_extra

        return result


def get_attribution_context() -> AttributionContext | None:
    """
    Get the current attribution context.

    Returns:
        Current AttributionContext or None if not set
    """
    return _attribution_context.get()


def set_attribution_context(ctx: AttributionContext | None) -> Token[AttributionContext | None]:
    """
    Set the attribution context.

    Args:
        ctx: AttributionContext to set, or None to clear

    Returns:
        Token that can be used to reset the context
    """
    if ctx is not None:
        _last_context_set_time.set(time.monotonic())
    return _attribution_context.set(ctx)


def reset_attribution_context(token: Token[AttributionContext | None]) -> None:
    """
    Reset the attribution context to a previous state.

    Args:
        token: Token from a previous set_attribution_context call
    """
    _last_context_cleared_time.set(time.monotonic())
    _attribution_context.reset(token)


def clear_attribution() -> None:
    """
    Clear the current attribution context.

    Use this to explicitly clear the context when using persistent mode
    or when you no longer need attribution to apply to subsequent calls.

    Example:
        # Set persistent context for streaming
        with attribution(user_id="user123", persistent=True):
            response = await framework.stream(...)

        async for chunk in response:
            yield chunk

        # Explicitly clear when done
        tokenledger.clear_attribution()
    """
    _last_context_cleared_time.set(time.monotonic())
    _attribution_context.set(None)


def check_attribution_context_warning() -> None:
    """
    Check if attribution context may have been lost and log a warning.

    This is called internally when an LLM event is recorded with no
    attribution context. If a context was recently set and then cleared,
    this may indicate the streaming/lazy response issue.
    """
    ctx = get_attribution_context()
    if ctx is not None:
        return  # Context exists, no warning needed

    set_time = _last_context_set_time.get()
    cleared_time = _last_context_cleared_time.get()

    if set_time is not None and cleared_time is not None:
        # Context was set then cleared recently
        time_since_set = time.monotonic() - set_time
        time_since_cleared = time.monotonic() - cleared_time

        # If context was set recently (within 5 seconds) and then cleared,
        # this may indicate the streaming issue
        if time_since_set < 5.0 and time_since_cleared < time_since_set:
            logger.warning(
                "LLM event recorded with no attribution context. "
                f"A context was set {time_since_set:.1f}s ago but cleared {time_since_cleared:.1f}s ago. "
                "If using streaming/lazy responses, use `persistent=True` or "
                "`set_attribution_context()` directly. "
                "See: https://github.com/ged1182/TokenLedger#streaming-with-attribution"
            )


class attribution:
    """
    Context manager and decorator for setting attribution context.

    Can be used as a synchronous context manager:
        with attribution(user_id="user_123", feature="summarize"):
            response = openai.chat.completions.create(...)

    Can be used as an async context manager:
        async with attribution(user_id="user_123", feature="summarize"):
            response = await client.chat.completions.create(...)

    Can be used as a decorator:
        @attribution(feature="chat", team="ml")
        def handle_chat(message: str):
            return client.messages.create(...)

        @attribution(feature="async_chat", team="ml")
        async def handle_async_chat(message: str):
            return await client.messages.create(...)

    Nested usage merges contexts (inner values override outer):
        with attribution(team="platform", feature="api"):
            with attribution(user_id="user_123"):
                # Both team, feature, and user_id are set
                ...

    Persistent mode for streaming/lazy responses:
        Many LLM frameworks return lazy/streaming responses where the actual API
        call happens when you iterate over the response, not when you call the
        method. In this case, the context manager may exit before the API call:

        # PROBLEM: Context exits before stream is consumed
        async with attribution(user_id="user123"):
            response = await framework.stream(...)  # Returns lazy response
        # Context exits here!
        async for chunk in response:  # API call happens here, no context!
            yield chunk

        # SOLUTION: Use persistent=True
        async with attribution(user_id="user123", persistent=True):
            response = await framework.stream(...)
        async for chunk in response:  # Context still active!
            yield chunk
        clear_attribution()  # Explicitly clear when done

    Args:
        persistent: If True, the context will NOT be automatically cleared
            when the context manager exits. You must call clear_attribution()
            manually when done. Useful for streaming/lazy responses.

    Note: For async code, either `with` or `async with` will work correctly.
    The async form is provided for API consistency and to make async intent explicit.
    """

    def __init__(
        self,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        organization_id: str | None = None,
        feature: str | None = None,
        page: str | None = None,
        component: str | None = None,
        team: str | None = None,
        project: str | None = None,
        cost_center: str | None = None,
        persistent: bool = False,
        **metadata_extra: Any,
    ) -> None:
        self._context = AttributionContext(
            user_id=user_id,
            session_id=session_id,
            organization_id=organization_id,
            feature=feature,
            page=page,
            component=component,
            team=team,
            project=project,
            cost_center=cost_center,
            metadata_extra=metadata_extra,
        )
        self._token: Token[AttributionContext | None] | None = None
        self._persistent = persistent

    def _enter(self) -> AttributionContext:
        """Common enter logic for both sync and async context managers."""
        # Get existing context and merge if present
        existing = get_attribution_context()
        merged = existing.merge(self._context) if existing is not None else self._context

        self._token = set_attribution_context(merged)
        return merged

    def _exit(self) -> None:
        """Common exit logic for both sync and async context managers."""
        if self._persistent:
            # In persistent mode, don't clear the context automatically
            # User must call clear_attribution() when done
            return
        if self._token is not None:
            reset_attribution_context(self._token)
            self._token = None

    def __enter__(self) -> AttributionContext:
        return self._enter()

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        # Unused parameters required by context manager protocol
        del exc_type, exc_val, exc_tb
        self._exit()

    async def __aenter__(self) -> AttributionContext:
        """Async context manager entry. Works identically to sync version."""
        return self._enter()

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        """Async context manager exit. Works identically to sync version."""
        # Unused parameters required by context manager protocol
        del exc_type, exc_val, exc_tb
        self._exit()

    def __call__(self, func: Any) -> Any:
        """Support use as a decorator."""
        import functools
        import inspect

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                async with self:
                    return await func(*args, **kwargs)

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                with self:
                    return func(*args, **kwargs)

            return sync_wrapper
