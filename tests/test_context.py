"""Tests for TokenLedger attribution context."""

from __future__ import annotations

import asyncio

import pytest

from tokenledger.context import (
    AttributionContext,
    attribution,
    get_attribution_context,
    reset_attribution_context,
    set_attribution_context,
)


class TestAttributionContext:
    """Tests for AttributionContext dataclass."""

    def test_default_values(self) -> None:
        """Test that default values are None."""
        ctx = AttributionContext()

        assert ctx.user_id is None
        assert ctx.session_id is None
        assert ctx.organization_id is None
        assert ctx.feature is None
        assert ctx.team is None
        assert ctx.project is None
        assert ctx.cost_center is None
        assert ctx.metadata_extra == {}

    def test_set_values(self) -> None:
        """Test setting context values."""
        ctx = AttributionContext(
            user_id="user_123",
            feature="summarize",
            team="ml",
            cost_center="CC-001",
        )

        assert ctx.user_id == "user_123"
        assert ctx.feature == "summarize"
        assert ctx.team == "ml"
        assert ctx.cost_center == "CC-001"

    def test_merge_basic(self) -> None:
        """Test merging two contexts."""
        ctx1 = AttributionContext(user_id="user_123", team="platform")
        ctx2 = AttributionContext(feature="chat", team="ml")

        merged = ctx1.merge(ctx2)

        assert merged.user_id == "user_123"  # From ctx1
        assert merged.feature == "chat"  # From ctx2
        assert merged.team == "ml"  # ctx2 overrides

    def test_merge_metadata_extra(self) -> None:
        """Test merging metadata_extra dicts."""
        ctx1 = AttributionContext(metadata_extra={"key1": "value1"})
        ctx2 = AttributionContext(metadata_extra={"key2": "value2"})

        merged = ctx1.merge(ctx2)

        assert merged.metadata_extra == {"key1": "value1", "key2": "value2"}

    def test_merge_metadata_extra_override(self) -> None:
        """Test that ctx2 metadata_extra values override ctx1."""
        ctx1 = AttributionContext(metadata_extra={"key": "old"})
        ctx2 = AttributionContext(metadata_extra={"key": "new"})

        merged = ctx1.merge(ctx2)

        assert merged.metadata_extra == {"key": "new"}

    def test_to_dict_excludes_none(self) -> None:
        """Test that to_dict excludes None values."""
        ctx = AttributionContext(user_id="user_123", feature="chat")

        d = ctx.to_dict()

        assert d == {"user_id": "user_123", "feature": "chat"}
        assert "team" not in d
        assert "project" not in d


class TestContextVars:
    """Tests for context variable operations."""

    def test_get_context_default_none(self) -> None:
        """Test that default context is None."""
        # Reset any existing context first
        token = set_attribution_context(None)
        try:
            ctx = get_attribution_context()
            assert ctx is None
        finally:
            reset_attribution_context(token)

    def test_set_and_get_context(self) -> None:
        """Test setting and getting context."""
        ctx = AttributionContext(user_id="user_123")
        token = set_attribution_context(ctx)

        try:
            retrieved = get_attribution_context()
            assert retrieved is not None
            assert retrieved.user_id == "user_123"
        finally:
            reset_attribution_context(token)

    def test_reset_context(self) -> None:
        """Test resetting context to previous state."""
        original = get_attribution_context()

        ctx = AttributionContext(user_id="user_123")
        token = set_attribution_context(ctx)

        reset_attribution_context(token)

        assert get_attribution_context() == original


class TestAttributionContextManager:
    """Tests for attribution context manager."""

    def test_basic_usage(self) -> None:
        """Test basic context manager usage."""
        with attribution(user_id="user_123", feature="chat") as ctx:
            assert ctx.user_id == "user_123"
            assert ctx.feature == "chat"

            retrieved = get_attribution_context()
            assert retrieved is not None
            assert retrieved.user_id == "user_123"

    def test_context_cleanup(self) -> None:
        """Test that context is cleaned up after exit."""
        # Store original context
        original = get_attribution_context()

        with attribution(user_id="user_123"):
            pass  # Context should be set

        # Context should be restored
        assert get_attribution_context() == original

    def test_nested_contexts(self) -> None:
        """Test nested context managers merge properly."""
        with attribution(team="platform", feature="api") as outer:
            assert outer.team == "platform"
            assert outer.feature == "api"

            with attribution(user_id="user_123", team="ml") as inner:
                # Inner should have merged values
                assert inner.user_id == "user_123"
                assert inner.feature == "api"  # From outer
                assert inner.team == "ml"  # Overridden

            # After inner exits, should be back to outer
            ctx = get_attribution_context()
            assert ctx is not None
            assert ctx.team == "platform"
            assert ctx.feature == "api"

    def test_metadata_extra_kwargs(self) -> None:
        """Test that extra kwargs become metadata_extra."""
        with attribution(custom_field="value") as ctx:
            assert ctx.metadata_extra == {"custom_field": "value"}


class TestAttributionDecorator:
    """Tests for attribution used as a decorator."""

    def test_sync_function(self) -> None:
        """Test decorating a sync function."""
        @attribution(feature="test", team="platform")
        def my_func() -> str:
            ctx = get_attribution_context()
            assert ctx is not None
            return f"{ctx.feature}-{ctx.team}"

        result = my_func()
        assert result == "test-platform"

    def test_async_function(self) -> None:
        """Test decorating an async function."""
        @attribution(feature="async_test", team="ml")
        async def my_async_func() -> str:
            ctx = get_attribution_context()
            assert ctx is not None
            return f"{ctx.feature}-{ctx.team}"

        result = asyncio.run(my_async_func())
        assert result == "async_test-ml"

    def test_decorator_cleanup(self) -> None:
        """Test that decorator cleans up context after function."""
        original = get_attribution_context()

        @attribution(feature="test")
        def my_func() -> None:
            pass

        my_func()

        assert get_attribution_context() == original

    def test_decorator_with_exception(self) -> None:
        """Test that context is cleaned up even on exception."""
        original = get_attribution_context()

        @attribution(feature="test")
        def my_func() -> None:
            raise ValueError("test error")

        with pytest.raises(ValueError):
            my_func()

        # Context should still be cleaned up
        assert get_attribution_context() == original
