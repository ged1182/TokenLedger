"""Tests for the middleware module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestFastAPIMiddleware:
    """Tests for FastAPIMiddleware."""

    @pytest.fixture
    def mock_tracker(self):
        """Create a mock tracker."""
        tracker = MagicMock()
        tracker.config.default_metadata = {}
        return tracker

    @pytest.fixture
    def mock_app(self):
        """Create a mock ASGI app."""

        async def app(scope, receive, send):
            pass

        return app

    @pytest.mark.asyncio
    async def test_non_http_request_passes_through(self, mock_app) -> None:
        """Test that non-HTTP requests pass through unchanged."""
        from tokenledger.middleware import FastAPIMiddleware

        middleware = FastAPIMiddleware(mock_app)
        scope = {"type": "websocket"}

        # Non-HTTP should pass through without touching tracker
        await middleware(scope, None, None)

    @pytest.mark.asyncio
    async def test_http_request_sets_context(self, mock_tracker) -> None:
        """Test that HTTP requests set attribution context."""
        from tokenledger.middleware import FastAPIMiddleware

        app_called = False

        async def mock_app(scope, receive, send):
            nonlocal app_called
            app_called = True

        middleware = FastAPIMiddleware(mock_app)
        scope = {
            "type": "http",
            "path": "/api/test",
            "method": "POST",
            "headers": [
                (b"x-user-id", b"user123"),
                (b"x-session-id", b"session456"),
                (b"x-organization-id", b"org789"),
                (b"x-feature", b"chat"),
                (b"x-team", b"engineering"),
                (b"x-project", b"chatbot"),
                (b"x-cost-center", b"CC001"),
            ],
        }

        with (
            patch("tokenledger.tracker.get_tracker", return_value=mock_tracker),
            patch("tokenledger.middleware.set_attribution_context") as mock_set_ctx,
            patch("tokenledger.middleware.reset_attribution_context") as mock_reset_ctx,
        ):
            await middleware(scope, None, None)

            assert app_called
            mock_set_ctx.assert_called_once()
            mock_reset_ctx.assert_called_once()

            # Check the context was set with correct values
            ctx = mock_set_ctx.call_args[0][0]
            assert ctx.user_id == "user123"
            assert ctx.session_id == "session456"
            assert ctx.organization_id == "org789"
            assert ctx.feature == "chat"
            assert ctx.team == "engineering"
            assert ctx.project == "chatbot"
            assert ctx.cost_center == "CC001"
            assert ctx.page == "/api/test"

    @pytest.mark.asyncio
    async def test_http_request_updates_tracker_metadata(self, mock_tracker) -> None:
        """Test that HTTP requests update tracker metadata."""
        from tokenledger.middleware import FastAPIMiddleware

        async def mock_app(scope, receive, send):
            # Check metadata was set during request
            assert mock_tracker.config.default_metadata["http_path"] == "/api/test"
            assert mock_tracker.config.default_metadata["http_method"] == "GET"
            assert mock_tracker.config.default_metadata["request_user_id"] == "user123"

        middleware = FastAPIMiddleware(mock_app)
        scope = {
            "type": "http",
            "path": "/api/test",
            "method": "GET",
            "headers": [
                (b"x-user-id", b"user123"),
            ],
        }

        with (
            patch("tokenledger.tracker.get_tracker", return_value=mock_tracker),
            patch("tokenledger.middleware.set_attribution_context"),
            patch("tokenledger.middleware.reset_attribution_context"),
        ):
            await middleware(scope, None, None)

    @pytest.mark.asyncio
    async def test_http_request_restores_metadata_on_exception(self, mock_tracker) -> None:
        """Test that metadata is restored even when app raises exception."""
        from tokenledger.middleware import FastAPIMiddleware

        mock_tracker.config.default_metadata = {"original": "value"}

        async def mock_app(scope, receive, send):
            raise ValueError("Test error")

        middleware = FastAPIMiddleware(mock_app)
        scope = {
            "type": "http",
            "path": "/api/test",
            "method": "GET",
            "headers": [],
        }

        with (
            patch("tokenledger.tracker.get_tracker", return_value=mock_tracker),
            patch("tokenledger.middleware.set_attribution_context"),
            patch("tokenledger.middleware.reset_attribution_context") as mock_reset,
            pytest.raises(ValueError, match="Test error"),
        ):
            await middleware(scope, None, None)

        # Context should still be reset
        mock_reset.assert_called_once()

    @pytest.mark.asyncio
    async def test_custom_header_names(self, mock_tracker) -> None:
        """Test middleware with custom header names."""
        from tokenledger.middleware import FastAPIMiddleware

        async def mock_app(scope, receive, send):
            pass

        middleware = FastAPIMiddleware(
            mock_app,
            user_id_header="X-Custom-User",
            session_id_header="X-Custom-Session",
        )
        scope = {
            "type": "http",
            "path": "/api/test",
            "method": "GET",
            "headers": [
                (b"x-custom-user", b"customuser"),
                (b"x-custom-session", b"customsession"),
            ],
        }

        with (
            patch("tokenledger.tracker.get_tracker", return_value=mock_tracker),
            patch("tokenledger.middleware.set_attribution_context") as mock_set_ctx,
            patch("tokenledger.middleware.reset_attribution_context"),
        ):
            await middleware(scope, None, None)

            ctx = mock_set_ctx.call_args[0][0]
            assert ctx.user_id == "customuser"
            assert ctx.session_id == "customsession"

    @pytest.mark.asyncio
    async def test_missing_headers_use_none(self, mock_tracker) -> None:
        """Test that missing headers result in None values."""
        from tokenledger.middleware import FastAPIMiddleware

        async def mock_app(scope, receive, send):
            pass

        middleware = FastAPIMiddleware(mock_app)
        scope = {
            "type": "http",
            "path": "/api/test",
            "method": "GET",
            "headers": [],
        }

        with (
            patch("tokenledger.tracker.get_tracker", return_value=mock_tracker),
            patch("tokenledger.middleware.set_attribution_context") as mock_set_ctx,
            patch("tokenledger.middleware.reset_attribution_context"),
        ):
            await middleware(scope, None, None)

            ctx = mock_set_ctx.call_args[0][0]
            assert ctx.user_id is None
            assert ctx.session_id is None
            assert ctx.organization_id is None


class TestFlaskMiddleware:
    """Tests for FlaskMiddleware."""

    def test_init_without_app(self) -> None:
        """Test initialization without app."""
        from tokenledger.middleware import FlaskMiddleware

        middleware = FlaskMiddleware()
        assert middleware.user_id_header == "X-User-ID"
        assert middleware.session_id_header == "X-Session-ID"

    def test_init_with_custom_headers(self) -> None:
        """Test initialization with custom headers."""
        from tokenledger.middleware import FlaskMiddleware

        middleware = FlaskMiddleware(
            user_id_header="X-Custom-User",
            feature_header="X-Custom-Feature",
        )
        assert middleware.user_id_header == "X-Custom-User"
        assert middleware.feature_header == "X-Custom-Feature"

    def test_tokenledger_alias(self) -> None:
        """Test that TokenLedger is an alias for FlaskMiddleware."""
        from tokenledger.middleware import FlaskMiddleware, TokenLedger

        assert TokenLedger is FlaskMiddleware

    def test_init_with_app(self) -> None:
        """Test initialization with Flask app."""
        from unittest.mock import MagicMock

        from tokenledger.middleware import FlaskMiddleware

        mock_app = MagicMock()
        FlaskMiddleware(app=mock_app)

        # init_app should have been called
        assert mock_app.before_request.called
        assert mock_app.after_request.called

    def test_init_app_registers_handlers(self) -> None:
        """Test that init_app registers before_request and after_request handlers."""
        from unittest.mock import MagicMock

        from tokenledger.middleware import FlaskMiddleware

        mock_app = MagicMock()
        middleware = FlaskMiddleware()
        middleware.init_app(mock_app)

        # Handlers should be registered
        mock_app.before_request.assert_called_once()
        mock_app.after_request.assert_called_once()

    def test_flask_integration_with_mock_request(self) -> None:
        """Test Flask middleware with mock request context."""
        from unittest.mock import MagicMock

        from tokenledger.middleware import FlaskMiddleware

        # Create middleware
        mock_app = MagicMock()
        before_request_fn = None
        after_request_fn = None

        def capture_before_request(fn):
            nonlocal before_request_fn
            before_request_fn = fn

        def capture_after_request(fn):
            nonlocal after_request_fn
            after_request_fn = fn

        mock_app.before_request = capture_before_request
        mock_app.after_request = capture_after_request

        FlaskMiddleware(app=mock_app)

        # Now we have the handlers
        assert before_request_fn is not None
        assert after_request_fn is not None

    def test_flask_all_header_fields(self) -> None:
        """Test that all Flask header fields are correctly configured."""
        from tokenledger.middleware import FlaskMiddleware

        middleware = FlaskMiddleware(
            user_id_header="X-User",
            session_id_header="X-Session",
            org_id_header="X-Org",
            feature_header="X-Feature",
            team_header="X-Team",
            project_header="X-Project",
            cost_center_header="X-Cost",
        )

        assert middleware.user_id_header == "X-User"
        assert middleware.session_id_header == "X-Session"
        assert middleware.org_id_header == "X-Org"
        assert middleware.feature_header == "X-Feature"
        assert middleware.team_header == "X-Team"
        assert middleware.project_header == "X-Project"
        assert middleware.cost_center_header == "X-Cost"
