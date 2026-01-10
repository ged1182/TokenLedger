"""
TokenLedger Middleware
Middleware for popular Python web frameworks.
"""

import logging

logger = logging.getLogger("tokenledger.middleware")


class FastAPIMiddleware:
    """
    FastAPI middleware for TokenLedger.

    Adds user context and request tracking to all LLM calls
    made during a request.

    Example:
        >>> from fastapi import FastAPI
        >>> from tokenledger.middleware import FastAPIMiddleware
        >>>
        >>> app = FastAPI()
        >>> app.add_middleware(FastAPIMiddleware)
    """

    def __init__(
        self,
        app,
        user_id_header: str = "X-User-ID",
        session_id_header: str = "X-Session-ID",
        org_id_header: str = "X-Organization-ID",
    ):
        self.app = app
        self.user_id_header = user_id_header
        self.session_id_header = session_id_header
        self.org_id_header = org_id_header

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        from .tracker import get_tracker

        # Extract headers
        headers = dict(scope.get("headers", []))

        user_id = headers.get(self.user_id_header.lower().encode(), b"").decode() or None

        session_id = headers.get(self.session_id_header.lower().encode(), b"").decode() or None

        org_id = headers.get(self.org_id_header.lower().encode(), b"").decode() or None

        # Store in context for use by tracker
        tracker = get_tracker()

        # Add endpoint info to default metadata temporarily
        path = scope.get("path", "")
        method = scope.get("method", "")

        original_metadata = tracker.config.default_metadata.copy()
        tracker.config.default_metadata.update(
            {
                "http_path": path,
                "http_method": method,
            }
        )

        if user_id:
            tracker.config.default_metadata["request_user_id"] = user_id
        if session_id:
            tracker.config.default_metadata["request_session_id"] = session_id
        if org_id:
            tracker.config.default_metadata["request_org_id"] = org_id

        try:
            await self.app(scope, receive, send)
        finally:
            # Restore original metadata
            tracker.config.default_metadata = original_metadata


class FlaskMiddleware:
    """
    Flask middleware/extension for TokenLedger.

    Example:
        >>> from flask import Flask
        >>> from tokenledger.middleware import FlaskMiddleware
        >>>
        >>> app = Flask(__name__)
        >>> TokenLedger(app)
    """

    def __init__(
        self,
        app=None,
        user_id_header: str = "X-User-ID",
        session_id_header: str = "X-Session-ID",
        org_id_header: str = "X-Organization-ID",
    ):
        self.user_id_header = user_id_header
        self.session_id_header = session_id_header
        self.org_id_header = org_id_header

        if app is not None:
            self.init_app(app)

    def init_app(self, app):
        """Initialize with Flask app"""

        @app.before_request
        def before_request():
            from flask import g, request

            from .tracker import get_tracker

            tracker = get_tracker()

            # Store original metadata
            g._tokenledger_original_metadata = tracker.config.default_metadata.copy()

            # Add request context
            tracker.config.default_metadata.update(
                {
                    "http_path": request.path,
                    "http_method": request.method,
                }
            )

            user_id = request.headers.get(self.user_id_header)
            if user_id:
                tracker.config.default_metadata["request_user_id"] = user_id

            session_id = request.headers.get(self.session_id_header)
            if session_id:
                tracker.config.default_metadata["request_session_id"] = session_id

            org_id = request.headers.get(self.org_id_header)
            if org_id:
                tracker.config.default_metadata["request_org_id"] = org_id

        @app.after_request
        def after_request(response):
            from flask import g

            from .tracker import get_tracker

            # Restore original metadata
            if hasattr(g, "_tokenledger_original_metadata"):
                tracker = get_tracker()
                tracker.config.default_metadata = g._tokenledger_original_metadata

            return response


# Alias for Flask
TokenLedger = FlaskMiddleware
