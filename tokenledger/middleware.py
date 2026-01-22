"""
TokenLedger Middleware
Middleware for popular Python web frameworks.
"""

import logging

from .context import AttributionContext, reset_attribution_context, set_attribution_context

logger = logging.getLogger("tokenledger.middleware")


class FastAPIMiddleware:
    """
    FastAPI middleware for TokenLedger.

    Adds user context and request tracking to all LLM calls
    made during a request. Sets attribution context from headers.

    Supported headers:
        - X-User-ID: User identifier
        - X-Session-ID: Session identifier
        - X-Organization-ID: Organization identifier
        - X-Feature: Feature/capability name
        - X-Team: Team responsible
        - X-Project: Project name
        - X-Cost-Center: Billing code

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
        feature_header: str = "X-Feature",
        team_header: str = "X-Team",
        project_header: str = "X-Project",
        cost_center_header: str = "X-Cost-Center",
    ):
        self.app = app
        self.user_id_header = user_id_header
        self.session_id_header = session_id_header
        self.org_id_header = org_id_header
        self.feature_header = feature_header
        self.team_header = team_header
        self.project_header = project_header
        self.cost_center_header = cost_center_header

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        from .tracker import get_tracker

        # Extract headers
        headers = dict(scope.get("headers", []))

        def get_header(name: str) -> str | None:
            return headers.get(name.lower().encode(), b"").decode() or None

        user_id = get_header(self.user_id_header)
        session_id = get_header(self.session_id_header)
        org_id = get_header(self.org_id_header)
        feature = get_header(self.feature_header)
        team = get_header(self.team_header)
        project = get_header(self.project_header)
        cost_center = get_header(self.cost_center_header)

        # Set attribution context
        path = scope.get("path", "")
        method = scope.get("method", "")

        ctx = AttributionContext(
            user_id=user_id,
            session_id=session_id,
            organization_id=org_id,
            feature=feature,
            page=path,
            team=team,
            project=project,
            cost_center=cost_center,
            metadata_extra={"http_method": method} if method else {},
        )

        token = set_attribution_context(ctx)

        # Also update tracker default metadata for backward compatibility
        tracker = get_tracker()
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
            # Restore original metadata and reset context
            tracker.config.default_metadata = original_metadata
            reset_attribution_context(token)


class FlaskMiddleware:
    """
    Flask middleware/extension for TokenLedger.

    Sets attribution context from request headers.

    Supported headers:
        - X-User-ID: User identifier
        - X-Session-ID: Session identifier
        - X-Organization-ID: Organization identifier
        - X-Feature: Feature/capability name
        - X-Team: Team responsible
        - X-Project: Project name
        - X-Cost-Center: Billing code

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
        feature_header: str = "X-Feature",
        team_header: str = "X-Team",
        project_header: str = "X-Project",
        cost_center_header: str = "X-Cost-Center",
    ):
        self.user_id_header = user_id_header
        self.session_id_header = session_id_header
        self.org_id_header = org_id_header
        self.feature_header = feature_header
        self.team_header = team_header
        self.project_header = project_header
        self.cost_center_header = cost_center_header

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

            # Extract headers
            user_id = request.headers.get(self.user_id_header)
            session_id = request.headers.get(self.session_id_header)
            org_id = request.headers.get(self.org_id_header)
            feature = request.headers.get(self.feature_header)
            team = request.headers.get(self.team_header)
            project = request.headers.get(self.project_header)
            cost_center = request.headers.get(self.cost_center_header)

            # Set attribution context
            ctx = AttributionContext(
                user_id=user_id,
                session_id=session_id,
                organization_id=org_id,
                feature=feature,
                page=request.path,
                team=team,
                project=project,
                cost_center=cost_center,
                metadata_extra={"http_method": request.method} if request.method else {},
            )

            g._tokenledger_context_token = set_attribution_context(ctx)

            # Add request context to metadata for backward compatibility
            tracker.config.default_metadata.update(
                {
                    "http_path": request.path,
                    "http_method": request.method,
                }
            )

            if user_id:
                tracker.config.default_metadata["request_user_id"] = user_id
            if session_id:
                tracker.config.default_metadata["request_session_id"] = session_id
            if org_id:
                tracker.config.default_metadata["request_org_id"] = org_id

        @app.after_request
        def after_request(response):
            from flask import g

            from .tracker import get_tracker

            # Restore original metadata and reset context
            if hasattr(g, "_tokenledger_original_metadata"):
                tracker = get_tracker()
                tracker.config.default_metadata = g._tokenledger_original_metadata

            if hasattr(g, "_tokenledger_context_token"):
                reset_attribution_context(g._tokenledger_context_token)

            return response


# Alias for Flask
TokenLedger = FlaskMiddleware
