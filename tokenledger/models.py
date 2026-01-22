"""
TokenLedger Pydantic Models

Provides validated data models for LLM events with fast construction paths.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .pricing import calculate_cost

logger = logging.getLogger("tokenledger.models")


def _generate_event_id() -> str:
    """Generate a new UUID for event_id."""
    return str(uuid.uuid4())


def _generate_timestamp() -> datetime:
    """Generate current timestamp."""
    return datetime.now(UTC)


class LLMEvent(BaseModel):
    """
    Represents a single LLM API call event.

    Use `LLMEvent.fast_construct()` for high-performance creation without validation.
    Use `create_event_safe()` for creation with silent fallback on validation errors.
    """

    model_config = ConfigDict(
        validate_assignment=False,
        extra="ignore",
    )

    # Identifiers - use default_factory for auto-generation
    event_id: str = Field(default_factory=_generate_event_id)
    trace_id: str | None = None
    span_id: str | None = None
    parent_span_id: str | None = None

    # Timing - use default_factory for auto-generation
    timestamp: datetime = Field(default_factory=_generate_timestamp)
    duration_ms: float | None = None

    # Provider & Model
    provider: str = ""
    model: str = ""

    # Token counts
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0

    # Cost
    cost_usd: float | None = None

    # Request details
    endpoint: str | None = None
    request_type: str = "chat"  # chat, completion, embedding, etc.

    # User & context
    user_id: str | None = None
    session_id: str | None = None
    organization_id: str | None = None

    # Application context
    app_name: str | None = None
    environment: str | None = None

    # Status
    status: str = "success"  # success, error, timeout
    error_type: str | None = None
    error_message: str | None = None

    # Custom metadata
    metadata: dict[str, Any] | None = None

    # Request/Response (optional, for debugging)
    request_preview: str | None = None  # First N chars of prompt
    response_preview: str | None = None  # First N chars of response

    # Attribution fields (first-class columns)
    feature: str | None = None  # Feature/capability (e.g., "summarize", "chat", "search")
    page: str | None = None  # Page/route (e.g., "/dashboard", "/api/chat")
    component: str | None = None  # UI component (e.g., "ChatWidget", "SearchBar")
    team: str | None = None  # Team responsible (e.g., "platform", "ml", "product")
    project: str | None = None  # Project name (e.g., "api", "web-app")
    cost_center: str | None = None  # Billing code (e.g., "CC-001", "ENG-ML")
    metadata_extra: dict[str, Any] | None = None  # JSONB overflow for custom data

    @model_validator(mode="after")
    def compute_derived_fields(self) -> LLMEvent:
        """Compute total_tokens and cost_usd after initialization."""
        # Calculate total tokens
        self.total_tokens = self.input_tokens + self.output_tokens

        # Calculate cost if not provided
        if self.cost_usd is None and self.model:
            self.cost_usd = calculate_cost(
                self.model, self.input_tokens, self.output_tokens, self.cached_tokens, self.provider
            )

        return self

    @classmethod
    def fast_construct(
        cls,
        *,
        event_id: str | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
        parent_span_id: str | None = None,
        timestamp: datetime | None = None,
        duration_ms: float | None = None,
        provider: str = "",
        model: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int | None = None,
        cached_tokens: int = 0,
        cost_usd: float | None = None,
        endpoint: str | None = None,
        request_type: str = "chat",
        user_id: str | None = None,
        session_id: str | None = None,
        organization_id: str | None = None,
        app_name: str | None = None,
        environment: str | None = None,
        status: str = "success",
        error_type: str | None = None,
        error_message: str | None = None,
        metadata: dict[str, Any] | None = None,
        request_preview: str | None = None,
        response_preview: str | None = None,
        feature: str | None = None,
        page: str | None = None,
        component: str | None = None,
        team: str | None = None,
        project: str | None = None,
        cost_center: str | None = None,
        metadata_extra: dict[str, Any] | None = None,
    ) -> LLMEvent:
        """
        Fast construction without validation.

        Use this in hot paths where performance matters and data is trusted.
        Derived fields (total_tokens, cost_usd) are computed if not provided.
        """
        # Compute defaults for required derived values
        final_event_id = event_id or str(uuid.uuid4())
        final_timestamp = timestamp or datetime.now(UTC)
        final_total_tokens = (
            total_tokens if total_tokens is not None else (input_tokens + output_tokens)
        )
        final_cost_usd = cost_usd
        if final_cost_usd is None and model:
            final_cost_usd = calculate_cost(
                model, input_tokens, output_tokens, cached_tokens, provider
            )

        return cls.model_construct(
            event_id=final_event_id,
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            timestamp=final_timestamp,
            duration_ms=duration_ms,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=final_total_tokens,
            cached_tokens=cached_tokens,
            cost_usd=final_cost_usd,
            endpoint=endpoint,
            request_type=request_type,
            user_id=user_id,
            session_id=session_id,
            organization_id=organization_id,
            app_name=app_name,
            environment=environment,
            status=status,
            error_type=error_type,
            error_message=error_message,
            metadata=metadata,
            request_preview=request_preview,
            response_preview=response_preview,
            feature=feature,
            page=page,
            component=component,
            team=team,
            project=project,
            cost_center=cost_center,
            metadata_extra=metadata_extra,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for database insertion."""
        d = self.model_dump()
        if d.get("timestamp"):
            d["timestamp"] = d["timestamp"].isoformat()
        if d.get("metadata") is not None:
            d["metadata"] = json.dumps(d["metadata"])
        if d.get("metadata_extra") is not None:
            d["metadata_extra"] = json.dumps(d["metadata_extra"])
        return d


def create_event_safe(
    *,
    debug: bool = False,
    **kwargs: Any,
) -> LLMEvent:
    """
    Create an LLMEvent with silent fallback on validation errors.

    In production, this never blocks - validation errors are logged (in debug mode)
    and a minimal event is returned using fast_construct.

    Args:
        debug: If True, log validation warnings
        **kwargs: Event fields

    Returns:
        LLMEvent instance (never raises)
    """
    try:
        return LLMEvent(**kwargs)
    except Exception as e:
        if debug:
            logger.warning(f"Event validation failed, using fallback: {e}")

        # Create minimal event with fast_construct
        return LLMEvent.fast_construct(
            provider=kwargs.get("provider", "unknown"),
            model=kwargs.get("model", "unknown"),
            input_tokens=kwargs.get("input_tokens", 0),
            output_tokens=kwargs.get("output_tokens", 0),
            cached_tokens=kwargs.get("cached_tokens", 0),
            status=kwargs.get("status", "success"),
            error_type=kwargs.get("error_type"),
            error_message=kwargs.get("error_message"),
        )
