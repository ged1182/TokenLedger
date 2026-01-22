"""
TokenLedger - LLM Cost Analytics for Postgres
Know exactly what your AI features cost, per user, per endpoint, per day.
"""

__version__ = "0.1.0"

from .config import configure, get_config
from .context import (
    AttributionContext,
    attribution,
    get_attribution_context,
    reset_attribution_context,
    set_attribution_context,
)
from .decorators import track_cost, track_llm
from .interceptors.anthropic import patch_anthropic, unpatch_anthropic
from .interceptors.google import patch_google, unpatch_google
from .interceptors.openai import patch_openai, unpatch_openai
from .models import LLMEvent, create_event_safe
from .tracker import (
    AsyncTokenTracker,
    TokenTracker,
    get_async_tracker,
    get_tracker,
    track_event_async,
)

__all__ = [
    "AsyncTokenTracker",
    "AttributionContext",
    "LLMEvent",
    "TokenTracker",
    "attribution",
    "configure",
    "create_event_safe",
    "get_async_tracker",
    "get_attribution_context",
    "get_config",
    "get_tracker",
    "patch_anthropic",
    "patch_google",
    "patch_openai",
    "reset_attribution_context",
    "set_attribution_context",
    "track_cost",
    "track_event_async",
    "track_llm",
    "unpatch_anthropic",
    "unpatch_google",
    "unpatch_openai",
]
