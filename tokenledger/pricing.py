"""
LLM Pricing Configuration
Prices in USD per 1M tokens (updated January 2026)
"""

from dataclasses import dataclass


@dataclass
class ModelPricing:
    """Pricing for a specific model"""

    input_price_per_1m: float  # USD per 1M input tokens
    output_price_per_1m: float  # USD per 1M output tokens
    cached_input_price_per_1m: float | None = None  # For cached prompts


# OpenAI Pricing (as of January 2026)
OPENAI_PRICING: dict[str, ModelPricing] = {
    # GPT-4o series
    "gpt-4o": ModelPricing(2.50, 10.00, 0.63),
    "gpt-4o-2024-11-20": ModelPricing(2.50, 10.00, 0.63),
    "gpt-4o-2024-08-06": ModelPricing(2.50, 10.00, 0.63),
    "gpt-4o-2024-05-13": ModelPricing(5.00, 15.00),
    "gpt-4o-mini": ModelPricing(0.15, 0.60, 0.04),
    "gpt-4o-mini-2024-07-18": ModelPricing(0.15, 0.60, 0.04),
    # GPT-4 Turbo
    "gpt-4-turbo": ModelPricing(10.00, 30.00),
    "gpt-4-turbo-2024-04-09": ModelPricing(10.00, 30.00),
    "gpt-4-turbo-preview": ModelPricing(10.00, 30.00),
    # GPT-4
    "gpt-4": ModelPricing(30.00, 60.00),
    "gpt-4-0613": ModelPricing(30.00, 60.00),
    "gpt-4-32k": ModelPricing(60.00, 120.00),
    # GPT-3.5 Turbo
    "gpt-3.5-turbo": ModelPricing(0.50, 1.50),
    "gpt-3.5-turbo-0125": ModelPricing(0.50, 1.50),
    "gpt-3.5-turbo-1106": ModelPricing(1.00, 2.00),
    "gpt-3.5-turbo-instruct": ModelPricing(1.50, 2.00),
    # o1 reasoning models
    "o1": ModelPricing(15.00, 60.00),
    "o1-2024-12-17": ModelPricing(15.00, 60.00),
    "o1-preview": ModelPricing(15.00, 60.00),
    "o1-mini": ModelPricing(3.00, 12.00),
    "o1-mini-2024-09-12": ModelPricing(3.00, 12.00),
    # Embeddings
    "text-embedding-3-small": ModelPricing(0.02, 0.0),
    "text-embedding-3-large": ModelPricing(0.13, 0.0),
    "text-embedding-ada-002": ModelPricing(0.10, 0.0),
}

# Anthropic Pricing (as of January 2026)
ANTHROPIC_PRICING: dict[str, ModelPricing] = {
    # Claude 3.5 series
    "claude-3-5-sonnet-20241022": ModelPricing(3.00, 15.00, 0.30),
    "claude-3-5-sonnet-20240620": ModelPricing(3.00, 15.00, 0.30),
    "claude-3-5-haiku-20241022": ModelPricing(0.80, 4.00, 0.08),
    # Claude 3 series
    "claude-3-opus-20240229": ModelPricing(15.00, 75.00, 1.50),
    "claude-3-sonnet-20240229": ModelPricing(3.00, 15.00, 0.30),
    "claude-3-haiku-20240307": ModelPricing(0.25, 1.25, 0.03),
    # Claude 4 series (anticipated)
    "claude-sonnet-4-20250514": ModelPricing(3.00, 15.00, 0.30),
    "claude-opus-4-20250514": ModelPricing(15.00, 75.00, 1.50),
    # Aliases
    "claude-3-5-sonnet-latest": ModelPricing(3.00, 15.00, 0.30),
    "claude-3-5-haiku-latest": ModelPricing(0.80, 4.00, 0.08),
    "claude-3-opus-latest": ModelPricing(15.00, 75.00, 1.50),
}

# Google/Gemini Pricing (as of January 2026)
# Note: Gemini 1.0 and 1.5 models were retired in late 2025
# Pricing kept for historical cost calculations
GOOGLE_PRICING: dict[str, ModelPricing] = {
    "gemini-1.5-pro": ModelPricing(1.25, 5.00),  # Deprecated
    "gemini-1.5-flash": ModelPricing(0.075, 0.30),  # Deprecated
    "gemini-1.0-pro": ModelPricing(0.50, 1.50),  # Deprecated
}

# Mistral Pricing (as of January 2026)
MISTRAL_PRICING: dict[str, ModelPricing] = {
    "mistral-large-latest": ModelPricing(0.50, 1.50),
    "mistral-medium-latest": ModelPricing(0.40, 2.00),
    "mistral-small-latest": ModelPricing(0.06, 0.18),
    "open-mistral-7b": ModelPricing(0.15, 0.15),  # Ministral 8B equivalent
    "open-mixtral-8x7b": ModelPricing(0.70, 0.70),
}


def get_pricing(model: str, provider: str = "auto") -> ModelPricing | None:
    """
    Get pricing for a model.

    Args:
        model: The model name/ID
        provider: 'openai', 'anthropic', 'google', 'mistral', or 'auto' to detect

    Returns:
        ModelPricing if found, None otherwise
    """
    if provider == "auto":
        # Try to detect provider from model name
        if model.startswith("gpt-") or model.startswith("o1") or model.startswith("text-embedding"):
            provider = "openai"
        elif model.startswith("claude"):
            provider = "anthropic"
        elif model.startswith("gemini"):
            provider = "google"
        elif "mistral" in model or "mixtral" in model:
            provider = "mistral"

    pricing_map = {
        "openai": OPENAI_PRICING,
        "anthropic": ANTHROPIC_PRICING,
        "google": GOOGLE_PRICING,
        "mistral": MISTRAL_PRICING,
    }

    pricing_dict = pricing_map.get(provider, {})
    return pricing_dict.get(model)


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
    provider: str = "auto",
) -> float | None:
    """
    Calculate the cost of an LLM call.

    Args:
        model: The model name/ID
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        cached_tokens: Number of cached input tokens (for Anthropic prompt caching)
        provider: Provider name or 'auto' to detect

    Returns:
        Cost in USD, or None if model pricing not found
    """
    pricing = get_pricing(model, provider)
    if not pricing:
        return None

    # Calculate input cost (excluding cached tokens)
    non_cached_input = input_tokens - cached_tokens
    input_cost = (non_cached_input / 1_000_000) * pricing.input_price_per_1m

    # Calculate cached input cost if applicable
    cached_cost = 0.0
    if cached_tokens > 0 and pricing.cached_input_price_per_1m:
        cached_cost = (cached_tokens / 1_000_000) * pricing.cached_input_price_per_1m

    # Calculate output cost
    output_cost = (output_tokens / 1_000_000) * pricing.output_price_per_1m

    return input_cost + cached_cost + output_cost


def estimate_monthly_cost(
    daily_calls: int, avg_input_tokens: int, avg_output_tokens: int, model: str
) -> float | None:
    """Estimate monthly cost based on usage patterns."""
    per_call = calculate_cost(model, avg_input_tokens, avg_output_tokens)
    if per_call is None:
        return None
    return per_call * daily_calls * 30
