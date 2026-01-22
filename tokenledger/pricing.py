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
    # GPT-5 series
    "gpt-5.2": ModelPricing(1.75, 14.00, 0.175),
    "gpt-5.1": ModelPricing(1.25, 10.00, 0.125),
    "gpt-5": ModelPricing(1.25, 10.00, 0.125),
    "gpt-5-mini": ModelPricing(0.25, 2.00, 0.025),
    "gpt-5-nano": ModelPricing(0.05, 0.40, 0.005),
    "gpt-5-pro": ModelPricing(15.00, 120.00),
    # GPT-4.1 series (1M context window)
    "gpt-4.1": ModelPricing(2.00, 8.00, 0.50),
    "gpt-4.1-mini": ModelPricing(0.40, 1.60, 0.10),
    "gpt-4.1-nano": ModelPricing(0.10, 0.40, 0.025),
    # GPT-4o series (128K context)
    "gpt-4o": ModelPricing(2.50, 10.00, 1.25),
    "gpt-4o-2024-11-20": ModelPricing(2.50, 10.00, 1.25),
    "gpt-4o-2024-08-06": ModelPricing(2.50, 10.00, 1.25),
    "gpt-4o-2024-05-13": ModelPricing(5.00, 15.00),
    "gpt-4o-mini": ModelPricing(0.15, 0.60, 0.075),
    "gpt-4o-mini-2024-07-18": ModelPricing(0.15, 0.60, 0.075),
    # GPT-4 Turbo (legacy)
    "gpt-4-turbo": ModelPricing(10.00, 30.00),
    "gpt-4-turbo-2024-04-09": ModelPricing(10.00, 30.00),
    "gpt-4-turbo-preview": ModelPricing(10.00, 30.00),
    # Legacy GPT-4 models
    "gpt-4": ModelPricing(30.00, 60.00),
    "gpt-4-0613": ModelPricing(30.00, 60.00),
    "gpt-4-32k": ModelPricing(60.00, 120.00),
    # GPT-3.5 Turbo (legacy)
    "gpt-3.5-turbo": ModelPricing(0.50, 1.50),
    "gpt-3.5-turbo-0125": ModelPricing(0.50, 1.50),
    "gpt-3.5-turbo-1106": ModelPricing(1.00, 2.00),
    "gpt-3.5-turbo-instruct": ModelPricing(1.50, 2.00),
    # o-series reasoning models
    "o1": ModelPricing(15.00, 60.00, 7.50),
    "o1-2024-12-17": ModelPricing(15.00, 60.00, 7.50),
    "o1-preview": ModelPricing(15.00, 60.00, 7.50),
    "o1-mini": ModelPricing(1.10, 4.40, 0.55),
    "o1-mini-2024-09-12": ModelPricing(1.10, 4.40, 0.55),
    "o3": ModelPricing(2.00, 8.00, 0.50),
    "o3-mini": ModelPricing(1.10, 4.40, 0.55),
    "o3-pro": ModelPricing(20.00, 80.00),
    "o3-deep-research": ModelPricing(10.00, 40.00, 2.50),
    "o4-mini": ModelPricing(1.10, 4.40, 0.275),
    # Embeddings
    "text-embedding-3-small": ModelPricing(0.02, 0.0),
    "text-embedding-3-large": ModelPricing(0.13, 0.0),
    "text-embedding-ada-002": ModelPricing(0.10, 0.0),
}

# Anthropic Pricing (as of January 2026)
ANTHROPIC_PRICING: dict[str, ModelPricing] = {
    # Claude 4.5 series
    "claude-opus-4-5-20251101": ModelPricing(5.00, 25.00, 0.50),
    "claude-sonnet-4-5-20250929": ModelPricing(3.00, 15.00, 0.30),
    "claude-haiku-4-5-20251001": ModelPricing(1.00, 5.00, 0.10),
    # Claude 4 series
    "claude-sonnet-4-20250514": ModelPricing(3.00, 15.00, 0.30),
    "claude-opus-4-1-20250805": ModelPricing(15.00, 75.00, 1.50),
    "claude-opus-4-20250514": ModelPricing(15.00, 75.00, 1.50),
    # Claude 3.5 series
    "claude-3-5-sonnet-20241022": ModelPricing(3.00, 15.00, 0.30),
    "claude-3-5-sonnet-20240620": ModelPricing(3.00, 15.00, 0.30),
    "claude-3-5-haiku-20241022": ModelPricing(0.80, 4.00, 0.08),
    # Claude 3 series
    "claude-3-opus-20240229": ModelPricing(15.00, 75.00, 1.50),
    "claude-3-sonnet-20240229": ModelPricing(3.00, 15.00, 0.30),
    "claude-3-haiku-20240307": ModelPricing(0.25, 1.25, 0.03),
    # Claude 3.7 series
    "claude-3-7-sonnet-20250219": ModelPricing(3.00, 15.00, 0.30),
    # Aliases
    "claude-opus-4-5-latest": ModelPricing(5.00, 25.00, 0.50),
    "claude-sonnet-4-5-latest": ModelPricing(3.00, 15.00, 0.30),
    "claude-haiku-4-5-latest": ModelPricing(1.00, 5.00, 0.10),
    "claude-opus-4-1-latest": ModelPricing(15.00, 75.00, 1.50),
    "claude-opus-4-latest": ModelPricing(15.00, 75.00, 1.50),
    "claude-sonnet-4-latest": ModelPricing(3.00, 15.00, 0.30),
    "claude-3-7-sonnet-latest": ModelPricing(3.00, 15.00, 0.30),
    "claude-3-5-sonnet-latest": ModelPricing(3.00, 15.00, 0.30),
    "claude-3-5-haiku-latest": ModelPricing(0.80, 4.00, 0.08),
    "claude-3-opus-latest": ModelPricing(15.00, 75.00, 1.50),
}

# Google/Gemini Pricing (as of January 2026)
GOOGLE_PRICING: dict[str, ModelPricing] = {
    # Gemini 3 series
    "gemini-3-pro-preview": ModelPricing(2.00, 12.00, 0.20),
    "gemini-3-flash-preview": ModelPricing(0.50, 4.00, 0.05),
    # Gemini 2.5 series
    "gemini-2.5-pro": ModelPricing(1.25, 10.00, 0.125),
    "gemini-2.5-pro-preview": ModelPricing(1.25, 10.00, 0.125),
    "gemini-2.5-flash": ModelPricing(0.30, 2.50, 0.03),
    "gemini-2.5-flash-preview": ModelPricing(0.30, 2.50, 0.03),
    "gemini-2.5-flash-lite": ModelPricing(0.10, 0.40, 0.01),
    # Gemini 2.0 series
    "gemini-2.0-flash": ModelPricing(0.10, 0.40, 0.025),
    "gemini-2.0-flash-exp": ModelPricing(0.10, 0.40, 0.025),
    "gemini-2.0-flash-lite": ModelPricing(0.075, 0.30, 0.01875),
    # Legacy models (deprecated - kept for historical cost calculations)
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
        if (
            model.startswith("gpt-")
            or model.startswith("o1")
            or model.startswith("o3")
            or model.startswith("o4")
            or model.startswith("text-embedding")
        ):
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


# =============================================================================
# OpenAI Audio Pricing (per minute for transcription/translation)
# =============================================================================

OPENAI_AUDIO_PRICING: dict[str, float] = {
    # Transcription/Translation models (price per minute)
    "whisper-1": 0.006,
    "gpt-4o-transcribe": 0.006,
    "gpt-4o-mini-transcribe": 0.003,
    "gpt-4o-transcribe-diarization": 0.012,  # With speaker diarization
}


def calculate_audio_cost(
    model: str,
    duration_minutes: float | None = None,
) -> float | None:
    """
    Calculate the cost of an audio transcription/translation call.

    Note: Duration is not always available from the API response, so this
    returns None when duration is not provided. The actual cost will be
    recorded as null in the database.

    Args:
        model: The model name (e.g., "whisper-1")
        duration_minutes: Audio duration in minutes (if known)

    Returns:
        Cost in USD, or None if model not found or duration unknown
    """
    price_per_minute = OPENAI_AUDIO_PRICING.get(model)
    if price_per_minute is None:
        return None

    # If duration is not provided, we can't calculate exact cost
    # Return None to indicate unknown cost (will be recorded as null in DB)
    if duration_minutes is None:
        return None

    return price_per_minute * duration_minutes


# =============================================================================
# OpenAI TTS Pricing (per 1K characters)
# =============================================================================

OPENAI_TTS_PRICING: dict[str, float] = {
    # TTS models (price per 1K characters)
    "tts-1": 0.015,
    "tts-1-hd": 0.030,
    "gpt-4o-mini-tts": 0.010,
}


def calculate_tts_cost(model: str, character_count: int) -> float | None:
    """
    Calculate the cost of a TTS (text-to-speech) call.

    Args:
        model: The model name (e.g., "tts-1")
        character_count: Number of characters in the input text

    Returns:
        Cost in USD, or None if model not found
    """
    price_per_1k = OPENAI_TTS_PRICING.get(model)
    if price_per_1k is None:
        return None

    return (character_count / 1000) * price_per_1k


# =============================================================================
# OpenAI Image Pricing (per image)
# =============================================================================

# Image generation pricing by model, quality, and size
OPENAI_IMAGE_PRICING: dict[str, dict[str, dict[str, float]]] = {
    "dall-e-3": {
        "standard": {
            "1024x1024": 0.040,
            "1024x1792": 0.080,
            "1792x1024": 0.080,
        },
        "hd": {
            "1024x1024": 0.080,
            "1024x1792": 0.120,
            "1792x1024": 0.120,
        },
    },
    "dall-e-2": {
        "standard": {
            "1024x1024": 0.020,
            "512x512": 0.018,
            "256x256": 0.016,
        },
    },
    "gpt-image-1": {
        "standard": {
            "1024x1024": 0.040,
            "1024x1792": 0.080,
            "1792x1024": 0.080,
        },
        "hd": {
            "1024x1024": 0.080,
            "1024x1792": 0.120,
            "1792x1024": 0.120,
        },
    },
    "gpt-image-1-mini": {
        "standard": {
            "1024x1024": 0.020,
            "512x512": 0.018,
            "256x256": 0.016,
        },
    },
}


def calculate_image_cost(
    model: str,
    n: int = 1,
    size: str = "1024x1024",
    quality: str = "standard",
) -> float | None:
    """
    Calculate the cost of an image generation call.

    Args:
        model: The model name (e.g., "dall-e-3")
        n: Number of images generated
        size: Image size (e.g., "1024x1024")
        quality: Image quality ("standard" or "hd")

    Returns:
        Cost in USD, or None if pricing not found
    """
    model_pricing = OPENAI_IMAGE_PRICING.get(model)
    if model_pricing is None:
        return None

    quality_pricing = model_pricing.get(quality, model_pricing.get("standard"))
    if quality_pricing is None:
        return None

    price_per_image = quality_pricing.get(size)
    if price_per_image is None:
        # Try to find a default size
        if size in quality_pricing:
            price_per_image = quality_pricing[size]
        else:
            # Use largest available size as fallback
            price_per_image = max(quality_pricing.values())

    return price_per_image * n
