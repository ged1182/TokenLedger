"""Tests for the pricing module."""

from __future__ import annotations

from tokenledger.pricing import calculate_cost, get_pricing


class TestGetPricing:
    """Tests for get_pricing function."""

    def test_openai_model(self) -> None:
        """Test getting pricing for OpenAI models."""
        pricing = get_pricing("gpt-4o", provider="openai")
        assert pricing is not None
        assert pricing.input_price_per_1m == 2.50
        assert pricing.output_price_per_1m == 10.00

    def test_anthropic_model(self) -> None:
        """Test getting pricing for Anthropic models."""
        pricing = get_pricing("claude-3-5-sonnet-20241022", provider="anthropic")
        assert pricing is not None
        assert pricing.input_price_per_1m == 3.00
        assert pricing.output_price_per_1m == 15.00
        assert pricing.cached_input_price_per_1m == 0.30

    def test_auto_detect_provider(self) -> None:
        """Test auto-detection of provider from model name."""
        # OpenAI models
        assert get_pricing("gpt-4o") is not None
        assert get_pricing("o1-mini") is not None

        # Anthropic models
        assert get_pricing("claude-3-opus-20240229") is not None

    def test_unknown_model(self) -> None:
        """Test that unknown models return None."""
        assert get_pricing("unknown-model") is None


class TestCalculateCost:
    """Tests for calculate_cost function."""

    def test_basic_cost_calculation(self) -> None:
        """Test basic cost calculation."""
        cost = calculate_cost(
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
            provider="openai",
        )
        assert cost is not None
        # 1000 input tokens @ $2.50/1M = $0.0025
        # 500 output tokens @ $10.00/1M = $0.005
        expected = (1000 / 1_000_000 * 2.50) + (500 / 1_000_000 * 10.00)
        assert abs(cost - expected) < 0.0001

    def test_cached_tokens(self) -> None:
        """Test cost calculation with cached tokens."""
        cost = calculate_cost(
            model="claude-3-5-sonnet-20241022",
            input_tokens=1000,
            output_tokens=500,
            cached_tokens=500,
            provider="anthropic",
        )
        assert cost is not None
        # 500 non-cached input @ $3.00/1M
        # 500 cached input @ $0.30/1M
        # 500 output @ $15.00/1M
        non_cached = 500 / 1_000_000 * 3.00
        cached = 500 / 1_000_000 * 0.30
        output = 500 / 1_000_000 * 15.00
        expected = non_cached + cached + output
        assert abs(cost - expected) < 0.0001

    def test_unknown_model_returns_none(self) -> None:
        """Test that unknown models return None for cost."""
        cost = calculate_cost(
            model="unknown-model",
            input_tokens=1000,
            output_tokens=500,
        )
        assert cost is None
