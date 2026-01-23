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

    def test_no_cached_price_uses_zero(self) -> None:
        """Test that models without cached pricing don't add cached cost."""
        # GPT-4 Turbo doesn't have cached pricing
        cost = calculate_cost(
            model="gpt-4-turbo",
            input_tokens=1000,
            output_tokens=500,
            cached_tokens=500,
            provider="openai",
        )
        assert cost is not None
        # Only non-cached input (500) charged at input rate, cached tokens ignored
        # 500 non-cached input @ $10.00/1M + 500 output @ $30.00/1M
        expected = (500 / 1_000_000 * 10.00) + (500 / 1_000_000 * 30.00)
        assert abs(cost - expected) < 0.0001


class TestAutoDetectProvider:
    """Tests for provider auto-detection."""

    def test_detect_google_gemini(self) -> None:
        """Test auto-detection of Google Gemini models."""
        pricing = get_pricing("gemini-2.0-flash")
        assert pricing is not None
        assert pricing.input_price_per_1m == 0.10

    def test_detect_mistral(self) -> None:
        """Test auto-detection of Mistral models."""
        pricing = get_pricing("mistral-large-latest")
        assert pricing is not None
        assert pricing.input_price_per_1m == 0.50

    def test_detect_mixtral(self) -> None:
        """Test auto-detection of Mixtral models."""
        pricing = get_pricing("open-mixtral-8x7b")
        assert pricing is not None

    def test_detect_o3_models(self) -> None:
        """Test auto-detection of o3 reasoning models."""
        pricing = get_pricing("o3-mini")
        assert pricing is not None

    def test_detect_o4_models(self) -> None:
        """Test auto-detection of o4 reasoning models."""
        pricing = get_pricing("o4-mini")
        assert pricing is not None

    def test_detect_text_embedding(self) -> None:
        """Test auto-detection of OpenAI embedding models."""
        pricing = get_pricing("text-embedding-3-small")
        assert pricing is not None
        assert pricing.output_price_per_1m == 0.0  # Embeddings have no output cost


class TestEstimateMonthlyCost:
    """Tests for estimate_monthly_cost function."""

    def test_basic_estimate(self) -> None:
        """Test basic monthly cost estimation."""
        from tokenledger.pricing import estimate_monthly_cost

        cost = estimate_monthly_cost(
            daily_calls=100,
            avg_input_tokens=1000,
            avg_output_tokens=500,
            model="gpt-4o",
        )
        assert cost is not None
        per_call = (1000 / 1_000_000 * 2.50) + (500 / 1_000_000 * 10.00)
        expected = per_call * 100 * 30
        assert abs(cost - expected) < 0.01

    def test_unknown_model_returns_none(self) -> None:
        """Test that unknown models return None."""
        from tokenledger.pricing import estimate_monthly_cost

        cost = estimate_monthly_cost(
            daily_calls=100,
            avg_input_tokens=1000,
            avg_output_tokens=500,
            model="unknown-model",
        )
        assert cost is None


class TestCalculateAudioCost:
    """Tests for calculate_audio_cost function."""

    def test_whisper_cost(self) -> None:
        """Test Whisper transcription cost calculation."""
        from tokenledger.pricing import calculate_audio_cost

        cost = calculate_audio_cost("whisper-1", duration_minutes=10.0)
        assert cost is not None
        assert abs(cost - 0.06) < 0.001  # $0.006/min * 10 min

    def test_unknown_model_returns_none(self) -> None:
        """Test that unknown audio models return None."""
        from tokenledger.pricing import calculate_audio_cost

        cost = calculate_audio_cost("unknown-audio-model", duration_minutes=10.0)
        assert cost is None

    def test_no_duration_returns_none(self) -> None:
        """Test that missing duration returns None."""
        from tokenledger.pricing import calculate_audio_cost

        cost = calculate_audio_cost("whisper-1", duration_minutes=None)
        assert cost is None


class TestCalculateTtsCost:
    """Tests for calculate_tts_cost function."""

    def test_tts_1_cost(self) -> None:
        """Test TTS-1 cost calculation."""
        from tokenledger.pricing import calculate_tts_cost

        cost = calculate_tts_cost("tts-1", character_count=5000)
        assert cost is not None
        assert abs(cost - 0.075) < 0.001  # $0.015/1K * 5K chars

    def test_tts_1_hd_cost(self) -> None:
        """Test TTS-1-HD cost calculation."""
        from tokenledger.pricing import calculate_tts_cost

        cost = calculate_tts_cost("tts-1-hd", character_count=2000)
        assert cost is not None
        assert abs(cost - 0.06) < 0.001  # $0.030/1K * 2K chars

    def test_unknown_model_returns_none(self) -> None:
        """Test that unknown TTS models return None."""
        from tokenledger.pricing import calculate_tts_cost

        cost = calculate_tts_cost("unknown-tts-model", character_count=1000)
        assert cost is None


class TestCalculateImageCost:
    """Tests for calculate_image_cost function."""

    def test_dalle3_standard(self) -> None:
        """Test DALL-E 3 standard quality cost."""
        from tokenledger.pricing import calculate_image_cost

        cost = calculate_image_cost("dall-e-3", n=1, size="1024x1024", quality="standard")
        assert cost is not None
        assert abs(cost - 0.04) < 0.001

    def test_dalle3_hd(self) -> None:
        """Test DALL-E 3 HD quality cost."""
        from tokenledger.pricing import calculate_image_cost

        cost = calculate_image_cost("dall-e-3", n=2, size="1024x1024", quality="hd")
        assert cost is not None
        assert abs(cost - 0.16) < 0.001  # $0.08 * 2

    def test_dalle3_wide(self) -> None:
        """Test DALL-E 3 wide image cost."""
        from tokenledger.pricing import calculate_image_cost

        cost = calculate_image_cost("dall-e-3", n=1, size="1792x1024", quality="standard")
        assert cost is not None
        assert abs(cost - 0.08) < 0.001

    def test_dalle2_cost(self) -> None:
        """Test DALL-E 2 cost calculation."""
        from tokenledger.pricing import calculate_image_cost

        cost = calculate_image_cost("dall-e-2", n=1, size="512x512", quality="standard")
        assert cost is not None
        assert abs(cost - 0.018) < 0.001

    def test_unknown_model_returns_none(self) -> None:
        """Test that unknown image models return None."""
        from tokenledger.pricing import calculate_image_cost

        cost = calculate_image_cost("unknown-image-model", n=1)
        assert cost is None

    def test_unknown_size_uses_fallback(self) -> None:
        """Test that unknown sizes fall back to max price."""
        from tokenledger.pricing import calculate_image_cost

        cost = calculate_image_cost("dall-e-3", n=1, size="2048x2048", quality="standard")
        assert cost is not None
        # Should use max available price (0.08 for wide images)
        assert cost == 0.08

    def test_unknown_quality_falls_back_to_standard(self) -> None:
        """Test that unknown quality falls back to standard."""
        from tokenledger.pricing import calculate_image_cost

        cost = calculate_image_cost("dall-e-2", n=1, size="1024x1024", quality="ultra")
        assert cost is not None
        assert abs(cost - 0.02) < 0.001
