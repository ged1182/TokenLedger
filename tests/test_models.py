"""Tests for TokenLedger Pydantic models."""

from __future__ import annotations

from datetime import datetime

from tokenledger.models import LLMEvent, create_event_safe


class TestLLMEvent:
    """Tests for LLMEvent model."""

    def test_default_values(self) -> None:
        """Test that default values are set correctly."""
        event = LLMEvent(provider="openai", model="gpt-4o")

        assert event.provider == "openai"
        assert event.model == "gpt-4o"
        assert event.event_id  # Should be auto-generated UUID
        assert event.timestamp  # Should be auto-generated
        assert event.input_tokens == 0
        assert event.output_tokens == 0
        assert event.total_tokens == 0
        assert event.status == "success"
        assert event.request_type == "chat"

    def test_total_tokens_computed(self) -> None:
        """Test that total_tokens is computed from input + output."""
        event = LLMEvent(
            provider="openai",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
        )

        assert event.total_tokens == 150

    def test_cost_usd_computed(self) -> None:
        """Test that cost_usd is computed when model is known."""
        event = LLMEvent(
            provider="openai",
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
        )

        assert event.cost_usd is not None
        assert event.cost_usd > 0

    def test_attribution_fields(self) -> None:
        """Test that attribution fields can be set."""
        event = LLMEvent(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            feature="summarize",
            team="ml",
            project="api",
            cost_center="CC-001",
        )

        assert event.feature == "summarize"
        assert event.team == "ml"
        assert event.project == "api"
        assert event.cost_center == "CC-001"

    def test_to_dict(self) -> None:
        """Test converting event to dictionary."""
        event = LLMEvent(
            provider="openai",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            metadata={"key": "value"},
        )

        d = event.to_dict()

        assert d["provider"] == "openai"
        assert d["model"] == "gpt-4o"
        assert d["input_tokens"] == 100
        assert d["output_tokens"] == 50
        assert d["total_tokens"] == 150
        assert isinstance(d["timestamp"], str)  # ISO format
        assert isinstance(d["metadata"], str)  # JSON string


class TestFastConstruct:
    """Tests for LLMEvent.fast_construct()."""

    def test_basic_construction(self) -> None:
        """Test basic fast construction."""
        event = LLMEvent.fast_construct(
            provider="openai",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
        )

        assert event.provider == "openai"
        assert event.model == "gpt-4o"
        assert event.input_tokens == 100
        assert event.output_tokens == 50

    def test_derived_fields_computed(self) -> None:
        """Test that derived fields are computed in fast_construct."""
        event = LLMEvent.fast_construct(
            provider="openai",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
        )

        assert event.total_tokens == 150
        assert event.cost_usd is not None

    def test_event_id_generated(self) -> None:
        """Test that event_id is generated if not provided."""
        event = LLMEvent.fast_construct(provider="openai", model="gpt-4o")

        assert event.event_id
        assert len(event.event_id) == 36  # UUID format

    def test_timestamp_generated(self) -> None:
        """Test that timestamp is generated if not provided."""
        event = LLMEvent.fast_construct(provider="openai", model="gpt-4o")

        assert event.timestamp
        assert isinstance(event.timestamp, datetime)

    def test_attribution_fields(self) -> None:
        """Test attribution fields in fast_construct."""
        event = LLMEvent.fast_construct(
            provider="openai",
            model="gpt-4o",
            feature="chat",
            team="platform",
            project="web-app",
            cost_center="ENG-001",
            metadata_extra={"custom": "data"},
        )

        assert event.feature == "chat"
        assert event.team == "platform"
        assert event.project == "web-app"
        assert event.cost_center == "ENG-001"
        assert event.metadata_extra == {"custom": "data"}


class TestCreateEventSafe:
    """Tests for create_event_safe factory."""

    def test_valid_event(self) -> None:
        """Test creating a valid event."""
        event = create_event_safe(
            provider="openai",
            model="gpt-4o",
            input_tokens=100,
        )

        assert event.provider == "openai"
        assert event.model == "gpt-4o"
        assert event.input_tokens == 100

    def test_invalid_data_fallback(self) -> None:
        """Test that invalid data falls back to minimal event."""
        # Pass an invalid timestamp type to trigger validation error
        event = create_event_safe(
            provider="openai",
            model="gpt-4o",
            timestamp="not-a-datetime",  # Invalid type
        )

        # Should not raise, should return a valid event
        assert event.provider == "openai"
        assert event.model == "gpt-4o"

    def test_debug_logging(self) -> None:
        """Test debug logging on validation error."""
        # This should not raise even with bad data
        event = create_event_safe(
            debug=True,
            provider="test",
            model="test-model",
        )

        assert event.provider == "test"
