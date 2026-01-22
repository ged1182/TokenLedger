"""Integration tests for TokenTracker with real PostgreSQL database.

These tests verify the full end-to-end workflow of tracking LLM events.
"""

from __future__ import annotations

import pytest

from tests.conftest import requires_postgres
from tokenledger import configure, get_tracker, track_cost
from tokenledger.config import TokenLedgerConfig
from tokenledger.tracker import LLMEvent, TokenTracker


@pytest.mark.integration
@requires_postgres
class TestTrackerIntegration:
    """Integration tests for TokenTracker."""

    def test_track_single_event(
        self,
        integration_config: TokenLedgerConfig,
        clean_events_table: None,
    ) -> None:
        """Test tracking a single event end-to-end."""
        tracker = TokenTracker(config=integration_config)

        event = LLMEvent(
            provider="openai",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.00075,
            user_id="test-user-1",
        )

        tracker.track(event)
        tracker.flush()

        # Verify event was written to database
        import psycopg2

        conn = psycopg2.connect(integration_config.database_url)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT provider, model, input_tokens, output_tokens, user_id "
                "FROM token_ledger_events WHERE event_id = %s",
                (event.event_id,),
            )
            row = cur.fetchone()

        conn.close()

        assert row is not None
        assert row[0] == "openai"
        assert row[1] == "gpt-4o"
        assert row[2] == 100
        assert row[3] == 50
        assert row[4] == "test-user-1"

    def test_track_multiple_events_batched(
        self,
        integration_config: TokenLedgerConfig,
        clean_events_table: None,
    ) -> None:
        """Test that multiple events are batched and written together."""
        tracker = TokenTracker(config=integration_config)

        events = []
        for i in range(15):  # More than batch_size=10
            event = LLMEvent(
                provider="anthropic",
                model="claude-3-5-sonnet",
                input_tokens=50 + i,
                output_tokens=25 + i,
                user_id=f"user-{i}",
            )
            tracker.track(event)
            events.append(event)

        tracker.flush()

        # Verify all events were written
        import psycopg2

        conn = psycopg2.connect(integration_config.database_url)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM token_ledger_events WHERE model = %s",
                ("claude-3-5-sonnet",),
            )
            count = cur.fetchone()[0]

        conn.close()

        assert count == 15

    def test_track_event_with_metadata(
        self,
        integration_config: TokenLedgerConfig,
        clean_events_table: None,
    ) -> None:
        """Test tracking events with JSON metadata."""
        tracker = TokenTracker(config=integration_config)

        metadata = {
            "feature": "chat",
            "version": "2.0",
            "tags": ["production", "premium"],
        }

        event = LLMEvent(
            provider="openai",
            model="gpt-4o-mini",
            input_tokens=200,
            output_tokens=100,
            metadata=metadata,
        )

        tracker.track(event)
        tracker.flush()

        # Verify metadata was stored correctly
        import psycopg2

        conn = psycopg2.connect(integration_config.database_url)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT metadata FROM token_ledger_events WHERE event_id = %s",
                (event.event_id,),
            )
            row = cur.fetchone()

        conn.close()

        assert row is not None
        stored_metadata = row[0]
        assert stored_metadata["feature"] == "chat"
        assert stored_metadata["version"] == "2.0"
        assert "production" in stored_metadata["tags"]

    def test_track_error_event(
        self,
        integration_config: TokenLedgerConfig,
        clean_events_table: None,
    ) -> None:
        """Test tracking an error event."""
        tracker = TokenTracker(config=integration_config)

        event = LLMEvent(
            provider="openai",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=0,
            status="error",
            error_type="RateLimitError",
            error_message="Rate limit exceeded",
        )

        tracker.track(event)
        tracker.flush()

        # Verify error details were stored
        import psycopg2

        conn = psycopg2.connect(integration_config.database_url)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT status, error_type, error_message "
                "FROM token_ledger_events WHERE event_id = %s",
                (event.event_id,),
            )
            row = cur.fetchone()

        conn.close()

        assert row is not None
        assert row[0] == "error"
        assert row[1] == "RateLimitError"
        assert row[2] == "Rate limit exceeded"

    def test_tracker_shutdown_flushes_events(
        self,
        integration_config: TokenLedgerConfig,
        clean_events_table: None,
    ) -> None:
        """Test that shutdown() properly flushes pending events."""
        tracker = TokenTracker(config=integration_config)

        event = LLMEvent(
            provider="google",
            model="gemini-2.0-flash",
            input_tokens=150,
            output_tokens=75,
        )

        tracker.track(event)
        tracker.shutdown()

        # Verify event was written during shutdown
        import psycopg2

        conn = psycopg2.connect(integration_config.database_url)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM token_ledger_events WHERE event_id = %s",
                (event.event_id,),
            )
            count = cur.fetchone()[0]

        conn.close()

        assert count == 1


@pytest.mark.integration
@requires_postgres
class TestTrackCostIntegration:
    """Integration tests for track_cost helper function."""

    def test_track_cost_function(
        self,
        integration_config: TokenLedgerConfig,
        clean_events_table: None,
    ) -> None:
        """Test the track_cost convenience function."""
        # Configure global tracker
        configure(
            database_url=integration_config.database_url,
            app_name="track-cost-test",
            environment="test",
            async_mode=False,
            batch_size=1,  # Immediate flush
            schema_name="public",  # Use public schema for integration tests
        )

        track_cost(
            input_tokens=500,
            output_tokens=200,
            model="gpt-4o",
            provider="openai",
            user_id="cost-test-user",
        )

        # Force flush
        tracker = get_tracker()
        tracker.flush()

        # Verify event was recorded
        import psycopg2

        conn = psycopg2.connect(integration_config.database_url)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT input_tokens, output_tokens, model, user_id "
                "FROM token_ledger_events WHERE user_id = %s",
                ("cost-test-user",),
            )
            row = cur.fetchone()

        conn.close()

        assert row is not None
        assert row[0] == 500
        assert row[1] == 200
        assert row[2] == "gpt-4o"
        assert row[3] == "cost-test-user"


@pytest.mark.integration
@requires_postgres
class TestQueryIntegration:
    """Integration tests for querying tracked events."""

    def test_daily_costs_view(
        self,
        integration_config: TokenLedgerConfig,
        clean_events_table: None,
    ) -> None:
        """Test the daily costs view works correctly."""
        tracker = TokenTracker(config=integration_config)

        # Track several events
        for _ in range(5):
            event = LLMEvent(
                provider="openai",
                model="gpt-4o",
                input_tokens=100,
                output_tokens=50,
                cost_usd=0.001,
            )
            tracker.track(event)

        tracker.flush()

        # Query the daily costs view
        import psycopg2

        conn = psycopg2.connect(integration_config.database_url)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT request_count, total_cost FROM token_ledger_daily_costs WHERE model = %s",
                ("gpt-4o",),
            )
            row = cur.fetchone()

        conn.close()

        assert row is not None
        assert row[0] == 5  # request_count
        assert float(row[1]) == pytest.approx(0.005, rel=1e-6)  # total_cost

    def test_user_costs_view(
        self,
        integration_config: TokenLedgerConfig,
        clean_events_table: None,
    ) -> None:
        """Test the user costs view works correctly."""
        tracker = TokenTracker(config=integration_config)

        # Track events for specific user
        for _ in range(3):
            event = LLMEvent(
                provider="anthropic",
                model="claude-3-opus",
                input_tokens=200,
                output_tokens=100,
                total_tokens=300,
                cost_usd=0.01,
                user_id="power-user",
            )
            tracker.track(event)

        tracker.flush()

        # Query the user costs view
        import psycopg2

        conn = psycopg2.connect(integration_config.database_url)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT request_count, total_tokens, total_cost "
                "FROM token_ledger_user_costs WHERE user_id = %s",
                ("power-user",),
            )
            row = cur.fetchone()

        conn.close()

        assert row is not None
        assert row[0] == 3  # request_count
        assert row[1] == 900  # total_tokens (300 * 3)
        assert float(row[2]) == pytest.approx(0.03, rel=1e-6)  # total_cost
