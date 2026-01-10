"""Tests for TokenLedger analytics queries."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from decimal import Decimal

import psycopg2
import pytest

from tokenledger.queries import (
    AsyncTokenLedgerQueries,
    CostSummary,
    DailyCost,
    HourlyCost,
    ModelCost,
    TokenLedgerQueries,
    UserCost,
    execute_query,
)


# Test database URL - uses local PostgreSQL with trust auth
TEST_DATABASE_URL = "postgresql://postgres@/tokenledger_test"
TEST_TABLE_NAME = "token_ledger_events"


@pytest.fixture(scope="module")
def db_connection():
    """Create a database connection for the test module."""
    conn = psycopg2.connect(TEST_DATABASE_URL)
    conn.autocommit = False
    yield conn
    conn.close()


@pytest.fixture(autouse=True)
def clean_table(db_connection):
    """Clean the table before each test and rollback after."""
    with db_connection.cursor() as cur:
        cur.execute(f"DELETE FROM {TEST_TABLE_NAME}")
    db_connection.commit()
    yield
    db_connection.rollback()


@pytest.fixture
def queries(db_connection) -> TokenLedgerQueries:
    """Create a TokenLedgerQueries instance with test connection."""
    return TokenLedgerQueries(connection=db_connection, table_name=TEST_TABLE_NAME)


def insert_event(
    conn,
    provider: str = "openai",
    model: str = "gpt-4",
    input_tokens: int = 100,
    output_tokens: int = 50,
    cost_usd: float = 0.01,
    user_id: str | None = "user-1",
    app_name: str = "test-app",
    status: str = "success",
    error_type: str | None = None,
    error_message: str | None = None,
    duration_ms: float | None = 100.0,
    timestamp: datetime | None = None,
) -> uuid.UUID:
    """Helper to insert a test event."""
    event_id = uuid.uuid4()
    ts = timestamp or datetime.now()

    with conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO {TEST_TABLE_NAME} (
                event_id, provider, model, input_tokens, output_tokens,
                total_tokens, cost_usd, user_id, app_name, status,
                error_type, error_message, duration_ms, timestamp
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            """,
            (
                str(event_id),
                provider,
                model,
                input_tokens,
                output_tokens,
                input_tokens + output_tokens,
                cost_usd,
                user_id,
                app_name,
                status,
                error_type,
                error_message,
                duration_ms,
                ts,
            ),
        )
    conn.commit()
    return event_id


@pytest.mark.integration
class TestTokenLedgerQueries:
    """Tests for TokenLedgerQueries class."""

    class TestGetCostSummary:
        """Tests for get_cost_summary method."""

        def test_empty_table_returns_zeros(self, queries):
            """Test that empty table returns zero values."""
            summary = queries.get_cost_summary(days=30)

            assert isinstance(summary, CostSummary)
            assert summary.total_cost == 0.0
            assert summary.total_tokens == 0
            assert summary.total_input_tokens == 0
            assert summary.total_output_tokens == 0
            assert summary.total_requests == 0
            assert summary.avg_cost_per_request == 0.0
            assert summary.avg_tokens_per_request == 0.0

        def test_single_event_summary(self, queries, db_connection):
            """Test summary with a single event."""
            insert_event(
                db_connection,
                input_tokens=100,
                output_tokens=50,
                cost_usd=0.015,
            )

            summary = queries.get_cost_summary(days=30)

            assert summary.total_cost == pytest.approx(0.015, rel=1e-6)
            assert summary.total_tokens == 150
            assert summary.total_input_tokens == 100
            assert summary.total_output_tokens == 50
            assert summary.total_requests == 1
            assert summary.avg_cost_per_request == pytest.approx(0.015, rel=1e-6)
            assert summary.avg_tokens_per_request == 150.0

        def test_multiple_events_aggregation(self, queries, db_connection):
            """Test that multiple events are aggregated correctly."""
            insert_event(db_connection, input_tokens=100, output_tokens=50, cost_usd=0.01)
            insert_event(db_connection, input_tokens=200, output_tokens=100, cost_usd=0.02)
            insert_event(db_connection, input_tokens=300, output_tokens=150, cost_usd=0.03)

            summary = queries.get_cost_summary(days=30)

            assert summary.total_cost == pytest.approx(0.06, rel=1e-6)
            assert summary.total_tokens == 900
            assert summary.total_input_tokens == 600
            assert summary.total_output_tokens == 300
            assert summary.total_requests == 3
            assert summary.avg_cost_per_request == pytest.approx(0.02, rel=1e-6)
            assert summary.avg_tokens_per_request == 300.0

        def test_filter_by_user_id(self, queries, db_connection):
            """Test filtering by user_id."""
            insert_event(db_connection, user_id="user-a", cost_usd=0.01)
            insert_event(db_connection, user_id="user-a", cost_usd=0.02)
            insert_event(db_connection, user_id="user-b", cost_usd=0.05)

            summary = queries.get_cost_summary(days=30, user_id="user-a")

            assert summary.total_requests == 2
            assert summary.total_cost == pytest.approx(0.03, rel=1e-6)

        def test_filter_by_model(self, queries, db_connection):
            """Test filtering by model."""
            insert_event(db_connection, model="gpt-4", cost_usd=0.01)
            insert_event(db_connection, model="gpt-4", cost_usd=0.02)
            insert_event(db_connection, model="gpt-3.5-turbo", cost_usd=0.001)

            summary = queries.get_cost_summary(days=30, model="gpt-4")

            assert summary.total_requests == 2
            assert summary.total_cost == pytest.approx(0.03, rel=1e-6)

        def test_filter_by_app_name(self, queries, db_connection):
            """Test filtering by app_name."""
            insert_event(db_connection, app_name="app-1", cost_usd=0.01)
            insert_event(db_connection, app_name="app-2", cost_usd=0.02)

            summary = queries.get_cost_summary(days=30, app_name="app-1")

            assert summary.total_requests == 1
            assert summary.total_cost == pytest.approx(0.01, rel=1e-6)

        def test_days_filter_excludes_old_events(self, queries, db_connection):
            """Test that old events are excluded by days filter."""
            # Recent event
            insert_event(db_connection, cost_usd=0.01, timestamp=datetime.now())
            # Old event (40 days ago)
            old_time = datetime.now() - timedelta(days=40)
            insert_event(db_connection, cost_usd=0.05, timestamp=old_time)

            summary = queries.get_cost_summary(days=30)

            assert summary.total_requests == 1
            assert summary.total_cost == pytest.approx(0.01, rel=1e-6)

    class TestGetCostsByModel:
        """Tests for get_costs_by_model method."""

        def test_empty_table_returns_empty_list(self, queries):
            """Test that empty table returns empty list."""
            result = queries.get_costs_by_model(days=30)
            assert result == []

        def test_single_model(self, queries, db_connection):
            """Test with a single model."""
            insert_event(db_connection, model="gpt-4", cost_usd=0.01)
            insert_event(db_connection, model="gpt-4", cost_usd=0.02)

            result = queries.get_costs_by_model(days=30)

            assert len(result) == 1
            assert isinstance(result[0], ModelCost)
            assert result[0].model == "gpt-4"
            assert result[0].provider == "openai"
            assert result[0].total_cost == pytest.approx(0.03, rel=1e-6)
            assert result[0].total_requests == 2

        def test_multiple_models_ordered_by_cost(self, queries, db_connection):
            """Test that results are ordered by cost descending."""
            insert_event(db_connection, model="gpt-3.5-turbo", cost_usd=0.001)
            insert_event(db_connection, model="gpt-4", cost_usd=0.05)
            insert_event(db_connection, model="claude-3-opus", provider="anthropic", cost_usd=0.10)

            result = queries.get_costs_by_model(days=30)

            assert len(result) == 3
            assert result[0].model == "claude-3-opus"
            assert result[1].model == "gpt-4"
            assert result[2].model == "gpt-3.5-turbo"

        def test_limit_parameter(self, queries, db_connection):
            """Test that limit parameter works."""
            insert_event(db_connection, model="model-1", cost_usd=0.01)
            insert_event(db_connection, model="model-2", cost_usd=0.02)
            insert_event(db_connection, model="model-3", cost_usd=0.03)

            result = queries.get_costs_by_model(days=30, limit=2)

            assert len(result) == 2

    class TestGetCostsByUser:
        """Tests for get_costs_by_user method."""

        def test_empty_table_returns_empty_list(self, queries):
            """Test that empty table returns empty list."""
            result = queries.get_costs_by_user(days=30)
            assert result == []

        def test_single_user(self, queries, db_connection):
            """Test with a single user."""
            insert_event(db_connection, user_id="user-1", cost_usd=0.01)
            insert_event(db_connection, user_id="user-1", cost_usd=0.02)

            result = queries.get_costs_by_user(days=30)

            assert len(result) == 1
            assert isinstance(result[0], UserCost)
            assert result[0].user_id == "user-1"
            assert result[0].total_cost == pytest.approx(0.03, rel=1e-6)
            assert result[0].total_requests == 2

        def test_anonymous_user(self, queries, db_connection):
            """Test that null user_id becomes 'anonymous'."""
            insert_event(db_connection, user_id=None, cost_usd=0.01)

            result = queries.get_costs_by_user(days=30)

            assert len(result) == 1
            assert result[0].user_id == "anonymous"

        def test_multiple_users_ordered_by_cost(self, queries, db_connection):
            """Test that results are ordered by cost descending."""
            insert_event(db_connection, user_id="low-spender", cost_usd=0.01)
            insert_event(db_connection, user_id="high-spender", cost_usd=0.50)
            insert_event(db_connection, user_id="medium-spender", cost_usd=0.10)

            result = queries.get_costs_by_user(days=30)

            assert result[0].user_id == "high-spender"
            assert result[1].user_id == "medium-spender"
            assert result[2].user_id == "low-spender"

    class TestGetDailyCosts:
        """Tests for get_daily_costs method."""

        def test_empty_table_returns_empty_list(self, queries):
            """Test that empty table returns empty list."""
            result = queries.get_daily_costs(days=30)
            assert result == []

        def test_single_day(self, queries, db_connection):
            """Test with events on a single day."""
            now = datetime.now()
            insert_event(db_connection, cost_usd=0.01, timestamp=now)
            insert_event(db_connection, cost_usd=0.02, timestamp=now)

            result = queries.get_daily_costs(days=30)

            assert len(result) == 1
            assert isinstance(result[0], DailyCost)
            assert result[0].total_cost == pytest.approx(0.03, rel=1e-6)
            assert result[0].total_requests == 2

        def test_multiple_days_ordered_chronologically(self, queries, db_connection):
            """Test that results are ordered by date ascending."""
            now = datetime.now()
            yesterday = now - timedelta(days=1)
            two_days_ago = now - timedelta(days=2)

            insert_event(db_connection, cost_usd=0.01, timestamp=now)
            insert_event(db_connection, cost_usd=0.02, timestamp=yesterday)
            insert_event(db_connection, cost_usd=0.03, timestamp=two_days_ago)

            result = queries.get_daily_costs(days=30)

            assert len(result) == 3
            # Should be ordered oldest first
            assert result[0].date < result[1].date < result[2].date

        def test_filter_by_user_id(self, queries, db_connection):
            """Test filtering by user_id."""
            insert_event(db_connection, user_id="user-a", cost_usd=0.01)
            insert_event(db_connection, user_id="user-b", cost_usd=0.02)

            result = queries.get_daily_costs(days=30, user_id="user-a")

            assert len(result) == 1
            assert result[0].total_cost == pytest.approx(0.01, rel=1e-6)

    class TestGetHourlyCosts:
        """Tests for get_hourly_costs method."""

        def test_empty_table_returns_empty_list(self, queries):
            """Test that empty table returns empty list."""
            result = queries.get_hourly_costs(hours=24)
            assert result == []

        def test_single_hour(self, queries, db_connection):
            """Test with events in a single hour."""
            now = datetime.now()
            insert_event(db_connection, cost_usd=0.01, timestamp=now)
            insert_event(db_connection, cost_usd=0.02, timestamp=now)

            result = queries.get_hourly_costs(hours=24)

            assert len(result) == 1
            assert isinstance(result[0], HourlyCost)
            assert result[0].total_cost == pytest.approx(0.03, rel=1e-6)
            assert result[0].total_requests == 2

    class TestGetErrorRate:
        """Tests for get_error_rate method."""

        def test_empty_table_returns_zeros(self, queries):
            """Test that empty table returns zero error rate."""
            result = queries.get_error_rate(days=7)

            assert result["total_requests"] == 0
            assert result["errors"] == 0
            assert result["error_rate"] == 0

        def test_no_errors(self, queries, db_connection):
            """Test with all successful requests."""
            insert_event(db_connection, status="success")
            insert_event(db_connection, status="success")

            result = queries.get_error_rate(days=7)

            assert result["total_requests"] == 2
            assert result["errors"] == 0
            assert result["error_rate"] == 0

        def test_some_errors(self, queries, db_connection):
            """Test error rate calculation with some errors."""
            insert_event(db_connection, status="success")
            insert_event(db_connection, status="success")
            insert_event(db_connection, status="error")
            insert_event(db_connection, status="error")

            result = queries.get_error_rate(days=7)

            assert result["total_requests"] == 4
            assert result["errors"] == 2
            assert result["error_rate"] == pytest.approx(0.5, rel=1e-6)
            assert result["status_breakdown"] == {"success": 2, "error": 2}

    class TestGetTopErrors:
        """Tests for get_top_errors method."""

        def test_empty_table_returns_empty_list(self, queries):
            """Test that empty table returns empty list."""
            result = queries.get_top_errors(days=7)
            assert result == []

        def test_no_errors(self, queries, db_connection):
            """Test with no error events."""
            insert_event(db_connection, status="success")

            result = queries.get_top_errors(days=7)
            assert result == []

        def test_single_error_type(self, queries, db_connection):
            """Test with a single error type."""
            insert_event(
                db_connection,
                status="error",
                error_type="RateLimitError",
                error_message="Rate limit exceeded",
                model="gpt-4",
            )

            result = queries.get_top_errors(days=7)

            assert len(result) == 1
            assert result[0]["error_type"] == "RateLimitError"
            assert result[0]["error_message"] == "Rate limit exceeded"
            assert result[0]["model"] == "gpt-4"
            assert result[0]["count"] == 1

        def test_errors_ordered_by_count(self, queries, db_connection):
            """Test that errors are ordered by count descending."""
            # Insert 3 rate limit errors
            for _ in range(3):
                insert_event(
                    db_connection,
                    status="error",
                    error_type="RateLimitError",
                    error_message="Rate limit exceeded",
                    model="gpt-4",
                )
            # Insert 1 auth error
            insert_event(
                db_connection,
                status="error",
                error_type="AuthenticationError",
                error_message="Invalid API key",
                model="gpt-4",
            )

            result = queries.get_top_errors(days=7)

            assert len(result) == 2
            assert result[0]["count"] == 3
            assert result[1]["count"] == 1

    class TestGetLatencyPercentiles:
        """Tests for get_latency_percentiles method."""

        def test_empty_table_returns_zeros(self, queries):
            """Test that empty table returns zero latencies."""
            result = queries.get_latency_percentiles(days=7)

            assert result["p50_ms"] == 0.0
            assert result["p90_ms"] == 0.0
            assert result["p95_ms"] == 0.0
            assert result["p99_ms"] == 0.0
            assert result["avg_ms"] == 0.0

        def test_single_request_latency(self, queries, db_connection):
            """Test with a single request."""
            insert_event(db_connection, duration_ms=100.0)

            result = queries.get_latency_percentiles(days=7)

            assert result["p50_ms"] == 100.0
            assert result["avg_ms"] == 100.0

        def test_multiple_requests_latency(self, queries, db_connection):
            """Test percentile calculations with multiple requests."""
            # Insert latencies: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
            for i in range(1, 11):
                insert_event(db_connection, duration_ms=float(i * 10))

            result = queries.get_latency_percentiles(days=7)

            assert result["avg_ms"] == 55.0
            # p50 should be around 50-55
            assert 45.0 <= result["p50_ms"] <= 60.0
            # p90 should be around 90
            assert 85.0 <= result["p90_ms"] <= 95.0

    class TestGetProjectedMonthlyCost:
        """Tests for get_projected_monthly_cost method."""

        def test_empty_table_returns_zero(self, queries):
            """Test that empty table returns zero projection."""
            result = queries.get_projected_monthly_cost(based_on_days=7)
            assert result == 0.0

        def test_projects_monthly_cost(self, queries, db_connection):
            """Test monthly cost projection calculation."""
            # Insert $7 of costs (should project to $30)
            for _ in range(7):
                insert_event(db_connection, cost_usd=1.0)

            result = queries.get_projected_monthly_cost(based_on_days=7)

            # $7 over 7 days = $1/day * 30 days = $30
            assert result == pytest.approx(30.0, rel=1e-6)


@pytest.mark.integration
class TestExecuteQuery:
    """Tests for the execute_query function."""

    def test_simple_select(self, db_connection):
        """Test simple SELECT query."""
        insert_event(db_connection, cost_usd=0.01)

        # Need to patch config for this test
        import tokenledger.config as config_module
        original_config = config_module._config

        try:
            config_module._config = config_module.TokenLedgerConfig(
                database_url=TEST_DATABASE_URL,
            )

            result = execute_query(
                f"SELECT cost_usd FROM {TEST_TABLE_NAME}",
            )

            assert len(result) == 1
            assert float(result[0][0]) == pytest.approx(0.01, rel=1e-6)
        finally:
            config_module._config = original_config

    def test_parameterized_query(self, db_connection):
        """Test parameterized query."""
        insert_event(db_connection, model="gpt-4", cost_usd=0.01)
        insert_event(db_connection, model="gpt-3.5-turbo", cost_usd=0.001)

        import tokenledger.config as config_module
        original_config = config_module._config

        try:
            config_module._config = config_module.TokenLedgerConfig(
                database_url=TEST_DATABASE_URL,
            )

            result = execute_query(
                f"SELECT model, cost_usd FROM {TEST_TABLE_NAME} WHERE model = %s",
                ["gpt-4"],
            )

            assert len(result) == 1
            assert result[0][0] == "gpt-4"
        finally:
            config_module._config = original_config


@pytest.mark.integration
class TestDataclasses:
    """Tests for query result dataclasses."""

    def test_cost_summary_dataclass(self):
        """Test CostSummary dataclass."""
        summary = CostSummary(
            total_cost=100.50,
            total_tokens=10000,
            total_input_tokens=7000,
            total_output_tokens=3000,
            total_requests=50,
            avg_cost_per_request=2.01,
            avg_tokens_per_request=200.0,
        )

        assert summary.total_cost == 100.50
        assert summary.total_tokens == 10000
        assert summary.total_requests == 50

    def test_model_cost_dataclass(self):
        """Test ModelCost dataclass."""
        model_cost = ModelCost(
            model="gpt-4",
            provider="openai",
            total_cost=50.0,
            total_requests=100,
            total_tokens=50000,
            avg_cost_per_request=0.50,
        )

        assert model_cost.model == "gpt-4"
        assert model_cost.provider == "openai"

    def test_user_cost_dataclass(self):
        """Test UserCost dataclass."""
        user_cost = UserCost(
            user_id="user-123",
            total_cost=25.0,
            total_requests=50,
            total_tokens=25000,
        )

        assert user_cost.user_id == "user-123"
        assert user_cost.total_cost == 25.0

    def test_daily_cost_dataclass(self):
        """Test DailyCost dataclass."""
        now = datetime.now()
        daily_cost = DailyCost(
            date=now,
            total_cost=10.0,
            total_requests=20,
            total_tokens=10000,
        )

        assert daily_cost.date == now
        assert daily_cost.total_cost == 10.0

    def test_hourly_cost_dataclass(self):
        """Test HourlyCost dataclass."""
        now = datetime.now()
        hourly_cost = HourlyCost(
            hour=now,
            total_cost=1.0,
            total_requests=5,
        )

        assert hourly_cost.hour == now
        assert hourly_cost.total_requests == 5
