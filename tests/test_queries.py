"""Tests for the queries module."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from tokenledger.queries import (
    AsyncTokenLedgerQueries,
    CostSummary,
    DailyCost,
    HourlyCost,
    ModelCost,
    TokenLedgerQueries,
    UserCost,
)


class TestDataclasses:
    """Tests for query result dataclasses."""

    def test_cost_summary(self) -> None:
        """Test CostSummary dataclass."""
        summary = CostSummary(
            total_cost=100.50,
            total_tokens=50000,
            total_input_tokens=30000,
            total_output_tokens=20000,
            total_requests=100,
            avg_cost_per_request=1.005,
            avg_tokens_per_request=500,
        )

        assert summary.total_cost == 100.50
        assert summary.total_tokens == 50000
        assert summary.total_input_tokens == 30000
        assert summary.total_output_tokens == 20000
        assert summary.total_requests == 100
        assert summary.avg_cost_per_request == 1.005
        assert summary.avg_tokens_per_request == 500

    def test_model_cost(self) -> None:
        """Test ModelCost dataclass."""
        model_cost = ModelCost(
            model="gpt-4o",
            provider="openai",
            total_cost=50.25,
            total_requests=50,
            total_tokens=25000,
            avg_cost_per_request=1.005,
        )

        assert model_cost.model == "gpt-4o"
        assert model_cost.provider == "openai"
        assert model_cost.total_cost == 50.25
        assert model_cost.total_requests == 50
        assert model_cost.total_tokens == 25000
        assert model_cost.avg_cost_per_request == 1.005

    def test_user_cost(self) -> None:
        """Test UserCost dataclass."""
        user_cost = UserCost(
            user_id="user-123",
            total_cost=25.00,
            total_requests=25,
            total_tokens=12500,
        )

        assert user_cost.user_id == "user-123"
        assert user_cost.total_cost == 25.00
        assert user_cost.total_requests == 25
        assert user_cost.total_tokens == 12500

    def test_daily_cost(self) -> None:
        """Test DailyCost dataclass."""
        date = datetime(2024, 1, 15)
        daily_cost = DailyCost(
            date=date,
            total_cost=10.00,
            total_requests=10,
            total_tokens=5000,
        )

        assert daily_cost.date == date
        assert daily_cost.total_cost == 10.00
        assert daily_cost.total_requests == 10
        assert daily_cost.total_tokens == 5000

    def test_hourly_cost(self) -> None:
        """Test HourlyCost dataclass."""
        hour = datetime(2024, 1, 15, 14, 0, 0)
        hourly_cost = HourlyCost(
            hour=hour,
            total_cost=1.50,
            total_requests=3,
        )

        assert hourly_cost.hour == hour
        assert hourly_cost.total_cost == 1.50
        assert hourly_cost.total_requests == 3


class TestTokenLedgerQueries:
    """Tests for TokenLedgerQueries class."""

    def test_initialization_with_connection(self) -> None:
        """Test initialization with provided connection."""
        mock_conn = MagicMock()
        queries = TokenLedgerQueries(connection=mock_conn)

        assert queries._connection == mock_conn

    def test_initialization_with_custom_table_name(self) -> None:
        """Test initialization with custom table name."""
        mock_conn = MagicMock()
        queries = TokenLedgerQueries(connection=mock_conn, table_name="custom.events")

        assert queries.table_name == "custom.events"

    def test_initialization_uses_config_table_name(self) -> None:
        """Test initialization uses config table name by default."""
        from tokenledger.config import configure

        configure(
            database_url="postgresql://test:test@localhost/test",
            table_name="config_events",
            schema_name="my_schema",
        )

        mock_conn = MagicMock()
        queries = TokenLedgerQueries(connection=mock_conn)

        assert queries.table_name == "my_schema.config_events"

    def test_get_cost_summary(self) -> None:
        """Test get_cost_summary query execution."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchone.return_value = (
            Decimal("100.50"),  # total_cost
            50000,  # total_tokens
            30000,  # total_input_tokens
            20000,  # total_output_tokens
            100,  # total_requests
        )

        queries = TokenLedgerQueries(connection=mock_conn)
        summary = queries.get_cost_summary(days=30)

        assert summary.total_cost == 100.50
        assert summary.total_tokens == 50000
        assert summary.total_input_tokens == 30000
        assert summary.total_output_tokens == 20000
        assert summary.total_requests == 100
        assert summary.avg_cost_per_request == pytest.approx(1.005)
        assert summary.avg_tokens_per_request == 500

    def test_get_cost_summary_with_filters(self) -> None:
        """Test get_cost_summary with various filters."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchone.return_value = (Decimal("50.00"), 25000, 15000, 10000, 50)

        queries = TokenLedgerQueries(connection=mock_conn)
        queries.get_cost_summary(
            days=7,
            user_id="user-123",
            model="gpt-4o",
            app_name="my-app",
        )

        # Verify query was called with filters
        mock_cursor.execute.assert_called_once()
        call_args = mock_cursor.execute.call_args
        assert call_args[0][1] == [7, "user-123", "gpt-4o", "my-app"]

    def test_get_cost_summary_zero_requests(self) -> None:
        """Test get_cost_summary handles zero requests."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchone.return_value = (None, None, None, None, 0)

        queries = TokenLedgerQueries(connection=mock_conn)
        summary = queries.get_cost_summary(days=30)

        assert summary.total_cost == 0
        assert summary.total_tokens == 0
        assert summary.total_requests == 0
        assert summary.avg_cost_per_request == 0
        assert summary.avg_tokens_per_request == 0

    def test_get_costs_by_model(self) -> None:
        """Test get_costs_by_model query."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = [
            ("gpt-4o", "openai", Decimal("50.00"), 100, 50000),
            ("claude-3-5-sonnet-20241022", "anthropic", Decimal("30.00"), 50, 25000),
        ]

        queries = TokenLedgerQueries(connection=mock_conn)
        results = queries.get_costs_by_model(days=30, limit=10)

        assert len(results) == 2
        assert results[0].model == "gpt-4o"
        assert results[0].provider == "openai"
        assert results[0].total_cost == 50.00
        assert results[0].total_requests == 100
        assert results[0].total_tokens == 50000
        assert results[0].avg_cost_per_request == 0.50

    def test_get_costs_by_user(self) -> None:
        """Test get_costs_by_user query."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = [
            ("user-123", Decimal("25.00"), 50, 25000),
            ("user-456", Decimal("15.00"), 30, 15000),
        ]

        queries = TokenLedgerQueries(connection=mock_conn)
        results = queries.get_costs_by_user(days=30, limit=20)

        assert len(results) == 2
        assert results[0].user_id == "user-123"
        assert results[0].total_cost == 25.00
        assert results[0].total_requests == 50
        assert results[1].user_id == "user-456"

    def test_get_daily_costs(self) -> None:
        """Test get_daily_costs query."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = [
            (datetime(2024, 1, 14), Decimal("10.00"), 20, 10000),
            (datetime(2024, 1, 15), Decimal("12.00"), 25, 12500),
        ]

        queries = TokenLedgerQueries(connection=mock_conn)
        results = queries.get_daily_costs(days=30)

        assert len(results) == 2
        assert results[0].date == datetime(2024, 1, 14)
        assert results[0].total_cost == 10.00
        assert results[0].total_requests == 20
        assert results[1].total_cost == 12.00

    def test_get_daily_costs_with_user_filter(self) -> None:
        """Test get_daily_costs with user filter."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = []

        queries = TokenLedgerQueries(connection=mock_conn)
        queries.get_daily_costs(days=7, user_id="user-123")

        call_args = mock_cursor.execute.call_args
        assert call_args[0][1] == [7, "user-123"]

    def test_get_hourly_costs(self) -> None:
        """Test get_hourly_costs query."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = [
            (datetime(2024, 1, 15, 10, 0), Decimal("1.00"), 5),
            (datetime(2024, 1, 15, 11, 0), Decimal("2.00"), 10),
        ]

        queries = TokenLedgerQueries(connection=mock_conn)
        results = queries.get_hourly_costs(hours=24)

        assert len(results) == 2
        assert results[0].hour == datetime(2024, 1, 15, 10, 0)
        assert results[0].total_cost == 1.00
        assert results[0].total_requests == 5

    def test_get_error_rate(self) -> None:
        """Test get_error_rate query."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = [
            ("success", 95),
            ("error", 5),
        ]

        queries = TokenLedgerQueries(connection=mock_conn)
        result = queries.get_error_rate(days=7)

        assert result["total_requests"] == 100
        assert result["errors"] == 5
        assert result["error_rate"] == 0.05
        assert result["status_breakdown"] == {"success": 95, "error": 5}

    def test_get_error_rate_no_errors(self) -> None:
        """Test get_error_rate with no errors."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = [("success", 100)]

        queries = TokenLedgerQueries(connection=mock_conn)
        result = queries.get_error_rate(days=7)

        assert result["errors"] == 0
        assert result["error_rate"] == 0

    def test_get_error_rate_zero_requests(self) -> None:
        """Test get_error_rate with zero requests."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = []

        queries = TokenLedgerQueries(connection=mock_conn)
        result = queries.get_error_rate(days=7)

        assert result["total_requests"] == 0
        assert result["error_rate"] == 0

    def test_get_top_errors(self) -> None:
        """Test get_top_errors query."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = [
            ("RateLimitError", "Rate limit exceeded", "gpt-4o", 10),
            ("APIError", "Service unavailable", "claude-3-5-sonnet", 5),
        ]

        queries = TokenLedgerQueries(connection=mock_conn)
        results = queries.get_top_errors(days=7, limit=10)

        assert len(results) == 2
        assert results[0]["error_type"] == "RateLimitError"
        assert results[0]["error_message"] == "Rate limit exceeded"
        assert results[0]["model"] == "gpt-4o"
        assert results[0]["count"] == 10

    def test_get_latency_percentiles(self) -> None:
        """Test get_latency_percentiles query."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchone.return_value = (
            Decimal("100.5"),  # p50
            Decimal("250.0"),  # p90
            Decimal("500.0"),  # p95
            Decimal("1000.0"),  # p99
            Decimal("150.0"),  # avg
        )

        queries = TokenLedgerQueries(connection=mock_conn)
        result = queries.get_latency_percentiles(days=7)

        assert result["p50_ms"] == 100.5
        assert result["p90_ms"] == 250.0
        assert result["p95_ms"] == 500.0
        assert result["p99_ms"] == 1000.0
        assert result["avg_ms"] == 150.0

    def test_get_projected_monthly_cost(self) -> None:
        """Test get_projected_monthly_cost calculation."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        # 7 days of data, $70 total = $10/day avg = $300/month projected
        mock_cursor.fetchone.return_value = (Decimal("70.00"), 35000, 21000, 14000, 70)

        queries = TokenLedgerQueries(connection=mock_conn)
        projection = queries.get_projected_monthly_cost(based_on_days=7)

        assert projection == pytest.approx(300.0)

    def test_get_projected_monthly_cost_zero_days(self) -> None:
        """Test get_projected_monthly_cost with zero days."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchone.return_value = (Decimal("0"), 0, 0, 0, 0)

        queries = TokenLedgerQueries(connection=mock_conn)
        projection = queries.get_projected_monthly_cost(based_on_days=0)

        assert projection == 0


class TestAsyncTokenLedgerQueries:
    """Tests for AsyncTokenLedgerQueries class."""

    def test_initialization_with_db(self) -> None:
        """Test initialization with provided db."""
        mock_db = MagicMock()
        queries = AsyncTokenLedgerQueries(db=mock_db)

        assert queries._db == mock_db

    def test_initialization_with_custom_table_name(self) -> None:
        """Test initialization with custom table name."""
        mock_db = MagicMock()
        queries = AsyncTokenLedgerQueries(db=mock_db, table_name="custom.async_events")

        assert queries.table_name == "custom.async_events"

    @pytest.mark.asyncio
    async def test_async_get_cost_summary(self) -> None:
        """Test async get_cost_summary query."""
        mock_db = AsyncMock()
        mock_db.fetchrow = AsyncMock(
            return_value=[
                Decimal("100.50"),  # total_cost
                50000,  # total_tokens
                30000,  # total_input_tokens
                20000,  # total_output_tokens
                100,  # total_requests
            ]
        )

        queries = AsyncTokenLedgerQueries(db=mock_db)
        summary = await queries.get_cost_summary(days=30)

        assert summary.total_cost == 100.50
        assert summary.total_tokens == 50000
        assert summary.total_requests == 100

    @pytest.mark.asyncio
    async def test_async_get_cost_summary_with_filters(self) -> None:
        """Test async get_cost_summary with filters."""
        mock_db = AsyncMock()
        mock_db.fetchrow = AsyncMock(return_value=[Decimal("50.00"), 25000, 15000, 10000, 50])

        queries = AsyncTokenLedgerQueries(db=mock_db)
        await queries.get_cost_summary(
            days=7,
            user_id="user-123",
            model="gpt-4o",
            app_name="my-app",
        )

        # Verify fetchrow was called with appropriate params
        mock_db.fetchrow.assert_called_once()
        call_args = mock_db.fetchrow.call_args
        assert "user-123" in call_args[0]
        assert "gpt-4o" in call_args[0]
        assert "my-app" in call_args[0]

    @pytest.mark.asyncio
    async def test_async_get_costs_by_model(self) -> None:
        """Test async get_costs_by_model query."""
        mock_db = AsyncMock()
        mock_db.fetch = AsyncMock(
            return_value=[
                ["gpt-4o", "openai", Decimal("50.00"), 100, 50000],
                ["claude-3-5-sonnet", "anthropic", Decimal("30.00"), 50, 25000],
            ]
        )

        queries = AsyncTokenLedgerQueries(db=mock_db)
        results = await queries.get_costs_by_model(days=30, limit=10)

        assert len(results) == 2
        assert results[0].model == "gpt-4o"
        assert results[0].total_cost == 50.00

    @pytest.mark.asyncio
    async def test_async_get_costs_by_user(self) -> None:
        """Test async get_costs_by_user query."""
        mock_db = AsyncMock()
        mock_db.fetch = AsyncMock(
            return_value=[
                ["user-123", Decimal("25.00"), 50, 25000],
                ["user-456", Decimal("15.00"), 30, 15000],
            ]
        )

        queries = AsyncTokenLedgerQueries(db=mock_db)
        results = await queries.get_costs_by_user(days=30, limit=20)

        assert len(results) == 2
        assert results[0].user_id == "user-123"
        assert results[0].total_cost == 25.00

    @pytest.mark.asyncio
    async def test_async_get_daily_costs(self) -> None:
        """Test async get_daily_costs query."""
        mock_db = AsyncMock()
        mock_db.fetch = AsyncMock(
            return_value=[
                [datetime(2024, 1, 14), Decimal("10.00"), 20, 10000],
                [datetime(2024, 1, 15), Decimal("12.00"), 25, 12500],
            ]
        )

        queries = AsyncTokenLedgerQueries(db=mock_db)
        results = await queries.get_daily_costs(days=30)

        assert len(results) == 2
        assert results[0].date == datetime(2024, 1, 14)
        assert results[0].total_cost == 10.00

    @pytest.mark.asyncio
    async def test_async_get_hourly_costs(self) -> None:
        """Test async get_hourly_costs query."""
        mock_db = AsyncMock()
        mock_db.fetch = AsyncMock(
            return_value=[
                [datetime(2024, 1, 15, 10, 0), Decimal("1.00"), 5],
                [datetime(2024, 1, 15, 11, 0), Decimal("2.00"), 10],
            ]
        )

        queries = AsyncTokenLedgerQueries(db=mock_db)
        results = await queries.get_hourly_costs(hours=24)

        assert len(results) == 2
        assert results[0].total_cost == 1.00

    @pytest.mark.asyncio
    async def test_async_get_error_rate(self) -> None:
        """Test async get_error_rate query."""
        mock_db = AsyncMock()
        mock_db.fetch = AsyncMock(
            return_value=[
                ["success", 95],
                ["error", 5],
            ]
        )

        queries = AsyncTokenLedgerQueries(db=mock_db)
        result = await queries.get_error_rate(days=7)

        assert result["total_requests"] == 100
        assert result["errors"] == 5
        assert result["error_rate"] == 0.05

    @pytest.mark.asyncio
    async def test_async_get_top_errors(self) -> None:
        """Test async get_top_errors query."""
        mock_db = AsyncMock()
        mock_db.fetch = AsyncMock(
            return_value=[
                ["RateLimitError", "Rate limit exceeded", "gpt-4o", 10],
            ]
        )

        queries = AsyncTokenLedgerQueries(db=mock_db)
        results = await queries.get_top_errors(days=7, limit=10)

        assert len(results) == 1
        assert results[0]["error_type"] == "RateLimitError"
        assert results[0]["count"] == 10

    @pytest.mark.asyncio
    async def test_async_get_latency_percentiles(self) -> None:
        """Test async get_latency_percentiles query."""
        mock_db = AsyncMock()
        mock_db.fetchrow = AsyncMock(
            return_value=[
                Decimal("100.5"),
                Decimal("250.0"),
                Decimal("500.0"),
                Decimal("1000.0"),
                Decimal("150.0"),
            ]
        )

        queries = AsyncTokenLedgerQueries(db=mock_db)
        result = await queries.get_latency_percentiles(days=7)

        assert result["p50_ms"] == 100.5
        assert result["p99_ms"] == 1000.0

    @pytest.mark.asyncio
    async def test_async_get_projected_monthly_cost(self) -> None:
        """Test async get_projected_monthly_cost calculation."""
        mock_db = AsyncMock()
        mock_db.fetchrow = AsyncMock(return_value=[Decimal("70.00"), 35000, 21000, 14000, 70])

        queries = AsyncTokenLedgerQueries(db=mock_db)
        projection = await queries.get_projected_monthly_cost(based_on_days=7)

        assert projection == pytest.approx(300.0)
