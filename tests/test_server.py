"""Tests for the server module."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@dataclass
class MockCostSummary:
    """Mock cost summary dataclass."""

    total_cost: float = 100.0
    total_tokens: int = 50000
    total_input_tokens: int = 30000
    total_output_tokens: int = 20000
    total_requests: int = 100
    avg_cost_per_request: float = 1.0
    avg_tokens_per_request: float = 500.0


@dataclass
class MockModelCost:
    """Mock model cost dataclass."""

    model: str = "gpt-4o"
    provider: str = "openai"
    total_cost: float = 50.0
    total_requests: int = 50
    total_tokens: int = 25000
    avg_cost_per_request: float = 1.0


@dataclass
class MockUserCost:
    """Mock user cost dataclass."""

    user_id: str = "user123"
    total_cost: float = 25.0
    total_requests: int = 25
    total_tokens: int = 12500


@dataclass
class MockDailyCost:
    """Mock daily cost dataclass."""

    date: date = date(2026, 1, 20)
    total_cost: float = 10.0
    total_requests: int = 10
    total_tokens: int = 5000


@dataclass
class MockHourlyCost:
    """Mock hourly cost dataclass."""

    hour: datetime = datetime(2026, 1, 20, 14, 0, 0)
    total_cost: float = 5.0
    total_requests: int = 5


@pytest.fixture
def mock_queries():
    """Create mock queries object."""
    queries = MagicMock()
    queries.get_cost_summary.return_value = MockCostSummary()
    queries.get_projected_monthly_cost.return_value = 3000.0
    queries.get_costs_by_model.return_value = [
        MockModelCost(),
        MockModelCost(model="claude-3-5-sonnet"),
    ]
    queries.get_costs_by_user.return_value = [MockUserCost(), MockUserCost(user_id="user456")]
    queries.get_daily_costs.return_value = [MockDailyCost(), MockDailyCost(date=date(2026, 1, 21))]
    queries.get_hourly_costs.return_value = [MockHourlyCost()]
    queries.get_error_rate.return_value = {
        "total_requests": 100,
        "errors": 5,
        "error_rate": 0.05,
        "status_breakdown": {"success": 95, "error": 5},
    }
    queries.get_top_errors.return_value = [{"error": "rate_limit", "count": 3}]
    queries.get_latency_percentiles.return_value = {
        "p50_ms": 100.0,
        "p90_ms": 200.0,
        "p95_ms": 250.0,
        "p99_ms": 400.0,
        "avg_ms": 120.0,
    }
    return queries


@pytest.fixture
def client(mock_queries):
    """Create test client with mocked queries."""
    with (
        patch("tokenledger.server.get_queries", return_value=mock_queries),
        patch("tokenledger.server.USE_ASYNCPG", False),
    ):
        from tokenledger.server import app

        yield TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check_success(self, client, mock_queries) -> None:
        """Test health check returns ok when database is connected."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["database"] == "connected"
        assert data["version"] == "0.1.0"

    def test_health_check_database_error(self, mock_queries) -> None:
        """Test health check returns degraded when database fails."""
        mock_queries.get_cost_summary.side_effect = Exception("Connection failed")

        with (
            patch("tokenledger.server.get_queries", return_value=mock_queries),
            patch("tokenledger.server.USE_ASYNCPG", False),
        ):
            from tokenledger.server import app

            client = TestClient(app)
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "degraded"
            assert "error" in data["database"]


class TestSummaryEndpoint:
    """Tests for summary endpoint."""

    def test_get_summary_default(self, client, mock_queries) -> None:
        """Test getting summary with default parameters."""
        response = client.get("/api/v1/summary")
        assert response.status_code == 200
        data = response.json()
        assert data["total_cost"] == 100.0
        assert data["total_tokens"] == 50000
        assert data["total_requests"] == 100
        assert data["projected_monthly_cost"] == 3000.0

    def test_get_summary_with_filters(self, client, mock_queries) -> None:
        """Test getting summary with filters."""
        response = client.get("/api/v1/summary?days=7&user_id=user123&model=gpt-4o")
        assert response.status_code == 200
        mock_queries.get_cost_summary.assert_called_with(
            days=7, user_id="user123", model="gpt-4o", app_name=None
        )

    def test_get_summary_invalid_days(self, client) -> None:
        """Test that invalid days parameter returns error."""
        response = client.get("/api/v1/summary?days=0")
        assert response.status_code == 422  # Validation error

        response = client.get("/api/v1/summary?days=500")
        assert response.status_code == 422


class TestCostsByModelEndpoint:
    """Tests for costs by model endpoint."""

    def test_get_costs_by_model(self, client, mock_queries) -> None:
        """Test getting costs broken down by model."""
        response = client.get("/api/v1/costs/by-model")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["model"] == "gpt-4o"
        assert data[0]["provider"] == "openai"
        assert "percentage_of_total" in data[0]

    def test_get_costs_by_model_with_params(self, client, mock_queries) -> None:
        """Test getting costs by model with custom parameters."""
        response = client.get("/api/v1/costs/by-model?days=14&limit=5")
        assert response.status_code == 200
        mock_queries.get_costs_by_model.assert_called_with(days=14, limit=5)


class TestCostsByUserEndpoint:
    """Tests for costs by user endpoint."""

    def test_get_costs_by_user(self, client, mock_queries) -> None:
        """Test getting costs broken down by user."""
        response = client.get("/api/v1/costs/by-user")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["user_id"] == "user123"
        assert data[0]["total_cost"] == 25.0


class TestDailyCostsEndpoint:
    """Tests for daily costs endpoint."""

    def test_get_daily_costs(self, client, mock_queries) -> None:
        """Test getting daily cost trends."""
        response = client.get("/api/v1/costs/daily")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["date"] == "2026-01-20"
        assert data[0]["total_cost"] == 10.0

    def test_get_daily_costs_with_user_filter(self, client, mock_queries) -> None:
        """Test getting daily costs filtered by user."""
        response = client.get("/api/v1/costs/daily?user_id=user123")
        assert response.status_code == 200
        mock_queries.get_daily_costs.assert_called_with(days=30, user_id="user123")


class TestHourlyCostsEndpoint:
    """Tests for hourly costs endpoint."""

    def test_get_hourly_costs(self, client, mock_queries) -> None:
        """Test getting hourly cost trends."""
        response = client.get("/api/v1/costs/hourly")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["total_cost"] == 5.0


class TestErrorsEndpoint:
    """Tests for errors endpoint."""

    def test_get_error_stats(self, client, mock_queries) -> None:
        """Test getting error statistics."""
        response = client.get("/api/v1/errors")
        assert response.status_code == 200
        data = response.json()
        assert data["total_requests"] == 100
        assert data["errors"] == 5
        assert data["error_rate"] == 0.05

    def test_get_top_errors(self, client, mock_queries) -> None:
        """Test getting top errors."""
        response = client.get("/api/v1/errors/top")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["error"] == "rate_limit"


class TestLatencyEndpoint:
    """Tests for latency endpoint."""

    def test_get_latency_stats(self, client, mock_queries) -> None:
        """Test getting latency percentiles."""
        response = client.get("/api/v1/latency")
        assert response.status_code == 200
        data = response.json()
        assert data["p50_ms"] == 100.0
        assert data["p90_ms"] == 200.0
        assert data["p95_ms"] == 250.0
        assert data["p99_ms"] == 400.0
        assert data["avg_ms"] == 120.0


class TestErrorHandling:
    """Tests for error handling in endpoints."""

    def test_summary_handles_exception(self, mock_queries) -> None:
        """Test that summary endpoint handles exceptions."""
        mock_queries.get_cost_summary.side_effect = Exception("Database error")

        with (
            patch("tokenledger.server.get_queries", return_value=mock_queries),
            patch("tokenledger.server.USE_ASYNCPG", False),
        ):
            from tokenledger.server import app

            client = TestClient(app, raise_server_exceptions=False)
            response = client.get("/api/v1/summary")
            assert response.status_code == 500
            assert "Database error" in response.json()["detail"]


class TestResponseModels:
    """Tests for response model validation."""

    def test_cost_summary_response_model(self) -> None:
        """Test CostSummaryResponse model."""
        from tokenledger.server import CostSummaryResponse

        response = CostSummaryResponse(
            total_cost=100.0,
            total_tokens=50000,
            total_input_tokens=30000,
            total_output_tokens=20000,
            total_requests=100,
            avg_cost_per_request=1.0,
            avg_tokens_per_request=500.0,
            projected_monthly_cost=3000.0,
        )
        assert response.total_cost == 100.0

    def test_model_cost_response_model(self) -> None:
        """Test ModelCostResponse model."""
        from tokenledger.server import ModelCostResponse

        response = ModelCostResponse(
            model="gpt-4o",
            provider="openai",
            total_cost=50.0,
            total_requests=50,
            total_tokens=25000,
            avg_cost_per_request=1.0,
            percentage_of_total=50.0,
        )
        assert response.model == "gpt-4o"

    def test_health_response_model(self) -> None:
        """Test HealthResponse model."""
        from tokenledger.server import HealthResponse

        response = HealthResponse(status="ok", database="connected", version="0.1.0")
        assert response.status == "ok"
