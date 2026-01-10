"""
TokenLedger Dashboard API
FastAPI server providing analytics endpoints for the dashboard.
"""

from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import get_config
from .queries import TokenLedgerQueries

# Initialize FastAPI
app = FastAPI(
    title="TokenLedger API",
    description="LLM Cost Analytics API",
    version="0.1.0",
)

# Add CORS middleware for dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Response models
class CostSummaryResponse(BaseModel):
    total_cost: float
    total_tokens: int
    total_input_tokens: int
    total_output_tokens: int
    total_requests: int
    avg_cost_per_request: float
    avg_tokens_per_request: float
    projected_monthly_cost: float


class ModelCostResponse(BaseModel):
    model: str
    provider: str
    total_cost: float
    total_requests: int
    total_tokens: int
    avg_cost_per_request: float
    percentage_of_total: float


class UserCostResponse(BaseModel):
    user_id: str
    total_cost: float
    total_requests: int
    total_tokens: int


class DailyCostResponse(BaseModel):
    date: str
    total_cost: float
    total_requests: int
    total_tokens: int


class HourlyCostResponse(BaseModel):
    hour: str
    total_cost: float
    total_requests: int


class ErrorStatsResponse(BaseModel):
    total_requests: int
    errors: int
    error_rate: float
    status_breakdown: dict[str, int]


class LatencyResponse(BaseModel):
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    avg_ms: float


class HealthResponse(BaseModel):
    status: str
    database: str
    version: str


# Module-level connection for reuse
_connection = None


def _get_connection():
    """Get or create a reusable database connection"""
    global _connection
    if _connection is None:
        import psycopg2

        config = get_config()
        _connection = psycopg2.connect(config.database_url)
    return _connection


def get_queries() -> TokenLedgerQueries:
    """Get queries instance with database connection"""
    return TokenLedgerQueries(_get_connection())


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        queries = get_queries()
        # Simple query to test connection
        queries.get_cost_summary(days=1)
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {e!s}"

    return HealthResponse(
        status="ok" if db_status == "connected" else "degraded", database=db_status, version="0.1.0"
    )


@app.get("/api/v1/summary", response_model=CostSummaryResponse)
async def get_summary(
    days: int = Query(30, ge=1, le=365),
    user_id: str | None = None,
    model: str | None = None,
    app_name: str | None = None,
):
    """Get cost summary for a time period"""
    try:
        queries = get_queries()
        summary = queries.get_cost_summary(
            days=days,
            user_id=user_id,
            model=model,
            app_name=app_name,
        )
        projected = queries.get_projected_monthly_cost(based_on_days=min(days, 7))

        return CostSummaryResponse(
            total_cost=summary.total_cost,
            total_tokens=summary.total_tokens,
            total_input_tokens=summary.total_input_tokens,
            total_output_tokens=summary.total_output_tokens,
            total_requests=summary.total_requests,
            avg_cost_per_request=summary.avg_cost_per_request,
            avg_tokens_per_request=summary.avg_tokens_per_request,
            projected_monthly_cost=projected,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/costs/by-model", response_model=list[ModelCostResponse])
async def get_costs_by_model(
    days: int = Query(30, ge=1, le=365),
    limit: int = Query(10, ge=1, le=50),
):
    """Get cost breakdown by model"""
    try:
        queries = get_queries()
        models = queries.get_costs_by_model(days=days, limit=limit)

        # Calculate total for percentages
        total_cost = sum(m.total_cost for m in models)

        return [
            ModelCostResponse(
                model=m.model,
                provider=m.provider,
                total_cost=m.total_cost,
                total_requests=m.total_requests,
                total_tokens=m.total_tokens,
                avg_cost_per_request=m.avg_cost_per_request,
                percentage_of_total=(m.total_cost / total_cost * 100) if total_cost > 0 else 0,
            )
            for m in models
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/costs/by-user", response_model=list[UserCostResponse])
async def get_costs_by_user(
    days: int = Query(30, ge=1, le=365),
    limit: int = Query(20, ge=1, le=100),
):
    """Get cost breakdown by user"""
    try:
        queries = get_queries()
        users = queries.get_costs_by_user(days=days, limit=limit)

        return [
            UserCostResponse(
                user_id=u.user_id,
                total_cost=u.total_cost,
                total_requests=u.total_requests,
                total_tokens=u.total_tokens,
            )
            for u in users
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/costs/daily", response_model=list[DailyCostResponse])
async def get_daily_costs(
    days: int = Query(30, ge=1, le=365),
    user_id: str | None = None,
):
    """Get daily cost trends"""
    try:
        queries = get_queries()
        daily = queries.get_daily_costs(days=days, user_id=user_id)

        return [
            DailyCostResponse(
                date=d.date.isoformat() if hasattr(d.date, "isoformat") else str(d.date),
                total_cost=d.total_cost,
                total_requests=d.total_requests,
                total_tokens=d.total_tokens,
            )
            for d in daily
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/costs/hourly", response_model=list[HourlyCostResponse])
async def get_hourly_costs(
    hours: int = Query(24, ge=1, le=168),
):
    """Get hourly cost trends"""
    try:
        queries = get_queries()
        hourly = queries.get_hourly_costs(hours=hours)

        return [
            HourlyCostResponse(
                hour=h.hour.isoformat() if hasattr(h.hour, "isoformat") else str(h.hour),
                total_cost=h.total_cost,
                total_requests=h.total_requests,
            )
            for h in hourly
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/errors", response_model=ErrorStatsResponse)
async def get_error_stats(
    days: int = Query(7, ge=1, le=90),
):
    """Get error rate statistics"""
    try:
        queries = get_queries()
        stats = queries.get_error_rate(days=days)

        return ErrorStatsResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/errors/top", response_model=list[dict[str, Any]])
async def get_top_errors(
    days: int = Query(7, ge=1, le=90),
    limit: int = Query(10, ge=1, le=50),
):
    """Get most common errors"""
    try:
        queries = get_queries()
        return queries.get_top_errors(days=days, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/latency", response_model=LatencyResponse)
async def get_latency_stats(
    days: int = Query(7, ge=1, le=90),
):
    """Get latency percentiles"""
    try:
        queries = get_queries()
        stats = queries.get_latency_percentiles(days=days)

        return LatencyResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


def run_server(host: str = "0.0.0.0", port: int = 8765):
    """Run the dashboard API server"""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
