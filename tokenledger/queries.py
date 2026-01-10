"""
TokenLedger Analytics Queries
Pre-built SQL queries for common analytics needs.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .config import get_config


@dataclass
class CostSummary:
    """Summary of costs for a period"""

    total_cost: float
    total_tokens: int
    total_input_tokens: int
    total_output_tokens: int
    total_requests: int
    avg_cost_per_request: float
    avg_tokens_per_request: float


@dataclass
class ModelCost:
    """Cost breakdown by model"""

    model: str
    provider: str
    total_cost: float
    total_requests: int
    total_tokens: int
    avg_cost_per_request: float


@dataclass
class UserCost:
    """Cost breakdown by user"""

    user_id: str
    total_cost: float
    total_requests: int
    total_tokens: int


@dataclass
class DailyCost:
    """Daily cost data"""

    date: datetime
    total_cost: float
    total_requests: int
    total_tokens: int


@dataclass
class HourlyCost:
    """Hourly cost data"""

    hour: datetime
    total_cost: float
    total_requests: int


class TokenLedgerQueries:
    """
    Pre-built analytics queries for TokenLedger data.

    Example:
        >>> from tokenledger.queries import TokenLedgerQueries
        >>>
        >>> queries = TokenLedgerQueries(connection)
        >>> summary = queries.get_cost_summary(days=30)
        >>> print(f"Last 30 days: ${summary.total_cost:.2f}")
    """

    def __init__(self, connection=None, table_name: str | None = None):
        self._connection = connection
        config = get_config()
        self.table_name = table_name or config.full_table_name

    def _get_connection(self):
        if self._connection is None:
            import psycopg2

            config = get_config()
            self._connection = psycopg2.connect(config.database_url)
        return self._connection

    def get_cost_summary(
        self,
        days: int = 30,
        user_id: str | None = None,
        model: str | None = None,
        app_name: str | None = None,
    ) -> CostSummary:
        """
        Get cost summary for a time period.

        Args:
            days: Number of days to look back
            user_id: Filter by user
            model: Filter by model
            app_name: Filter by app

        Returns:
            CostSummary with totals and averages
        """
        conn = self._get_connection()

        where_clauses = ["timestamp >= NOW() - INTERVAL '%s days'"]
        params: list[Any] = [days]

        if user_id:
            where_clauses.append("user_id = %s")
            params.append(user_id)
        if model:
            where_clauses.append("model = %s")
            params.append(model)
        if app_name:
            where_clauses.append("app_name = %s")
            params.append(app_name)

        where_sql = " AND ".join(where_clauses)

        sql = f"""
            SELECT
                COALESCE(SUM(cost_usd), 0) as total_cost,
                COALESCE(SUM(total_tokens), 0) as total_tokens,
                COALESCE(SUM(input_tokens), 0) as total_input_tokens,
                COALESCE(SUM(output_tokens), 0) as total_output_tokens,
                COUNT(*) as total_requests
            FROM {self.table_name}
            WHERE {where_sql}
        """

        with conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()

        total_cost = float(row[0] or 0)
        total_tokens = int(row[1] or 0)
        total_input = int(row[2] or 0)
        total_output = int(row[3] or 0)
        total_requests = int(row[4] or 0)

        return CostSummary(
            total_cost=total_cost,
            total_tokens=total_tokens,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_requests=total_requests,
            avg_cost_per_request=total_cost / total_requests if total_requests > 0 else 0,
            avg_tokens_per_request=total_tokens / total_requests if total_requests > 0 else 0,
        )

    def get_costs_by_model(
        self,
        days: int = 30,
        limit: int = 10,
    ) -> list[ModelCost]:
        """Get cost breakdown by model"""
        conn = self._get_connection()

        sql = f"""
            SELECT
                model,
                provider,
                COALESCE(SUM(cost_usd), 0) as total_cost,
                COUNT(*) as total_requests,
                COALESCE(SUM(total_tokens), 0) as total_tokens
            FROM {self.table_name}
            WHERE timestamp >= NOW() - INTERVAL '%s days'
            GROUP BY model, provider
            ORDER BY total_cost DESC
            LIMIT %s
        """

        with conn.cursor() as cur:
            cur.execute(sql, [days, limit])
            rows = cur.fetchall()

        results = []
        for row in rows:
            total_cost = float(row[2] or 0)
            total_requests = int(row[3] or 0)
            results.append(
                ModelCost(
                    model=row[0],
                    provider=row[1],
                    total_cost=total_cost,
                    total_requests=total_requests,
                    total_tokens=int(row[4] or 0),
                    avg_cost_per_request=total_cost / total_requests if total_requests > 0 else 0,
                )
            )

        return results

    def get_costs_by_user(
        self,
        days: int = 30,
        limit: int = 20,
    ) -> list[UserCost]:
        """Get cost breakdown by user"""
        conn = self._get_connection()

        sql = f"""
            SELECT
                COALESCE(user_id, 'anonymous') as user_id,
                COALESCE(SUM(cost_usd), 0) as total_cost,
                COUNT(*) as total_requests,
                COALESCE(SUM(total_tokens), 0) as total_tokens
            FROM {self.table_name}
            WHERE timestamp >= NOW() - INTERVAL '%s days'
            GROUP BY user_id
            ORDER BY total_cost DESC
            LIMIT %s
        """

        with conn.cursor() as cur:
            cur.execute(sql, [days, limit])
            rows = cur.fetchall()

        return [
            UserCost(
                user_id=row[0],
                total_cost=float(row[1] or 0),
                total_requests=int(row[2] or 0),
                total_tokens=int(row[3] or 0),
            )
            for row in rows
        ]

    def get_daily_costs(
        self,
        days: int = 30,
        user_id: str | None = None,
    ) -> list[DailyCost]:
        """Get daily cost trends"""
        conn = self._get_connection()

        where_clauses = ["timestamp >= NOW() - INTERVAL '%s days'"]
        params: list[Any] = [days]

        if user_id:
            where_clauses.append("user_id = %s")
            params.append(user_id)

        where_sql = " AND ".join(where_clauses)

        sql = f"""
            SELECT
                DATE(timestamp) as date,
                COALESCE(SUM(cost_usd), 0) as total_cost,
                COUNT(*) as total_requests,
                COALESCE(SUM(total_tokens), 0) as total_tokens
            FROM {self.table_name}
            WHERE {where_sql}
            GROUP BY DATE(timestamp)
            ORDER BY date ASC
        """

        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        return [
            DailyCost(
                date=row[0],
                total_cost=float(row[1] or 0),
                total_requests=int(row[2] or 0),
                total_tokens=int(row[3] or 0),
            )
            for row in rows
        ]

    def get_hourly_costs(
        self,
        hours: int = 24,
    ) -> list[HourlyCost]:
        """Get hourly cost trends"""
        conn = self._get_connection()

        sql = f"""
            SELECT
                DATE_TRUNC('hour', timestamp) as hour,
                COALESCE(SUM(cost_usd), 0) as total_cost,
                COUNT(*) as total_requests
            FROM {self.table_name}
            WHERE timestamp >= NOW() - INTERVAL '%s hours'
            GROUP BY DATE_TRUNC('hour', timestamp)
            ORDER BY hour ASC
        """

        with conn.cursor() as cur:
            cur.execute(sql, [hours])
            rows = cur.fetchall()

        return [
            HourlyCost(
                hour=row[0],
                total_cost=float(row[1] or 0),
                total_requests=int(row[2] or 0),
            )
            for row in rows
        ]

    def get_error_rate(
        self,
        days: int = 7,
    ) -> dict[str, Any]:
        """Get error rate statistics"""
        conn = self._get_connection()

        sql = f"""
            SELECT
                status,
                COUNT(*) as count
            FROM {self.table_name}
            WHERE timestamp >= NOW() - INTERVAL '%s days'
            GROUP BY status
        """

        with conn.cursor() as cur:
            cur.execute(sql, [days])
            rows = cur.fetchall()

        status_counts = {row[0]: row[1] for row in rows}
        total = sum(status_counts.values())
        errors = status_counts.get("error", 0)

        return {
            "total_requests": total,
            "errors": errors,
            "error_rate": errors / total if total > 0 else 0,
            "status_breakdown": status_counts,
        }

    def get_top_errors(
        self,
        days: int = 7,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get most common errors"""
        conn = self._get_connection()

        sql = f"""
            SELECT
                error_type,
                error_message,
                model,
                COUNT(*) as count
            FROM {self.table_name}
            WHERE timestamp >= NOW() - INTERVAL '%s days'
              AND status = 'error'
            GROUP BY error_type, error_message, model
            ORDER BY count DESC
            LIMIT %s
        """

        with conn.cursor() as cur:
            cur.execute(sql, [days, limit])
            rows = cur.fetchall()

        return [
            {
                "error_type": row[0],
                "error_message": row[1],
                "model": row[2],
                "count": row[3],
            }
            for row in rows
        ]

    def get_latency_percentiles(
        self,
        days: int = 7,
    ) -> dict[str, float]:
        """Get latency percentiles"""
        conn = self._get_connection()

        sql = f"""
            SELECT
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY duration_ms) as p50,
                PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY duration_ms) as p90,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms) as p95,
                PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY duration_ms) as p99,
                AVG(duration_ms) as avg
            FROM {self.table_name}
            WHERE timestamp >= NOW() - INTERVAL '%s days'
              AND duration_ms IS NOT NULL
        """

        with conn.cursor() as cur:
            cur.execute(sql, [days])
            row = cur.fetchone()

        return {
            "p50_ms": float(row[0] or 0),
            "p90_ms": float(row[1] or 0),
            "p95_ms": float(row[2] or 0),
            "p99_ms": float(row[3] or 0),
            "avg_ms": float(row[4] or 0),
        }

    def get_projected_monthly_cost(
        self,
        based_on_days: int = 7,
    ) -> float:
        """Project monthly cost based on recent usage"""
        summary = self.get_cost_summary(days=based_on_days)
        daily_avg = summary.total_cost / based_on_days if based_on_days > 0 else 0
        return daily_avg * 30


# Convenience function for raw SQL queries
def execute_query(sql: str, params: list | None = None) -> list[tuple]:
    """
    Execute a raw SQL query against the TokenLedger table.

    Args:
        sql: SQL query (use %s for parameters)
        params: Query parameters

    Returns:
        List of result tuples
    """
    import psycopg2

    config = get_config()

    conn = psycopg2.connect(config.database_url)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params or [])
            return cur.fetchall()
    finally:
        conn.close()
