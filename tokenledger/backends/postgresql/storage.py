"""
PostgreSQL Storage Backend

Synchronous storage backend using psycopg2 or psycopg3.
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, Any

from tokenledger.backends.base import BaseStorageBackend
from tokenledger.backends.exceptions import ConnectionError, DriverNotFoundError
from tokenledger.backends.protocol import BackendCapabilities

from .schema import (
    get_full_schema_sql,
    get_health_check_sql,
    get_insert_sql_psycopg2,
    get_insert_sql_psycopg3,
)

if TYPE_CHECKING:
    from tokenledger.config import TokenLedgerConfig  # noqa: F401

logger = logging.getLogger("tokenledger.backends.postgresql")


class PostgreSQLBackend(BaseStorageBackend):
    """
    PostgreSQL storage backend using psycopg2 or psycopg3.

    Automatically selects the available driver (prefers psycopg2 for
    compatibility, but works with either).

    Example:
        >>> from tokenledger.backends.postgresql import PostgreSQLBackend
        >>> from tokenledger.config import TokenLedgerConfig
        >>>
        >>> config = TokenLedgerConfig(database_url="postgresql://localhost/mydb")
        >>> backend = PostgreSQLBackend()
        >>> backend.initialize(config)
        >>>
        >>> events = [{"event_id": "...", "provider": "openai", ...}]
        >>> backend.write_events(events)
        >>> backend.close()
    """

    def __init__(self) -> None:
        super().__init__()
        self._use_psycopg2: bool = False
        self._execute_values: Any = None  # psycopg2's execute_values function
        self._driver_name: str = "unknown"
        self._driver_version: str = "unknown"

    @property
    def name(self) -> str:
        return "PostgreSQL"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_jsonb=True,
            supports_uuid=True,
            supports_decimal=True,
            supports_upsert=True,
            supports_returning=True,
            supports_batch_insert=True,
            supports_window_functions=True,
            supports_cte=True,
            supports_percentile=True,
            max_batch_size=10000,
            recommended_batch_size=1000,
        )

    def _connect(self) -> None:
        """Establish database connection using psycopg2 or psycopg3."""
        if not self._config:
            raise ConnectionError("No configuration provided", backend_name=self.name)

        # Try psycopg2 first (more common)
        try:
            import psycopg2
            from psycopg2.extras import execute_values

            self._connection = psycopg2.connect(self._config.database_url)
            self._execute_values = execute_values
            self._use_psycopg2 = True
            self._driver_name = "psycopg2"
            try:
                self._driver_version = psycopg2.__version__
            except AttributeError:
                self._driver_version = "unknown"
            logger.debug("Connected using psycopg2")
            return
        except ImportError:
            pass

        # Fall back to psycopg3
        try:
            import psycopg

            self._connection = psycopg.connect(self._config.database_url)
            self._use_psycopg2 = False
            self._driver_name = "psycopg"
            try:
                self._driver_version = psycopg.__version__
            except AttributeError:
                self._driver_version = "unknown"
            logger.debug("Connected using psycopg3")
            return
        except ImportError:
            pass

        # Neither driver found
        raise DriverNotFoundError(
            driver_name="psycopg2 or psycopg",
            install_hint="pip install tokenledger[postgres]",
            backend_name=self.name,
        )

    def _create_schema(self) -> None:
        """Create the events table and indexes."""
        if not self._config or not self._connection:
            return

        sql = get_full_schema_sql(self._config)

        with self._connection.cursor() as cur:
            cur.execute(sql)
        self._connection.commit()

        if self._config.debug:
            logger.info(f"Created schema for table {self._config.full_table_name}")

    def _write_batch(self, events: list[dict[str, Any]]) -> int:
        """Write a batch of events using the appropriate driver."""
        if not self._config or not self._connection:
            return 0

        values = [self._prepare_event_values(event) for event in events]

        try:
            with self._connection.cursor() as cur:
                if self._use_psycopg2 and self._execute_values:
                    sql = get_insert_sql_psycopg2(self._config, self.COLUMNS)
                    self._execute_values(cur, sql, values)
                else:
                    sql = get_insert_sql_psycopg3(self._config, self.COLUMNS)
                    cur.executemany(sql, values)
            self._connection.commit()
            return len(events)
        except Exception as e:
            logger.error(f"Error writing batch: {e}")
            self._connection.rollback()
            raise

    def _health_check(self) -> bool:
        """Check if the database connection is healthy."""
        if not self._connection:
            return False

        try:
            with self._connection.cursor() as cur:
                cur.execute(get_health_check_sql())
                result = cur.fetchone()
                return result is not None and result[0] == 1
        except Exception:
            return False

    def _close(self) -> None:
        """Close the database connection."""
        if self._connection:
            self._connection.close()

    def _get_version(self) -> str:
        """Get PostgreSQL server version."""
        if not self._connection:
            return "unknown"

        try:
            with self._connection.cursor() as cur:
                cur.execute("SELECT version()")
                result = cur.fetchone()
                if result:
                    # Parse "PostgreSQL 15.4 ..." to just "15.4"
                    version_str = result[0]
                    parts = version_str.split()
                    if len(parts) >= 2:
                        return parts[1]
                    return version_str
        except Exception:
            pass
        return "unknown"

    def _get_driver_name(self) -> str:
        """Get the driver name."""
        return f"{self._driver_name} {self._driver_version}"

    def reconnect(self) -> None:
        """Reconnect to the database if the connection was lost."""
        if self._connection:
            with contextlib.suppress(Exception):
                self._connection.close()
            self._connection = None

        self._connect()
        logger.info("Reconnected to PostgreSQL")
