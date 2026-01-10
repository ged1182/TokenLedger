"""
TokenLedger Configuration
"""

import os
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse


@dataclass
class TokenLedgerConfig:
    """Configuration for TokenLedger"""

    # Database connection
    database_url: str = ""

    # Table configuration
    table_name: str = "token_ledger_events"
    schema_name: str = "public"

    # Batching configuration
    batch_size: int = 100
    flush_interval_seconds: float = 5.0

    # Async configuration
    async_mode: bool = True
    max_queue_size: int = 10000

    # Sampling (for high-volume apps)
    sample_rate: float = 1.0  # 1.0 = log everything, 0.1 = log 10%

    # Default metadata to include with every event
    default_metadata: dict[str, Any] = field(default_factory=dict)

    # Application identification
    app_name: str | None = None
    environment: str | None = None

    # Debug mode
    debug: bool = False

    def __post_init__(self):
        # Try to load from environment if not set
        if not self.database_url:
            self.database_url = os.getenv("TOKENLEDGER_DATABASE_URL", os.getenv("DATABASE_URL", ""))

        if not self.app_name:
            self.app_name = os.getenv("TOKENLEDGER_APP_NAME")

        if not self.environment:
            self.environment = os.getenv(
                "TOKENLEDGER_ENVIRONMENT", os.getenv("ENVIRONMENT", "development")
            )

        if os.getenv("TOKENLEDGER_DEBUG"):
            self.debug = os.getenv("TOKENLEDGER_DEBUG", "").lower() in ("1", "true", "yes")

    @property
    def is_supabase(self) -> bool:
        """Check if using Supabase"""
        if not self.database_url:
            return False
        parsed = urlparse(self.database_url)
        return "supabase" in parsed.hostname if parsed.hostname else False

    @property
    def full_table_name(self) -> str:
        """Get fully qualified table name"""
        return f"{self.schema_name}.{self.table_name}"


# Global configuration instance
_config: TokenLedgerConfig | None = None


def configure(database_url: str | None = None, **kwargs: Any) -> TokenLedgerConfig:
    """
    Configure TokenLedger.

    Args:
        database_url: PostgreSQL connection string
        **kwargs: Additional configuration options

    Returns:
        The configuration instance

    Example:
        >>> import tokenledger
        >>> tokenledger.configure(
        ...     database_url="postgresql://user:pass@localhost/db",
        ...     app_name="my-app",
        ...     environment="production"
        ... )
    """
    global _config

    config_kwargs: dict[str, Any] = {}
    if database_url:
        config_kwargs["database_url"] = database_url
    config_kwargs.update(kwargs)

    _config = TokenLedgerConfig(**config_kwargs)
    return _config


def get_config() -> TokenLedgerConfig:
    """Get the current configuration, creating default if needed."""
    global _config
    if _config is None:
        _config = TokenLedgerConfig()
    return _config
