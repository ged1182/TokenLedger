"""
TokenLedger Backends Package

Provides pluggable storage and query backends for TokenLedger.

Usage:
    >>> from tokenledger.backends import registry
    >>> backend = registry.create_storage_backend("postgresql", config)

    Or use auto-detection from URL:
    >>> backend = registry.get_backend_for_url("postgresql://localhost/mydb", config)

Available backends (built-in):
    - postgresql / postgres: PostgreSQL with psycopg2/psycopg3
    - asyncpg: PostgreSQL with asyncpg (async)

Additional backends available via plugins:
    - pip install tokenledger-clickhouse
    - pip install tokenledger-bigquery
    - pip install tokenledger-snowflake
"""

from .base import BaseAsyncStorageBackend, BaseStorageBackend
from .exceptions import (
    BackendError,
    BackendNotFoundError,
    BackendNotInitializedError,
    ConnectionError,
    DriverNotFoundError,
    InitializationError,
    ReadError,
    SchemaError,
    WriteError,
)
from .protocol import (
    AsyncQueryBackend,
    AsyncStorageBackend,
    BackendCapabilities,
    BackendInfo,
    QueryBackend,
    StorageBackend,
)
from .registry import (
    clear_registered_backends,
    create_storage_backend,
    get_available_query_backends,
    get_available_storage_backends,
    get_backend_for_url,
    load_query_backend,
    load_storage_backend,
    parse_database_url,
    register_query_backend,
    register_storage_backend,
    unregister_query_backend,
    unregister_storage_backend,
)

__all__ = [
    "AsyncQueryBackend",
    "AsyncStorageBackend",
    "BackendCapabilities",
    "BackendError",
    "BackendInfo",
    "BackendNotFoundError",
    "BackendNotInitializedError",
    "BaseAsyncStorageBackend",
    "BaseStorageBackend",
    "ConnectionError",
    "DriverNotFoundError",
    "InitializationError",
    "QueryBackend",
    "ReadError",
    "SchemaError",
    "StorageBackend",
    "WriteError",
    "clear_registered_backends",
    "create_storage_backend",
    "get_available_query_backends",
    "get_available_storage_backends",
    "get_backend_for_url",
    "load_query_backend",
    "load_storage_backend",
    "parse_database_url",
    "register_query_backend",
    "register_storage_backend",
    "unregister_query_backend",
    "unregister_storage_backend",
]
