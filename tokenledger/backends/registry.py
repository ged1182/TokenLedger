"""
TokenLedger Backend Registry

Provides plugin discovery and loading for storage and query backends.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, cast

from .exceptions import BackendNotFoundError, DriverNotFoundError

if TYPE_CHECKING:
    from tokenledger.config import TokenLedgerConfig

    from .protocol import AsyncStorageBackend, StorageBackend

logger = logging.getLogger("tokenledger.backends")

# Entry point group names
STORAGE_BACKEND_GROUP = "tokenledger.backends"
QUERY_BACKEND_GROUP = "tokenledger.query_backends"

# Built-in backends (always available)
_BUILTIN_STORAGE_BACKENDS: dict[str, str] = {
    "postgresql": "tokenledger.backends.postgresql:PostgreSQLBackend",
    "postgres": "tokenledger.backends.postgresql:PostgreSQLBackend",
    "asyncpg": "tokenledger.backends.postgresql:AsyncPostgreSQLBackend",
}

_BUILTIN_QUERY_BACKENDS: dict[str, str] = {
    "postgresql": "tokenledger.backends.postgresql:PostgreSQLQueryBackend",
    "postgres": "tokenledger.backends.postgresql:PostgreSQLQueryBackend",
    "asyncpg": "tokenledger.backends.postgresql:AsyncPostgreSQLQueryBackend",
}

# Runtime-registered backends (for testing or custom usage)
_REGISTERED_STORAGE_BACKENDS: dict[str, str] = {}
_REGISTERED_QUERY_BACKENDS: dict[str, str] = {}


def register_storage_backend(name: str, module_path: str, class_name: str) -> None:
    """
    Register a storage backend at runtime without setuptools.

    Useful for testing or embedding custom backends.

    Args:
        name: Backend identifier (e.g., "clickhouse")
        module_path: Full module path (e.g., "myapp.backends.clickhouse")
        class_name: Class name within the module (e.g., "ClickHouseBackend")

    Example:
        >>> from tokenledger.backends import registry
        >>> registry.register_storage_backend("mydb", "myapp.backends", "MyDBBackend")
    """
    _REGISTERED_STORAGE_BACKENDS[name] = f"{module_path}:{class_name}"
    logger.debug(f"Registered storage backend: {name} -> {module_path}:{class_name}")


def register_query_backend(name: str, module_path: str, class_name: str) -> None:
    """
    Register a query backend at runtime without setuptools.

    Args:
        name: Backend identifier (e.g., "clickhouse")
        module_path: Full module path
        class_name: Class name within the module
    """
    _REGISTERED_QUERY_BACKENDS[name] = f"{module_path}:{class_name}"
    logger.debug(f"Registered query backend: {name} -> {module_path}:{class_name}")


def unregister_storage_backend(name: str) -> bool:
    """
    Unregister a runtime-registered storage backend.

    Args:
        name: Backend identifier to unregister

    Returns:
        True if backend was unregistered, False if it wasn't registered
    """
    if name in _REGISTERED_STORAGE_BACKENDS:
        del _REGISTERED_STORAGE_BACKENDS[name]
        return True
    return False


def unregister_query_backend(name: str) -> bool:
    """
    Unregister a runtime-registered query backend.

    Args:
        name: Backend identifier to unregister

    Returns:
        True if backend was unregistered, False if it wasn't registered
    """
    if name in _REGISTERED_QUERY_BACKENDS:
        del _REGISTERED_QUERY_BACKENDS[name]
        return True
    return False


def get_available_storage_backends() -> dict[str, str]:
    """
    Get all available storage backends (built-in + installed + registered).

    Returns:
        Dictionary mapping backend names to their module:class paths
    """
    backends = dict(_BUILTIN_STORAGE_BACKENDS)
    backends.update(_REGISTERED_STORAGE_BACKENDS)

    # Discover installed plugins via entry points
    try:
        eps = entry_points(group=STORAGE_BACKEND_GROUP)
        for ep in eps:
            backends[ep.name] = ep.value
    except Exception as e:
        logger.warning(f"Error discovering storage backend entry points: {e}")

    return backends


def get_available_query_backends() -> dict[str, str]:
    """
    Get all available query backends (built-in + installed + registered).

    Returns:
        Dictionary mapping backend names to their module:class paths
    """
    backends = dict(_BUILTIN_QUERY_BACKENDS)
    backends.update(_REGISTERED_QUERY_BACKENDS)

    # Discover installed plugins via entry points
    try:
        eps = entry_points(group=QUERY_BACKEND_GROUP)
        for ep in eps:
            backends[ep.name] = ep.value
    except Exception as e:
        logger.warning(f"Error discovering query backend entry points: {e}")

    return backends


def _load_class(module_class_path: str) -> type:
    """
    Load a class from a module:class path string.

    Args:
        module_class_path: String in format "module.path:ClassName"

    Returns:
        The loaded class

    Raises:
        ImportError: If module cannot be imported
        AttributeError: If class not found in module
    """
    module_path, class_name = module_class_path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def load_storage_backend(name: str) -> type[StorageBackend] | type[AsyncStorageBackend]:
    """
    Load a storage backend class by name.

    Args:
        name: Backend identifier (e.g., "postgres", "clickhouse")

    Returns:
        The backend class (not an instance)

    Raises:
        BackendNotFoundError: If backend is not found
        DriverNotFoundError: If backend's driver is not installed
    """
    available = get_available_storage_backends()

    if name not in available:
        raise BackendNotFoundError(name, list(available.keys()))

    module_class = available[name]

    try:
        return _load_class(module_class)
    except ImportError as e:
        # Provide helpful error for missing drivers
        driver_hints = {
            "postgresql": "pip install tokenledger[postgres]",
            "postgres": "pip install tokenledger[postgres]",
            "asyncpg": "pip install tokenledger[asyncpg]",
            "clickhouse": "pip install tokenledger-clickhouse",
            "bigquery": "pip install tokenledger-bigquery",
            "snowflake": "pip install tokenledger-snowflake",
        }
        hint = driver_hints.get(name, f"pip install tokenledger-{name}")
        raise DriverNotFoundError(
            driver_name=str(e).split("'")[1] if "'" in str(e) else name,
            install_hint=hint,
            backend_name=name,
        ) from e


def load_query_backend(name: str) -> type:
    """
    Load a query backend class by name.

    Args:
        name: Backend identifier (e.g., "postgres", "clickhouse")

    Returns:
        The backend class (not an instance)

    Raises:
        BackendNotFoundError: If backend is not found
        DriverNotFoundError: If backend's driver is not installed
    """
    available = get_available_query_backends()

    if name not in available:
        raise BackendNotFoundError(name, list(available.keys()))

    module_class = available[name]

    try:
        return _load_class(module_class)
    except ImportError as e:
        raise DriverNotFoundError(
            driver_name=str(e).split("'")[1] if "'" in str(e) else name,
            install_hint=f"pip install tokenledger[{name}]",
            backend_name=name,
        ) from e


def create_storage_backend(
    name: str,
    config: TokenLedgerConfig,
    create_schema: bool = True,
) -> StorageBackend:
    """
    Create and initialize a storage backend instance.

    Args:
        name: Backend identifier
        config: TokenLedger configuration
        create_schema: Whether to create tables if they don't exist

    Returns:
        Initialized storage backend instance

    Example:
        >>> from tokenledger.config import TokenLedgerConfig
        >>> from tokenledger.backends import registry
        >>> config = TokenLedgerConfig(database_url="postgresql://...")
        >>> backend = registry.create_storage_backend("postgres", config)
    """
    backend_class = load_storage_backend(name)
    backend = backend_class()
    backend.initialize(config, create_schema=create_schema)
    return cast("StorageBackend", backend)


async def create_async_storage_backend(
    name: str,
    config: TokenLedgerConfig,
    create_schema: bool = True,
) -> AsyncStorageBackend:
    """
    Create and initialize an async storage backend instance.

    Args:
        name: Backend identifier (must support async)
        config: TokenLedger configuration
        create_schema: Whether to create tables if they don't exist

    Returns:
        Initialized async storage backend instance

    Raises:
        ValueError: If the backend doesn't support async
    """
    backend_class = load_storage_backend(name)
    backend = backend_class()

    # Check if it's an async backend
    if inspect.iscoroutinefunction(backend.initialize):
        await backend.initialize(config, create_schema=create_schema)
    else:
        raise ValueError(f"Backend '{name}' does not support async operations")

    return cast("AsyncStorageBackend", backend)


def parse_database_url(url: str) -> tuple[str, str]:
    """
    Parse a database URL to extract backend name and connection string.

    URL formats:
        - tokenledger+<backend>://<connection-string>
        - <backend>://<connection-string>

    Args:
        url: Full database URL

    Returns:
        Tuple of (backend_name, connection_string)

    Examples:
        >>> parse_database_url("tokenledger+clickhouse://localhost:9000/default")
        ('clickhouse', 'clickhouse://localhost:9000/default')
        >>> parse_database_url("postgresql://localhost/mydb")
        ('postgresql', 'postgresql://localhost/mydb')
    """
    if url.startswith("tokenledger+"):
        # Extract backend from URL
        rest = url[len("tokenledger+") :]
        backend_name, _, connection = rest.partition("://")
        return backend_name, f"{backend_name}://{connection}"

    # Infer backend from URL scheme
    scheme = url.split("://")[0].split("+")[0]
    backend_map = {
        "postgresql": "postgresql",
        "postgres": "postgresql",
        "clickhouse": "clickhouse",
        "bigquery": "bigquery",
        "snowflake": "snowflake",
    }
    return backend_map.get(scheme, scheme), url


def get_backend_for_url(url: str, config: TokenLedgerConfig) -> StorageBackend:
    """
    Get and initialize the appropriate backend for a database URL.

    Args:
        url: Database URL
        config: TokenLedger configuration

    Returns:
        Initialized storage backend

    Example:
        >>> backend = get_backend_for_url("postgresql://localhost/mydb", config)
    """
    backend_name, connection_url = parse_database_url(url)

    # Update config with parsed URL if needed
    if config.database_url != connection_url:
        # Create a new config with the parsed URL
        from dataclasses import replace

        config = replace(config, database_url=connection_url)

    return create_storage_backend(backend_name, config)


def clear_registered_backends() -> None:
    """Clear all runtime-registered backends. Useful for testing."""
    _REGISTERED_STORAGE_BACKENDS.clear()
    _REGISTERED_QUERY_BACKENDS.clear()
