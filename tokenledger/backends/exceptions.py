"""
TokenLedger Backend Exceptions

Custom exceptions for backend operations.
"""

from __future__ import annotations


class BackendError(Exception):
    """Base exception for all backend errors."""

    def __init__(self, message: str, backend_name: str | None = None):
        self.backend_name = backend_name
        super().__init__(f"[{backend_name}] {message}" if backend_name else message)


class ConnectionError(BackendError):
    """Raised when unable to connect to the backend."""

    pass


class InitializationError(BackendError):
    """Raised when backend initialization fails."""

    pass


class SchemaError(BackendError):
    """Raised when schema creation or migration fails."""

    pass


class WriteError(BackendError):
    """Raised when a write operation fails."""

    def __init__(
        self,
        message: str,
        backend_name: str | None = None,
        events_count: int = 0,
        events_written: int = 0,
    ):
        self.events_count = events_count
        self.events_written = events_written
        super().__init__(message, backend_name)


class ReadError(BackendError):
    """Raised when a read/query operation fails."""

    pass


class BackendNotFoundError(BackendError):
    """Raised when a requested backend is not found."""

    def __init__(self, backend_name: str, available_backends: list[str] | None = None):
        self.available_backends = available_backends or []
        available_str = (
            ", ".join(sorted(self.available_backends)) if self.available_backends else "none"
        )
        super().__init__(
            f"Backend '{backend_name}' not found. Available backends: {available_str}",
            backend_name=None,
        )


class BackendNotInitializedError(BackendError):
    """Raised when trying to use an uninitialized backend."""

    def __init__(self, backend_name: str | None = None):
        super().__init__(
            "Backend not initialized. Call initialize() first.",
            backend_name=backend_name,
        )


class DriverNotFoundError(BackendError):
    """Raised when a required database driver is not installed."""

    def __init__(
        self,
        driver_name: str,
        install_hint: str | None = None,
        backend_name: str | None = None,
    ):
        self.driver_name = driver_name
        self.install_hint = install_hint
        message = f"Required driver '{driver_name}' not found."
        if install_hint:
            message += f" Install with: {install_hint}"
        super().__init__(message, backend_name=backend_name)
