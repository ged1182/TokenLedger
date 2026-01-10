"""Pytest configuration and fixtures for TokenLedger tests."""

from __future__ import annotations

import pytest

from tokenledger.config import TokenLedgerConfig


@pytest.fixture
def mock_config() -> TokenLedgerConfig:
    """Create a mock configuration for testing."""
    return TokenLedgerConfig(
        database_url="postgresql://test:test@localhost:5432/test",
        app_name="test-app",
        environment="test",
        async_mode=False,
        debug=True,
    )
