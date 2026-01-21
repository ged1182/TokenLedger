"""Tests for the config module."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from tokenledger.config import TokenLedgerConfig, configure, get_config


class TestTokenLedgerConfig:
    """Tests for TokenLedgerConfig dataclass."""

    def test_default_values(self) -> None:
        """Test that default values are set correctly."""
        config = TokenLedgerConfig()

        assert config.table_name == "token_ledger_events"
        assert config.schema_name == "public"
        assert config.batch_size == 100
        assert config.flush_interval_seconds == 5.0
        assert config.async_mode is True
        assert config.max_queue_size == 10000
        assert config.pool_min_size == 2
        assert config.pool_max_size == 10
        assert config.sample_rate == 1.0
        assert config.debug is False

    def test_custom_values(self) -> None:
        """Test that custom values are properly set."""
        config = TokenLedgerConfig(
            database_url="postgresql://custom:custom@localhost/custom",
            table_name="custom_events",
            schema_name="custom",
            batch_size=50,
            flush_interval_seconds=10.0,
            async_mode=False,
            sample_rate=0.5,
            app_name="custom-app",
            environment="staging",
            debug=True,
        )

        assert config.database_url == "postgresql://custom:custom@localhost/custom"
        assert config.table_name == "custom_events"
        assert config.schema_name == "custom"
        assert config.batch_size == 50
        assert config.flush_interval_seconds == 10.0
        assert config.async_mode is False
        assert config.sample_rate == 0.5
        assert config.app_name == "custom-app"
        assert config.environment == "staging"
        assert config.debug is True

    def test_full_table_name_property(self) -> None:
        """Test the full_table_name property."""
        config = TokenLedgerConfig(
            table_name="events",
            schema_name="analytics",
        )

        assert config.full_table_name == "analytics.events"

    def test_full_table_name_default(self) -> None:
        """Test the full_table_name with default values."""
        config = TokenLedgerConfig()

        assert config.full_table_name == "public.token_ledger_events"

    def test_is_supabase_detection(self) -> None:
        """Test Supabase detection from URL."""
        supabase_config = TokenLedgerConfig(
            database_url="postgresql://user:pass@db.supabase.co:5432/postgres"
        )
        assert supabase_config.is_supabase is True

        regular_config = TokenLedgerConfig(
            database_url="postgresql://user:pass@localhost:5432/mydb"
        )
        assert regular_config.is_supabase is False

    def test_is_supabase_empty_url(self) -> None:
        """Test is_supabase returns False for empty URL."""
        config = TokenLedgerConfig(database_url="")
        assert config.is_supabase is False

    def test_default_metadata(self) -> None:
        """Test default metadata initialization."""
        config = TokenLedgerConfig(
            default_metadata={"service": "api", "version": "1.0"}
        )

        assert config.default_metadata == {"service": "api", "version": "1.0"}

    def test_default_metadata_empty_by_default(self) -> None:
        """Test that default_metadata is empty by default."""
        config = TokenLedgerConfig()
        assert config.default_metadata == {}


class TestConfigEnvironmentVariables:
    """Tests for environment variable handling."""

    def test_database_url_from_env(self) -> None:
        """Test DATABASE_URL environment variable."""
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://env:env@localhost/env"}):
            config = TokenLedgerConfig()
            assert config.database_url == "postgresql://env:env@localhost/env"

    def test_tokenledger_database_url_takes_precedence(self) -> None:
        """Test TOKENLEDGER_DATABASE_URL takes precedence over DATABASE_URL."""
        with patch.dict(
            os.environ,
            {
                "DATABASE_URL": "postgresql://generic:generic@localhost/generic",
                "TOKENLEDGER_DATABASE_URL": "postgresql://specific:specific@localhost/specific",
            },
        ):
            config = TokenLedgerConfig()
            assert config.database_url == "postgresql://specific:specific@localhost/specific"

    def test_app_name_from_env(self) -> None:
        """Test TOKENLEDGER_APP_NAME environment variable."""
        with patch.dict(os.environ, {"TOKENLEDGER_APP_NAME": "env-app"}):
            config = TokenLedgerConfig()
            assert config.app_name == "env-app"

    def test_environment_from_env(self) -> None:
        """Test TOKENLEDGER_ENVIRONMENT environment variable."""
        with patch.dict(os.environ, {"TOKENLEDGER_ENVIRONMENT": "production"}):
            config = TokenLedgerConfig()
            assert config.environment == "production"

    def test_environment_fallback_to_generic(self) -> None:
        """Test ENVIRONMENT fallback when TOKENLEDGER_ENVIRONMENT not set."""
        with patch.dict(os.environ, {"ENVIRONMENT": "staging"}, clear=True):
            config = TokenLedgerConfig()
            assert config.environment == "staging"

    def test_environment_default_to_development(self) -> None:
        """Test default environment is development."""
        with patch.dict(os.environ, {}, clear=True):
            # Clear any existing env vars that might affect this
            os.environ.pop("TOKENLEDGER_ENVIRONMENT", None)
            os.environ.pop("ENVIRONMENT", None)
            config = TokenLedgerConfig()
            assert config.environment == "development"

    def test_debug_from_env_true_values(self) -> None:
        """Test TOKENLEDGER_DEBUG environment variable with true values."""
        for value in ["1", "true", "yes", "True", "YES"]:
            with patch.dict(os.environ, {"TOKENLEDGER_DEBUG": value}):
                config = TokenLedgerConfig()
                assert config.debug is True, f"Failed for value: {value}"

    def test_debug_from_env_false_values(self) -> None:
        """Test TOKENLEDGER_DEBUG environment variable with false values."""
        for value in ["0", "false", "no", ""]:
            with patch.dict(os.environ, {"TOKENLEDGER_DEBUG": value}):
                config = TokenLedgerConfig()
                assert config.debug is False, f"Failed for value: {value}"

    def test_pool_sizes_from_env(self) -> None:
        """Test pool size environment variables."""
        with patch.dict(
            os.environ,
            {
                "TOKENLEDGER_POOL_MIN_SIZE": "5",
                "TOKENLEDGER_POOL_MAX_SIZE": "20",
            },
        ):
            config = TokenLedgerConfig()
            assert config.pool_min_size == 5
            assert config.pool_max_size == 20

    def test_explicit_values_override_env(self) -> None:
        """Test that explicit values override environment variables."""
        with patch.dict(
            os.environ,
            {
                "DATABASE_URL": "postgresql://env:env@localhost/env",
                "TOKENLEDGER_APP_NAME": "env-app",
            },
        ):
            config = TokenLedgerConfig(
                database_url="postgresql://explicit:explicit@localhost/explicit",
                app_name="explicit-app",
            )
            assert config.database_url == "postgresql://explicit:explicit@localhost/explicit"
            assert config.app_name == "explicit-app"


class TestConfigure:
    """Tests for configure() function."""

    def test_configure_returns_config(self) -> None:
        """Test that configure returns a config instance."""
        config = configure(
            database_url="postgresql://test:test@localhost/test",
            app_name="test-app",
        )

        assert isinstance(config, TokenLedgerConfig)
        assert config.database_url == "postgresql://test:test@localhost/test"
        assert config.app_name == "test-app"

    def test_configure_sets_global_config(self) -> None:
        """Test that configure sets the global config."""
        configure(
            database_url="postgresql://test:test@localhost/test",
            app_name="configured-app",
        )

        config = get_config()
        assert config.app_name == "configured-app"

    def test_configure_with_kwargs(self) -> None:
        """Test configure with various kwargs."""
        config = configure(
            database_url="postgresql://test:test@localhost/test",
            batch_size=50,
            sample_rate=0.5,
            debug=True,
            environment="production",
        )

        assert config.batch_size == 50
        assert config.sample_rate == 0.5
        assert config.debug is True
        assert config.environment == "production"

    def test_configure_without_database_url(self) -> None:
        """Test configure can be called without database_url."""
        config = configure(app_name="app-only")
        assert config.app_name == "app-only"


class TestGetConfig:
    """Tests for get_config() function."""

    def test_get_config_creates_default(self) -> None:
        """Test that get_config creates a default config if none exists."""
        config = get_config()

        assert isinstance(config, TokenLedgerConfig)

    def test_get_config_returns_same_instance(self) -> None:
        """Test that get_config returns the same instance."""
        config1 = get_config()
        config2 = get_config()

        assert config1 is config2

    def test_get_config_returns_configured_instance(self) -> None:
        """Test that get_config returns previously configured instance."""
        configure(
            database_url="postgresql://test:test@localhost/test",
            app_name="pre-configured",
        )

        config = get_config()
        assert config.app_name == "pre-configured"
