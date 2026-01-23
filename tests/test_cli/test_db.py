"""Tests for the database CLI commands."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from tokenledger.cli.db import app, get_database_url

runner = CliRunner()


class TestGetDatabaseUrl:
    """Tests for get_database_url helper."""

    def test_returns_argument_if_provided(self) -> None:
        """Test that argument takes precedence."""
        url = get_database_url("postgresql://arg:pass@localhost/db")
        assert url == "postgresql://arg:pass@localhost/db"

    def test_returns_database_url_env_var(self) -> None:
        """Test DATABASE_URL environment variable."""
        with patch.dict("os.environ", {"DATABASE_URL": "postgresql://env:pass@localhost/db"}):
            url = get_database_url(None)
            assert url == "postgresql://env:pass@localhost/db"

    def test_returns_tokenledger_database_url_env_var(self) -> None:
        """Test TOKENLEDGER_DATABASE_URL environment variable."""
        with patch.dict(
            "os.environ",
            {"TOKENLEDGER_DATABASE_URL": "postgresql://tl:pass@localhost/db"},
            clear=True,
        ):
            url = get_database_url(None)
            assert url == "postgresql://tl:pass@localhost/db"

    def test_raises_exit_if_no_url(self) -> None:
        """Test that missing URL raises typer.Exit."""
        import typer

        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(typer.Exit) as exc_info,
        ):
            get_database_url(None)
        assert exc_info.value.exit_code == 1


class TestInitCommand:
    """Tests for db init command."""

    def test_init_no_database_url(self) -> None:
        """Test init fails without database URL."""
        with patch.dict("os.environ", {}, clear=True):
            result = runner.invoke(app, ["init"])
            assert result.exit_code == 1
            assert "No database URL" in result.output

    def test_init_with_database_url(self) -> None:
        """Test init with database URL."""
        mock_runner = MagicMock()
        with patch("tokenledger.migrations.MigrationRunner", return_value=mock_runner):
            result = runner.invoke(
                app, ["init", "--database-url", "postgresql://test:test@localhost/db"]
            )
            mock_runner.init.assert_called_once()
            assert result.exit_code == 0
            assert "initialized successfully" in result.output

    def test_init_import_error(self) -> None:
        """Test init handles ImportError gracefully."""
        with patch.dict("sys.modules", {"tokenledger.migrations": None}):
            result = runner.invoke(
                app, ["init", "--database-url", "postgresql://test:test@localhost/db"]
            )
            assert result.exit_code == 1
            assert "not fully implemented" in result.output

    def test_init_exception(self) -> None:
        """Test init handles general exceptions."""
        with patch("tokenledger.migrations.MigrationRunner", side_effect=Exception("DB Error")):
            result = runner.invoke(
                app, ["init", "--database-url", "postgresql://test:test@localhost/db"]
            )
            assert result.exit_code == 1
            assert "DB Error" in result.output


class TestUpgradeCommand:
    """Tests for db upgrade command."""

    def test_upgrade_no_database_url(self) -> None:
        """Test upgrade fails without database URL."""
        with patch.dict("os.environ", {}, clear=True):
            result = runner.invoke(app, ["upgrade"])
            assert result.exit_code == 1

    def test_upgrade_applies_migrations(self) -> None:
        """Test upgrade applies migrations."""
        mock_runner = MagicMock()
        mock_runner.upgrade.return_value = ["001_init", "002_add_index"]
        with patch("tokenledger.migrations.MigrationRunner", return_value=mock_runner):
            result = runner.invoke(
                app, ["upgrade", "--database-url", "postgresql://test:test@localhost/db"]
            )
            assert result.exit_code == 0
            assert "Applied 2 migration" in result.output

    def test_upgrade_already_up_to_date(self) -> None:
        """Test upgrade when already up to date."""
        mock_runner = MagicMock()
        mock_runner.upgrade.return_value = []
        with patch("tokenledger.migrations.MigrationRunner", return_value=mock_runner):
            result = runner.invoke(
                app, ["upgrade", "--database-url", "postgresql://test:test@localhost/db"]
            )
            assert result.exit_code == 0
            assert "up to date" in result.output

    def test_upgrade_dry_run(self) -> None:
        """Test upgrade with dry-run flag."""
        mock_runner = MagicMock()
        mock_runner.upgrade.return_value = ["001_init"]
        with patch("tokenledger.migrations.MigrationRunner", return_value=mock_runner):
            result = runner.invoke(
                app,
                ["upgrade", "--database-url", "postgresql://test:test@localhost/db", "--dry-run"],
            )
            assert result.exit_code == 0
            assert "DRY RUN" in result.output


class TestDowngradeCommand:
    """Tests for db downgrade command."""

    def test_downgrade_no_database_url(self) -> None:
        """Test downgrade fails without database URL."""
        with patch.dict("os.environ", {}, clear=True):
            result = runner.invoke(app, ["downgrade", "001"])
            assert result.exit_code == 1

    def test_downgrade_with_confirmation(self) -> None:
        """Test downgrade with confirmation."""
        mock_runner = MagicMock()
        mock_runner.downgrade.return_value = ["002_add_index"]
        with patch("tokenledger.migrations.MigrationRunner", return_value=mock_runner):
            result = runner.invoke(
                app,
                ["downgrade", "001", "--database-url", "postgresql://test:test@localhost/db"],
                input="y\n",
            )
            assert result.exit_code == 0
            assert "Reverted 1 migration" in result.output

    def test_downgrade_abort(self) -> None:
        """Test downgrade abort on confirmation."""
        result = runner.invoke(
            app,
            ["downgrade", "001", "--database-url", "postgresql://test:test@localhost/db"],
            input="n\n",
        )
        assert result.exit_code == 0
        assert "Aborted" in result.output

    def test_downgrade_with_yes_flag(self) -> None:
        """Test downgrade with --yes flag skips confirmation."""
        mock_runner = MagicMock()
        mock_runner.downgrade.return_value = []
        with patch("tokenledger.migrations.MigrationRunner", return_value=mock_runner):
            result = runner.invoke(
                app,
                [
                    "downgrade",
                    "001",
                    "--database-url",
                    "postgresql://test:test@localhost/db",
                    "--yes",
                ],
            )
            assert result.exit_code == 0


class TestCurrentCommand:
    """Tests for db current command."""

    def test_current_no_database_url(self) -> None:
        """Test current fails without database URL."""
        with patch.dict("os.environ", {}, clear=True):
            result = runner.invoke(app, ["current"])
            assert result.exit_code == 1

    def test_current_shows_version(self) -> None:
        """Test current shows version."""
        mock_runner = MagicMock()
        mock_runner.current.return_value = "003_latest"
        with patch("tokenledger.migrations.MigrationRunner", return_value=mock_runner):
            result = runner.invoke(
                app, ["current", "--database-url", "postgresql://test:test@localhost/db"]
            )
            assert result.exit_code == 0
            assert "003_latest" in result.output

    def test_current_no_migrations(self) -> None:
        """Test current when no migrations applied."""
        mock_runner = MagicMock()
        mock_runner.current.return_value = None
        with patch("tokenledger.migrations.MigrationRunner", return_value=mock_runner):
            result = runner.invoke(
                app, ["current", "--database-url", "postgresql://test:test@localhost/db"]
            )
            assert result.exit_code == 0
            assert "No migrations applied" in result.output


class TestStatusCommand:
    """Tests for db status command."""

    def test_status_no_database_url(self) -> None:
        """Test status fails without database URL."""
        with patch.dict("os.environ", {}, clear=True):
            result = runner.invoke(app, ["status"])
            assert result.exit_code == 1

    def test_status_shows_table(self) -> None:
        """Test status shows migration table."""
        mock_runner = MagicMock()
        mock_runner.status.return_value = [
            {"version": "001", "description": "Init", "applied": True, "applied_at": "2024-01-01"},
            {"version": "002", "description": "Index", "applied": False, "applied_at": None},
        ]
        with patch("tokenledger.migrations.MigrationRunner", return_value=mock_runner):
            result = runner.invoke(
                app, ["status", "--database-url", "postgresql://test:test@localhost/db"]
            )
            assert result.exit_code == 0
            assert "Migration Status" in result.output


class TestHistoryCommand:
    """Tests for db history command."""

    def test_history_no_database_url(self) -> None:
        """Test history fails without database URL."""
        with patch.dict("os.environ", {}, clear=True):
            result = runner.invoke(app, ["history"])
            assert result.exit_code == 1

    def test_history_shows_list(self) -> None:
        """Test history shows migration list."""
        mock_runner = MagicMock()
        mock_runner.history.return_value = [
            {"version": "001", "description": "Init", "applied_at": "2024-01-01"},
        ]
        with patch("tokenledger.migrations.MigrationRunner", return_value=mock_runner):
            result = runner.invoke(
                app, ["history", "--database-url", "postgresql://test:test@localhost/db"]
            )
            assert result.exit_code == 0
            assert "Migration History" in result.output

    def test_history_empty(self) -> None:
        """Test history when no migrations applied."""
        mock_runner = MagicMock()
        mock_runner.history.return_value = []
        with patch("tokenledger.migrations.MigrationRunner", return_value=mock_runner):
            result = runner.invoke(
                app, ["history", "--database-url", "postgresql://test:test@localhost/db"]
            )
            assert result.exit_code == 0
            assert "No migrations have been applied" in result.output

    def test_history_import_error(self) -> None:
        """Test history handles ImportError gracefully."""
        with patch.dict("sys.modules", {"tokenledger.migrations": None}):
            result = runner.invoke(
                app, ["history", "--database-url", "postgresql://test:test@localhost/db"]
            )
            assert result.exit_code == 1
            assert "not fully implemented" in result.output

    def test_history_exception(self) -> None:
        """Test history handles general exceptions."""
        with patch(
            "tokenledger.migrations.MigrationRunner", side_effect=Exception("History Error")
        ):
            result = runner.invoke(
                app, ["history", "--database-url", "postgresql://test:test@localhost/db"]
            )
            assert result.exit_code == 1
            assert "History Error" in result.output


class TestExceptionHandling:
    """Tests for exception handling in all CLI commands."""

    def test_upgrade_import_error(self) -> None:
        """Test upgrade handles ImportError gracefully."""
        with patch.dict("sys.modules", {"tokenledger.migrations": None}):
            result = runner.invoke(
                app, ["upgrade", "--database-url", "postgresql://test:test@localhost/db"]
            )
            assert result.exit_code == 1
            assert "not fully implemented" in result.output

    def test_upgrade_exception(self) -> None:
        """Test upgrade handles general exceptions."""
        with patch(
            "tokenledger.migrations.MigrationRunner", side_effect=Exception("Upgrade Error")
        ):
            result = runner.invoke(
                app, ["upgrade", "--database-url", "postgresql://test:test@localhost/db"]
            )
            assert result.exit_code == 1
            assert "Upgrade Error" in result.output

    def test_downgrade_import_error(self) -> None:
        """Test downgrade handles ImportError gracefully."""
        with patch.dict("sys.modules", {"tokenledger.migrations": None}):
            result = runner.invoke(
                app,
                [
                    "downgrade",
                    "001",
                    "--yes",
                    "--database-url",
                    "postgresql://test:test@localhost/db",
                ],
            )
            assert result.exit_code == 1
            assert "not fully implemented" in result.output

    def test_downgrade_exception(self) -> None:
        """Test downgrade handles general exceptions."""
        with patch(
            "tokenledger.migrations.MigrationRunner", side_effect=Exception("Downgrade Error")
        ):
            result = runner.invoke(
                app,
                [
                    "downgrade",
                    "001",
                    "--yes",
                    "--database-url",
                    "postgresql://test:test@localhost/db",
                ],
            )
            assert result.exit_code == 1
            assert "Downgrade Error" in result.output

    def test_current_import_error(self) -> None:
        """Test current handles ImportError gracefully."""
        with patch.dict("sys.modules", {"tokenledger.migrations": None}):
            result = runner.invoke(
                app, ["current", "--database-url", "postgresql://test:test@localhost/db"]
            )
            assert result.exit_code == 1
            assert "not fully implemented" in result.output

    def test_current_exception(self) -> None:
        """Test current handles general exceptions."""
        with patch(
            "tokenledger.migrations.MigrationRunner", side_effect=Exception("Current Error")
        ):
            result = runner.invoke(
                app, ["current", "--database-url", "postgresql://test:test@localhost/db"]
            )
            assert result.exit_code == 1
            assert "Current Error" in result.output

    def test_status_import_error(self) -> None:
        """Test status handles ImportError gracefully."""
        with patch.dict("sys.modules", {"tokenledger.migrations": None}):
            result = runner.invoke(
                app, ["status", "--database-url", "postgresql://test:test@localhost/db"]
            )
            assert result.exit_code == 1
            assert "not fully implemented" in result.output

    def test_status_exception(self) -> None:
        """Test status handles general exceptions."""
        with patch("tokenledger.migrations.MigrationRunner", side_effect=Exception("Status Error")):
            result = runner.invoke(
                app, ["status", "--database-url", "postgresql://test:test@localhost/db"]
            )
            assert result.exit_code == 1
            assert "Status Error" in result.output
