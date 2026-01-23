"""Tests for the migrations runner module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestIsTokenledgerAlembicDir:
    """Tests for _is_tokenledger_alembic_dir helper."""

    def test_returns_true_for_valid_directory(self, tmp_path: Path) -> None:
        """Test returns True for directory with env.py and versions/."""
        from tokenledger.migrations.runner import _is_tokenledger_alembic_dir

        # Create required structure
        (tmp_path / "env.py").touch()
        (tmp_path / "versions").mkdir()

        assert _is_tokenledger_alembic_dir(tmp_path) is True

    def test_returns_false_for_missing_env_py(self, tmp_path: Path) -> None:
        """Test returns False if env.py is missing."""
        from tokenledger.migrations.runner import _is_tokenledger_alembic_dir

        (tmp_path / "versions").mkdir()

        assert _is_tokenledger_alembic_dir(tmp_path) is False

    def test_returns_false_for_missing_versions(self, tmp_path: Path) -> None:
        """Test returns False if versions/ is missing."""
        from tokenledger.migrations.runner import _is_tokenledger_alembic_dir

        (tmp_path / "env.py").touch()

        assert _is_tokenledger_alembic_dir(tmp_path) is False

    def test_returns_false_for_non_directory(self, tmp_path: Path) -> None:
        """Test returns False for non-directory path."""
        from tokenledger.migrations.runner import _is_tokenledger_alembic_dir

        file_path = tmp_path / "file.txt"
        file_path.touch()

        assert _is_tokenledger_alembic_dir(file_path) is False


class TestGetAlembicConfig:
    """Tests for get_alembic_config function."""

    def test_raises_if_alembic_dir_not_found(self, tmp_path: Path) -> None:
        """Test raises FileNotFoundError if alembic dir not found."""
        from tokenledger.migrations.runner import get_alembic_config

        with (
            patch("tokenledger.migrations.runner.Path") as mock_path,
            pytest.raises(FileNotFoundError, match="Alembic migrations directory not found"),
        ):
            # Make both paths invalid
            mock_path.return_value.parent.parent = tmp_path
            mock_path.return_value.parent = tmp_path
            get_alembic_config()

    def test_uses_database_url_argument(self) -> None:
        """Test uses provided database_url."""
        from tokenledger.migrations.runner import get_alembic_config

        with (
            patch("tokenledger.migrations.runner._is_tokenledger_alembic_dir", return_value=True),
            patch("alembic.config.Config") as mock_config_class,
        ):
            mock_config = MagicMock()
            mock_config_class.return_value = mock_config

            get_alembic_config(database_url="postgresql://test:test@localhost/db")

            mock_config.set_main_option.assert_any_call(
                "sqlalchemy.url", "postgresql://test:test@localhost/db"
            )

    def test_uses_tokenledger_database_url_env_var(self) -> None:
        """Test uses TOKENLEDGER_DATABASE_URL env var."""
        from tokenledger.migrations.runner import get_alembic_config

        with (
            patch("tokenledger.migrations.runner._is_tokenledger_alembic_dir", return_value=True),
            patch("alembic.config.Config") as mock_config_class,
            patch.dict("os.environ", {"TOKENLEDGER_DATABASE_URL": "postgresql://tl@localhost/db"}),
        ):
            mock_config = MagicMock()
            mock_config_class.return_value = mock_config

            get_alembic_config()

            mock_config.set_main_option.assert_any_call(
                "sqlalchemy.url", "postgresql://tl@localhost/db"
            )

    def test_uses_database_url_env_var_fallback(self) -> None:
        """Test uses DATABASE_URL env var as fallback."""
        from tokenledger.migrations.runner import get_alembic_config

        with (
            patch("tokenledger.migrations.runner._is_tokenledger_alembic_dir", return_value=True),
            patch("alembic.config.Config") as mock_config_class,
            patch.dict(
                "os.environ",
                {"DATABASE_URL": "postgresql://db@localhost/db"},
                clear=True,
            ),
        ):
            mock_config = MagicMock()
            mock_config_class.return_value = mock_config

            get_alembic_config()

            mock_config.set_main_option.assert_any_call(
                "sqlalchemy.url", "postgresql://db@localhost/db"
            )

    def test_sets_schema_in_cmd_opts(self) -> None:
        """Test sets schema in cmd_opts.x."""
        from tokenledger.migrations.runner import get_alembic_config

        with (
            patch("tokenledger.migrations.runner._is_tokenledger_alembic_dir", return_value=True),
            patch("alembic.config.Config") as mock_config_class,
        ):
            mock_config = MagicMock()
            mock_config_class.return_value = mock_config

            config = get_alembic_config(schema="custom_schema")

            assert config.cmd_opts.x == ["schema=custom_schema"]


class TestMigrationRunner:
    """Tests for MigrationRunner class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config."""
        config = MagicMock()
        config.get_main_option.return_value = "postgresql://test@localhost/db"
        return config

    def test_init(self) -> None:
        """Test MigrationRunner initialization."""
        with patch("tokenledger.migrations.runner.get_alembic_config") as mock_get_config:
            mock_get_config.return_value = MagicMock()

            from tokenledger.migrations.runner import MigrationRunner

            runner = MigrationRunner(database_url="postgresql://test@localhost/db")

            assert runner.database_url == "postgresql://test@localhost/db"
            assert runner.schema == "token_ledger"
            mock_get_config.assert_called_once_with(
                "postgresql://test@localhost/db", "token_ledger"
            )

    def test_init_custom_schema(self) -> None:
        """Test MigrationRunner with custom schema."""
        with patch("tokenledger.migrations.runner.get_alembic_config") as mock_get_config:
            mock_get_config.return_value = MagicMock()

            from tokenledger.migrations.runner import MigrationRunner

            runner = MigrationRunner(schema="custom_schema")

            assert runner.schema == "custom_schema"

    def test_safe_walk_revisions_returns_list(self, mock_config) -> None:
        """Test _safe_walk_revisions returns revision list."""
        with patch("tokenledger.migrations.runner.get_alembic_config", return_value=mock_config):
            from tokenledger.migrations.runner import MigrationRunner

            runner = MigrationRunner()

            mock_script = MagicMock()
            mock_rev = MagicMock()
            mock_script.walk_revisions.return_value = [mock_rev]

            result = runner._safe_walk_revisions(mock_script, "head", "base")

            assert result == [mock_rev]

    def test_safe_walk_revisions_handles_exception_with_fallback(self, mock_config) -> None:
        """Test _safe_walk_revisions uses fallback on exception."""
        with patch("tokenledger.migrations.runner.get_alembic_config", return_value=mock_config):
            from tokenledger.migrations.runner import MigrationRunner

            runner = MigrationRunner()

            mock_script = MagicMock()
            mock_script.walk_revisions.side_effect = Exception("Walk failed")
            mock_rev = MagicMock()
            mock_script.get_revision.return_value = mock_rev

            result = runner._safe_walk_revisions(
                mock_script, "head", "base", fallback=["rev1"]
            )

            assert result == [mock_rev]
            mock_script.get_revision.assert_called_with("rev1")

    def test_safe_walk_revisions_handles_exception_no_fallback(self, mock_config) -> None:
        """Test _safe_walk_revisions returns empty list on exception without fallback."""
        with patch("tokenledger.migrations.runner.get_alembic_config", return_value=mock_config):
            from tokenledger.migrations.runner import MigrationRunner

            runner = MigrationRunner()

            mock_script = MagicMock()
            mock_script.walk_revisions.side_effect = Exception("Walk failed")

            result = runner._safe_walk_revisions(mock_script, "head", "base")

            assert result == []

    def test_init_creates_schema_if_not_public(self, mock_config) -> None:
        """Test init creates schema if not public."""
        with patch("tokenledger.migrations.runner.get_alembic_config", return_value=mock_config):
            from tokenledger.migrations.runner import MigrationRunner

            runner = MigrationRunner(schema="custom_schema")
            runner._create_schema = MagicMock()
            runner.upgrade = MagicMock(return_value=[])

            runner.init()

            runner._create_schema.assert_called_once()
            runner.upgrade.assert_called_once_with("head")

    def test_init_skips_schema_creation_for_public(self, mock_config) -> None:
        """Test init skips schema creation for public schema."""
        with patch("tokenledger.migrations.runner.get_alembic_config", return_value=mock_config):
            from tokenledger.migrations.runner import MigrationRunner

            runner = MigrationRunner(schema="public")
            runner._create_schema = MagicMock()
            runner.upgrade = MagicMock(return_value=[])

            runner.init()

            runner._create_schema.assert_not_called()

    def test_create_schema_raises_without_url(self, mock_config) -> None:
        """Test _create_schema raises if no database URL."""
        with (
            patch("tokenledger.migrations.runner.get_alembic_config", return_value=mock_config),
            patch.dict("os.environ", {}, clear=True),
        ):
            from tokenledger.migrations.runner import MigrationRunner

            runner = MigrationRunner(database_url=None, schema="custom")

            with pytest.raises(ValueError, match="No database URL configured"):
                runner._create_schema()

    def test_create_schema_executes_sql(self, mock_config) -> None:
        """Test _create_schema executes CREATE SCHEMA."""
        with (
            patch("tokenledger.migrations.runner.get_alembic_config", return_value=mock_config),
            patch("sqlalchemy.create_engine") as mock_engine,
        ):
            from tokenledger.migrations.runner import MigrationRunner

            runner = MigrationRunner(
                database_url="postgresql://test@localhost/db", schema="custom"
            )

            mock_conn = MagicMock()
            mock_engine.return_value.connect.return_value.__enter__ = MagicMock(
                return_value=mock_conn
            )
            mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=None)

            runner._create_schema()

            mock_conn.execute.assert_called_once()
            mock_conn.commit.assert_called_once()

    def test_upgrade_dry_run(self, mock_config) -> None:
        """Test upgrade with dry_run generates SQL."""
        with (
            patch("tokenledger.migrations.runner.get_alembic_config", return_value=mock_config),
            patch("alembic.command.upgrade") as mock_upgrade,
        ):
            from tokenledger.migrations.runner import MigrationRunner

            runner = MigrationRunner()

            result = runner.upgrade("head", dry_run=True)

            mock_upgrade.assert_called_once_with(mock_config, "head", sql=True)
            assert result == []

    def test_upgrade_runs_migrations(self, mock_config) -> None:
        """Test upgrade runs migrations and returns applied."""
        with (
            patch("tokenledger.migrations.runner.get_alembic_config", return_value=mock_config),
            patch("alembic.command.upgrade") as mock_upgrade,
            patch("alembic.script.ScriptDirectory.from_config") as mock_script_dir,
        ):
            from tokenledger.migrations.runner import MigrationRunner

            runner = MigrationRunner()
            runner.current = MagicMock(side_effect=[None, "abc123"])

            mock_script = MagicMock()
            mock_script_dir.return_value = mock_script
            mock_rev = MagicMock()
            mock_rev.revision = "abc123"
            mock_script.walk_revisions.return_value = [mock_rev]

            result = runner.upgrade()

            mock_upgrade.assert_called_once()
            assert "abc123" in result

    def test_downgrade_no_current_revision(self, mock_config) -> None:
        """Test downgrade returns empty when no current revision."""
        with (
            patch("tokenledger.migrations.runner.get_alembic_config", return_value=mock_config),
            patch("alembic.script.ScriptDirectory.from_config") as mock_script_dir,
        ):
            from tokenledger.migrations.runner import MigrationRunner

            mock_script_dir.return_value = MagicMock()
            runner = MigrationRunner()
            runner.current = MagicMock(return_value=None)

            result = runner.downgrade("001")

            assert result == []

    def test_downgrade_runs_downgrade(self, mock_config) -> None:
        """Test downgrade runs alembic downgrade."""
        with (
            patch("tokenledger.migrations.runner.get_alembic_config", return_value=mock_config),
            patch("alembic.command.downgrade") as mock_downgrade,
            patch("alembic.script.ScriptDirectory.from_config") as mock_script_dir,
        ):
            from tokenledger.migrations.runner import MigrationRunner

            runner = MigrationRunner()
            runner.current = MagicMock(side_effect=["abc123", "001"])

            mock_script = MagicMock()
            mock_script_dir.return_value = mock_script
            mock_rev = MagicMock()
            mock_rev.revision = "abc123"
            mock_script.walk_revisions.return_value = [mock_rev]

            result = runner.downgrade("001")

            mock_downgrade.assert_called_once_with(mock_config, "001")
            assert "abc123" in result

    def test_current_raises_without_url(self, mock_config) -> None:
        """Test current raises if no database URL."""
        mock_config.get_main_option.return_value = None
        with (
            patch("tokenledger.migrations.runner.get_alembic_config", return_value=mock_config),
            patch.dict("os.environ", {}, clear=True),
        ):
            from tokenledger.migrations.runner import MigrationRunner

            runner = MigrationRunner(database_url=None)

            with pytest.raises(ValueError, match="No database URL configured"):
                runner.current()

    def test_current_returns_revision(self, mock_config) -> None:
        """Test current returns current revision."""
        with (
            patch("tokenledger.migrations.runner.get_alembic_config", return_value=mock_config),
            patch("sqlalchemy.create_engine") as mock_engine,
            patch("alembic.runtime.migration.MigrationContext.configure") as mock_ctx_cfg,
        ):
            from tokenledger.migrations.runner import MigrationRunner

            runner = MigrationRunner()

            mock_conn = MagicMock()
            mock_engine.return_value.connect.return_value.__enter__ = MagicMock(
                return_value=mock_conn
            )
            mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=None)

            mock_ctx = MagicMock()
            mock_ctx.get_current_revision.return_value = "abc123"
            mock_ctx_cfg.return_value = mock_ctx

            result = runner.current()

            assert result == "abc123"

    def test_status_returns_migration_list(self, mock_config) -> None:
        """Test status returns list of migration statuses."""
        with (
            patch("tokenledger.migrations.runner.get_alembic_config", return_value=mock_config),
            patch("alembic.script.ScriptDirectory.from_config") as mock_script_dir,
        ):
            from tokenledger.migrations.runner import MigrationRunner

            runner = MigrationRunner()
            runner.current = MagicMock(return_value="002")

            mock_script = MagicMock()
            mock_script_dir.return_value = mock_script

            mock_rev1 = MagicMock()
            mock_rev1.revision = "001"
            mock_rev1.doc = "Initial migration"

            mock_rev2 = MagicMock()
            mock_rev2.revision = "002"
            mock_rev2.doc = "Add index"

            mock_script.walk_revisions.side_effect = [
                [mock_rev2, mock_rev1],  # head to base
                [mock_rev2, mock_rev1],  # current to base
            ]

            result = runner.status()

            assert len(result) == 2
            assert result[0]["version"] == "001"
            assert result[1]["version"] == "002"

    def test_history_returns_applied_only(self, mock_config) -> None:
        """Test history returns only applied migrations."""
        with patch("tokenledger.migrations.runner.get_alembic_config", return_value=mock_config):
            from tokenledger.migrations.runner import MigrationRunner

            runner = MigrationRunner()
            runner.status = MagicMock(
                return_value=[
                    {"version": "001", "applied": True},
                    {"version": "002", "applied": False},
                ]
            )

            result = runner.history()

            assert len(result) == 1
            assert result[0]["version"] == "001"

    def test_heads_returns_head_revisions(self, mock_config) -> None:
        """Test heads returns head revision list."""
        with (
            patch("tokenledger.migrations.runner.get_alembic_config", return_value=mock_config),
            patch("alembic.script.ScriptDirectory.from_config") as mock_script_dir,
        ):
            from tokenledger.migrations.runner import MigrationRunner

            runner = MigrationRunner()

            mock_script = MagicMock()
            mock_script.get_heads.return_value = ["abc123"]
            mock_script_dir.return_value = mock_script

            result = runner.heads()

            assert result == ["abc123"]

    def test_stamp_stamps_database(self, mock_config) -> None:
        """Test stamp stamps database with revision."""
        with (
            patch("tokenledger.migrations.runner.get_alembic_config", return_value=mock_config),
            patch("alembic.command.stamp") as mock_stamp,
        ):
            from tokenledger.migrations.runner import MigrationRunner

            runner = MigrationRunner()

            runner.stamp("head")

            mock_stamp.assert_called_once_with(mock_config, "head")
