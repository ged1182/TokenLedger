"""
TokenLedger Migration Runner

Wraps Alembic for programmatic database migrations.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("tokenledger.migrations")


def _is_tokenledger_alembic_dir(path: Path) -> bool:
    """Check if a directory is a valid TokenLedger alembic migrations directory."""
    return path.is_dir() and (path / "env.py").is_file() and (path / "versions").is_dir()


def get_alembic_config(database_url: str | None = None, schema: str = "public") -> Any:
    """
    Create an Alembic Config object for programmatic usage.

    Args:
        database_url: Database URL (uses env vars if not provided)
        schema: Schema name for TokenLedger tables (default: public)

    Returns:
        Configured alembic.config.Config object
    """
    from alembic.config import Config

    # Find the alembic directory (shipped with the package or in repo root)
    package_dir = Path(__file__).parent.parent  # tokenledger/
    repo_root = package_dir.parent  # TokenLedger/ (or site-packages/ when installed)

    # Check package-relative path FIRST (for installed packages)
    # This avoids collision with site-packages/alembic (the alembic package itself)
    alembic_dir = package_dir / "alembic"
    alembic_ini = package_dir / "alembic.ini"

    if not _is_tokenledger_alembic_dir(alembic_dir):
        # Fall back to repo root (for development)
        alembic_dir = repo_root / "alembic"
        alembic_ini = repo_root / "alembic.ini"

    if not _is_tokenledger_alembic_dir(alembic_dir):
        raise FileNotFoundError(
            f"Alembic migrations directory not found. "
            f"Checked: {package_dir / 'alembic'} and {repo_root / 'alembic'}. "
            f"Directory must contain env.py and versions/ subdirectory."
        )

    # Create config from alembic.ini if it exists, otherwise programmatically
    if alembic_ini.exists():
        config = Config(str(alembic_ini))
    else:
        config = Config()
        config.set_main_option("script_location", str(alembic_dir))

    # Override script_location to use our bundled migrations
    config.set_main_option("script_location", str(alembic_dir))

    # Set database URL
    if database_url:
        config.set_main_option("sqlalchemy.url", database_url)
    elif os.environ.get("TOKENLEDGER_DATABASE_URL"):
        config.set_main_option("sqlalchemy.url", os.environ["TOKENLEDGER_DATABASE_URL"])
    elif os.environ.get("DATABASE_URL"):
        config.set_main_option("sqlalchemy.url", os.environ["DATABASE_URL"])

    # Pass schema to migrations via -x argument
    # This is accessed in env.py via context.get_x_argument()
    config.cmd_opts = type("obj", (object,), {"x": [f"schema={schema}"]})()

    return config


class MigrationRunner:
    """
    Manages database migrations for TokenLedger using Alembic.

    Provides a simplified interface over Alembic commands.
    """

    def __init__(self, database_url: str | None = None, schema: str = "token_ledger"):
        """
        Initialize the migration runner.

        Args:
            database_url: PostgreSQL connection string
            schema: Schema name for TokenLedger tables (default: token_ledger)
                   Note: Schema support requires updating Alembic migrations
        """
        self.database_url = database_url
        self.schema = schema
        self._config = get_alembic_config(database_url, schema)

    def _safe_walk_revisions(
        self, script: Any, upper: str, lower: str, fallback: list[str] | None = None
    ) -> list[Any]:
        """
        Safely walk revisions between two points, handling edge cases.

        Args:
            script: ScriptDirectory instance
            upper: Upper revision (e.g., "head" or specific revision)
            lower: Lower revision (e.g., "base" or specific revision)
            fallback: Fallback revision list if walk fails

        Returns:
            List of revision objects
        """
        try:
            return list(script.walk_revisions(upper, lower))
        except Exception:
            # walk_revisions can fail with invalid ranges
            if fallback is not None:
                return [script.get_revision(r) for r in fallback if script.get_revision(r)]
            return []

    def init(self) -> None:
        """
        Initialize the database for TokenLedger.

        Creates the schema (if using dedicated schema) and runs all migrations.
        This is equivalent to `tokenledger db upgrade head`.
        """
        # Create schema if not using public
        if self.schema != "public":
            self._create_schema()

        # Run all migrations
        self.upgrade("head")
        logger.info("Database initialized successfully")

    def _create_schema(self) -> None:
        """Create the dedicated schema if it doesn't exist."""
        from sqlalchemy import create_engine, text

        url = (
            self.database_url
            or os.environ.get("TOKENLEDGER_DATABASE_URL")
            or os.environ.get("DATABASE_URL")
        )
        if not url:
            raise ValueError("No database URL configured")

        engine = create_engine(url)
        with engine.connect() as conn:
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {self.schema}"))
            conn.commit()
        logger.info(f"Created schema: {self.schema}")

    def upgrade(self, revision: str = "head", dry_run: bool = False) -> list[str]:
        """
        Run pending migrations up to the target revision.

        Args:
            revision: Target revision or "head" for latest
            dry_run: If True, show SQL without executing (offline mode)

        Returns:
            List of applied migration revision IDs
        """
        from alembic.script import ScriptDirectory

        from alembic import command

        if dry_run:
            # Generate SQL without executing
            command.upgrade(self._config, revision, sql=True)
            return []

        # Get current state before upgrade
        script = ScriptDirectory.from_config(self._config)
        current = self.current()

        # Run upgrade
        command.upgrade(self._config, revision)

        # Determine what was applied by comparing before/after state
        new_current = self.current()
        applied = []

        if new_current and new_current != current:
            base_rev = current if current else "base"
            fallback = [new_current] if new_current else None
            for rev in self._safe_walk_revisions(script, new_current, base_rev, fallback):
                if rev and rev.revision != current:
                    applied.append(rev.revision)
            applied.reverse()

        return applied

    def downgrade(self, revision: str) -> list[str]:
        """
        Revert migrations down to the target revision.

        Args:
            revision: Target revision to downgrade to (e.g., "001" or "-1" for previous)

        Returns:
            List of reverted migration revision IDs
        """
        from alembic.script import ScriptDirectory

        from alembic import command

        # Get current state before downgrade
        script = ScriptDirectory.from_config(self._config)
        current = self.current()

        if current is None:
            logger.info("No migrations to revert")
            return []

        # Run downgrade
        command.downgrade(self._config, revision)

        # Determine what was reverted
        new_current = self.current()
        reverted = []
        fallback = [current] if current and current != new_current else None
        for rev in self._safe_walk_revisions(script, current, new_current or "base", fallback):
            if rev and rev.revision != new_current:
                reverted.append(rev.revision)

        return reverted

    def current(self) -> str | None:
        """
        Get the current migration revision.

        Returns:
            The current revision ID, or None if no migrations applied
        """
        from alembic.runtime.migration import MigrationContext
        from sqlalchemy import create_engine

        url = (
            self._config.get_main_option("sqlalchemy.url")
            or self.database_url
            or os.environ.get("TOKENLEDGER_DATABASE_URL")
            or os.environ.get("DATABASE_URL")
        )
        if not url:
            raise ValueError("No database URL configured")

        engine = create_engine(url)
        with engine.connect() as conn:
            context = MigrationContext.configure(conn, opts={"version_table_schema": self.schema})
            return context.get_current_revision()

    def status(self) -> list[dict[str, Any]]:
        """
        Get the status of all migrations.

        Returns:
            List of migration status dictionaries with version, description, applied
        """
        from alembic.script import ScriptDirectory

        script = ScriptDirectory.from_config(self._config)
        current = self.current()

        result = []
        # Get all revisions - walk from head to base
        revisions = self._safe_walk_revisions(script, "head", "base")
        revisions.reverse()  # Oldest first

        # Determine which revisions are applied
        applied_revisions = set()
        if current:
            for rev in self._safe_walk_revisions(script, current, "base", [current]):
                if rev:
                    applied_revisions.add(rev.revision)

        for rev in revisions:
            result.append(
                {
                    "version": rev.revision,
                    "description": rev.doc or "",
                    "applied": rev.revision in applied_revisions,
                    "applied_at": None,  # Alembic doesn't track this by default
                }
            )

        return result

    def history(self) -> list[dict[str, Any]]:
        """
        Get the migration history (applied migrations only).

        Returns:
            List of applied migration dictionaries
        """
        return [m for m in self.status() if m["applied"]]

    def heads(self) -> list[str]:
        """
        Get the head revision(s).

        Returns:
            List of head revision IDs
        """
        from alembic.script import ScriptDirectory

        script = ScriptDirectory.from_config(self._config)
        return list(script.get_heads())

    def stamp(self, revision: str) -> None:
        """
        Stamp the database with a revision without running migrations.

        Useful for marking a database as already having certain migrations applied.

        Args:
            revision: Revision to stamp (e.g., "head" or specific revision)
        """
        from alembic import command

        command.stamp(self._config, revision)
        logger.info(f"Stamped database with revision: {revision}")
