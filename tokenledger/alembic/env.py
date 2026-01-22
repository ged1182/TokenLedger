"""
Alembic Environment Configuration for TokenLedger

This module configures how Alembic connects to the database and runs migrations.
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

# Alembic Config object - provides access to alembic.ini values
config = context.config

# Set up Python logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Target metadata for autogenerate support
# We don't use SQLAlchemy ORM models, so this is None
target_metadata = None


def get_database_url() -> str:
    """
    Get database URL from environment or config.

    Priority:
    1. TOKENLEDGER_DATABASE_URL environment variable
    2. DATABASE_URL environment variable
    3. sqlalchemy.url from alembic.ini
    """
    url = os.environ.get("TOKENLEDGER_DATABASE_URL")
    if url:
        return url

    url = os.environ.get("DATABASE_URL")
    if url:
        return url

    return config.get_main_option("sqlalchemy.url", "")


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    This generates SQL scripts without connecting to the database.
    Useful for reviewing migrations before applying them.

    Usage:
        alembic upgrade head --sql > migration.sql
    """
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.

    This connects to the database and applies migrations directly.
    """
    url = get_database_url()

    if not url or url == "driver://user:pass@localhost/dbname":
        raise ValueError(
            "Database URL not configured. Set TOKENLEDGER_DATABASE_URL or DATABASE_URL "
            "environment variable, or update sqlalchemy.url in alembic.ini"
        )

    connectable = create_engine(
        url,
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
