# TokenLedger CLI Migration System - Implementation Plan

**Status:** In Progress
**Created:** 2026-01-22
**Author:** Claude (assisted implementation)

---

## Overview

Add a CLI migration system to TokenLedger that allows users to manage database schema migrations similar to Alembic, with support for a dedicated PostgreSQL schema for better isolation.

### Goals

1. **Dedicated Schema**: TokenLedger tables live in `token_ledger` schema (configurable), not `public`
2. **CLI Commands**: `uv run tokenledger db upgrade head`, `tokenledger db current`, etc.
3. **Migration Tracking**: Track applied migrations in a version table
4. **Backward Compatible**: Existing users can migrate from `public` schema

### Schema Status

The Alembic migrations in `alembic/versions/` are the source of truth:
- `001` - Initial schema with events table and indexes
- `002` - Attribution columns including `metadata_extra` (JSONB)

Note: The raw SQL files in `migrations/` are for manual use but Alembic is the preferred method.

---

## Design Decisions

### Why Dedicated Schema?

- **Isolation**: TokenLedger tables don't pollute user's application schema
- **Clarity**: Easy to identify TokenLedger tables (`token_ledger.*`)
- **Permissions**: Can grant/revoke schema-level permissions
- **Portability**: Can `pg_dump` just the TokenLedger schema
- **Multi-tenancy**: Multiple TokenLedger instances in same database (different schemas)

### Using Alembic Programmatically

We build on top of Alembic (not reinventing it) because:

1. **Battle-tested**: Alembic is the industry standard for SQLAlchemy migrations
2. **Version tracking**: Uses `alembic_version` table for tracking applied migrations
3. **Rollback support**: Full downgrade support with dependency tracking
4. **Existing setup**: Project already has `alembic/` directory with migrations

The `MigrationRunner` class wraps Alembic's programmatic API to:
- `tokenledger db upgrade` wraps `alembic upgrade` with nicer CLI/output
- Find bundled migrations automatically (from package or repo)
- Create dedicated schema if needed before running migrations
- Provide simplified status/history queries

### CLI Framework Choice: Typer

- Modern, uses type hints
- Clean subcommand structure
- Good error messages
- Already common in Python ecosystem
- Pairs well with Rich for output formatting

---

## Architecture

### New Files

```
tokenledger/
├── cli/
│   ├── __init__.py           # CLI entry point (typer app)
│   ├── db.py                  # Database subcommands (upgrade, downgrade, etc.)
│   └── server.py              # Server subcommand (moved from current entry point)
├── migrations/
│   ├── __init__.py            # Migration registry and runner
│   ├── versions/              # Individual migration modules
│   │   ├── __init__.py
│   │   ├── v001_initial_schema.py
│   │   └── v002_attribution_columns.py
│   └── runner.py              # Migration execution logic
└── (existing files)

migrations/                    # Keep SQL files for reference/manual use
├── 001_initial.sql
└── 002_add_attribution_columns.sql
```

### Migration Version Table

```sql
CREATE TABLE IF NOT EXISTS token_ledger.schema_versions (
    version VARCHAR(20) PRIMARY KEY,
    description VARCHAR(255) NOT NULL,
    applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    checksum VARCHAR(64)  -- SHA256 of migration content for integrity
);
```

### Migration Module Structure

```python
# tokenledger/migrations/versions/v001_initial_schema.py

VERSION = "001"
DESCRIPTION = "Initial schema with events table and indexes"
DEPENDS_ON = None  # or "000" for chaining

def upgrade(conn, schema: str = "token_ledger") -> None:
    """Apply this migration."""
    # SQL execution with schema substitution
    ...

def downgrade(conn, schema: str = "token_ledger") -> None:
    """Revert this migration."""
    # DROP statements
    ...
```

---

## CLI Commands

### Command Structure

```
tokenledger
├── db                          # Database management
│   ├── init                    # Create schema and version table
│   ├── upgrade [revision]      # Run migrations (default: head)
│   ├── downgrade <revision>    # Revert to revision
│   ├── current                 # Show current version
│   ├── history                 # Show migration history
│   └── status                  # Show pending migrations
├── serve                       # Start API server (moved)
│   ├── --host                  # Host to bind
│   ├── --port                  # Port to bind
│   └── --reload                # Auto-reload on changes
└── version                     # Show TokenLedger version
```

### Example Usage

```bash
# Initialize database (creates schema + version table)
uv run tokenledger db init

# Run all pending migrations
uv run tokenledger db upgrade head

# Check current state
uv run tokenledger db current
# Output: Current version: 002 (attribution_columns)

# Show what would run
uv run tokenledger db status
# Output:
#   ✓ 001 - Initial schema (applied 2026-01-15)
#   ✓ 002 - Attribution columns (applied 2026-01-20)
#   • 003 - Streaming metadata (pending)

# Start server
uv run tokenledger serve --port 8765
```

---

## Implementation Phases

### Phase 1: CLI Foundation

**Files to create/modify:**
- `tokenledger/cli/__init__.py` - Main typer app
- `tokenledger/cli/server.py` - Moved server command
- `pyproject.toml` - Update entry point

**Tasks:**
1. Add `typer` and `rich` to dependencies
2. Create CLI structure with `tokenledger` main command
3. Add `serve` subcommand (wrapper for existing `run_server`)
4. Add `version` command
5. Update entry point in pyproject.toml
6. Ensure backward compatibility: `tokenledger` without args shows help

**Acceptance:**
- `uv run tokenledger --help` shows commands
- `uv run tokenledger serve` starts server
- `uv run tokenledger version` shows version

### Phase 2: Migration Infrastructure

**Files to create:**
- `tokenledger/migrations/__init__.py` - Migration registry
- `tokenledger/migrations/runner.py` - Migration runner
- `tokenledger/cli/db.py` - Database CLI commands

**Tasks:**
1. Create `MigrationRunner` class that:
   - Connects to database
   - Creates schema if not exists (`CREATE SCHEMA IF NOT EXISTS token_ledger`)
   - Creates version table if not exists
   - Tracks applied/pending migrations
2. Implement `db init` command
3. Implement `db current` command
4. Implement `db status` command
5. Add tests for migration infrastructure

**Acceptance:**
- `tokenledger db init` creates schema and version table
- `tokenledger db current` shows "No migrations applied" initially
- `tokenledger db status` shows all migrations as pending

### Phase 3: Migration Versions

**Files to create:**
- `tokenledger/migrations/versions/__init__.py`
- `tokenledger/migrations/versions/v001_initial_schema.py`
- `tokenledger/migrations/versions/v002_attribution_columns.py`

**Tasks:**
1. Convert `001_initial.sql` to Python migration module
2. Convert `002_add_attribution_columns.sql` to Python migration module
3. Add `upgrade` and `downgrade` functions to each
4. Register migrations in version registry
5. Add schema substitution (`{schema}` placeholders)

**Key Design:**
```python
# Each migration has:
def upgrade(conn, schema: str) -> None:
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {schema}.token_ledger_events (
            ...
        )
    """)

def downgrade(conn, schema: str) -> None:
    conn.execute(f"DROP TABLE IF EXISTS {schema}.token_ledger_events")
```

**Acceptance:**
- Migration modules importable and have correct VERSION/DESCRIPTION
- upgrade/downgrade functions executable

### Phase 4: Upgrade/Downgrade Commands

**Files to modify:**
- `tokenledger/migrations/runner.py`
- `tokenledger/cli/db.py`

**Tasks:**
1. Implement `db upgrade` command:
   - Accept optional revision (default: "head")
   - Run pending migrations in order
   - Record each in version table
   - Show progress with rich output
2. Implement `db downgrade` command:
   - Accept required revision
   - Run downgrade functions in reverse order
   - Remove from version table
   - Confirm before destructive action
3. Handle errors gracefully (transaction per migration)
4. Add dry-run mode (`--dry-run`)

**Acceptance:**
- `tokenledger db upgrade head` runs all pending
- `tokenledger db upgrade 001` stops at version 001
- `tokenledger db downgrade 001` reverts to version 001
- Errors roll back individual migration, not all

### Phase 5: Schema Configuration

**Files to modify:**
- `tokenledger/config.py`
- `tokenledger/cli/db.py`
- `tokenledger/migrations/runner.py`

**Tasks:**
1. Update `TokenLedgerConfig`:
   - Default `schema_name` to `"token_ledger"` (was `"public"`)
   - Add `TOKENLEDGER_SCHEMA` environment variable
2. CLI accepts `--schema` option for all db commands
3. Migration runner uses configured schema
4. Update all SQL generation to use schema prefix
5. Add migration path from `public` to dedicated schema (optional utility)

**Acceptance:**
- Default schema is `token_ledger`
- `tokenledger db upgrade --schema=my_schema` works
- `TOKENLEDGER_SCHEMA=custom tokenledger db upgrade` works

### Phase 6: Documentation & Polish

**Files to create/modify:**
- `docs/migrations.md` - New documentation
- `docs/quickstart.md` - Update with CLI instructions
- `README.md` - Update quick start

**Tasks:**
1. Document all CLI commands
2. Add migration authoring guide (for contributors)
3. Update quickstart to use `tokenledger db upgrade`
4. Add troubleshooting section
5. Update examples

---

## Configuration Changes

### Updated Config Defaults

```python
@dataclass
class TokenLedgerConfig:
    # Database
    database_url: str = ""
    schema_name: str = "token_ledger"  # Changed from "public"
    table_name: str = "token_ledger_events"

    @property
    def full_table_name(self) -> str:
        return f"{self.schema_name}.{self.table_name}"
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | - | PostgreSQL connection string |
| `TOKENLEDGER_SCHEMA` | `token_ledger` | Schema name for all tables |
| `TOKENLEDGER_TABLE` | `token_ledger_events` | Events table name |

---

## Migration to Dedicated Schema (Existing Users)

For users with existing data in `public` schema:

```bash
# Option 1: Move existing data
tokenledger db migrate-schema --from=public --to=token_ledger

# Option 2: Keep using public (explicit)
tokenledger db upgrade --schema=public
```

The `migrate-schema` command would:
1. Create new schema
2. Copy table structure
3. Copy data with INSERT...SELECT
4. Create indexes
5. Verify row counts
6. Optionally drop old table

---

## Testing Strategy

### Unit Tests
- Migration module loading
- Version ordering
- Schema substitution
- CLI argument parsing

### Integration Tests
- Full upgrade cycle on test database
- Downgrade and re-upgrade
- Schema isolation (two schemas in same DB)
- Concurrent migration attempts (locking)

### Test Database
```python
# tests/conftest.py
@pytest.fixture
def test_db():
    """Create isolated test database/schema."""
    schema = f"test_{uuid.uuid4().hex[:8]}"
    # Create schema, yield, cleanup
```

---

## Dependencies to Add

```toml
# pyproject.toml
dependencies = [
    # ... existing ...
    "typer>=0.9.0",
    "rich>=13.0.0",
]
```

Note: `typer` already includes `click` as a dependency.

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking existing `public` schema users | High | Default to `token_ledger` only for new installs; add migration utility |
| Migration failures mid-way | Medium | Transaction per migration; clear error messages |
| Schema permission issues | Medium | Document required permissions; helpful error messages |
| Async vs sync database access | Low | Migration runner uses sync (psycopg2); async not needed for migrations |

---

## Success Criteria

1. **New users**: `pip install tokenledger && tokenledger db upgrade` works out of box
2. **Existing users**: Can continue using `public` schema with `--schema=public`
3. **CI/CD friendly**: Non-interactive, clear exit codes
4. **Safe**: Migrations are transactional, downgrades possible
5. **Observable**: Clear output showing what's happening

---

## Timeline Estimate

| Phase | Effort |
|-------|--------|
| Phase 1: CLI Foundation | Small |
| Phase 2: Migration Infrastructure | Medium |
| Phase 3: Migration Versions | Small |
| Phase 4: Upgrade/Downgrade | Medium |
| Phase 5: Schema Configuration | Small |
| Phase 6: Documentation | Small |

---

## References

- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [Typer Documentation](https://typer.tiangolo.com/)
- [PostgreSQL Schemas](https://www.postgresql.org/docs/current/ddl-schemas.html)
