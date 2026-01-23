# CLAUDE.md - TokenLedger

## Project Overview

TokenLedger is a self-hosted LLM cost analytics solution for tracking AI API costs with complete data ownership. It integrates with existing Postgres databases, automatically tracks OpenAI/Anthropic API calls, and provides a React dashboard for visualization.

## Tech Stack

- **Backend**: Python 3.11+, FastAPI, Uvicorn
- **Database**: PostgreSQL 16 (psycopg2 sync, asyncpg async)
- **Frontend**: React 18, Vite 5, TailwindCSS 3, Recharts
- **LLM SDKs**: openai >= 1.0.0, anthropic >= 0.18.0
- **Build**: Hatchling
- **Testing**: pytest, pytest-asyncio, pytest-cov
- **Linting**: ruff, mypy

## Common Commands

```bash
# Install
pip install -e ".[dev]"              # Development setup with all deps

# Test
pytest                               # Run all tests
pytest -m "not integration"          # Skip integration tests
pytest --cov=tokenledger             # Run with coverage

# Lint & Format
ruff check tokenledger               # Lint check
ruff format tokenledger              # Auto-format
mypy tokenledger                     # Type checking

# Dashboard
cd dashboard && npm install && npm run dev   # Dev server (port 3000)
cd dashboard && npm run build                # Production build

# Run server
python -m tokenledger.server         # API server (port 8765)
docker compose up                    # Full stack
```

## Project Structure

```
tokenledger/              # Main Python package
├── __init__.py           # Public API exports
├── config.py             # Configuration (dataclass-based)
├── tracker.py            # Core event tracking (TokenTracker, AsyncTokenTracker)
├── pricing.py            # LLM pricing data per provider
├── queries.py            # Analytics SQL queries
├── decorators.py         # @track_llm, @track_cost decorators
├── middleware.py         # FastAPI/Flask middleware
├── server.py             # FastAPI dashboard API
├── async_db.py           # AsyncPG connection pooling
└── interceptors/         # SDK monkey patches
    ├── openai.py         # OpenAI SDK auto-tracking
    └── anthropic.py      # Anthropic SDK auto-tracking

dashboard/                # React frontend
├── src/App.jsx           # Main dashboard component
├── vite.config.js
└── tailwind.config.js

tests/                    # Test suite
├── conftest.py           # Pytest fixtures
└── test_pricing.py       # Pricing tests
```

## Code Conventions

- **Type hints**: Required for all public APIs (PEP 484)
- **Dataclasses**: Used for config and data structures (`TokenLedgerConfig`, `LLMEvent`)
- **Line length**: 100 characters
- **Quote style**: Double quotes
- **Import order**: Enforced by ruff (isort rules)
- **Docstrings**: Module-level and for public functions

## Architecture Patterns

- **Singleton**: Global `_config` instance via `get_config()`
- **Monkey patching**: Interceptors wrap SDK methods for auto-tracking
- **Batching**: Events queued and flushed in configurable batches
- **Dual-mode**: Sync tracker with background thread, pure async tracker
- **Middleware**: Framework-agnostic for FastAPI/Flask integration

## Key Files

- `tokenledger/tracker.py:582` - Core tracking logic, `TokenTracker` class
- `tokenledger/pricing.py` - Model pricing (USD per 1M tokens)
- `tokenledger/queries.py` - Pre-built analytics queries
- `tokenledger/alembic/versions/` - Database migrations (run via `tokenledger db init`)

## Database

Main table: `token_ledger_events` with UUID primary keys, JSONB metadata, composite indexes. Views: `token_ledger_daily_costs`, `token_ledger_user_costs`.

## Environment Variables

- `DATABASE_URL` - PostgreSQL connection string
- `TOKENLEDGER_ENVIRONMENT` - Deployment environment
- `TOKENLEDGER_DEBUG` - Debug mode flag

## License

Elastic License 2.0 (ELv2)
