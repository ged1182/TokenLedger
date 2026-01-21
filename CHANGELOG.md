# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- CI workflow with ruff, mypy, and pytest
- Single-source versioning via hatchling

### Changed
- Fixed Python version badge in README (3.11+)

### Fixed
- Code formatting issues in `async_db.py` and `anthropic.py`

## [0.1.0] - 2025-01-21

### Added
- Core event tracking with `TokenTracker` and `AsyncTokenTracker`
- OpenAI SDK auto-tracking via `patch_openai()`
- Anthropic SDK auto-tracking via `patch_anthropic()`
- FastAPI middleware integration
- Flask middleware integration
- `@track_llm` and `@track_cost` decorators
- React dashboard for cost visualization
- Async and sync database support (psycopg2, asyncpg)
- Configurable batching and sampling
- Built-in pricing for OpenAI, Anthropic, Google, and Mistral models
- SQL migrations for PostgreSQL
- Docker Compose setup for local development
- Pre-built analytics queries via `TokenLedgerQueries`

[Unreleased]: https://github.com/ged1182/tokenledger/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ged1182/tokenledger/releases/tag/v0.1.0
