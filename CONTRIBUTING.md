# Contributing to TokenLedger

Thank you for your interest in contributing to TokenLedger! This document provides guidelines and instructions for contributing.

## Development Setup

### Quick Start (Recommended)

```bash
# Clone and enter the project
git clone https://github.com/ged1182/tokenledger.git
cd tokenledger

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# One-command setup: installs deps + starts test database
make setup

# Verify everything works
make test-all
```

### Manual Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/ged1182/tokenledger.git
   cd tokenledger
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -e ".[all,dev]"
   ```

4. **Install pre-commit hooks** (optional)
   ```bash
   pre-commit install
   ```

5. **Start the test database**
   ```bash
   make db-start
   # Or manually: docker compose -f docker-compose.test.yml up -d
   ```

## Code Quality

We use the following tools to maintain code quality:

- **Ruff** - Linting and formatting
- **Mypy** - Type checking
- **Pytest** - Testing

### Running Checks

```bash
# Format code
ruff format tokenledger tests

# Lint code
ruff check tokenledger tests

# Type check
mypy tokenledger

# Run tests
pytest

# Run tests with coverage
pytest --cov=tokenledger --cov-report=term-missing
```

### Pre-commit Hooks

Pre-commit hooks run automatically on `git commit`. To run them manually:

```bash
pre-commit run --all-files
```

## Testing

TokenLedger has two types of tests:
- **Unit tests** - Fast, no external dependencies
- **Integration tests** - Require PostgreSQL, test real database operations

### Makefile Commands (Recommended)

```bash
make test              # Run unit tests only (fast, no database)
make test-integration  # Run integration tests (starts database automatically)
make test-all          # Run all tests
make test-cov          # Run all tests with coverage report
```

### Unit Tests

Unit tests run without any database and use mocks:

```bash
pytest -m "not integration"
```

These tests cover:
- Configuration parsing
- Event creation and validation
- Pricing calculations
- Protocol compliance
- Mock database operations

### Integration Tests

Integration tests verify the complete workflow with a real PostgreSQL database.

**Setup:**
```bash
# Start the test database (port 5433)
make db-start

# Run integration tests
make test-integration
```

**What's tested:**
- `TokenTracker` with real database writes
- `PostgreSQLBackend` and `AsyncPostgreSQLBackend`
- Batch writing and flushing
- Database views (daily costs, user costs)
- Error handling and duplicate event handling

**Database Configuration:**

The test database runs on **port 5433** (not 5432) to avoid conflicts with any development database:

| Setting | Value |
|---------|-------|
| Host | localhost |
| Port | 5433 |
| Database | tokenledger_test |
| User | tokenledger |
| Password | tokenledger |

**Connection string:**
```
postgresql://tokenledger:tokenledger@localhost:5433/tokenledger_test
```

### Test Database Commands

```bash
make db-start   # Start PostgreSQL container
make db-stop    # Stop PostgreSQL container
make db-shell   # Open psql shell to test database
make db-logs    # View container logs
make db-reset   # Reset database (stop, remove volume, restart)
```

### Writing Integration Tests

Use the `@pytest.mark.integration` decorator and `requires_postgres` marker:

```python
import pytest
from tests.conftest import requires_postgres

@pytest.mark.integration
@requires_postgres
class TestMyFeature:
    def test_something(self, integration_config, clean_events_table):
        # integration_config provides a TokenLedgerConfig
        # clean_events_table truncates the table before/after each test
        pass
```

Available fixtures:
- `integration_config` - Sync `TokenLedgerConfig` pointing to test database
- `async_integration_config` - Async `TokenLedgerConfig`
- `integration_database_url` - Raw connection string
- `clean_events_table` - Truncates events table before/after test

### Skipped Tests

Integration tests are automatically skipped if the database is unavailable:

```
SKIPPED [1] tests/integration/test_tracker.py - PostgreSQL not available
         (run: docker compose -f docker-compose.test.yml up -d)
```

## Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **Make your changes**
   - Write tests for new functionality
   - Update documentation as needed
   - Follow the existing code style

3. **Run all checks**
   ```bash
   ruff format tokenledger tests
   ruff check tokenledger tests
   mypy tokenledger
   pytest
   ```

4. **Commit your changes**
   - Use [Conventional Commits](https://www.conventionalcommits.org/) format
   - Examples: `feat: add new feature`, `fix: resolve bug`, `docs: update README`

5. **Push and create a PR**
   ```bash
   git push -u origin feat/your-feature-name
   ```

## Code Style

- **Line length**: 100 characters
- **Quotes**: Double quotes
- **Type hints**: Required for all public APIs
- **Docstrings**: Required for modules and public functions

## Architecture

See the [Architecture documentation](.planning/03_ARCHITECTURE.md) for details on the codebase structure and design decisions.

## Questions?

- Open an issue for bugs or feature requests
- Check existing issues before creating new ones

## License

By contributing to TokenLedger, you agree that your contributions will be licensed under the Elastic License 2.0 (ELv2).
