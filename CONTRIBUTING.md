# Contributing to TokenLedger

Thank you for your interest in contributing to TokenLedger! This document provides guidelines and instructions for contributing.

## Development Setup

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

4. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

5. **Start the development database**
   ```bash
   docker compose up -d postgres
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

### Unit Tests

Unit tests should not require a database connection:

```bash
pytest -m "not integration"
```

### Integration Tests

Integration tests require a running PostgreSQL database:

```bash
# Start the database
docker compose up -d postgres

# Run integration tests
pytest -m integration
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
