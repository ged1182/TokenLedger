.PHONY: help install dev test test-cov test-integration lint format typecheck check build clean dashboard docker-up docker-down

# Default target
help:
	@echo "TokenLedger Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install production dependencies"
	@echo "  make dev              Install all dependencies (dev + all extras)"
	@echo ""
	@echo "Testing:"
	@echo "  make test             Run unit tests"
	@echo "  make test-cov         Run tests with coverage report"
	@echo "  make test-integration Run integration tests (requires Postgres)"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             Run ruff linter"
	@echo "  make format           Format code with ruff"
	@echo "  make typecheck        Run mypy type checker"
	@echo "  make check            Run all checks (format, lint, typecheck, test)"
	@echo ""
	@echo "Build:"
	@echo "  make build            Build package (wheel + sdist)"
	@echo "  make clean            Remove build artifacts"
	@echo ""
	@echo "Dashboard:"
	@echo "  make dashboard        Build React dashboard"
	@echo "  make dashboard-dev    Start dashboard dev server"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up        Start Postgres container"
	@echo "  make docker-down      Stop Postgres container"

# =============================================================================
# Setup
# =============================================================================
install:
	pip install -e .

dev:
	pip install -e ".[all,dev]"
	pre-commit install

# =============================================================================
# Testing
# =============================================================================
test:
	pytest -m "not integration"

test-cov:
	pytest -m "not integration" --cov=tokenledger --cov-report=term-missing --cov-report=html

test-integration:
	pytest -m integration

# =============================================================================
# Code Quality
# =============================================================================
lint:
	ruff check tokenledger tests

format:
	ruff format tokenledger tests
	ruff check --fix tokenledger tests

typecheck:
	mypy tokenledger

check: format lint typecheck test
	@echo "All checks passed!"

# =============================================================================
# Build
# =============================================================================
build:
	python -m build

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# =============================================================================
# Dashboard
# =============================================================================
dashboard:
	cd dashboard && npm install && npm run build

dashboard-dev:
	cd dashboard && npm install && npm run dev

# =============================================================================
# Docker
# =============================================================================
docker-up:
	docker compose up -d postgres

docker-down:
	docker compose down
