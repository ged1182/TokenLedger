.PHONY: help install dev setup test test-cov test-integration test-all lint format typecheck check build clean dashboard docker-up docker-down db-start db-stop db-shell db-logs

# Default target
help:
	@echo "TokenLedger Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make setup            First-time setup (install deps + start test db)"
	@echo "  make install          Install production dependencies"
	@echo "  make dev              Install all dependencies (dev + all extras)"
	@echo ""
	@echo "Testing:"
	@echo "  make test             Run unit tests (no database required)"
	@echo "  make test-integration Run integration tests (requires database)"
	@echo "  make test-all         Run all tests (unit + integration)"
	@echo "  make test-cov         Run all tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             Run ruff linter"
	@echo "  make format           Format code with ruff"
	@echo "  make typecheck        Run mypy type checker"
	@echo "  make check            Run all checks (format, lint, typecheck, test)"
	@echo ""
	@echo "Database:"
	@echo "  make db-start         Start PostgreSQL test database (port 5433)"
	@echo "  make db-stop          Stop PostgreSQL test database"
	@echo "  make db-shell         Open psql shell to test database"
	@echo "  make db-logs          View database container logs"
	@echo ""
	@echo "Build:"
	@echo "  make build            Build package (wheel + sdist)"
	@echo "  make clean            Remove build artifacts and containers"
	@echo ""
	@echo "Dashboard:"
	@echo "  make dashboard        Build React dashboard"
	@echo "  make dashboard-dev    Start dashboard dev server"
	@echo ""
	@echo "Full Stack:"
	@echo "  make docker-up        Start full stack (postgres + api + dashboard)"
	@echo "  make docker-down      Stop full stack"

# =============================================================================
# Setup
# =============================================================================
install:
	pip install -e .

dev:
	pip install -e ".[all,dev]"
	-pre-commit install

setup: dev db-start
	@echo ""
	@echo "✓ Setup complete!"
	@echo ""
	@echo "Run 'make test-all' to verify everything works."
	@echo "Run 'make test' for quick unit tests (no database)."

# =============================================================================
# Testing
# =============================================================================
test:
	pytest -m "not integration" -v

test-integration: db-start db-wait
	pytest -m integration -v

test-all: db-start db-wait
	pytest -v

test-cov: db-start db-wait
	pytest --cov=tokenledger --cov-report=term-missing --cov-report=html

# =============================================================================
# Database (for integration tests)
# =============================================================================
db-start:
	@echo "Starting PostgreSQL test database..."
	@docker compose -f docker-compose.test.yml up -d
	@echo "PostgreSQL starting on localhost:5433"

db-stop:
	docker compose -f docker-compose.test.yml down

db-wait:
	@echo "Waiting for PostgreSQL to be ready..."
	@until docker compose -f docker-compose.test.yml exec -T postgres pg_isready -U tokenledger -d tokenledger_test > /dev/null 2>&1; do \
		sleep 1; \
	done
	@echo "✓ PostgreSQL is ready"

db-shell:
	docker compose -f docker-compose.test.yml exec postgres psql -U tokenledger -d tokenledger_test

db-logs:
	docker compose -f docker-compose.test.yml logs -f postgres

db-reset: db-stop
	docker volume rm tokenledger_postgres_test_data 2>/dev/null || true
	$(MAKE) db-start db-wait

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
	-docker compose -f docker-compose.test.yml down -v 2>/dev/null
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleaned up"

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
