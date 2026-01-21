"""Integration tests for TokenLedger.

These tests require a running PostgreSQL database.
Start the test database with:

    docker compose -f docker-compose.test.yml up -d

Run integration tests:

    pytest -m integration

Or run all tests including integration:

    pytest
"""
