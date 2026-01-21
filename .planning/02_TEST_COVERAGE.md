# TokenLedger Test Coverage Analysis

**Date:** January 21, 2026
**Overall Coverage:** 11%
**Test Files:** 1 (tests/test_pricing.py)
**Total Tests:** 7

---

## Executive Summary

TokenLedger's test suite is critically underdeveloped, with only **11% code coverage**. The project has a single test file covering only the `pricing.py` module, while the core functionality (event tracking, database operations, SDK interceptors, API server) remains entirely untested.

**Risk Assessment: HIGH** - The codebase is not production-ready from a testing perspective. Critical paths involving database operations, concurrent event tracking, and SDK monkey-patching have zero test coverage.

---

## Coverage Breakdown by Module

| Module | Statements | Missed | Coverage | Status |
|--------|------------|--------|----------|--------|
| `__init__.py` | 7 | 0 | **100%** | Adequate |
| `pricing.py` | 39 | 6 | **82%** | Good |
| `config.py` | 54 | 26 | **39%** | Needs work |
| `tracker.py` | 269 | 194 | **21%** | Critical gap |
| `decorators.py` | 85 | 75 | **9%** | Critical gap |
| `interceptors/openai.py` | 183 | 165 | **8%** | Critical gap |
| `interceptors/anthropic.py` | 231 | 214 | **6%** | Critical gap |
| `async_db.py` | 103 | 103 | **0%** | Untested |
| `middleware.py` | 64 | 64 | **0%** | Untested |
| `queries.py` | 244 | 244 | **0%** | Untested |
| `server.py` | 187 | 187 | **0%** | Untested |
| `interceptors/__init__.py` | 3 | 0 | **100%** | N/A (exports only) |

---

## Untested Critical Paths

### 1. Core Tracking System (`tracker.py`) - HIGHEST PRIORITY

The heart of TokenLedger is completely untested:

- **`LLMEvent` dataclass**: Event creation, serialization, auto cost calculation
- **`TokenTracker` class**:
  - Database connection management (psycopg2/psycopg3 fallback)
  - Table creation SQL
  - Event batching and queue management
  - Background flush thread
  - Sampling logic
  - Trace context propagation
  - Graceful shutdown
- **`AsyncTokenTracker` class**:
  - Async lock management
  - Connection pool integration
  - Async batch writing

**Lines Never Executed:** 78-582 (virtually the entire file beyond dataclass definition)

### 2. SDK Interceptors (`interceptors/`) - HIGH PRIORITY

Zero integration tests for the monkey-patching mechanism:

- **OpenAI Interceptor**:
  - Token extraction from responses
  - Request/response preview extraction
  - Sync `chat.completions.create` wrapper
  - Async `chat.completions.create` wrapper
  - Embeddings wrapper
  - Patch/unpatch lifecycle

- **Anthropic Interceptor**:
  - Token extraction (including cache_read_input_tokens)
  - Content block parsing
  - Messages create wrapper
  - Streaming response tracking (TrackedStream, TrackedStreamIterator)
  - Patch/unpatch lifecycle

### 3. Async Database Layer (`async_db.py`) - HIGH PRIORITY

- Connection pool initialization with asyncpg
- Table creation in async context
- `insert_events` bulk insert with proper JSON serialization
- Connection acquire/release lifecycle
- Pool closure and cleanup

### 4. API Server (`server.py`) - MEDIUM PRIORITY

- FastAPI endpoint handlers
- Pydantic response model serialization
- Error handling and HTTP exceptions
- Health check endpoint
- CORS middleware configuration
- Sync vs async mode switching

### 5. Analytics Queries (`queries.py`) - MEDIUM PRIORITY

- SQL query generation and parameterization
- Result dataclass mapping
- Both sync (`TokenLedgerQueries`) and async (`AsyncTokenLedgerQueries`) variants
- All 9 query methods (get_cost_summary, get_costs_by_model, etc.)

### 6. Middleware (`middleware.py`) - LOWER PRIORITY

- FastAPI ASGI middleware
- Flask extension integration
- Header extraction for user context
- Metadata restoration after request

---

## Test Quality Assessment

### Current Test File Analysis (`tests/test_pricing.py`)

**Strengths:**
- Clean class-based organization (`TestGetPricing`, `TestCalculateCost`)
- Type hints present
- Docstrings for each test
- Proper assertion patterns
- Edge cases covered (unknown models)

**Weaknesses:**
- No fixtures utilized (mock_config in conftest.py is unused)
- No parameterized tests despite repetitive patterns
- Missing tests for `estimate_monthly_cost` function
- No tests for provider auto-detection edge cases

### Test Infrastructure (`tests/conftest.py`)

The conftest defines a `mock_config` fixture that is **never used**:

```python
@pytest.fixture
def mock_config() -> TokenLedgerConfig:
    return TokenLedgerConfig(
        database_url="postgresql://test:test@localhost:5432/test",
        ...
    )
```

This suggests tests were planned but not implemented.

### Anti-Patterns Identified

1. **No mocking framework usage**: Complex external dependencies (psycopg2, asyncpg, openai SDK, anthropic SDK) would need mocking for unit tests.

2. **No database test fixtures**: Integration tests require proper test database setup/teardown.

3. **No test markers utilized**: The `integration` marker is defined but no tests use it.

4. **Global state not reset between tests**: Singletons like `_tracker`, `_config`, `_async_db` could leak state.

5. **No async test coverage**: Despite `pytest-asyncio` being installed and configured, no async tests exist.

---

## Recommendations (Prioritized)

### P0 - Production Blockers (Must Have)

1. **Add unit tests for `LLMEvent` dataclass**
   - Event creation with various field combinations
   - Auto cost calculation via `__post_init__`
   - `to_dict()` serialization

2. **Add unit tests for `TokenTracker` (with mocked database)**
   - Event tracking with mocked connection
   - Batch accumulation and flush trigger
   - Sampling behavior (sample_rate < 1.0)
   - Context manager `trace()` functionality

3. **Add unit tests for interceptors (with mocked SDKs)**
   - Token extraction helpers
   - Wrapper behavior with mock responses
   - Error handling paths
   - Patch/unpatch idempotency

### P1 - High Value (Should Have)

4. **Add integration tests for database operations**
   - Use pytest fixtures with Docker PostgreSQL (via testcontainers or similar)
   - Test table creation
   - Test event insertion and retrieval
   - Test batch insert performance

5. **Add API server tests**
   - Use FastAPI's TestClient
   - Test each endpoint with mock data
   - Test error responses

6. **Add async tracker tests**
   - Test with mocked asyncpg pool
   - Test concurrent event tracking
   - Test graceful shutdown

### P2 - Coverage Improvement (Nice to Have)

7. **Add config module tests**
   - Environment variable loading
   - `is_supabase` property detection
   - `full_table_name` construction

8. **Add decorator tests**
   - `@track_llm` with sync functions
   - `@track_llm` with async functions
   - `track_cost()` manual tracking

9. **Add query module tests**
   - Test SQL generation (can use snapshot testing)
   - Test result dataclass mapping

---

## Suggested Test Cases for Production Readiness

### `tests/test_tracker.py`

```python
# Unit Tests (no database)
class TestLLMEvent:
    def test_event_creation_defaults()
    def test_event_auto_cost_calculation()
    def test_event_to_dict_serialization()
    def test_event_with_metadata_json_encoding()

class TestTokenTracker:
    def test_track_event_adds_to_batch(mock_connection)
    def test_batch_flush_at_threshold(mock_connection)
    def test_sampling_drops_events_below_rate()
    def test_trace_context_propagation()
    def test_shutdown_flushes_pending_events()

class TestAsyncTokenTracker:
    async def test_async_track_event()
    async def test_concurrent_tracking_thread_safety()
```

### `tests/test_interceptors.py`

```python
class TestOpenAIInterceptor:
    def test_extract_tokens_from_response()
    def test_extract_tokens_with_cached_prompt()
    def test_wrap_chat_completions_tracks_event(mock_tracker)
    def test_wrap_handles_api_error(mock_tracker)
    def test_patch_unpatch_idempotent()

class TestAnthropicInterceptor:
    def test_extract_tokens_with_cache_read()
    def test_wrap_streaming_aggregates_tokens()
    def test_content_block_parsing()
```

### `tests/test_server.py`

```python
class TestServerEndpoints:
    def test_health_check_returns_connected()
    def test_health_check_returns_degraded_on_db_error()
    def test_summary_endpoint_returns_valid_response()
    def test_costs_by_model_calculates_percentages()
    def test_invalid_days_parameter_returns_422()
```

### `tests/test_integration.py` (marked with `@pytest.mark.integration`)

```python
@pytest.mark.integration
class TestDatabaseIntegration:
    def test_table_creation()
    def test_event_roundtrip()
    def test_batch_insert_performance()
    def test_query_cost_summary()
```

---

## Coverage Target Recommendations

| Phase | Coverage Target | Timeframe |
|-------|----------------|-----------|
| Minimum Viable | 50% | Week 1-2 |
| Production Ready | 70% | Week 3-4 |
| Enterprise Ready | 85% | Ongoing |

**Current state:** 11% coverage is a significant liability for a data-handling library where correctness is critical.

---

## Test Infrastructure Improvements Needed

1. **Add test database fixture** using `pytest-postgresql` or `testcontainers-python`

2. **Add mock fixtures** for:
   - `psycopg2.connect`
   - `asyncpg.create_pool`
   - `openai.chat.completions.create`
   - `anthropic.messages.create`

3. **Configure CI/CD** to:
   - Run tests on every PR
   - Enforce minimum coverage threshold
   - Run integration tests against real PostgreSQL

4. **Add mutation testing** (via `mutmut`) for pricing.py to validate test quality

5. **Add property-based testing** (via `hypothesis`) for:
   - Cost calculation edge cases
   - Event serialization roundtrips

---

## Conclusion

TokenLedger requires significant testing investment before it can be considered production-ready. The single test file covering pricing logic is well-written but represents only a small fraction of the codebase. The core value proposition (automatic LLM call tracking) is completely untested, which poses both reliability and maintainability risks.

Priority should be given to testing the `tracker.py` and `interceptors/` modules, as these contain the most complex logic and highest-risk code paths (database operations, threading, monkey-patching).
