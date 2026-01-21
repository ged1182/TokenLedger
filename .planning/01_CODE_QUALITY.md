# TokenLedger Code Quality Report

**Date:** 2026-01-21
**Reviewer:** Claude Code (Opus 4.5)
**Repository:** TokenLedger - LLM Cost Analytics for Postgres

---

## Executive Summary

| Tool | Status | Issues |
|------|--------|--------|
| **Ruff Linter** | PASS | 0 errors, 0 warnings |
| **Ruff Formatter** | FAIL | 2 files need formatting |
| **Mypy Type Checker** | PASS | 0 errors (with dependencies installed) |
| **Security Review** | PASS (with notes) | No critical vulnerabilities; minor SQL injection considerations |
| **Code Style** | PASS | Consistent conventions, good docstring coverage |
| **Error Handling** | PASS | Comprehensive error handling patterns |

**Overall Assessment:** The codebase demonstrates professional-grade quality with consistent patterns, good type safety, and thorough error handling. Minor formatting issues exist but are easily fixable.

---

## Detailed Findings

### 1. Ruff Linter Analysis

**Command:** `ruff check tokenledger`
**Result:** All checks passed!

The codebase passes all 16 enabled rule categories:
- E/W: pycodestyle (errors/warnings)
- F: Pyflakes
- I: isort
- B: flake8-bugbear
- C4: flake8-comprehensions
- UP: pyupgrade
- ARG: flake8-unused-arguments
- SIM: flake8-simplify
- TCH: flake8-type-checking
- PTH: flake8-use-pathlib
- ERA: eradicate
- PL: Pylint
- RUF: Ruff-specific

The `pyproject.toml` configuration shows thoughtful rule selection with appropriate ignores:
- `PLR0913` (too many arguments) - necessary for flexible APIs
- `B008` (function call in default argument) - required for FastAPI dependency injection
- `PLW0603` (global statement) - acceptable for singleton pattern

### 2. Ruff Formatter Analysis

**Command:** `ruff format --check tokenledger`
**Result:** 2 files would be reformatted

#### File: `/home/george-dekermenjian/git/TokenLedger/tokenledger/async_db.py`

**Lines 143-145:** Multi-line string should be single line
```python
# Current:
raise RuntimeError(
    "Database not initialized. Call await db.initialize() first."
)

# Expected:
raise RuntimeError("Database not initialized. Call await db.initialize() first.")
```

**Line 234:** Missing spaces around operator in f-string
```python
# Current:
placeholders = ", ".join(f"${i+1}" for i in range(len(columns)))

# Expected:
placeholders = ", ".join(f"${i + 1}" for i in range(len(columns)))
```

#### File: `/home/george-dekermenjian/git/TokenLedger/tokenledger/interceptors/anthropic.py`

**Lines 376-378:** Multi-line string should be single line
```python
# Current:
raise ImportError(
    "Anthropic SDK not installed. Run: pip install anthropic"
) from err

# Expected:
raise ImportError("Anthropic SDK not installed. Run: pip install anthropic") from err
```

### 3. Mypy Type Checker Analysis

**Command:** `mypy tokenledger`
**Result:** Success: no issues found in 12 source files

**Note:** Initial run showed 8 import-not-found errors for optional dependencies (fastapi, pydantic, openai, anthropic). These resolved after installing the `[all]` dependencies, confirming the type stubs are properly available.

The codebase uses modern Python 3.11+ type hints effectively:
- Union types use `X | None` syntax
- Generic types use `list[T]`, `dict[K, V]` syntax
- Type annotations on all public API methods
- Proper use of `Any` for truly dynamic values

### 4. Code Style Observations

#### Strengths

1. **Consistent Naming Conventions**
   - Classes: PascalCase (`TokenTracker`, `LLMEvent`, `AsyncDatabase`)
   - Functions/methods: snake_case (`get_tracker`, `track_event`)
   - Private methods: underscore prefix (`_get_connection`, `_flush_batch`)
   - Constants: SCREAMING_SNAKE_CASE (`OPENAI_PRICING`, `USE_ASYNCPG`)

2. **Module-Level Docstrings**
   - All 12 Python files have descriptive module docstrings
   - Example from `tracker.py`:
     ```python
     """
     TokenLedger Core Tracker
     Handles event logging to PostgreSQL with batching and async support.
     """
     ```

3. **Public API Docstrings**
   - Comprehensive docstrings with Args, Returns, and Example sections
   - Example from `queries.py`:
     ```python
     def get_cost_summary(
         self,
         days: int = 30,
         user_id: str | None = None,
         ...
     ) -> CostSummary:
         """
         Get cost summary for a time period.

         Args:
             days: Number of days to look back
             user_id: Filter by user
             ...

         Returns:
             CostSummary with totals and averages
         """
     ```

4. **Dataclass Usage**
   - Consistent use of `@dataclass` for data structures
   - Files: `config.py`, `tracker.py`, `queries.py`, `pricing.py`
   - Clean field definitions with type hints and defaults

5. **Import Organization**
   - Standard library first, third-party second, local imports last
   - Enforced by ruff isort rules

#### Minor Observations

1. **Line Length:** Configured at 100 characters (reasonable for modern displays)

2. **Quote Style:** Double quotes consistently used (enforced by ruff format)

3. **Lazy Imports:** Used appropriately for optional dependencies
   - `psycopg2` imported inside methods to allow `psycopg` fallback
   - `asyncpg` imported only when async features used

### 5. Security Observations

#### SQL Injection Analysis

**Risk Level:** LOW (with caveats)

1. **Table Name Interpolation**

   Multiple files use f-strings to interpolate table names into SQL:

   ```python
   # tracker.py line 161
   create_sql = f"""
   CREATE TABLE IF NOT EXISTS {self.config.full_table_name} (
   ...
   """

   # queries.py line 128
   sql = f"""
       SELECT ... FROM {self.table_name}
       WHERE {where_sql}
   """
   ```

   **Analysis:** The table name comes from `TokenLedgerConfig` which:
   - Defaults to `"token_ledger_events"` (hardcoded)
   - Can be overridden via constructor (internal API only)
   - Not exposed to user input in any endpoint

   **Verdict:** Acceptable pattern for internal configuration. Table names cannot contain user-controlled data in normal usage.

2. **Parameterized Queries**

   All user-filterable parameters use proper parameterization:

   ```python
   # queries.py - Sync (psycopg2 style)
   if user_id:
       where_clauses.append("user_id = %s")
       params.append(user_id)

   # queries.py - Async (asyncpg style)
   if user_id:
       where_clauses.append(f"user_id = ${param_idx}")
       params.append(user_id)
   ```

   **Verdict:** Correct parameterization prevents SQL injection for user-supplied values.

3. **INTERVAL Interpolation**

   ```python
   # queries.py line 467
   where_clauses = [f"timestamp >= NOW() - INTERVAL '{days} days'"]
   ```

   **Analysis:** The `days` parameter comes from FastAPI's `Query(30, ge=1, le=365)` which:
   - Enforces integer type
   - Validates range (1-365)
   - Cannot contain SQL injection payload

   **Verdict:** Safe due to type enforcement at API layer.

#### CORS Configuration

```python
# server.py line 50
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    ...
)
```

**Observation:** Wide-open CORS is flagged with a comment. Acceptable for development/self-hosted scenarios but should be restricted in production deployments.

#### Error Message Exposure

```python
# server.py line 168
db_status = f"error: {e!s}"
```

**Observation:** Health check exposes database error messages. This is useful for debugging but could leak information in production. The `/health` endpoint is typically internal-only.

### 6. Error Handling Patterns

#### Strengths

1. **Consistent Exception Handling in Interceptors**
   ```python
   try:
       response = original_method(*args, **kwargs)
       # ... success handling
       event.status = "success"
       tracker.track(event)
       return response
   except Exception as e:
       event.status = "error"
       event.error_type = type(e).__name__
       event.error_message = str(e)[:1000]  # Truncate to prevent overflow
       tracker.track(event)
       raise  # Re-raise to not swallow errors
   ```

2. **Graceful Degradation**
   ```python
   # tracker.py line 266
   try:
       self._queue.put_nowait(event)
   except Exception:
       logger.warning("Event queue full, dropping event")
   ```

   Events are dropped rather than blocking the main application.

3. **Database Rollback on Error**
   ```python
   # tracker.py line 371
   except Exception as e:
       logger.error(f"Error writing batch: {e}")
       conn.rollback()
       return 0
   ```

4. **Import Error Handling for Optional Dependencies**
   ```python
   try:
       import psycopg2
       ...
   except ImportError:
       try:
           import psycopg
           ...
       except ImportError as err:
           raise ImportError(
               "No PostgreSQL driver found..."
           ) from err
   ```

5. **FastAPI Exception Chaining**
   ```python
   except Exception as e:
       raise HTTPException(status_code=500, detail=str(e)) from e
   ```

---

## Recommendations

### P0 - Critical (Fix Before Merge)

None identified. The codebase is production-ready from a code quality perspective.

### P1 - Important (Fix Soon)

1. **Run Formatter**
   ```bash
   ruff format tokenledger
   ```
   Fixes 2 files with minor formatting inconsistencies.

2. **Add Pre-commit Hook**
   The `pyproject.toml` includes `pre-commit>=3.5.0` in dev dependencies, but no `.pre-commit-config.yaml` was found. Create one to enforce formatting on commit:
   ```yaml
   repos:
     - repo: https://github.com/astral-sh/ruff-pre-commit
       rev: v0.8.0
       hooks:
         - id: ruff
         - id: ruff-format
   ```

3. **Expand Test Coverage**
   Current tests only cover the `pricing.py` module. Recommend adding:
   - Unit tests for `TokenTracker` and `AsyncTokenTracker`
   - Unit tests for `LLMEvent` dataclass
   - Integration tests for database operations (marked as such)
   - Tests for decorators and middleware

### P2 - Nice to Have (Backlog)

1. **Document CORS Configuration**
   Add explicit documentation or environment variable for production CORS settings:
   ```python
   CORS_ORIGINS = os.getenv("TOKENLEDGER_CORS_ORIGINS", "*").split(",")
   ```

2. **Consider Identifier Validation**
   While not exploitable in current usage, consider adding validation for `table_name` and `schema_name` in `TokenLedgerConfig`:
   ```python
   import re
   IDENTIFIER_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')

   def __post_init__(self):
       if not IDENTIFIER_PATTERN.match(self.table_name):
           raise ValueError(f"Invalid table name: {self.table_name}")
   ```

3. **Add `py.typed` Marker**
   Create an empty `tokenledger/py.typed` file to indicate the package supports type checking (PEP 561).

4. **Consider Structured Logging**
   Replace format-string logging with structured logging for better observability:
   ```python
   # Current
   logger.info(f"Flushed {len(events)} events to database")

   # Suggested
   logger.info("Flushed events to database", extra={"event_count": len(events)})
   ```

5. **Add Streaming Response Support for OpenAI**
   The Anthropic interceptor handles streaming (`_wrap_streaming_messages`), but the OpenAI interceptor does not. Consider adding `_wrap_stream_chat_completions` for feature parity.

---

## Files Reviewed

| File | Lines | Status |
|------|-------|--------|
| `tokenledger/__init__.py` | 37 | Clean |
| `tokenledger/config.py` | 123 | Clean |
| `tokenledger/tracker.py` | 583 | Clean |
| `tokenledger/pricing.py` | 199 | Clean |
| `tokenledger/queries.py` | 803 | Clean |
| `tokenledger/decorators.py` | 220 | Clean |
| `tokenledger/middleware.py` | 159 | Clean |
| `tokenledger/server.py` | 393 | Clean |
| `tokenledger/async_db.py` | 314 | Minor formatting |
| `tokenledger/interceptors/__init__.py` | 12 | Clean |
| `tokenledger/interceptors/openai.py` | 353 | Clean |
| `tokenledger/interceptors/anthropic.py` | 434 | Minor formatting |
| `tests/conftest.py` | 20 | Clean |
| `tests/test_pricing.py` | 84 | Clean |
| `migrations/001_initial.sql` | 113 | Clean |

---

## Conclusion

TokenLedger demonstrates strong software engineering practices:

- **Clean Architecture:** Clear separation between config, tracking, persistence, and API layers
- **Type Safety:** Full type annotations with mypy passing
- **Code Quality:** Zero ruff linter warnings
- **Error Handling:** Comprehensive and consistent patterns
- **Documentation:** Module and function docstrings throughout
- **Security:** Proper SQL parameterization, no critical vulnerabilities

The codebase is ready for production use. The only immediate action required is running `ruff format` to fix two minor formatting inconsistencies.
