# TokenLedger Quick Start Guide

Get LLM cost visibility in under 5 minutes.

## Prerequisites

- Python 3.11+
- PostgreSQL database (or Supabase)
- OpenAI or Anthropic API key

## Installation

```bash
pip install tokenledger
```

---

## Easy: Auto-Track Everything (2 Lines)

The simplest way to get started. Just patch the SDK and all calls are automatically tracked.

```python
import tokenledger
import openai

# Add these 2 lines
tokenledger.configure(database_url="postgresql://user:pass@localhost/db")
tokenledger.patch_openai()

# Your existing code works unchanged
client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
# Automatically tracked with tokens, cost, and latency
```

Works with Anthropic too:

```python
import tokenledger
import anthropic

tokenledger.configure(database_url="postgresql://...")
tokenledger.patch_anthropic()

client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

---

## Medium: Track by User & Feature (Attribution)

Know exactly **who** is costing you money and **what feature** is driving costs.

### Context Manager

```python
import tokenledger
from tokenledger import attribution
import openai

tokenledger.configure(database_url="postgresql://...")
tokenledger.patch_openai()

client = openai.OpenAI()

# Track costs by user and feature
with attribution(user_id="user_123", feature="summarize"):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Summarize this article..."}]
    )
    # Event automatically tagged with user_id and feature
```

### Decorator

```python
from tokenledger import attribution

@attribution(feature="chat", team="product")
def handle_chat(user_id: str, message: str):
    with attribution(user_id=user_id):  # Nest contexts - they merge!
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": message}]
        )
    return response

# Every call inside is tagged with feature="chat", team="product", and user_id
```

### All Attribution Fields

```python
with attribution(
    user_id="user_123",        # Who made the request
    session_id="sess_abc",     # Session identifier
    organization_id="org_456", # Multi-tenant org
    feature="summarize",       # Feature name (summarize, chat, search)
    team="ml",                 # Team responsible (ml, product, platform)
    project="api-v2",          # Project name
    cost_center="CC-001",      # Billing/cost center code
    custom_field="value",      # Any extra fields go to metadata_extra
):
    # All LLM calls inside are attributed
    ...
```

### Query by Attribution

```sql
-- Cost by feature
SELECT feature, SUM(cost_usd) as cost
FROM token_ledger_events
WHERE timestamp >= NOW() - INTERVAL '30 days'
GROUP BY feature
ORDER BY cost DESC;

-- Cost by team
SELECT team, SUM(cost_usd) as cost, COUNT(*) as requests
FROM token_ledger_events
WHERE timestamp >= NOW() - INTERVAL '30 days'
GROUP BY team;

-- Top users by cost
SELECT user_id, feature, SUM(cost_usd) as cost
FROM token_ledger_events
WHERE timestamp >= NOW() - INTERVAL '7 days'
GROUP BY user_id, feature
ORDER BY cost DESC
LIMIT 10;
```

---

## Hard: Full Production Setup

Complete setup with middleware, async tracking, and database optimization.

### 1. Database Setup

**Database Driver Selection:**

TokenLedger supports two database drivers:

- **psycopg2** (default): Best for sync code or when using `async_mode=True` (background thread). Install with `pip install tokenledger[postgres]`
- **asyncpg**: Recommended for purely async applications using `async`/`await`. Install with `pip install tokenledger[asyncpg]`

> **Note:** `async_mode=True` (the default) uses a background thread for non-blocking writes with psycopg2. This is different from true async I/O which requires asyncpg.

Run the migrations to create tables with proper indexes:

```bash
# Using Alembic (recommended)
alembic upgrade head

# Or manually run SQL
psql $DATABASE_URL < migrations/001_initial.sql
psql $DATABASE_URL < migrations/002_add_attribution_columns.sql
```

The schema includes:
- `token_ledger_events` table with all attribution columns
- Optimized indexes for time-series and attribution queries
- Helper views: `token_ledger_daily_costs`, `token_ledger_user_costs`, `token_ledger_team_costs`, `token_ledger_cost_center_costs`

### 2. FastAPI Integration

```python
from fastapi import FastAPI, Request
from tokenledger import configure, patch_openai
from tokenledger.middleware import FastAPIMiddleware

app = FastAPI()

# Configure TokenLedger
configure(
    database_url="postgresql://...",
    app_name="my-api",
    environment="production",
    async_mode=True,       # Non-blocking writes
    batch_size=100,        # Batch inserts for performance
    flush_interval_seconds=5,
)

# Patch SDKs
patch_openai()

# Add middleware - automatically extracts user_id and attribution from headers
app.add_middleware(FastAPIMiddleware)

@app.post("/chat")
async def chat(request: Request):
    # X-User-ID header -> user_id
    # X-Feature header -> feature
    # X-Team header -> team
    # X-Cost-Center header -> cost_center
    ...
```

### 3. Manual Event Tracking

For custom LLM providers or fine-grained control:

```python
from tokenledger import get_tracker, attribution, LLMEvent

tracker = get_tracker()

# Track with full control
with attribution(user_id="user_123", feature="custom"):
    event = LLMEvent(
        provider="custom",
        model="my-model",
        input_tokens=100,
        output_tokens=50,
        duration_ms=250.0,
        status="success",
    )
    tracker.track(event)
```

### 4. Async Tracking

For high-throughput applications:

```python
import asyncio
from tokenledger import configure, get_async_tracker, LLMEvent

configure(database_url="postgresql://...", async_mode=True)

async def track_events():
    tracker = await get_async_tracker()

    event = LLMEvent.fast_construct(  # Skip validation for speed
        provider="openai",
        model="gpt-4o",
        input_tokens=100,
        output_tokens=50,
        user_id="user_123",
        feature="batch-process",
    )
    await tracker.track_async(event)

    # Flush before shutdown
    await tracker.flush_async()

asyncio.run(track_events())
```

### 5. Environment Variables

```bash
# Required
export DATABASE_URL="postgresql://user:pass@localhost/db"

# Optional
export TOKENLEDGER_APP_NAME="my-app"
export TOKENLEDGER_ENVIRONMENT="production"
export TOKENLEDGER_DEBUG="false"
export TOKENLEDGER_BATCH_SIZE="100"
export TOKENLEDGER_FLUSH_INTERVAL="5"
export TOKENLEDGER_SAMPLE_RATE="1.0"  # 0.1 = sample 10%
```

### 6. Monitoring Queries

```sql
-- Daily cost trend
SELECT * FROM token_ledger_daily_costs
ORDER BY date DESC
LIMIT 30;

-- User cost leaderboard
SELECT * FROM token_ledger_user_costs
ORDER BY total_cost DESC
LIMIT 20;

-- Cost by feature and team (last 7 days)
SELECT
    feature,
    team,
    SUM(cost_usd) as cost,
    COUNT(*) as requests,
    AVG(duration_ms) as avg_latency_ms
FROM token_ledger_events
WHERE timestamp >= NOW() - INTERVAL '7 days'
GROUP BY feature, team
ORDER BY cost DESC;

-- Detect cost anomalies (users spending 2x their average)
WITH user_avg AS (
    SELECT user_id, AVG(daily_cost) as avg_daily
    FROM (
        SELECT user_id, DATE(timestamp), SUM(cost_usd) as daily_cost
        FROM token_ledger_events
        WHERE timestamp >= NOW() - INTERVAL '30 days'
        GROUP BY user_id, DATE(timestamp)
    ) daily
    GROUP BY user_id
)
SELECT
    e.user_id,
    DATE(e.timestamp) as date,
    SUM(e.cost_usd) as daily_cost,
    ua.avg_daily
FROM token_ledger_events e
JOIN user_avg ua ON e.user_id = ua.user_id
WHERE e.timestamp >= NOW() - INTERVAL '1 day'
GROUP BY e.user_id, DATE(e.timestamp), ua.avg_daily
HAVING SUM(e.cost_usd) > ua.avg_daily * 2;
```

---

## What's Tracked

For every LLM call, TokenLedger captures:

| Field | Description |
|-------|-------------|
| `timestamp` | When the call was made |
| `provider` | openai, anthropic, etc. |
| `model` | gpt-4o, claude-3-5-sonnet, etc. |
| `input_tokens` | Prompt tokens |
| `output_tokens` | Completion tokens |
| `cached_tokens` | Cached/prompt tokens (Anthropic) |
| `cost_usd` | Calculated cost |
| `duration_ms` | Response latency |
| `status` | success or error |
| **Attribution** | |
| `user_id` | Your user identifier |
| `session_id` | Session identifier |
| `organization_id` | Multi-tenant org ID |
| `feature` | Feature name |
| `team` | Team responsible |
| `project` | Project name |
| `cost_center` | Billing code |
| `metadata_extra` | Custom JSON data |

## Supabase Setup

1. Go to Supabase Dashboard → Settings → Database
2. Copy the connection string
3. Run migrations in SQL Editor:
   - `migrations/001_initial.sql`
   - Or use Alembic: `alembic upgrade head`
4. Configure TokenLedger:

```python
tokenledger.configure(
    database_url="postgresql://postgres:[password]@db.[ref].supabase.co:5432/postgres"
)
```

## Next Steps

- View the [Dashboard](dashboard.md) for visualization
- Check [API Reference](api-reference.md) for full documentation
- See [Framework Integration](frameworks.md) for Flask, Django, etc.
