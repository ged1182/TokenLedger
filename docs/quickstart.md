# TokenLedger Quick Start Guide

Get LLM cost visibility in under 5 minutes.

## Prerequisites

- Python 3.9+
- PostgreSQL database (or Supabase)
- OpenAI or Anthropic API key

## Step 1: Install

```bash
pip install tokenledger[postgres]

# Or with all extras
pip install tokenledger[all]
```

## Step 2: Create the Database Table

Run this SQL in your PostgreSQL database:

```sql
-- Copy from migrations/001_initial.sql
CREATE TABLE IF NOT EXISTS token_ledger_events (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    provider VARCHAR(50) NOT NULL,
    model VARCHAR(100) NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd DECIMAL(12, 8),
    user_id VARCHAR(255),
    session_id VARCHAR(255),
    organization_id VARCHAR(255),
    app_name VARCHAR(100),
    environment VARCHAR(50),
    status VARCHAR(20) DEFAULT 'success',
    duration_ms DOUBLE PRECISION,
    metadata JSONB
);

CREATE INDEX idx_token_ledger_timestamp ON token_ledger_events (timestamp DESC);
CREATE INDEX idx_token_ledger_user ON token_ledger_events (user_id, timestamp DESC);
CREATE INDEX idx_token_ledger_org ON token_ledger_events (organization_id, timestamp DESC);
```

## Step 3: Integrate (2 lines!)

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
```

## Step 4: View Your Data

### Option A: SQL

```sql
SELECT 
    DATE(timestamp) as date,
    model,
    SUM(cost_usd) as cost,
    COUNT(*) as requests
FROM token_ledger_events
GROUP BY date, model
ORDER BY date DESC;
```

### Option B: Python API

```python
from tokenledger.queries import TokenLedgerQueries

queries = TokenLedgerQueries()
summary = queries.get_cost_summary(days=7)
print(f"This week: ${summary.total_cost:.2f}")
```

### Option C: Dashboard

```bash
# Start the dashboard
docker compose up
# Open http://localhost:3000
```

## Environment Variables

Instead of passing `database_url` directly, you can use environment variables:

```bash
export DATABASE_URL="postgresql://user:pass@localhost/db"
export TOKENLEDGER_APP_NAME="my-app"
export TOKENLEDGER_ENVIRONMENT="production"
```

## Supabase Setup

1. Go to Supabase Dashboard → Settings → Database
2. Copy the connection string
3. Run the migration SQL in SQL Editor
4. Configure TokenLedger:

```python
tokenledger.configure(
    database_url="postgresql://postgres:[password]@db.[ref].supabase.co:5432/postgres"
)
```

## Adding User & Cost Attribution

The real power of TokenLedger is attributing costs to users, organizations, features, and other dimensions. Here's how:

### Using OpenAI's `user` Parameter

The easiest way—pass the `user` parameter that OpenAI already supports:

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
    user="user_123"  # TokenLedger captures this as user_id
)
```

### Setting Default Metadata

For metadata that applies to all requests (org, feature area, service name):

```python
tokenledger.configure(
    database_url="postgresql://...",
    app_name="chat-service",
    environment="production",
    default_metadata={
        "organization_id": "org_456",
        "feature": "customer-support",
        "team": "platform"
    }
)
```

### Using Decorators for Full Control

When you need per-request attribution with more fields:

```python
from tokenledger import track_llm

@track_llm(
    provider="openai",
    user_id="user_123",
    metadata={
        "feature": "summarization",
        "session_id": "sess_abc",
        "customer_tier": "enterprise"
    }
)
def summarize_document(doc: str):
    return client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"Summarize: {doc}"}]
    )
```

### Manual Tracking

For streaming responses or custom integrations:

```python
from tokenledger import track_cost

# After counting tokens from a streaming response
track_cost(
    input_tokens=150,
    output_tokens=500,
    model="gpt-4o",
    user_id="user_123",
    metadata={
        "session_id": "sess_abc",
        "feature": "chat",
        "conversation_id": "conv_xyz"
    }
)
```

### Available Attribution Fields

| Field | Type | Description |
|-------|------|-------------|
| `user_id` | string | Your user identifier |
| `session_id` | string | Session or conversation ID |
| `organization_id` | string | Org/tenant for multi-tenant apps |
| `metadata` | JSON | Any custom dimensions |

### Querying by Attribution

```sql
-- Cost by user
SELECT user_id, SUM(cost_usd) as total_cost
FROM token_ledger_events
WHERE timestamp > NOW() - INTERVAL '7 days'
GROUP BY user_id
ORDER BY total_cost DESC;

-- Cost by feature (from metadata)
SELECT metadata->>'feature' as feature, SUM(cost_usd) as cost
FROM token_ledger_events
WHERE metadata->>'feature' IS NOT NULL
GROUP BY feature;

-- Cost by organization
SELECT organization_id, COUNT(*) as requests, SUM(cost_usd) as cost
FROM token_ledger_events
GROUP BY organization_id;
```

## What's Tracked

For every LLM call, TokenLedger captures:

| Field | Description |
|-------|-------------|
| `timestamp` | When the call was made |
| `provider` | openai, anthropic, etc. |
| `model` | gpt-4o, claude-3-5-sonnet, etc. |
| `input_tokens` | Prompt tokens |
| `output_tokens` | Completion tokens |
| `cost_usd` | Calculated cost |
| `duration_ms` | Response latency |
| `user_id` | Your user identifier |
| `session_id` | Session or conversation ID |
| `organization_id` | Org/tenant identifier |
| `status` | success or error |
| `metadata` | Custom JSON data (feature, team, etc.) |

## Next Steps

- [Full API Reference](api-reference.md)
- [Framework Integration](frameworks.md)
- [Dashboard Setup](dashboard.md)
- [Cost Optimization Tips](optimization.md)
