# 🪙 TokenLedger

**Know exactly what your AI features cost, per user, per endpoint, per day.**

[![CI](https://github.com/ged1182/tokenledger/actions/workflows/ci.yml/badge.svg)](https://github.com/ged1182/tokenledger/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ged1182/tokenledger/branch/main/graph/badge.svg)](https://codecov.io/gh/ged1182/tokenledger)
[![PyPI version](https://badge.fury.io/py/tokenledger.svg)](https://badge.fury.io/py/tokenledger)
[![License: ELv2](https://img.shields.io/badge/License-Elastic%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

<p align="center">
  <img src="docs/dashboard-preview.png" alt="TokenLedger Dashboard" width="800"/>
</p>

TokenLedger is a **self-hosted** LLM cost analytics solution that runs on your existing Postgres database. Zero external dependencies, complete data ownership, works with Supabase out of the box.

## ✨ Why TokenLedger?

Every startup building AI features is bleeding money on LLM APIs with **zero visibility**:

- 💸 "How much are we spending on OpenAI?" → *Nobody knows*
- 📊 "Which users are costing us the most?" → *No idea*
- 🎯 "What's our cost per feature?" → *Can't tell you*

Existing solutions (Helicone, LangSmith, Langfuse) are either:
- **SaaS** — Your data leaves your infrastructure
- **Heavy** — Require significant setup and infrastructure
- **Expensive** — Per-seat pricing adds up fast

**TokenLedger is different:**
- ✅ **Postgres-native** — Works with your existing database (Supabase, Neon, RDS)
- ✅ **Self-hosted** — Your data never leaves your infrastructure  
- ✅ **Zero overhead** — 2-line integration, async batching
- ✅ **Cost-aware** — Automatic cost calculation with up-to-date pricing

## 🚀 Quick Start

### Installation

```bash
pip install tokenledger
```

### 2-Line Integration

```python
import tokenledger
import openai

# Configure once
tokenledger.configure(database_url="postgresql://...")
tokenledger.patch_openai()

# That's it! All calls are now tracked
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

Every OpenAI call is now automatically logged to your Postgres database with:
- Token counts (input, output, cached)
- Cost in USD
- Latency
- Model used
- User ID (if provided)
- Full request/response metadata

### Works with Anthropic too

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

### Cost Attribution

Know exactly **who** is spending money and **which features** are driving costs:

```python
from tokenledger import attribution

# Context manager - all calls inside are attributed
with attribution(user_id="user_123", feature="summarize", team="ml"):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Summarize this..."}]
    )

# Decorator - attribute entire functions
@attribution(feature="chat", cost_center="CC-001")
def handle_chat(user_id: str, message: str):
    with attribution(user_id=user_id):  # Contexts nest and merge
        return client.chat.completions.create(...)
```

Query your costs by any dimension:

```sql
SELECT feature, team, SUM(cost_usd) as cost
FROM token_ledger_events
WHERE timestamp >= NOW() - INTERVAL '7 days'
GROUP BY feature, team
ORDER BY cost DESC;
```

## 📊 Dashboard

TokenLedger includes a beautiful React dashboard:

```bash
# Start with Docker
docker compose up

# Open http://localhost:3000
```

Or run the API server standalone:

```bash
pip install tokenledger[server]
python -m tokenledger.server
```

## 🔧 Configuration Options

```python
import tokenledger

tokenledger.configure(
    # Database connection
    database_url="postgresql://user:pass@localhost/db",
    
    # App identification
    app_name="my-app",
    environment="production",
    
    # Performance tuning
    batch_size=100,           # Events per batch write
    flush_interval_seconds=5,  # How often to flush
    async_mode=True,          # Background logging
    
    # Sampling for high-volume apps
    sample_rate=1.0,          # 1.0 = log everything
)
```

## 📈 Querying Your Data

### Using the Python API

```python
from tokenledger.queries import TokenLedgerQueries

queries = TokenLedgerQueries()

# Get cost summary
summary = queries.get_cost_summary(days=30)
print(f"Last 30 days: ${summary.total_cost:.2f}")
print(f"Total requests: {summary.total_requests}")

# Cost by model
models = queries.get_costs_by_model(days=30)
for m in models:
    print(f"{m.model}: ${m.total_cost:.2f} ({m.total_requests} requests)")

# Cost by user
users = queries.get_costs_by_user(days=30)
for u in users[:5]:
    print(f"{u.user_id}: ${u.total_cost:.2f}")

# Daily trends
daily = queries.get_daily_costs(days=7)
for d in daily:
    print(f"{d.date}: ${d.total_cost:.2f}")
```

### Direct SQL

```sql
-- Daily costs by model
SELECT 
    DATE(timestamp) as date,
    model,
    SUM(cost_usd) as total_cost,
    COUNT(*) as requests
FROM token_ledger_events
WHERE timestamp >= NOW() - INTERVAL '30 days'
GROUP BY DATE(timestamp), model
ORDER BY date DESC, total_cost DESC;

-- Top 10 users by cost
SELECT 
    user_id,
    SUM(cost_usd) as total_cost,
    COUNT(*) as requests
FROM token_ledger_events
WHERE timestamp >= NOW() - INTERVAL '30 days'
GROUP BY user_id
ORDER BY total_cost DESC
LIMIT 10;

-- Projected monthly cost
SELECT 
    (SUM(cost_usd) / 7) * 30 as projected_monthly
FROM token_ledger_events
WHERE timestamp >= NOW() - INTERVAL '7 days';
```

## 🔌 Framework Integration

### FastAPI

```python
from fastapi import FastAPI
from tokenledger.middleware import FastAPIMiddleware

app = FastAPI()
app.add_middleware(FastAPIMiddleware)

# User ID from X-User-ID header is automatically tracked
```

### Flask

```python
from flask import Flask
from tokenledger.middleware import TokenLedger

app = Flask(__name__)
TokenLedger(app)
```

### Manual Tracking

```python
from tokenledger import track_cost

# Track manually if you need to
track_cost(
    input_tokens=150,
    output_tokens=500,
    model="gpt-4o",
    user_id="user_123",
)
```

## 🐘 Supabase Setup

TokenLedger works perfectly with Supabase:

1. Get your connection string from Supabase Dashboard → Settings → Database

2. Run the migration:
```sql
-- Copy contents of migrations/001_initial.sql to Supabase SQL Editor
```

3. Configure TokenLedger:
```python
tokenledger.configure(
    database_url="postgresql://postgres:password@db.xxx.supabase.co:5432/postgres"
)
```

## 📁 Project Structure

```
tokenledger/
├── tokenledger/           # Python package
│   ├── __init__.py       # Main exports
│   ├── config.py         # Configuration
│   ├── tracker.py        # Core tracking logic
│   ├── pricing.py        # LLM pricing data
│   ├── queries.py        # Analytics queries
│   ├── decorators.py     # @track_llm decorator
│   ├── middleware.py     # FastAPI/Flask middleware
│   ├── server.py         # Dashboard API server
│   └── interceptors/     # SDK patches
│       ├── openai.py
│       └── anthropic.py
├── dashboard/            # React dashboard
├── migrations/           # SQL migrations
└── examples/             # Usage examples
```

## 💰 Supported Models & Pricing

TokenLedger includes up-to-date pricing for:

**OpenAI:**
- GPT-4o, GPT-4o-mini
- GPT-4-turbo, GPT-4
- GPT-3.5-turbo
- o1, o1-mini
- Embeddings (text-embedding-3-small/large)

**Anthropic:**
- Claude 3.5 Sonnet, Claude 3.5 Haiku
- Claude 3 Opus, Claude 3 Sonnet, Claude 3 Haiku
- Prompt caching support

**Coming Soon:**
- Google Gemini
- Mistral
- Custom/self-hosted models

## 🛠 Development

```bash
# Clone the repo
git clone https://github.com/yourusername/tokenledger
cd tokenledger

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Start local development
docker compose up postgres
python -m tokenledger.server
```

## 🗺 Roadmap

- [ ] Alerts & notifications (budget thresholds)
- [x] Cost allocation tags (feature, team, project, cost_center)
- [x] Team/project grouping via attribution context
- [ ] Grafana integration
- [ ] CLI for querying
- [ ] More LLM providers (Gemini, Mistral, Cohere)
- [ ] Streaming cost tracking improvements
- [ ] TimescaleDB optimization guide

## 📜 License

TokenLedger is licensed under the [Elastic License 2.0 (ELv2)](LICENSE).

**What this means:**
- ✅ **Free to use** — Use TokenLedger in your projects, even commercial ones
- ✅ **Modify freely** — Fork it, extend it, make it yours
- ✅ **Self-host** — Run it on your own infrastructure
- ❌ **No SaaS** — You cannot offer TokenLedger as a hosted/managed service

This license protects the project while keeping it free for the community.

## 🙏 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) first.

---

<p align="center">
  Built with ❤️ for the AI startup community
</p>
