#!/usr/bin/env python3
"""Generate realistic mock data for TokenLedger dashboard demo."""

import random
import uuid
from datetime import datetime, timedelta, timezone

import psycopg2

# Connect to local database
conn = psycopg2.connect("postgresql://tokenledger:tokenledger@localhost:5432/tokenledger")
cur = conn.cursor()

# Configuration
NUM_EVENTS = 1500
DAYS_BACK = 90

# Realistic data pools
USERS = ["user_alice", "user_bob", "user_charlie", "user_diana", "user_eve", "user_frank"]
FEATURES = ["chat", "summarize", "search", "code_review", "translate", "analyze"]
PAGES = ["/dashboard", "/api/chat", "/api/summarize", "/docs", "/playground", "/admin"]
TEAMS = ["platform", "ml", "product", "growth", "enterprise"]
PROJECTS = ["web-app", "api", "mobile", "internal-tools"]
COST_CENTERS = ["ENG-001", "ENG-002", "ML-001", "PROD-001"]
ENVIRONMENTS = ["production", "staging", "development"]

# Models with realistic token ranges (2025/2026 models)
MODELS = {
    "openai": {
        "gpt-4.5-preview": {"input_range": (1000, 8000), "output_range": (500, 4000), "weight": 15},
        "gpt-4o": {"input_range": (500, 4000), "output_range": (100, 2000), "weight": 25},
        "gpt-4o-mini": {"input_range": (200, 2000), "output_range": (50, 1000), "weight": 30},
        "o1": {"input_range": (2000, 10000), "output_range": (1000, 8000), "weight": 10},
        "o1-mini": {"input_range": (1000, 5000), "output_range": (500, 3000), "weight": 15},
        "o3-mini": {"input_range": (1000, 6000), "output_range": (500, 4000), "weight": 5},
    },
    "anthropic": {
        "claude-opus-4": {"input_range": (2000, 12000), "output_range": (500, 5000), "weight": 10},
        "claude-sonnet-4": {"input_range": (1000, 8000), "output_range": (300, 3000), "weight": 20},
        "claude-3.5-sonnet": {"input_range": (1000, 6000), "output_range": (300, 2500), "weight": 35},
        "claude-3.5-haiku": {"input_range": (200, 2000), "output_range": (50, 1000), "weight": 25},
        "claude-3-opus": {"input_range": (2000, 10000), "output_range": (500, 4000), "weight": 5},
        "claude-3-haiku": {"input_range": (200, 2000), "output_range": (50, 1000), "weight": 5},
    },
    "google": {
        "gemini-2.0-flash": {"input_range": (500, 4000), "output_range": (200, 2000), "weight": 40},
        "gemini-2.0-pro": {"input_range": (1000, 8000), "output_range": (500, 4000), "weight": 30},
        "gemini-1.5-pro": {"input_range": (1000, 6000), "output_range": (300, 3000), "weight": 20},
        "gemini-1.5-flash": {"input_range": (300, 2000), "output_range": (100, 1000), "weight": 10},
    },
}

# Pricing per 1M tokens (input, output) - 2025/2026 pricing
PRICING = {
    # OpenAI
    "gpt-4.5-preview": (75.00, 150.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "o1": (15.00, 60.00),
    "o1-mini": (3.00, 12.00),
    "o3-mini": (1.10, 4.40),
    # Anthropic
    "claude-opus-4": (15.00, 75.00),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-3.5-sonnet": (3.00, 15.00),
    "claude-3.5-haiku": (0.80, 4.00),
    "claude-3-opus": (15.00, 75.00),
    "claude-3-haiku": (0.25, 1.25),
    # Google
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.0-pro": (1.25, 5.00),
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-1.5-flash": (0.075, 0.30),
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD."""
    if model not in PRICING:
        return 0.0
    input_price, output_price = PRICING[model]
    return (input_tokens * input_price / 1_000_000) + (output_tokens * output_price / 1_000_000)


def generate_event(base_time: datetime) -> dict:
    """Generate a single mock event."""
    # Pick provider with weights (OpenAI 40%, Anthropic 45%, Google 15%)
    provider = random.choices(
        list(MODELS.keys()),
        weights=[40, 45, 15],
        k=1
    )[0]

    # Pick model based on weights within provider
    models = list(MODELS[provider].keys())
    weights = [MODELS[provider][m]["weight"] for m in models]
    model = random.choices(models, weights=weights, k=1)[0]
    model_config = MODELS[provider][model]

    # Generate tokens
    input_tokens = random.randint(*model_config["input_range"])
    output_tokens = random.randint(*model_config["output_range"])
    total_tokens = input_tokens + output_tokens

    # Sometimes add cached tokens
    cached_tokens = random.randint(0, input_tokens // 3) if random.random() > 0.7 else 0

    # Calculate cost
    cost_usd = calculate_cost(model, input_tokens - cached_tokens, output_tokens)

    # Status (mostly success)
    status = random.choices(["success", "error", "timeout"], weights=[0.95, 0.03, 0.02])[0]
    error_type = None
    error_message = None
    if status == "error":
        error_type = random.choice(["rate_limit", "context_length", "invalid_request"])
        error_message = f"Error: {error_type} occurred"
    elif status == "timeout":
        error_type = "timeout"
        error_message = "Request timed out after 60s"

    # Random timestamp within the time window
    offset_seconds = random.randint(0, 86400)  # Within 24 hours
    timestamp = base_time + timedelta(seconds=offset_seconds)

    return {
        "event_id": str(uuid.uuid4()),
        "timestamp": timestamp.isoformat(),
        "provider": provider,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached_tokens,
        "cost_usd": round(cost_usd, 6),
        "request_type": "chat",
        "user_id": random.choice(USERS),
        "session_id": str(uuid.uuid4()),
        "app_name": "tokenledger-demo",
        "environment": random.choice(ENVIRONMENTS),
        "status": status,
        "error_type": error_type,
        "error_message": error_message,
        "feature": random.choice(FEATURES),
        "page": random.choice(PAGES),
        "team": random.choice(TEAMS),
        "project": random.choice(PROJECTS),
        "cost_center": random.choice(COST_CENTERS),
        "duration_ms": random.randint(200, 15000),
    }


def insert_events():
    """Generate and insert mock events."""
    # Clear existing data first
    print("Clearing existing data...")
    cur.execute("TRUNCATE token_ledger.token_ledger_events;")
    conn.commit()

    print(f"Generating {NUM_EVENTS} mock events over {DAYS_BACK} days...")

    events = []
    now = datetime.now(timezone.utc)

    for day_offset in range(DAYS_BACK):
        base_time = now - timedelta(days=day_offset)
        # More events on weekdays
        is_weekday = base_time.weekday() < 5
        daily_events = random.randint(15, 25) if is_weekday else random.randint(5, 12)

        for _ in range(daily_events):
            events.append(generate_event(base_time))

    # Insert in batches
    insert_sql = """
        INSERT INTO token_ledger.token_ledger_events (
            event_id, timestamp, provider, model, input_tokens, output_tokens,
            total_tokens, cached_tokens, cost_usd, request_type, user_id,
            session_id, app_name, environment, status, error_type, error_message,
            feature, page, team, project, cost_center, duration_ms
        ) VALUES (
            %(event_id)s, %(timestamp)s, %(provider)s, %(model)s, %(input_tokens)s,
            %(output_tokens)s, %(total_tokens)s, %(cached_tokens)s, %(cost_usd)s,
            %(request_type)s, %(user_id)s, %(session_id)s, %(app_name)s,
            %(environment)s, %(status)s, %(error_type)s, %(error_message)s,
            %(feature)s, %(page)s, %(team)s, %(project)s, %(cost_center)s,
            %(duration_ms)s
        )
    """

    for event in events:
        cur.execute(insert_sql, event)

    conn.commit()
    print(f"Inserted {len(events)} events successfully!")

    # Print summary
    cur.execute("SELECT COUNT(*), SUM(cost_usd) FROM token_ledger.token_ledger_events")
    count, total_cost = cur.fetchone()
    print(f"\nTotal events: {count}")
    print(f"Total cost: ${total_cost:.2f}")

    # Print by provider
    cur.execute("""
        SELECT provider, COUNT(*), SUM(cost_usd)
        FROM token_ledger.token_ledger_events
        GROUP BY provider
        ORDER BY SUM(cost_usd) DESC
    """)
    print("\nBy provider:")
    for provider, count, cost in cur.fetchall():
        print(f"  {provider}: {count} events, ${cost:.2f}")


if __name__ == "__main__":
    insert_events()
    cur.close()
    conn.close()
