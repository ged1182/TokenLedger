"""
TokenLedger Basic Usage Example
================================

This example shows how to integrate TokenLedger with your existing
OpenAI or Anthropic code in just 2 lines.
"""

import os

# ============================================
# SETUP: Run this once at startup
# ============================================

import tokenledger

# Option 1: Configure with connection string
tokenledger.configure(
    database_url=os.getenv("DATABASE_URL", "postgresql://localhost/mydb"),
    app_name="my-app",
    environment="development",
)

# Patch the SDK you're using (choose one or both)
tokenledger.patch_openai()      # For OpenAI
# tokenledger.patch_anthropic() # For Anthropic


# ============================================
# YOUR EXISTING CODE - No changes needed!
# ============================================

import openai

client = openai.OpenAI()

# This call is automatically tracked
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ],
    user="user_123"  # Optional: track by user
)

print(response.choices[0].message.content)


# ============================================
# ANTHROPIC EXAMPLE
# ============================================

# import anthropic
# 
# tokenledger.patch_anthropic()
# 
# client = anthropic.Anthropic()
# 
# response = client.messages.create(
#     model="claude-3-5-sonnet-20241022",
#     max_tokens=1024,
#     messages=[
#         {"role": "user", "content": "What is the capital of France?"}
#     ]
# )
# 
# print(response.content[0].text)


# ============================================
# QUERYING YOUR DATA
# ============================================

from tokenledger.queries import TokenLedgerQueries

queries = TokenLedgerQueries()

# Get cost summary for last 30 days
summary = queries.get_cost_summary(days=30)
print(f"\n📊 Last 30 Days Summary:")
print(f"   Total Cost: ${summary.total_cost:.2f}")
print(f"   Total Requests: {summary.total_requests}")
print(f"   Avg per Request: ${summary.avg_cost_per_request:.4f}")

# Get costs by model
print(f"\n📈 Costs by Model:")
for model in queries.get_costs_by_model(days=30, limit=5):
    print(f"   {model.model}: ${model.total_cost:.2f} ({model.total_requests} requests)")

# Get top users
print(f"\n👥 Top Users by Cost:")
for user in queries.get_costs_by_user(days=30, limit=5):
    print(f"   {user.user_id}: ${user.total_cost:.2f}")


# ============================================
# USING DECORATORS (Alternative approach)
# ============================================

from tokenledger import track_llm

@track_llm(provider="openai", user_id="user_456")
def my_ai_function(prompt: str):
    """Your AI function - automatically tracked"""
    return client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )


# ============================================
# MANUAL TRACKING (For edge cases)
# ============================================

from tokenledger import track_cost

# If you need to track something manually
track_cost(
    input_tokens=150,
    output_tokens=500,
    model="gpt-4o",
    user_id="user_789",
    metadata={"feature": "summarization"}
)


# ============================================
# FLUSH ON SHUTDOWN (Optional but recommended)
# ============================================

# TokenLedger batches writes for performance.
# It automatically flushes on shutdown, but you can force it:

from tokenledger.tracker import get_tracker
tracker = get_tracker()
tracker.flush()  # Ensure all events are written
