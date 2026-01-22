# Anthropic SDK Analysis for TokenLedger

> **SDK Version**: 0.76.0
> **Analysis Date**: 2026-01-22
> **Current Coverage**: ~13% (3 of 23 cost-bearing methods)

## Overview

The Anthropic Python SDK has **two separate Messages APIs**: standard (`client.messages`) and beta (`client.beta.messages`). This is critical because **pydantic-ai and most frameworks use the beta API**, which TokenLedger currently does NOT patch.

## Critical Finding

```
⚠️ PYDANTIC-AI USES client.beta.messages.create()
⚠️ TOKENLEDGER ONLY PATCHES client.messages.create()
⚠️ THESE ARE DIFFERENT CLASSES - NO COVERAGE FOR PYDANTIC-AI!
```

## SDK Structure

```
anthropic/resources/
├── messages/                    # Standard Messages API
│   ├── messages.py             # Messages, AsyncMessages (PATCHED)
│   └── batches.py              # Batches (NOT PATCHED)
├── beta/                        # Beta Features
│   ├── messages/               # Beta Messages API
│   │   ├── messages.py         # Messages, AsyncMessages (NOT PATCHED - CRITICAL!)
│   │   └── batches.py          # Beta Batches (NOT PATCHED)
│   ├── files.py
│   └── models.py
└── completions.py              # Legacy (NOT PATCHED)
```

## Two Separate Message Classes

| Aspect | Standard Messages | Beta Messages |
|--------|-------------------|---------------|
| **Access Path** | `client.messages.create()` | `client.beta.messages.create()` |
| **Module** | `anthropic.resources.messages.Messages` | `anthropic.resources.beta.messages.Messages` |
| **Used By** | Direct simple API usage | pydantic-ai, LangChain, frameworks |
| **Features** | Basic messages | Tools, structured output, MCP, thinking |
| **TokenLedger** | ✅ Patched | ❌ NOT Patched |

## Currently Patched Methods

| Resource | Method | Sync | Async | Module Path |
|----------|--------|------|-------|-------------|
| messages | `create()` | ✅ | ✅ | `anthropic.resources.messages.Messages` |
| messages | `stream()` | ✅ | ❌ | `anthropic.resources.messages.Messages` |

## Gap Analysis: Missing Methods

### Tier 1 - CRITICAL (pydantic-ai compatibility)

| Resource | Method | Sync | Async | Priority | Notes |
|----------|--------|------|-------|----------|-------|
| **beta.messages** | `create()` | ✅ | ✅ | 🔴 CRITICAL | pydantic-ai main entry point |
| **beta.messages** | `parse()` | ✅ | ✅ | 🔴 CRITICAL | Structured output |
| **beta.messages** | `stream()` | ✅ | ✅ | 🔴 CRITICAL | Beta streaming |
| **beta.messages** | `tool_runner()` | ✅ | ✅ | 🔴 CRITICAL | Tool execution |

### Tier 2 - High Priority

| Resource | Method | Sync | Async | Priority | Notes |
|----------|--------|------|-------|----------|-------|
| messages | `stream()` | ❌ | ✅ | 🟠 HIGH | Async streaming missing |
| messages | `count_tokens()` | ✅ | ✅ | 🟡 MEDIUM | Token estimation (free) |
| beta.messages | `count_tokens()` | ✅ | ✅ | 🟡 MEDIUM | Beta token estimation |
| messages.batches | `create()` | ✅ | ✅ | 🟡 MEDIUM | Batch processing |
| beta.messages.batches | `create()` | ✅ | ✅ | 🟡 MEDIUM | Beta batch processing |

### Tier 3 - Lower Priority

| Resource | Method | Sync | Async | Priority | Notes |
|----------|--------|------|-------|----------|-------|
| completions | `create()` | ✅ | ✅ | 🟢 LOW | Legacy, deprecated |

## Token Extraction

### Standard/Beta Message Response
```python
response.usage.input_tokens           # Input token count
response.usage.output_tokens          # Output token count
response.usage.cache_read_input_tokens # Cached tokens (prompt caching)
response.model                         # Actual model used
response.stop_reason                   # "end_turn", "max_tokens", "error"
```

### Streaming Response Chunks
```python
# message_start event
chunk.message.usage.input_tokens      # Input tokens at start

# content_block_delta events
chunk.delta.text                      # Accumulate response text

# message_delta event
chunk.usage.output_tokens             # Output tokens at end
```

### Beta ParsedMessage Response
```python
response.usage.input_tokens           # Same as standard
response.usage.output_tokens          # Same as standard
response.parsed                       # Structured parsed output (T)
```

## Implementation: Patching Beta Messages

Add to `patch_anthropic()`:

```python
def patch_anthropic(client=None, track_streaming=True):
    global _patched

    if _patched:
        return

    # Current: Standard Messages API
    from anthropic.resources import messages
    _original_methods["messages_create"] = messages.Messages.create
    messages.Messages.create = _wrap_messages_create(messages.Messages.create)

    _original_methods["async_messages_create"] = messages.AsyncMessages.create
    messages.AsyncMessages.create = _wrap_async_messages_create(messages.AsyncMessages.create)

    # NEW: Beta Messages API (CRITICAL for pydantic-ai)
    from anthropic.resources.beta import messages as beta_messages

    _original_methods["beta_messages_create"] = beta_messages.Messages.create
    beta_messages.Messages.create = _wrap_messages_create(beta_messages.Messages.create)

    _original_methods["beta_async_messages_create"] = beta_messages.AsyncMessages.create
    beta_messages.AsyncMessages.create = _wrap_async_messages_create(beta_messages.AsyncMessages.create)

    # Beta parse() method
    if hasattr(beta_messages.Messages, "parse"):
        _original_methods["beta_messages_parse"] = beta_messages.Messages.parse
        beta_messages.Messages.parse = _wrap_messages_create(beta_messages.Messages.parse)

    if hasattr(beta_messages.AsyncMessages, "parse"):
        _original_methods["beta_async_messages_parse"] = beta_messages.AsyncMessages.parse
        beta_messages.AsyncMessages.parse = _wrap_async_messages_create(beta_messages.AsyncMessages.parse)

    # Beta stream() method
    if track_streaming:
        if hasattr(beta_messages.Messages, "stream"):
            _original_methods["beta_messages_stream"] = beta_messages.Messages.stream
            beta_messages.Messages.stream = _wrap_streaming_messages(beta_messages.Messages.stream)

    _patched = True
```

## Coverage Summary

| API | create() | parse() | stream() | count_tokens() | batches |
|-----|----------|---------|----------|----------------|---------|
| **Standard Messages** | ✅ | N/A | ✅ sync only | ❌ | ❌ |
| **Standard Async** | ✅ | N/A | ❌ | ❌ | ❌ |
| **Beta Messages** | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Beta Async** | ❌ | ❌ | ❌ | ❌ | ❌ |

**Current**: 3/23 methods = 13%
**After Beta Fix**: 11/23 methods = 48%
**Full Coverage Target**: 23/23 methods = 100%

## Verification: pydantic-ai Uses Beta

From `/tmp/pydantic-ai/pydantic_ai_slim/pydantic_ai/models/anthropic.py`:

```python
# Line 413 - This is what pydantic-ai calls:
return await self.client.beta.messages.create(
    max_tokens=model_settings.get('max_tokens', 4096),
    system=system_prompt or OMIT,
    messages=anthropic_messages,
    model=self._model_name,
    tools=tools or OMIT,
    ...
)
```

## References

- Anthropic SDK: `/tmp/anthropic-sdk/src/anthropic/`
- TokenLedger interceptor: `tokenledger/interceptors/anthropic.py`
- pydantic-ai Anthropic model: `/tmp/pydantic-ai/pydantic_ai_slim/pydantic_ai/models/anthropic.py:413`
