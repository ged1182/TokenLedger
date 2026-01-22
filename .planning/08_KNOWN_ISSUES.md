# TokenLedger Known Issues & Future Fixes

**Status:** Tracking
**Last Updated:** January 2026

This document tracks known issues discovered during development and integration testing that should be addressed in future phases.

---

## Interceptor Issues

### 1. Anthropic `AsyncMessages.stream` Not Patched

**Priority:** HIGH
**File:** `tokenledger/interceptors/anthropic.py`
**Discovered:** January 2026 (Glass Box Portfolio integration)

**Issue:**
The main `messages.AsyncMessages.stream` method is NOT patched for tracking, while:
- `messages.Messages.stream` (sync) IS patched (line 733-735)
- `beta_messages.AsyncMessages.stream` IS patched (line 654-658)

This means async streaming for the regular Anthropic API is not tracked.

**Root Cause:**
Missing patch in `patch_anthropic()` function. The `_wrap_async_streaming_messages` wrapper exists but is only applied to beta.messages.

**Fix Required:**
Add after line 735 in `patch_anthropic()`:
```python
if track_streaming and hasattr(messages.AsyncMessages, "stream"):
    _original_methods["async_messages_stream"] = messages.AsyncMessages.stream
    messages.AsyncMessages.stream = _wrap_async_streaming_messages(
        messages.AsyncMessages.stream
    )
```

Also update `unpatch_anthropic()` to restore this method.

**Testing:**
```python
import asyncio
import anthropic
import tokenledger

tokenledger.configure(database_url="postgresql://...")
tokenledger.patch_anthropic()

async def test():
    client = anthropic.AsyncAnthropic()
    async with client.messages.stream(
        model="claude-3-5-sonnet-20241022",
        max_tokens=100,
        messages=[{"role": "user", "content": "Hello"}]
    ) as stream:
        async for text in stream.text_stream:
            print(text, end="")

asyncio.run(test())
# Verify event appears in token_ledger_events
```

---

### 2. OpenAI Streaming Not Implemented

**Priority:** MEDIUM
**File:** `tokenledger/interceptors/openai.py`
**Status:** Documented in README roadmap

**Issue:**
OpenAI streaming (`stream=True`) is not tracked. This is noted in the README:
> `- [ ] OpenAI streaming support (Anthropic & Google streaming complete)`

**Implementation Notes:**
- OpenAI streaming returns an iterator/async iterator
- Need to wrap similar to Google's `_wrap_generate_content_stream` / `_wrap_async_generate_content_stream`
- Token counts come in final chunk or via `stream_options={"include_usage": True}`

---

## Schema/Migration Issues

### 3. Attribution Columns Added in Separate Migration

**Priority:** LOW (RESOLVED in PR #30)
**Status:** Fixed

The attribution columns (`feature`, `page`, `component`, `team`, `project`, `cost_center`) were missing from `001_initial.sql` but present in `LLMEvent` and `AttributionContext`.

**Resolution:** Added `migrations/002_add_attribution_columns.sql`

---

## Documentation Issues

### 4. Database Driver Clarity

**Priority:** LOW (RESOLVED in PR #30)
**Status:** Fixed

Documentation didn't clearly explain when to use `psycopg2` vs `asyncpg`.

**Resolution:** Updated `docs/quickstart.md` with driver selection guidance.

---

## Testing Gaps

### 5. Integration Tests for Streaming APIs

**Priority:** MEDIUM

Need integration tests that verify:
- Google async streaming tracking works end-to-end
- Anthropic sync/async streaming tracking
- Token counts are correctly accumulated from stream chunks
- Events are recorded with correct cost after stream completes

---

## Performance Considerations

### 6. Stream Wrapper Memory Usage

**Priority:** LOW

The stream wrappers accumulate response text in memory (`self._response_text.append(text)`). For very long responses, this could be problematic.

**Potential Fix:**
- Limit accumulated text to first N characters
- Only store preview, not full response
- Current limit is 10000 chars during streaming, truncated to 500 for storage

---

## References

- PR #30: Fix async streaming and schema mismatch bugs
- Glass Box Portfolio integration notes
